"""
Autoresearch-robotics training script. Single-GPU, single-file.
SAC + HER for goal-conditioned robotic manipulation.
Usage: uv run train.py
"""

import os
import gc
import time
import copy
import json
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, ENV_ID,
    make_env, flatten_obs, get_obs_dim, get_action_dim, get_action_bounds,
)
from evaluate import (
    EVAL_EPISODES, RENDER_EPISODES,
    evaluate, render_episodes,
)

# ---------------------------------------------------------------------------
# EVERYTHING BELOW IS YOURS TO REWRITE
# ---------------------------------------------------------------------------
# This file contains a baseline SAC+HER implementation. You can rewrite any
# part of it: the RL algorithm, the policy architecture, the replay buffer,
# the optimizer, the training loop. The hyperparameters below and the classes
# that follow are a starting point, not a constraint.
# ---------------------------------------------------------------------------

# Hyperparameters (baseline values — feel free to change, add, remove, or
# replace with a completely different configuration scheme)

# Policy architecture
HIDDEN_DIM = 256            # hidden layer width
N_LAYERS = 3                # number of hidden layers
ACTIVATION = "relu"         # activation function: "relu", "tanh", "gelu"

# SAC optimization
LR_ACTOR = 3e-4             # actor learning rate
LR_CRITIC = 3e-4            # critic learning rate
GAMMA = 0.98                # discount factor
TAU = 0.005                 # soft target update rate
INIT_ALPHA = 0.2            # initial entropy coefficient
AUTO_ALPHA = True           # automatic entropy tuning
LR_ALPHA = 3e-4             # alpha learning rate (if AUTO_ALPHA)

# Replay buffer
BATCH_SIZE = 256            # minibatch size for updates
BUFFER_SIZE = 200_000       # replay buffer capacity

# HER (Hindsight Experience Replay)
USE_HER = True              # enable HER for sparse rewards
HER_K = 4                   # future goals per transition
HER_STRATEGY = "future"     # "future" or "final"

# Training schedule
STEPS_BEFORE_LEARNING = 1000    # random exploration steps before learning
UPDATE_EVERY = 1                # gradient steps per env step
N_UPDATES = 1                   # number of gradient updates per update cycle
WARMUP_STEPS = 0                # steps with random actions after learning starts

# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Online running mean/variance using Welford's algorithm."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        """Update with a single observation or batch of observations."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        """Normalize observation, clipping to [-5, 5]."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------

def get_activation(name):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[name]


class Actor(nn.Module):
    """Gaussian policy: obs -> (mean, log_std) -> action via reparameterization."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, action_dim, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, activation=ACTIVATION):
        super().__init__()
        act_cls = get_activation(activation)
        layers = [nn.Linear(obs_dim, hidden_dim), act_cls()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """Sample action with reparameterization trick. Returns (action, log_prob)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log prob with tanh squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs, deterministic=False):
        """Get action for a single observation (numpy in, numpy out)."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
            if deterministic:
                mean, _ = self.forward(obs_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.sample(obs_t)
            return action.cpu().numpy()[0]


class Critic(nn.Module):
    """Twin Q-networks: (obs, action) -> (Q1, Q2)."""

    def __init__(self, obs_dim, action_dim, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, activation=ACTIVATION):
        super().__init__()
        act_cls = get_activation(activation)
        input_dim = obs_dim + action_dim

        # Q1
        q1_layers = [nn.Linear(input_dim, hidden_dim), act_cls()]
        for _ in range(n_layers - 1):
            q1_layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2
        q2_layers = [nn.Linear(input_dim, hidden_dim), act_cls()]
        for _ in range(n_layers - 1):
            q2_layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# ---------------------------------------------------------------------------
# Replay buffer with HER (fixed episode tracking)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Replay buffer with optional Hindsight Experience Replay (HER).

    Stores full goal-conditioned transitions and applies HER relabeling on sample.
    Uses per-transition episode_id tracking for O(1) episode lookup and correct
    handling of circular buffer wraparound.
    """

    def __init__(self, capacity, obs_dim, action_dim, goal_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.ptr = 0
        self.size = 0

        # Store raw components (not flattened) for HER relabeling
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.desired_goals = np.zeros((capacity, goal_dim), dtype=np.float32)

        # Episode tracking for HER — per-transition episode ID for O(1) lookup
        self.episode_ids = np.full(capacity, -1, dtype=np.int32)
        self.episode_boundaries = {}  # episode_id -> (start_idx, length)
        self._current_episode_id = 0
        self._current_episode_start = 0

    def add(self, obs, action, reward, next_obs, done, achieved_goal,
            next_achieved_goal, desired_goal):
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = done
        self.achieved_goals[idx] = achieved_goal
        self.next_achieved_goals[idx] = next_achieved_goal
        self.desired_goals[idx] = desired_goal
        self.episode_ids[idx] = self._current_episode_id

        # Invalidate any old episode that occupied this slot
        # (happens after buffer wraparound)
        old_ep_id = self.episode_ids[idx]
        if old_ep_id != self._current_episode_id and old_ep_id in self.episode_boundaries:
            del self.episode_boundaries[old_ep_id]

        self.episode_ids[idx] = self._current_episode_id

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            # Compute episode length handling wraparound
            if idx >= self._current_episode_start:
                ep_len = idx - self._current_episode_start + 1
            else:
                # Wrapped around — this episode spans the boundary
                ep_len = (self.capacity - self._current_episode_start) + idx + 1

            if ep_len > 0:
                self.episode_boundaries[self._current_episode_id] = (
                    self._current_episode_start, ep_len
                )
            self._current_episode_id += 1
            self._current_episode_start = self.ptr

    def _compute_reward(self, achieved_goal, desired_goal):
        """Sparse reward: -1 if not at goal, 0 if at goal (distance < 0.05)."""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > 0.05).astype(np.float32)

    def sample(self, batch_size, device, obs_normalizer=None,
               use_her=USE_HER, her_k=HER_K):
        """Sample a batch, optionally with HER relabeling.

        Args:
            batch_size: number of transitions to sample
            device: torch device
            obs_normalizer: RunningMeanStd instance for observation normalization
            use_her: whether to apply HER relabeling
            her_k: number of future goals per transition for HER
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        obs = self.observations[indices].copy()
        actions = self.actions[indices]
        rewards = self.rewards[indices].copy()
        next_obs = self.next_observations[indices].copy()
        dones = self.dones[indices].copy()
        desired_goals = self.desired_goals[indices].copy()

        if use_her and self.episode_boundaries:
            # HER: relabel a fraction of transitions with future achieved goals
            n_her = int(batch_size * her_k / (her_k + 1))

            for i in range(n_her):
                idx = indices[i]
                ep_id = self.episode_ids[idx]

                # Skip if episode not tracked (invalidated by wraparound)
                if ep_id not in self.episode_boundaries:
                    continue

                ep_start, ep_len = self.episode_boundaries[ep_id]

                # Compute position within episode
                if idx >= ep_start:
                    pos_in_ep = idx - ep_start
                else:
                    # Wrapped episode
                    pos_in_ep = (self.capacity - ep_start) + idx

                if HER_STRATEGY == "future":
                    # Sample a future transition in the same episode
                    remaining = ep_len - pos_in_ep - 1
                    if remaining <= 0:
                        continue
                    future_offset = np.random.randint(1, remaining + 1)
                    future_idx = (idx + future_offset) % self.capacity
                else:  # "final"
                    future_idx = (ep_start + ep_len - 1) % self.capacity

                # Validate the future index still belongs to the same episode
                if self.episode_ids[future_idx] != ep_id:
                    continue

                # Relabel goal with future achieved goal
                new_goal = self.achieved_goals[future_idx]
                desired_goals[i] = new_goal

                # Recompute reward with new goal
                next_ag = self.next_achieved_goals[idx]
                rewards[i] = self._compute_reward(next_ag, new_goal)

                # Recompute done (success = distance < 0.05)
                dones[i] = float(
                    np.linalg.norm(next_ag - new_goal) < 0.05
                )

        # Flatten observations: concat obs + desired_goal
        flat_obs = np.concatenate([obs, desired_goals], axis=-1)
        flat_next_obs = np.concatenate([next_obs, desired_goals], axis=-1)

        # Apply observation normalization if available
        if obs_normalizer is not None:
            flat_obs = obs_normalizer.normalize(flat_obs)
            flat_next_obs = obs_normalizer.normalize(flat_next_obs)

        return (
            torch.FloatTensor(flat_obs).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(flat_next_obs).to(device),
            torch.FloatTensor(dones).to(device),
        )


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    def __init__(self, obs_dim, action_dim, device, obs_normalizer=None):
        self.device = device
        self.action_dim = action_dim
        self.obs_normalizer = obs_normalizer

        # Networks
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Entropy tuning
        if AUTO_ALPHA:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = INIT_ALPHA

    def select_action(self, obs_dict, deterministic=False):
        """Select action from observation dict (numpy in, numpy out)."""
        flat = flatten_obs(obs_dict)
        if self.obs_normalizer is not None:
            flat = self.obs_normalizer.normalize(flat)
        return self.actor.get_action(flat, deterministic=deterministic)

    def update(self, batch):
        """Single SAC update step. Returns dict of losses."""
        obs, actions, rewards, next_obs, dones = batch

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * GAMMA * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = 0.0
        if AUTO_ALPHA:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        # Soft target update
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1 - TAU)
                p_target.data.add_(TAU * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        }


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train SAC+HER on Fetch robotics tasks")
parser.add_argument("--headless", action="store_true",
                    help="Disable live MuJoCo window during rendering")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Environment info
obs_dim = get_obs_dim(ENV_ID)
action_dim = get_action_dim(ENV_ID)
action_low, action_high = get_action_bounds(ENV_ID)
print(f"Environment: {ENV_ID}")
print(f"Observation dim (flattened): {obs_dim}")
print(f"Action dim: {action_dim}")

# Goal dim (inferred)
env_tmp = make_env(ENV_ID)
obs_tmp, _ = env_tmp.reset()
goal_dim = obs_tmp["desired_goal"].shape[0]
raw_obs_dim = obs_tmp["observation"].shape[0]
env_tmp.close()

# Observation normalizer
obs_normalizer = RunningMeanStd(shape=(obs_dim,))

# Agent
agent = SACAgent(obs_dim, action_dim, device, obs_normalizer=obs_normalizer)
num_params = sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters())
print(f"Total parameters: {num_params:,}")

# Replay buffer
buffer = ReplayBuffer(BUFFER_SIZE, raw_obs_dim, action_dim, goal_dim)

# Training environment
env = make_env(ENV_ID)

print(f"Time budget: {TIME_BUDGET}s")
print(f"HER: {'enabled' if USE_HER else 'disabled'} (k={HER_K})")
print()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
total_steps = 0
total_episodes = 0
total_updates = 0
smooth_critic_loss = 0
smooth_actor_loss = 0

obs, info = env.reset()
episode_step = 0
episode_reward = 0.0

while True:
    t0 = time.time()

    # Select action
    if total_steps < STEPS_BEFORE_LEARNING:
        action = env.action_space.sample()
    else:
        action = agent.select_action(obs, deterministic=False)

    # Environment step
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_step += 1
    episode_reward += reward

    # Update observation normalizer with flattened obs
    flat_obs_for_norm = flatten_obs(obs)
    obs_normalizer.update(flat_obs_for_norm)

    # Store transition
    buffer.add(
        obs=obs["observation"],
        action=action,
        reward=reward,
        next_obs=next_obs["observation"],
        done=float(done),
        achieved_goal=obs["achieved_goal"],
        next_achieved_goal=next_obs["achieved_goal"],
        desired_goal=obs["desired_goal"],
    )

    obs = next_obs

    # Episode reset
    if done:
        total_episodes += 1
        obs, info = env.reset()
        episode_step = 0
        episode_reward = 0.0

    # Learning
    if total_steps >= STEPS_BEFORE_LEARNING and total_steps % UPDATE_EVERY == 0:
        for _ in range(N_UPDATES):
            batch = buffer.sample(BATCH_SIZE, device, obs_normalizer=obs_normalizer,
                                  use_her=USE_HER, her_k=HER_K)
            losses = agent.update(batch)
            total_updates += 1

            # Smoothed losses for logging
            ema = 0.99
            smooth_critic_loss = ema * smooth_critic_loss + (1 - ema) * losses["critic_loss"]
            smooth_actor_loss = ema * smooth_actor_loss + (1 - ema) * losses["actor_loss"]

    total_steps += 1

    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt

    # Logging (every 1000 steps)
    if total_steps % 1000 == 0:
        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)
        debiased_critic = smooth_critic_loss / (1 - 0.99 ** max(total_updates, 1))
        debiased_actor = smooth_actor_loss / (1 - 0.99 ** max(total_updates, 1))
        steps_per_sec = int(total_steps / total_training_time) if total_training_time > 0 else 0

        print(
            f"\rstep {total_steps:06d} ({pct_done:.1f}%) | "
            f"critic: {debiased_critic:.4f} | actor: {debiased_actor:.4f} | "
            f"alpha: {agent.alpha:.3f} | episodes: {total_episodes} | "
            f"buf: {buffer.size:,} | steps/s: {steps_per_sec:,} | "
            f"remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

    # GC management
    if total_steps == 100:
        gc.collect()
        gc.disable()

    # Time's up
    if total_training_time >= TIME_BUDGET:
        break

env.close()
print()  # newline after \r training log

# ---------------------------------------------------------------------------
# Evaluation + Rendering
# ---------------------------------------------------------------------------

print("\n--- Evaluation ---")

# Create policy function for evaluation
def policy_fn(obs_dict):
    return agent.select_action(obs_dict, deterministic=True)

# Quantitative evaluation
metrics = evaluate(policy_fn, ENV_ID, n_episodes=EVAL_EPISODES)
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Mean reward: {metrics['mean_reward']:.2f}")
print(f"Mean distance: {metrics['mean_distance']:.4f}")

# Render episodes
print("\n--- Rendering ---")
render_result = render_episodes(policy_fn, ENV_ID, n_episodes=RENDER_EPISODES,
                                output_dir="./renders", show_window=not args.headless)
print(f"Video: {render_result['video_path']}")
print(f"Key frames: {len(render_result['frame_paths'])}")

# ---------------------------------------------------------------------------
# Update experiment history
history_path = Path("./experiment_history.json")


if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
else:
    history = {
        "best_commit": None,
        "best_success_rate": 0.0,
        "total_experiments": 0,
        "experiments": [],
        "insights": [],
        "failed_directions": [],
        "promising_directions": [],
    }

# Check if this is a new best
is_new_best = metrics["success_rate"] > history["best_success_rate"]
if is_new_best:
    # Save best renders for future visual comparison
    best_dir = Path("./renders_best")
    best_dir.mkdir(exist_ok=True)
    for png in Path("./renders").glob("ep*_*.png"):
        shutil.copy2(str(png), str(best_dir / png.name))
    print(f"New best! Saved renders to {best_dir}")

# Append experiment entry (agent fills in hypothesis/lesson via experiment_history.json)
experiment_entry = {
    "id": history["total_experiments"] + 1,
    "commit": None,  # agent fills this after git commit
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "description": "",  # agent fills this
    "hypothesis": "",  # agent fills this
    "changes_made": [],  # agent fills this
    "metrics": {
        "success_rate": float(metrics["success_rate"]),
        "mean_reward": float(metrics["mean_reward"]),
        "mean_distance": float(metrics["mean_distance"]),
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "total_updates": total_updates,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
    },
    "vlm_feedback_summary": "",  # agent fills from visual_feedback.txt
    "vlm_failure_modes": [],  # agent fills
    "vlm_suggestions": [],  # agent fills
    "status": "",  # agent fills: keep/discard/crash
    "hypothesis_confirmed": None,  # agent fills: true/false/null
    "lesson_learned": "",  # agent fills
}

if is_new_best:
    history["best_success_rate"] = float(metrics["success_rate"])
    # best_commit updated by agent after git commit

history["total_experiments"] += 1
history["experiments"].append(experiment_entry)

with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"Experiment history updated: {history_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

print("\n---")
print(f"eval_success_rate: {metrics['success_rate']:.6f}")
print(f"eval_mean_reward:  {metrics['mean_reward']:.6f}")
print(f"eval_mean_distance:{metrics['mean_distance']:.6f}")
print(f"training_seconds:  {total_training_time:.1f}")
print(f"total_seconds:     {t_end - t_start:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"total_steps:       {total_steps}")
print(f"total_episodes:    {total_episodes}")
print(f"total_updates:     {total_updates}")
print(f"num_params:        {num_params:,}")
print(f"buffer_size:       {buffer.size:,}")
