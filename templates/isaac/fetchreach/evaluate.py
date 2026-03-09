"""
Fixed evaluation infrastructure for autoresearch-robotics (Isaac Sim backend).
Evaluation harness and rendering pipeline.

This file is READ-ONLY — the agent must NOT modify it.

PROTOTYPE: Isaac Sim rendering is not yet configured. The evaluate() function
works with any gymnasium-compatible env. The render_episodes() function needs
Isaac Sim-specific camera/viewport setup.

Usage:
    python evaluate.py --render     # verify rendering pipeline
"""

import os
import sys
from pathlib import Path

import numpy as np

from prepare import make_env, flatten_obs, get_action_dim, ENV_ID, FRAME_WIDTH, FRAME_HEIGHT

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

EVAL_EPISODES = 10           # episodes for quantitative evaluation
RENDER_EPISODES = 3          # episodes to render for visual analysis

# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(policy_fn, env_id=ENV_ID, n_episodes=EVAL_EPISODES):
    """Run evaluation episodes and compute metrics.

    Args:
        policy_fn: callable(obs_dict) -> action (np.ndarray)
        env_id: environment ID
        n_episodes: number of evaluation episodes

    Returns:
        dict with keys: success_rate, mean_reward, mean_distance, per_episode
    """
    env = make_env(env_id)
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        final_distance = np.linalg.norm(
            obs["achieved_goal"] - obs["desired_goal"]
        )
        success = info.get("is_success", float(final_distance < 0.05))

        results.append({
            "reward": episode_reward,
            "distance": final_distance,
            "success": float(success),
        })

    env.close()

    success_rate = np.mean([r["success"] for r in results])
    mean_reward = np.mean([r["reward"] for r in results])
    mean_distance = np.mean([r["distance"] for r in results])

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_distance": mean_distance,
        "per_episode": results,
    }

# ---------------------------------------------------------------------------
# Rendering pipeline (Isaac Sim)
# ---------------------------------------------------------------------------

def render_episodes(policy_fn, env_id=ENV_ID, n_episodes=RENDER_EPISODES,
                    output_dir="./renders", show_window=False):
    """Render evaluation episodes with Isaac Sim.

    PROTOTYPE: This needs Isaac Sim viewport/camera setup to capture frames.
    The interface matches the MuJoCo version so train.py works unchanged.

    Args:
        policy_fn: callable(obs_dict) -> action
        env_id: environment ID
        n_episodes: number of episodes to render
        output_dir: directory to save renders
        show_window: if True, show Isaac Sim viewport

    Returns:
        dict with paths to saved files
    """
    raise NotImplementedError(
        "Isaac Sim rendering not yet configured. "
        "To implement:\n"
        "  1. Set up Isaac Sim viewport/camera for the environment\n"
        "  2. Capture RGB frames during episode rollouts\n"
        "  3. Save video + key frame PNGs matching MuJoCo output format\n"
        "  4. Replace this function with the actual implementation"
    )


# ---------------------------------------------------------------------------
# Main (verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify evaluation infrastructure")
    parser.add_argument("--render", action="store_true", help="Test rendering pipeline")
    args = parser.parse_args()

    action_dim = get_action_dim(ENV_ID)
    def random_policy(obs):
        return np.random.uniform(-1, 1, size=action_dim)

    print("Running evaluation with random policy...")
    metrics = evaluate(random_policy, ENV_ID, n_episodes=5)
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Mean reward: {metrics['mean_reward']:.2f}")
    print(f"  Mean distance: {metrics['mean_distance']:.4f}")
    print()

    if args.render:
        print("Testing rendering pipeline...")
        render_result = render_episodes(random_policy, ENV_ID, n_episodes=2, output_dir="./renders")
        print(f"  Video: {render_result['video_path']}")
        print(f"  Frames: {len(render_result['frame_paths'])} key frames saved")
        print()

    print("Done!")
