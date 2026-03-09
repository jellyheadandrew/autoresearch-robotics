"""
Fixed infrastructure for autoresearch-robotics (Isaac Sim backend).
Environment factory and observation utilities.

This file is READ-ONLY — the agent must NOT modify it.

PROTOTYPE: Isaac Sim support is not yet configured. This file defines the
interface contract that all prepare.py implementations must satisfy.
Once Isaac Sim is set up, replace the NotImplementedError calls with
actual Isaac Sim environment creation logic.

Usage:
    python prepare.py              # verify environment works
"""

import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60              # training time budget in seconds
MAX_EPISODE_STEPS = 50        # episode length
FRAME_WIDTH = 640             # render resolution
FRAME_HEIGHT = 480
ENV_ID = "FetchReach-Isaac"   # Isaac Sim environment identifier

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_id=ENV_ID, render_mode=None):
    """Create an Isaac Sim environment.

    Must return a gymnasium-compatible env with goal-conditioned observation space:
        obs, info = env.reset()
        obs = {"observation": np.ndarray, "achieved_goal": np.ndarray, "desired_goal": np.ndarray}
        obs, reward, terminated, truncated, info = env.step(action)

    Args:
        env_id: environment identifier
        render_mode: None for training, "rgb_array" for rendering

    Returns:
        gymnasium-compatible Env with goal-conditioned observation space
    """
    raise NotImplementedError(
        "Isaac Sim environment not yet configured. "
        "To implement:\n"
        "  1. Install Isaac Sim (isaacsim package)\n"
        "  2. Create or wrap an Isaac Sim FetchReach environment\n"
        "  3. Ensure it returns goal-conditioned obs dicts\n"
        "  4. Replace this function with the actual implementation"
    )


def flatten_obs(obs_dict):
    """Flatten goal-conditioned observation dict into a single vector.

    Args:
        obs_dict: dict with "observation" and "desired_goal" keys

    Returns:
        np.ndarray: concatenated observation vector
    """
    return np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]])


def get_obs_dim(env_id=ENV_ID):
    """Get the flattened observation dimension for an environment."""
    env = make_env(env_id)
    obs, _ = env.reset()
    dim = flatten_obs(obs).shape[0]
    env.close()
    return dim


def get_action_dim(env_id=ENV_ID):
    """Get the action dimension for an environment."""
    env = make_env(env_id)
    dim = env.action_space.shape[0]
    env.close()
    return dim


def get_action_bounds(env_id=ENV_ID):
    """Get action space bounds."""
    env = make_env(env_id)
    low = env.action_space.low.copy()
    high = env.action_space.high.copy()
    env.close()
    return low, high


# ---------------------------------------------------------------------------
# Main (verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Environment: {ENV_ID}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Simulator: Isaac Sim (PROTOTYPE — not yet configured)")
    print()

    print("Verifying environment...")
    env = make_env(ENV_ID)
    obs, info = env.reset()
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Flattened obs dim: {flatten_obs(obs).shape[0]}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    env.close()
    print("  Environment OK!")
