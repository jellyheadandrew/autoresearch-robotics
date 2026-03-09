"""
Fixed infrastructure for autoresearch-robotics.
Environment factory and observation utilities.

This file is READ-ONLY — the agent must NOT modify it.

Usage:
    python prepare.py              # verify environment works
"""

import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Auto-configure MuJoCo OpenGL backend (must run before any MuJoCo import)
# ---------------------------------------------------------------------------

def _configure_mujoco_gl():
    """Auto-configure MuJoCo OpenGL backend for offscreen rendering.

    Default to EGL (GPU-accelerated offscreen) which works reliably everywhere
    including WSL2. Falls back to OSMesa (CPU software) if EGL is unavailable.
    Users can override by setting MUJOCO_GL explicitly.
    """
    if "MUJOCO_GL" in os.environ:
        return  # User explicitly set it, respect their choice
    # Default to EGL for reliable offscreen rendering
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
    except OSError:
        os.environ["MUJOCO_GL"] = "osmesa"

_configure_mujoco_gl()

if os.environ.get("MUJOCO_GL"):
    print(f"MuJoCo GL backend: {os.environ['MUJOCO_GL']}")

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600            # training time budget in seconds (10 minutes)
MAX_EPISODE_STEPS = 50       # Fetch environment default
FRAME_WIDTH = 640            # render resolution
FRAME_HEIGHT = 480
ENV_ID = "FetchPush-v4"          # default environment

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_id=ENV_ID, render_mode=None):
    """Create a Gymnasium Robotics environment.

    Args:
        env_id: Gymnasium environment ID (e.g. "FetchReach-v3", "FetchPickAndPlace-v3")
        render_mode: None for training, "rgb_array" for rendering

    Returns:
        gymnasium.Env with goal-conditioned observation space
    """
    import gymnasium
    import gymnasium_robotics

    gymnasium_robotics.register_robotics_envs()

    kwargs = {"max_episode_steps": MAX_EPISODE_STEPS}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
        kwargs["width"] = FRAME_WIDTH
        kwargs["height"] = FRAME_HEIGHT

    env = gymnasium.make(env_id, **kwargs)
    return env


def flatten_obs(obs_dict):
    """Flatten goal-conditioned observation dict into a single vector.

    Fetch environments return: {"observation": ..., "achieved_goal": ..., "desired_goal": ...}
    We concatenate observation + desired_goal for the policy input.

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
    print()

    # Verify environment
    print("Verifying environment...")
    env = make_env(ENV_ID)
    obs, info = env.reset()
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Observation shape: {obs['observation'].shape}")
    print(f"  Goal shape: {obs['desired_goal'].shape}")
    print(f"  Flattened obs dim: {flatten_obs(obs).shape[0]}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    print(f"  Action range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    env.close()
    print("  Environment OK!")
    print()

    print("Done! Environment verified.")
