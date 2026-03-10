"""
Fixed evaluation infrastructure for autoresearch-robotics.
Evaluation harness and rendering pipeline.

This file is READ-ONLY — the agent must NOT modify it.

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
            Takes a raw observation dict and returns an action.
        env_id: environment ID
        n_episodes: number of evaluation episodes

    Returns:
        dict with keys:
            success_rate: fraction of episodes achieving the goal
            mean_reward: average episode return
            mean_distance: average final distance to goal
            per_episode: list of per-episode dicts
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

        # Compute final distance to goal
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
# Rendering pipeline (headless MuJoCo)
# ---------------------------------------------------------------------------

def render_episodes(policy_fn, env_id=ENV_ID, n_episodes=RENDER_EPISODES,
                    output_dir="./renders", show_window=False):
    """Render evaluation episodes with MuJoCo.

    Captures frames at each timestep, saves a video and key frame PNGs.
    Optionally displays a live MuJoCo window.

    Args:
        policy_fn: callable(obs_dict) -> action
        env_id: environment ID
        n_episodes: number of episodes to render
        output_dir: directory to save renders
        show_window: if True, replay episodes in a live MuJoCo window after recording

    Returns:
        dict with paths to saved files
    """
    import imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id, render_mode="rgb_array")
    all_frames = []
    episode_boundaries = [0]

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        # Capture initial frame
        frame = env.render()
        all_frames.append(frame)

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            all_frames.append(frame)
            done = terminated or truncated

        episode_boundaries.append(len(all_frames))

    env.close()

    # Save video
    video_path = output_dir / "eval_video.mp4"
    imageio.mimsave(str(video_path), all_frames, fps=30)

    # Save key frames: first, middle, last of each episode
    frame_paths = []
    for ep in range(n_episodes):
        start = episode_boundaries[ep]
        end = episode_boundaries[ep + 1]
        mid = (start + end) // 2

        for label, idx in [("start", start), ("mid", mid), ("end", end - 1)]:
            fname = f"ep{ep}_{label}.png"
            fpath = output_dir / fname
            imageio.imwrite(str(fpath), all_frames[idx])
            frame_paths.append(str(fpath))

    # Live window display (optional)
    if show_window:
        print("Opening live MuJoCo window...")
        # Switch to GLFW backend for window display.
        # EGL (used for offscreen above) can't open windows.
        # WSLg provides a local X11 display at :0.
        prev_gl = os.environ.get("MUJOCO_GL")
        prev_display = os.environ.get("DISPLAY")
        os.environ["MUJOCO_GL"] = "glfw"
        os.environ["DISPLAY"] = ":0"

        env_human = make_env(env_id, render_mode="human")
        for ep in range(n_episodes):
            obs, info = env_human.reset()
            done = False
            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env_human.step(action)
                done = terminated or truncated
        env_human.close()

        # Restore previous backend
        if prev_gl is not None:
            os.environ["MUJOCO_GL"] = prev_gl
        else:
            os.environ.pop("MUJOCO_GL", None)
        if prev_display is not None:
            os.environ["DISPLAY"] = prev_display
        else:
            os.environ.pop("DISPLAY", None)

    return {
        "video_path": str(video_path),
        "frame_paths": frame_paths,
        "n_frames": len(all_frames),
    }

# ---------------------------------------------------------------------------
# Main (verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify evaluation infrastructure")
    parser.add_argument("--render", action="store_true", help="Test rendering pipeline")
    args = parser.parse_args()

    # Random policy for testing
    action_dim = get_action_dim(ENV_ID)
    def random_policy(obs):
        return np.random.uniform(-1, 1, size=action_dim)

    # Evaluate with random policy
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
        print(f"  Total frames captured: {render_result['n_frames']}")
        print()

    print("Done!")
