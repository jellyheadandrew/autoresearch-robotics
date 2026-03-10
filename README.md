# autoresearch-robotics

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) adapted for robotics -- autonomous overnight experiment optimization with robotics simulation feedback loop.

![teaser](assets/teaser.gif)

## Quickstart

Requirements: Python 3.10+, Claude Code (or any coding agent). (Tested on Nvidia RTX 2080)

```bash
git clone https://github.com/jellyheadandrew/autoresearch-robotics.git
cd autoresearch-robotics

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pick a task

```bash
# List available templates
python setup_task.py --list

# Set up an experiment directory
python setup_task.py mujoco/fetchreach my_experiment
# or: python setup_task.py mujoco/fetchpush my_experiment
# or: python setup_task.py mujoco/fetchpickplace my_experiment
```

### Install and verify

```bash
cd my_experiment
uv sync
uv run prepare.py
```

### Launch the autonomous research loop

```bash
claude --dangerously-skip-permissions
```

Or use any LLM-powered coding agent with similar autonomous permissions.

Then type:

```
Read program.md and let's set up a new experiment. Then start the experiment loop.
# Or, for headless mode:
Read program.md and let's set up a new experiment. Then start the experiment loop. Use --headless mode.
```

The agent reads `program.md`, creates a branch, runs the baseline, then loops: analyze → hypothesize → modify `train.py` → commit → train → evaluate → keep/discard → repeat. It runs indefinitely until you Ctrl+C. For overnight runs, use `tmux`.

Monitor from another terminal:

```bash
watch -n 60 'cat results.tsv'
watch -n 30 'git log --oneline -10'
```

## Available tasks

| Template | Task | Time budget | Experiments/hour |
|----------|------|-------------|------------------|
| `mujoco/fetchreach` | Reach a target position | 10 seconds | ~30 |
| `mujoco/fetchpush` | Push a cube to a goal | 10 minutes | ~5 |
| `mujoco/fetchpickplace` | Pick and place an object | 30 minutes | ~2 |
| `isaac/fetchreach` | Reach (Isaac Sim) | 60 seconds | prototype |

## Results

**FetchReach:**

![FetchReach Results](assets/results_plot_fetchreach.png)

**FetchPush, FetchPickPlace**: TBD Soon.

**VLA Experiments**: TBD after getting compute credits. (Support would be appreciated! [buymeacoffee.com/jellyheadandrew](https://buymeacoffee.com/jellyheadandrew))

## What changed from the original

[autoresearch](https://github.com/karpathy/autoresearch) targets LLM training, where the only feedback is loss curves. Robotics has a visual component: you can *see* what the robot is doing wrong.

Key adaptations:

- **Visual feedback loop.** After each experiment, MuJoCo renders the robot's behavior. The coding agent analyzes the rendered frames, getting qualitative feedback ("the arm overshoots and oscillates") alongside quantitative metrics — not just numbers.
- **MuJoCo + Gymnasium Robotics** instead of nanoGPT. SAC + HER as the baseline RL algorithm.
- **Template system.** Multiple tasks and simulators in one repo. `setup_task.py` assembles flat, self-contained experiment directories.
- **Simulator modularity.** MuJoCo is fully supported; Isaac Sim support is prototyped for future use.

## Project structure

```
core/                        — shared files (evaluate.py, train.py, pyproject.toml, program.md.template)
templates/
  mujoco/                    — MuJoCo task templates
    fetchreach/prepare.py    — FetchReach-v4, 10s budget
    fetchpush/prepare.py     — FetchPush-v4, 10min budget
    fetchpickplace/prepare.py — FetchPickAndPlace-v4, 30min budget
  isaac/                     — Isaac Sim task templates (prototype)
    fetchreach/              — prepare.py, evaluate.py, pyproject.toml overrides
setup_task.py                — assembles template → experiment directory
```

Each assembled experiment directory contains:
```
prepare.py              — env factory, obs utilities (do not modify)
evaluate.py             — evaluation, rendering (do not modify)
train.py                — policy, training algorithm, training loop (agent modifies this)
program.md              — agent instructions
pyproject.toml          — dependencies
```

## Credits

Built on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. Uses [Gymnasium Robotics](https://robotics.farama.org/) with [MuJoCo](https://mujoco.org/).

## License

MIT
