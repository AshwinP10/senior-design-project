# Air Hockey Reinforcement Learning Environment

## Precision striker and headless TACC training

See [the implementation plan and runbook](docs/PRECISION_STRIKER_PLAN.md) for the
senior-design scoring task, access status, GPU batch jobs and acceptance criteria.
The new `scripts.train_precision_striker` and `scripts.eval_precision_striker` entry
points support headless training and terminal playback. Existing tasks remain available.
Training checkpoints alone do not establish the report's 80% scoring target.

The v2 task trains **moving-puck interception and scoring around static obstacles**.
Read the [detailed model and results report](docs/TECHNICAL_REPORT.md),
the [PDF report](output/pdf/precision_striker_technical_report.pdf), and the
[model card](models/precision-striker-v2/README.md).
The bundled experimental PPO model completed 1,048,576 transitions on local CPU;
the report gives its measured performance and remaining failures. Remote TACC GPU
execution has not been verified.

To watch the actual learned model on Windows, double-click `watch_rl_policy.bat`
in an environment with the dependencies installed. It uses the local final run
when present and otherwise the bundled model. On other systems:

```sh
python -m scripts.watch_precision_striker --run models/precision-striker-v2
```

For a fresh graphical environment, install PyTorch and `requirements-viewer.in`.
Use `requirements-headless.in` on compute nodes. Do not install both graphical
and headless OpenCV packages in the same environment.

This contains an air hockey simulation environment powered by Box2D. It is fast (C++ back-end), capable of self-play, 1v1 play, and easy goal-conditioned reinforcement learning, resulting in a rich testbed for various algorithms.


The GIFs below and `results/sac/puck_touch/eval_*.gif` are **legacy task examples**,
not recordings of the new incoming-puck PPO checkpoint. Their episode boundaries
and objectives differ. Use `watch_rl_policy.bat` for the current experiment;
it holds the final state and outcome until N is pressed (A enables automatic next).

Legacy Upward Puck Velocity | Legacy Goal-Conditioned RL
:-------------------------:|:-------------------------:
![](assets/puck_vel.gif)  |  ![](assets/puck_goal_pos.gif)

## Installation

### Using uv
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Option A: sync with lock file
```bash
# Create virtual environment and sync dependencies from lock file
uv sync
# For training dependencies
uv sync --extra train
```

#### Option B: Install directly
```bash
# create uv virtual environment and activate
uv venv
source .venv/bin/activate

# Install the package in development mode
uv pip install -e .

# Or if you need training too:
uv pip install -e ".[train]"
```

### Using pip (legacy)
```bash
# Install with training dependencies
pip install -e .[train]

# Or just the base package
pip install -e .
```


## Other

#### Having this issue?
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
`echo 'export MUJOCO_GL="glx"' >> ~/.bashrc`
`source ~/.bashrc`

## How to Run
Most of the files use a configuration file (--cfg cmd argument), but is defaulted to one from `configs/`. Please see there to tune parameters for various scripts.
#### What the files do
- `airhockey2d.py`: base gym environment for air hockey
- `render.py`: renders the air hockey environment
- `train.py`: trains an agent via stable-baselines3 PPO.

Legacy:
- `demonstrate.py`: user plays a self-play air hockey environment using keyboard
- `play_trained_agent`: run after training, you can play against the trained agent

## Running on the Physical UR5
- Boot up the robot through the touchpad
    - Press physical power button
    - Press red power on touchpad in bottom left corner
    - power on the robot with touch button in the middle
    - open program "external_control.urp"
- run desired script in scripts/real
    - ex: python scripts/real/teleoperate.py --cfg configs/baseline_configs/puck_vel_real.yaml
- When prompted in the terminal, run the program using the play button in the bottom middle of the touchpad
- follow prompts on the terminal. Hold 'q' to end trajectories
