# Precision Striker models trained on TACC Lonestar6 (October 2026)

Pure-RL policies: a 128-128 MLP maps the observation (paddle, puck, obstacle slots, hit flag, time) straight
to a paddle velocity command. No controller runs at test time. Most were started by imitation (DAgger from
the strike controller) and then trained further with PPO (Stable-Baselines3 2.2.1).

Each folder has `policy.zip` + `vecnormalize.pkl` (keep them together), `manifest.json` (exact task and
training settings) and `evaluation.json`. Load model and pickle files only from a trusted source.

## Launch

From the repository root (Windows, Anaconda or any Python with the repo requirements):

```bat
demo_rl.bat                                        :: live demo, default model W1-seed2
demo_rl.bat models\tacc-2026-10\WB-seed5           :: any other folder here
```

or directly:

```sh
python -m scripts.demo_live --rl-run models/tacc-2026-10/W1-seed2 --start-rl     # live demo, click to place blocks
python -m scripts.eval_precision_striker --run models/tacc-2026-10/W1-seed2 --seed 30000 --layouts 10 --shots 20 --obstacles 1
python -m scripts.eval_suite --run models/tacc-2026-10/W1-seed2 --seed 50000    # strict one-contact suite
```

To keep training one of them: `python -m scripts.train_precision_striker --resume models/tacc-2026-10/W1-seed2 ...`
(add `--resume-new-task` when changing task options; see `scripts/tacc/strict_node.sh`).

## Results

Test: 200 shots (seed 30000), obstacle bounce rule, one obstacle anywhere a user may place it, paddle
restitution 1.0, a second paddle touch allowed. Models were chosen on validation seed 50000, not on test.

| Model | Training | Obstacles in training | Test goals |
|---|---|---|---|
| W1-seed2 (demo default) | DAgger + 4M PPO | 1 anywhere | 169/200 = 84.5% (validation 183/200) |
| WB-seed5 | DAgger (more data) + 4M PPO | 1 anywhere | 174/200 = 87.0% (validation 182/200) |
| WB-seed4 | DAgger (more data) + 4M PPO | 1 anywhere | 164/200 = 82.0% |
| W1-seed1 | DAgger + 4M PPO | 1 anywhere | 161/200 = 80.5% |
| W1-seed3 | DAgger + 4M PPO | 1 anywhere | 152/200 = 76.0% |
| WF-seed1 | 6M PPO from imitation-ppo-run2-12M | 1 anywhere | 154/200 = 77.0% |
| W3-seed1/2/3 | DAgger + 4M PPO | 1-3 anywhere | 60.0% / 69.0% / 57.0% |
| imitation-ppo-run2-12M | BC + 12M PPO | 1 in the 25-65 cm band | 159/200 = 79.5% |

Known limits (strict one-contact suite, `scripts/eval_suite.py`, validation seed 50000, paddle restitution
0.8, a second touch or a returning puck ends the shot): W1-seed2 scores 78.5% with one random block but only
7.0% when the block sits on the straight line to the goal, and 28.0% / 5.5% with 1-3 blocks. These nets were
not trained under those rules; `scripts/tacc/strict_sweep.slurm` trains successors on exactly that task.
