# Incoming-puck precision striker: current plan and runbook

The active task is `precision-striker-v2-incoming`: intercept a moving puck, send it into the opposing goal, and avoid static obstacles. Incoming position, direction, speed, and obstacle position vary. The paddle remains on its own half. This is planar air hockey with zero gravity; the puck does not drop between physical levels. Objects are randomized when a new episode starts and remain stationary during that episode.

The original stationary-puck experiment is archived in `ARCHIVE_V1_PLAN.md`. Its checkpoint cannot be used with the v2 observation/action contract. The detailed implemented model, measured results, limitations, and next experiments are in `TECHNICAL_REPORT.md` and `../output/pdf/precision_striker_technical_report.pdf`.

## Watch the actual learned model

From an Anaconda Prompt, change to this repository and run:

```bat
watch_rl_policy.bat
```

The default is the completed local run `runs/incoming-v2-final`, with fallback to `models/precision-striker-v2` on a fresh checkout. The bundled model is experimental; inspect its model card and measured failure rates. The portable equivalent on this workstation is:

```sh
python -m scripts.watch_precision_striker --run runs/incoming-v2-final
```

Space pauses; N starts another scenario; R repeats the seed; 1/2/3 changes obstacle count; +/- changes playback rate; Q closes. Terminal outcomes remain on screen until N is pressed; A toggles automatic advance after a two-second hold. Click within the permitted starting region to set puck position. The speed slider changes the next reset's incoming speed; zero restores randomized speed. The model was trained with one obstacle; two/three obstacles and unusual clicked starts are stress tests. The display runs the real Box2D state and PPO action, including misses and collisions.

To inspect a recorded contact-and-goal case that visibly crosses the centerline:

```sh
python -m scripts.watch_precision_striker --run models/precision-striker-v2 --scenario docs/experiment_data/example_contact_goal.json
```

This is a selected successful test case, not a representative success rate. The complete test scored 18/200, with only 9 contact-qualified goals. The old GIFs in `assets/` and `results/sac/puck_touch/` are pre-existing legacy demonstrations, not this PPO model.

For a terminal-only demonstration (choose a fresh output directory):

```sh
python -m scripts.eval_precision_striker --run runs/incoming-v2-final --output runs/terminal-demo --layouts 1 --shots 3 --obstacles 1 --terminal
```

## Train, resume, evaluate

Use Python 3.11. Install the CPU/CUDA PyTorch build appropriate to the machine before `requirements-headless.in`; use a graphical OpenCV build to watch locally. Run from the repository root.

```sh
python -m unittest discover -s tests -v
python -m scripts.train_precision_striker --steps 262144 --envs 4 --vector dummy --obstacles 1 --device cpu --output runs/incoming-v2
python -m scripts.train_precision_striker --steps 786432 --envs 8 --vector subproc --obstacles 1 --device cpu --resume runs/incoming-v2 --output runs/incoming-v2-final
python -m scripts.eval_precision_striker --run runs/incoming-v2-final --output runs/final-test --seed 30000 --layouts 10 --shots 20 --obstacles 1
```

The commands describe the two-stage local experiment; consult its saved manifests for exact historical arguments. Resume adds transitions to the existing count and retains optimizer and observation normalization. Use a new output directory for each run/evaluation. Keep `policy.zip` and `vecnormalize.pkl` together. A model loaded without its normalization is a different controller.

## Headless TACC path

Training has no display dependency. Box2D physics executes on CPUs; `--device cuda` moves the small neural networks to a GPU. This is not a GPU-native simulator. Benchmark throughput before reserving long GPU jobs. This delivery's measured run was on local CPU; remote GPU training has not been verified.

In Windows PowerShell or Command Prompt, first connect:

```sh
ssh YOUR_TACC_USERNAME@ls6.tacc.utexas.edu
```

Complete TACC's normal authentication. Only after the prompt changes to the remote Linux host, run:

```sh
hostname
qlimits
module -t list 2>&1
command -v python3
printf 'WORK=%s\n' "$WORK"
squeue -u "$USER"
```

`qlimits` and `module` are remote cluster commands; a `C:\Users\...>` prompt is still Windows. A logged-in external terminal does not automatically grant a separate process SSH access.

On the remote host, clone this repository and check out `codex/tacc-headless-plan`. Select a currently available Python >=3.10 module. Use `scripts/tacc/setup_env.sh` within an allocated compute job. Verify account eligibility and the current queue/node/GPU request with `qlimits` and the official system guide. The template still names Frontera `rtx-dev`; override it for a verified Lonestar6 partition. Do not copy a Frontera queue name into a Lonestar6 job.

```sh
mkdir -p logs
export TACC_PROJECT=YOUR_ALLOCATION
# Substitute a partition verified by qlimits for YOUR_GPU_PARTITION.
sbatch -A "$TACC_PROJECT" -p YOUR_GPU_PARTITION --export=ALL,TRAIN_STEPS=32768,OBSTACLES=1 scripts/tacc/train.slurm
squeue -u "$USER"
```

Start with the 32,768-step smoke job; inspect CUDA assertion, source manifest, tests, completed steps, checkpoint, and evaluation. Queue-specific resource requests may require additional options. After the smoke test, set a measured time/resource budget and use `TRAIN_STEPS=1048576,EVAL_LAYOUTS=10,EVAL_SHOTS=20`. Copy the run directory back with `scp -r` and use the same local viewer. A terminal can show training logs and ASCII playback without X forwarding; full graphics are displayed locally.

## Completion gates

1. **Software baseline:** reproducible environment, model/normalizer pair, training logs, evaluation trajectories, and an interactive actual-policy viewer. Implemented in this branch.
2. **Learning target:** at least 80% scoring over ten new layouts with twenty shots each and zero obstacle collision episodes. A completed training budget does not establish this target; use the report's measured outcome.
3. **Robustness:** repeat with at least three independent training seeds, wider speed bands, new obstacle counts, sensor noise, latency, and physical parameter uncertainty. Pending beyond the exploratory tests reported.
4. **Perception/control integration:** timestamped camera observations, calibration, state estimation, and verified robot workspace/controller dynamics. Pending.
5. **Hardware acceptance:** independently test the System Design report's tracking, latency, sustained control, and emergency-stop criteria. No physical robot validation is claimed.

Repository read/write access was verified by a successful push to this feature branch. TACC interactive access was shown in the supplied screenshot; this agent's separate SSH authentication, GPU allocation, modules, and remote execution remain unverified. See [Lonestar6 documentation](https://docs.tacc.utexas.edu/hpc/lonestar6/) and [Frontera documentation](https://docs.tacc.utexas.edu/hpc/frontera/) for current cluster requirements.
