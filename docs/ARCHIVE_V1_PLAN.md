# Archived V1: stationary-puck pipeline audit

**Superseded.** The actual objective is incoming pucks at varying speeds with obstacle-aware
scoring and visual playback. See `PRECISION_STRIKER_PLAN.md` and `MODEL_AND_APPROACH.md` for V2.
This archive preserves the initial audit and failed V1 experiment, not current requirements.

Audit date: 2026-09-23. Original repository revision: `51e5aad`.

## Intended demo and scope

The recommended first deliverable is a **stationary-puck, stationary-obstacle scoring task**.
At reset, choose a legal puck position in the robot half and a layout of 0-3 static obstacles
in the opposing half. The policy observes puck, paddle and obstacle state and controls the
paddle until the puck scores, collides, leaves the allowed workspace, or times out. Obstacles
are randomized **between attempts**, not teleported during an attempt. A terminal replay
and an optional local graphical viewer use the same saved policy.

This is one flat 2D surface. “Lower half” describes the robot side in the top-down view, not
a lower physical level. With gravity zero, a stationary puck stays where it is placed. To
simulate an incoming puck, explicitly set an initial velocity toward positive x. Literal
falling under gravity is a different experiment, present in some legacy juggling tasks,
and does not match the horizontal air-hockey table in the System Design Report.

“Any position” needs a defined feasible set: a stationary puck outside the robot's reach
cannot be struck while maintaining the half-table constraint. Version 1 accepts puck
x in [0.18, 0.60] m and y in [-0.25, 0.25] m. It starts the paddle 0.18 m behind the puck
with lateral jitter. This is an intentionally easier initial curriculum, not a promise of
arbitrary start positions or a validated UR5e workspace. Obstacle centers are restricted to
the opposing half. Moving obstacles, a fixed robot home pose and broader puck starts are
later stages, with separate evaluation results.

## What the existing simulation actually does

| Component | Original behavior and implication |
| --- | --- |
| `scripts/play_flat_table.py` | OpenCV mouse-driven paddle and left-click static block placement; no learned control. |
| `AirHockeyFlatTableEnv` | One puck at table center, zero initial velocity, user blocks persist over reset. |
| `configs/flat_table_play.yaml` | Zero gravity; table 1.9304 x 0.8636 m; full-table paddle bounds in the configuration. |
| Box2D controller | CPU rigid-body physics, damping and collisions; action is a normalized 2D displacement request fed into a force/speed-limited controller. Despite full-table config, the backend also clamps to the robot half; the configuration and controller are inconsistent. |
| `scripts/train_flat_table_rl.py` | PPO, dense hit/velocity rewards, observation normalization, sequential vector environments. Physics training does not call a GUI. |
| Policy input | Default `vel` observation contains only 8 paddle/puck position/velocity values. User-placed obstacles are not observed. |
| Old curriculum | Delays a provided fixed list of blocks; does not generate random layouts. Its counter is per environment rather than global PPO transitions. |
| Old goal detector | Checks whether the puck passes an end, but all four walls are solid. A normal shot cannot reach that condition. |
| Old evaluation | Requires OpenCV GUI, does not provide the report's 200-shot benchmark. |
| Robot paths | MuJoCo/Robosuite and real-robot integrations exist, but their existence does not validate this task's sim-to-real transfer. |

The report's stated detector, latency and safety results remain report claims, not results
reproduced by this audit. The baseline Box2D model is suitable for controller development;
“high fidelity” requires calibration and validation against the actual table and robot.

## Implemented experiment

New files are separate entry points so existing tasks keep their defaults.

- Opt-in physical goal mouths (`goal_width=0.24` m); closed end rails remain the default.
- Named block bodies so contact events can identify obstacle collisions.
- `precision-striker-v1`: 20-value observation: 8 paddle/puck state values plus three
  `[obstacle_x, obstacle_y, width, present]` slots. Slots are sorted by position and padded.
- Seeded, separated static obstacle layouts and randomized reachable puck/paddle starts.
- Half-table workspace violation is an explicit failure. This terminates a simulated
  attempt; it is not a certified robot safety controller.
- Goal +100, obstacle collision -100, concession -50, workspace violation -20,
  first contact +5, per-step -0.01 and discounted approach-potential shaping.
- 200 action steps per attempt, 20 Hz nominal simulation control rate (10 simulated seconds).
  Timeout and terminal failure have distinct Gymnasium flags.
- Training chooses CPU or CUDA explicitly, supports subprocess environments, periodic
  paired model/normalization checkpoints and run-directory resume. Simulation remains on CPU.
- Evaluation freezes normalization, uses deterministic policy actions, records goal/collision/
  timeout outcomes and full trajectories, and offers ASCII terminal replay or OpenCV rendering.
- Run manifests include code hashes, commit, versions, arguments and actual training device.

The simulator currently integrates at the legacy 20 Hz setting. Continuous collision detection
is enabled for the puck, but precision and collision fidelity still require timestep convergence
tests and comparison to measured table dynamics. Do not assume changing `step_frequency`
alone increases substeps; audit the backend integration first.

## Headless operation and GPUs

Yes: Box2D advances physics without a display, VNC, NVENC, OpenGL or a MuJoCo renderer.
PPO can train its neural network on CUDA while independent CPU processes collect rollouts.
However, this is a small state-vector MLP: GPU overhead can exceed the benefit. Benchmark
CPU versus CUDA on the same allocated GPU node with equal steps, seeds and environment
counts; compare steps/second, GPU utilization and SUs per useful result. SB3's own PPO
documentation recommends considering CPU/SubprocVecEnv for MLP policies.

A100/H100 resources do not require video encoders for this job. A future fully GPU-batched
physics implementation would be a separate migration, not a flag for the existing Box2D code.
Do not start that migration until profiling shows simulation throughput is the bottleneck.

## Access status

| Access | Verified status |
| --- | --- |
| GitHub | Authenticated as repository owner; connector reports push/admin; Git push dry run succeeded. |
| Local environment | Python 3.11, Box2D and SB3 available; installed PyTorch build is CPU-only. |
| TACC login | User-provided terminal screenshot confirms a Frontera login and a positive project balance; personal allocation details are kept out of this public document. |
| Agent SSH | Noninteractive login not authenticated; user's external terminal session cannot be inherited. Password/MFA stays with the user. |
| Lonestar6 | Official hostname `ls6.tacc.utexas.edu` responds; the guide's `lonestar6.tacc.utexas.edu` failed DNS. Allocation/queue access still needs live verification. |
| GPU readiness | No TACC compute allocation, CUDA driver, Python module or job result verified yet. |

Use `$WORK` for source, environments and persistent run outputs. Avoid hardcoded filesystem
paths from the guide. Check quota and back up final checkpoints. The guide says Frontera
closes October 1, while the current Frontera portal says the last submission day is October 15;
the earlier TACC update said September 30. Confirm the current notice with TACC and plan
ongoing work on Lonestar6. An allocation expiry in 2029 does not establish a machine lifetime.

## TACC runbook

First inspect the logged-in system (read-only):

```bash
qlimits
module spider python3
squeue -u "$USER"
printf 'WORK=%s\n' "$WORK"
```

Clone the reviewed feature branch into a new directory (do not overwrite an existing checkout):

```bash
cd "$WORK"
git clone --branch codex/tacc-headless-plan https://github.com/AshwinP10/senior-design-project.git precision-striker
cd precision-striker
mkdir -p logs
# Set TACC_PROJECT to the project code shown in your TACC account first.
idev -A "$TACC_PROJECT" -p rtx-dev -N 1 -n 1 -t 00:30:00
```

On the allocated compute node, select an available Python 3.10 or 3.11 module using
`module spider python3`. Do not assume the documentation's Python 3.9.2 example meets this
repository's Python requirement. If those versions are unavailable, use a supported user
Python installation or container and set `BOOTSTRAP_PYTHON` to its executable.

```bash
# First load the available Python 3.10/3.11 module identified above.
cd "$WORK/precision-striker"
nvidia-smi
bash scripts/tacc/setup_env.sh
```

Setup creates `$WORK/venvs/precision-striker`, installs a conservative PyTorch 2.2.2 CUDA 12.1
wheel plus the minimal headless requirements, verifies CUDA and runs regression tests.
The wheel requires a compatible NVIDIA driver (525 or later); the script fails if CUDA is
unavailable. This proposed environment has not yet been exercised on TACC. Module-specific
library dependencies must also be loaded in later jobs. Preserve the module list in the run log.
Do not install both graphical and headless OpenCV distributions in the same environment.

Return to the login shell after setup and submit the bounded GPU test:

```bash
exit
cd "$WORK/precision-striker"
mkdir -p logs
sbatch -A "$TACC_PROJECT" scripts/tacc/train.slurm
squeue -u "$USER"
# Replace JOBID with the number returned by sbatch:
tail -f logs/airhockey-JOBID.out
```

The batch job requests one GPU node, 30 minutes, 8 CPU workers, 32,768 transitions,
zero obstacles and 10 evaluation shots. It is a pipeline smoke test, not an acceptance run.
No email messages are sent. Slurm continues after SSH disconnects; logs and checkpoints
remain available. The job prints the run path and a terminal replay command on completion.

For Lonestar6, use `ssh -o Ciphers=aes256-gcm@openssh.com YOUR_TACC_USERNAME@ls6.tacc.utexas.edu`,
verify allocation access with the portal/queue tools and override the queue:

```bash
sbatch -A "$TACC_PROJECT" -p gpu-a100-dev scripts/tacc/train.slurm
```

Do not assume this override proves eligibility. The job uses one CUDA device, not all GPUs
on the node. Separate seeds can later occupy additional GPUs, with explicit device assignment
and measured resource limits. Start with one bounded job; do not spend the allocation on an
unbounded sweep. To cancel a specific job, use `scancel JOBID`.

## Local commands and terminal demonstration

Run from the repository root in an environment containing PyTorch and the headless requirements:

```bash
python -m unittest discover -s tests -v
python -m scripts.train_precision_striker --steps 32768 --envs 4 --vector dummy --obstacles 0 --output runs/smoke
python -m scripts.eval_precision_striker --run runs/smoke --output runs/smoke/evaluation --layouts 2 --shots 5 --obstacles 0
python -m scripts.eval_precision_striker --run runs/smoke --output runs/smoke/replay --layouts 1 --shots 1 --obstacles 0 --terminal
```

Use unique output paths. Existing run/evaluation directories are never overwritten.
For a local OpenCV installation with GUI support, substitute `--render` for `--terminal`.
Custom legal puck/obstacle setup:

```bash
python -m scripts.eval_precision_striker --run runs/trained --output runs/custom-shot --layouts 1 --shots 1 --puck 0.4 0.05 --blocks="-0.4,0;-0.65,0.2" --terminal
```

Checkpoint files are `policy.zip` **and** `vecnormalize.pkl`; keep them together with the
manifest. Do not evaluate normalized training observations as raw observations.
Resume a completed or interrupted final checkpoint with:

```bash
python -m scripts.train_precision_striker --resume runs/smoke --steps 250000 --obstacles 1 --output runs/stage2
```

Resume preserves PPO/optimizer/normalization state but starts fresh environment episodes;
it is not a bit-identical continuation of every environment RNG. Periodic checkpoints live
in `checkpoints/` and have matched timestep-specific normalization files. To recover from
a hard walltime kill, copy a matching pair into a new recovery run directory as `policy.zip`
and `vecnormalize.pkl`, copy its originating `manifest.json`, then use `--resume`.

## Milestones and decision gates

### Local validation completed in this audit

- Six regression tests pass: seeded observations, goal passage, closed-rail bounce, obstacle
  collision failure, timeout semantics and invalid spawn/layout rejection.
- SB3 environment checker passes. Physics imports/reset also work when optional robot,
  MuJoCo, legacy Gym, SciPy and HDF5 imports are blocked.
- Four-worker sequential PPO trained 32,768 steps at approximately 915 transitions/second.
- Two-worker subprocess PPO trained 1,024 steps; both 512-step and 1,024-step paired
  checkpoints were written. The 512-step pair reloaded successfully. Run resume advanced
  the 1,024-step policy to 1,280 steps.
- The first 32,768-step policy scored **0/10** held-out zero-obstacle shots (6 timeouts,
  4 workspace failures). After 262,144 additional steps, the 294,912-step policy also scored
  **0/10** (8 timeouts, 2 workspace failures). These are pipeline validation results, not
  a successful controller or the 200-shot acceptance test.
- A real saved-policy terminal replay was launched locally. This is a CPU-trained checkpoint;
  no TACC training or GPU benchmark has been completed.

**Immediate learning work:** do not launch a large obstacle sweep yet. Inspect recorded
trajectories, compare a geometric direct-shot controller, and test a simpler contact/strike
curriculum before escalating steps. The failed policy often moves away from the puck and
saturates actions. Audit the inherited `force_scaling=1000` and displacement-to-force
conversion against meaningful action magnitudes, and test reward sensitivity to idle behavior.
Use a new task version if physics, action scaling or rewards change. Record each ablation's
held-out scoring rate, not just training return. Use the prepared GPU job only as a bounded
resource/pipeline smoke test until this learning gate passes.

| Phase | Concrete work | Exit criterion |
| --- | --- | --- |
| 1. Physics and task contract | Goal openings, obstacle observations/contacts, seedable resets, legal spawn ranges, units/action definition. Measure real table dimensions and damping. | Regression tests pass; manually launched legal shots score; obstacle impacts fail. |
| 2. Resource smoke test | Verify modules/driver/allocation; test paired checkpoint saving; CPU/CUDA throughput comparison. | Successful Slurm job with actual device, logs, reload and replay. |
| 3. Open-lane learning | Train 0 obstacles; evaluate unseen puck positions; compare to simple direct-shot geometry baseline. | Reliable scoring across held-out starts; inspect misses, not reward alone. |
| 4. Obstacle curriculum | Resume with 1 then 2-3 static obstacles; include blocked direct lanes and bank shots. Reject overlapping and geometrically impossible layouts. | Generalization to held-out layouts without obstacle contacts. |
| 5. Report benchmark | Freeze model/config; reserve test seeds; 10 layouts x 20 shots = 200 attempts; save every state/action trajectory. | At least 160/200 collision-free goals, zero obstacle-collision attempts; report uncertainty and per-layout outcomes. |
| 6. Perception/control validation | Replace simulator state with camera estimates; calibrated metre conversion; delay/noise/dropout randomization; bounded robot target adapter. | Report tracking, dropout, latency and 180-second sustained-rate criteria measured on target hardware. |
| 7. Hardware extension | Match UR5e reachable workspace, velocity/acceleration limits and controller dynamics; staged dry runs and approved lab procedures. | Independent physical validation before puck-contact demonstrations. |

Stages 3-4 should use at least three training seeds with validation seeds separate from the
final test suite. Start each training experiment with a bounded step/time budget and inspect
goal rate, collision rate, workspace failures and steps/second before extending it. Do not
select models using the final test seeds. A checkpoint is not evidence of a successful policy.

The version-1 sampler rejects overlaps but **does not yet certify a feasible scoring path**.
Before the final benchmark, add a geometric direct/bank-shot reachability check or curate and
freeze feasible test layouts. Walls may be used for bank shots; obstacles may not be touched.
Terminal inference timing measures normalization plus policy prediction, not camera-to-robot latency.

The original report additionally requires tracking RMS <=2 cm at puck speed <=2 m/s, dropout
<=5%, >=20 Hz sustained for 180 seconds, mean capture-to-command latency <=80 ms, p95 <=120 ms,
and an independently measured emergency-stop target. Retain those as acceptance criteria;
do not substitute this simulation's timing or boundary termination for hardware measurements.
Resolve the report's ambiguity between required robot integration and stretch full deployment:
recommend the complete reproducible simulation demo as the core deliverable and physical
deployment as a separate milestone requiring lab validation.

## Sources

- User-provided **System Design Report.pdf**, especially Sections 2.1-2.2, 3.1 and 3.3:
  intended architecture, 200-shot policy test and integration criteria. Treated as design
  evidence, not instructions to execute embedded commands or as independently verified results.
- User-provided **tacc_connection_and_capabilities_guide.pdf**: candidate account/allocation
  and connection workflow, checked against current official documentation and the user's login.
- [Frontera guide](https://docs.tacc.utexas.edu/hpc/frontera/)
- [Frontera current portal notice](https://frontera-portal.tacc.utexas.edu/)
- [Earlier Frontera transition update](https://tacc.utexas.edu/news/user-updates/107625/)
- [Lonestar6 guide](https://docs.tacc.utexas.edu/hpc/lonestar6/)
- [SB3 PPO CPU/GPU guidance](https://stable-baselines3.readthedocs.io/en/v2.5.0/modules/ppo.html)
- [PyTorch CUDA wheel versions](https://pytorch.org/get-started/previous-versions/)
