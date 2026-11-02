# Clean scoring task

The live demo and policy viewer now end a shot on obstacle contact. A goal must follow a paddle
hit, and the tally uses `is_success`. The planner evaluator and one-contact evaluation suite also
use these rules by default. `eval_planner --obstacle-bounce` and
`eval_precision_striker --obstacle-bounce` explicitly replay the historical permissive rules.
Historical model weights and their training manifests are unchanged.

## Measured baseline (October 26, 2026)

Each entry uses 200 episodes and seed 30000. Obstacle contact is a failure.

| Obstacles | Strike controller | Default bank-shot RL model |
| --- | ---: | ---: |
| 1 | 88% | 49% |
| 2 | 80.5% | 43.5% |
| 3 | 71.5% | 20.5% |

These columns use different protocols: the controller uses the historical fixed-layout protocol
and original restitution; RL uses the wider obstacle region, restitution 0.8, and one-contact
rules. They diagnose each mode separately and are not a matched policy comparison.
The RL model is `models/tacc-2026-10/AB-seed3-bank`. Detailed summaries are in
[`results/clean-scoring-baseline.json`](../results/clean-scoring-baseline.json).
The previous 96% controller headline allowed obstacle rebounds. It does not establish 90% clean
scoring. The existing RL model also still makes unnecessary banks; changing scoring rules does
not retrain its weights.

## Train for the corrected objective

`--prefer-direct` changes aim shaping: if a clear direct shot exists, bank directions receive no
aim bonus. When every direct shot is blocked, clear bank shots still receive aim credit. The task
version gets a `-direct` suffix so runs with different reward rules remain distinguishable.
The existing `--obstacles-max` option samples counts uniformly, avoiding training only on one block.

With the training dependencies from `requirements-headless.in` installed, a new training run is:

```sh
python -m scripts.train_precision_striker --steps 2000000 --envs 4 --vector subproc --require-hit --timeout-penalty 10 --extra-obs --goal-shaping --wide-obstacles --obstacles 1 --obstacles-max 3 --single-hit --paddle-restitution 0.8 --aim-shaping 5 --prefer-direct --output runs/clean-scoring-seed0
```

This is a starting configuration, not a validated budget or a guarantee of 90% accuracy. It uses
unfiltered random scorable layouts and does not preselect successful episodes. Keep open-goal
examples in training instead of training exclusively on forced bank shots.

For fine-tuning the existing 22-input bank model, add
`--resume models/tacc-2026-10/AB-seed3-bank --resume-new-task`. Observation dimensions must match.
Use a separate output directory and evaluate before replacing a demo checkpoint.

Evaluate on validation seeds 50000 and 60000 while choosing checkpoints; keep 30000 as the historical
test and add a fresh test seed after selection. Report each obstacle count, not only their average:

```sh
python -m scripts.eval_suite --run runs/clean-scoring-seed0 --seed 50000 --only random_1 random_2 random_3 blocked_1 blocked_1to3 --shots 200 --output runs/clean-scoring-seed0/validation.json
```

Use `--controller` instead of `--run ...` for a matched controller comparison under the same rules.
The 90% target remains unmet until a newly evaluated checkpoint meets it separately at all three
obstacle counts. A 64-transition PPO smoke run verifies integration only; it is not accuracy evidence.
