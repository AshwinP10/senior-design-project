#!/bin/bash
# One node's share of imitation_sweep.slurm: does PPO add anything on top of imitation?
# Clean scoring task (one contact, paddle restitution 0.8, obstacle contact ends the shot, 1-3 obstacles
# anywhere). Teacher: the physics striker with simulator-in-the-loop aim correction (scripts/sim_striker.py,
# ~97% on the strict test suite).
#   1. DAgger: iteration 0 = pure imitation (behaviour cloning), then 9 DAgger iterations. Every iteration
#      is scored on the strict validation suite -> the imitation learning curve (manifest history).
#   2. Strict suite (200 shots per condition, validation seed 50000) on the best imitation network.
#   3. PPO fine-tuning of that network, 3 seeds in parallel.
#   4. Strict suite on every 4th PPO checkpoint (100 shots per condition) -> the PPO learning curve.
# Usage: imitation_node.sh GROUP OUT_DIR
#   D1  1M labelled steps per iteration, 256-wide network
#   D2  1M per iteration, 512-wide network
#   D3  2M per iteration (more data), 512-wide network
#   D4  1M per iteration, 256-wide network, straight line blocked in 50% of episodes (others 30%)
set -uo pipefail
GROUP="$1"; OUT="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
PPO_STEPS="${PPO_STEPS:-20000000}"
ITERS="${ITERS:-10}"
case "$GROUP" in
    D1) PER=1000000; NET=256; BLOCK=0.3 ;;
    D2) PER=1000000; NET=512; BLOCK=0.3 ;;
    D3) PER=2000000; NET=512; BLOCK=0.3 ;;
    D4) PER=1000000; NET=256; BLOCK=0.5 ;;
    *) echo "unknown group $GROUP"; exit 1 ;;
esac
SEED="${GROUP#D}"
ENV_FLAGS=(--extra-obs --goal-shaping --wide-obstacles --obstacles-max 3 --block-line-prob "$BLOCK")
CONDS=(random_1 random_2 random_3 blocked_1)
IM="$OUT/$GROUP-imitation"
echo "$(hostname) group $GROUP start $(date)"

# 1. Imitation (behaviour cloning, then DAgger)
"$PY" -u -m scripts.train_dagger --output "$IM" --seed "$SEED" --workers 120 --iterations "$ITERS" \
    --steps-per-iter "$PER" --epochs 8 --net "$NET" --clean --sim-teacher --val-shots 100 \
    "${ENV_FLAGS[@]}" > "$OUT/$GROUP-imitation.log" 2>&1 || { echo "DAgger failed"; exit 1; }

# 2. Best imitation network on the full validation suite
"$PY" -u -m scripts.eval_suite --run "$IM" --seed 50000 --workers 4 --only "${CONDS[@]}" \
    --output "$IM/suite_val50000.json" > "$IM/suite.log" 2>&1

# 3. PPO fine-tuning from the best imitation network, 3 seeds. Small exploration noise, low learning rate,
#    capped update size, value-network warm-up and fixed normalisation: with PPO defaults the noise alone
#    cut a 40% imitation network to 13% and PPO then degraded it (2-hour job 3547783).
PPO_FT=(--resume-log-std "${PPO_LOG_STD:--3.5}" --resume-lr "${PPO_LR:-3e-5}" --target-kl 0.02
        --critic-warmup "${PPO_WARMUP:-1000000}" --freeze-norm)
for s in 1 2 3; do
    name="$GROUP-ppo-seed$s"
    "$PY" -u -m scripts.train_precision_striker --envs 32 --vector subproc --rollout 256 --device cpu \
        --checkpoint-every 500000 --steps "$PPO_STEPS" --seed "$s" \
        --require-hit --timeout-penalty 10 --extra-obs --goal-shaping --wide-obstacles --obstacles 1 \
        --obstacles-max 3 --paddle-restitution 0.8 --single-hit --block-line-prob "$BLOCK" \
        --aim-shaping 5 --prefer-direct "${PPO_FT[@]}" \
        --resume "$IM" --resume-new-task --output "$OUT/$name" > "$OUT/$name.log" 2>&1 &
done
wait

# 4. PPO learning curves: every 4th checkpoint (every 2M steps) plus the final policy
for s in 1 2 3; do
    run="$OUT/$GROUP-ppo-seed$s"
    [ -f "$run/policy.zip" ] && "$PY" -u -m scripts.eval_suite --run "$run" --seed 50000 --workers 4 \
        --only "${CONDS[@]}" --output "$run/suite_val50000.json" > "$run/suite.log" 2>&1 &
    for n in $(ls "$run/checkpoints" 2>/dev/null | grep -o 'policy_[0-9]*_steps.zip' | grep -o '[0-9]*' | sort -n | awk 'NR % 4 == 0'); do
        d="$run/curve/ckpt-$n"; mkdir -p "$d"
        cp "$run/checkpoints/policy_${n}_steps.zip" "$d/policy.zip"
        cp "$run/checkpoints/policy_vecnormalize_${n}_steps.pkl" "$d/vecnormalize.pkl"
        cp "$run/manifest.json" "$d/"
        "$PY" -u -m scripts.eval_suite --run "$d" --seed 50000 --shots 100 --workers 4 --only "${CONDS[@]}" \
            --output "$d/suite_val50000.json" > "$d.log" 2>&1 &
        while [ "$(jobs -r | wc -l)" -ge 28 ]; do sleep 5; done
    done
done
wait
echo "$(hostname) group $GROUP done $(date)"
