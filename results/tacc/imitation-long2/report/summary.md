# Imitation vs imitation + PPO

Teacher (physics striker with simulator aim correction), validation mean: **98.8%**.

| Group | Pure imitation | + DAgger (200-shot val) | + PPO (best val) | PPO change | Labelled steps |
|---|---|---|---|---|---|
| D1: 1M steps/iter, 256 net | 10.2% | 36.4% | 63.5% | +27.1 pts | 10M |
| D2: 1M steps/iter, 512 net | 15.2% | 41.6% | 61.3% | +19.6 pts | 10M |
| D3: 2M steps/iter, 512 net | 20.0% | 46.2% | 67.5% | +21.2 pts | 20M |
| D4: 1M steps/iter, 256 net, 50% blocked | 16.5% | 38.2% | 58.0% | +19.8 pts | 10M |

## Test shots (seed 30000, chosen on validation)

- **Physics striker**: mean 97.1% (1 block 99.5%, 2 blocks 97.0%, 3 blocks 96.0%, block on straight line 96.0%)
- **Imitation**: mean 43.4% (1 block 60.0%, 2 blocks 41.0%, 3 blocks 27.0%, block on straight line 45.5%)
- **Imitation + PPO**: mean 64.0% (1 block 79.0%, 2 blocks 61.5%, 3 blocks 44.0%, block on straight line 71.5%)

## What this means

- **Imitation alone** reaches 46.2% at best, which is 52.5 points below the teacher it copies. The gap is the part of the teacher's skill the network has not learned to copy.
- **DAgger vs plain copying:** plain copying (iteration 0) scored 20.0% at best; letting the network play and correcting its mistakes raised that to 46.2%.
- **PPO:** the best PPO gain over its starting imitation network is +27.1 points (mean +21.9). Verdict: PPO clearly helps.

Figures: imitation_curve.png, ppo_curve.png, comparison.png, by_condition.png; GIFs in gifs/.
