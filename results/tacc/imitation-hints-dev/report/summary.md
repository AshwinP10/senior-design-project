# Imitation vs imitation + PPO

Teacher (physics striker with simulator aim correction), validation mean: **98.8%**.

| Group | Pure imitation | + DAgger (200-shot val) | + PPO (best val) | PPO change | Labelled steps |
|---|---|---|---|---|---|
| H1: hints, 1M examples per loop, 512 net | 18.5% | 66.4% | 84.5% | +18.1 pts | 5M |
| H2: hints, 0.5M examples per loop, 512 net | 17.8% | 62.5% | 82.8% | +20.2 pts | 5M |
| H3: hints, 2M examples per loop, 512 net | 27.5% | 70.2% | 85.0% | +14.8 pts | 10M |
| H4: hints, 1M per loop, PPO on 2-3 blocks | 8.8% | 72.1% | 83.1% | +11.0 pts | 5M |

## Test shots (seed 30000, chosen on validation)

- **Physics striker**: mean 97.1% (1 block 99.5%, 2 blocks 97.0%, 3 blocks 96.0%, block on straight line 96.0%)
- **Imitation**: mean 68.2% (1 block 81.5%, 2 blocks 69.0%, 3 blocks 56.0%, block on straight line 66.5%)
- **Imitation + PPO**: mean 80.4% (1 block 89.5%, 2 blocks 83.5%, 3 blocks 70.5%, block on straight line 78.0%)

## What this means

- **Imitation alone** reaches 72.1% at best, which is 26.6 points below the teacher it copies. The gap is the part of the teacher's skill the network has not learned to copy.
- **DAgger vs plain copying:** plain copying (iteration 0) scored 27.5% at best; letting the network play and correcting its mistakes raised that to 72.1%.
- **PPO:** the best PPO gain over its starting imitation network is +20.2 points (mean +16.0). Verdict: PPO clearly helps.

Figures: imitation_curve.png, ppo_curve.png, comparison.png, by_condition.png; GIFs in gifs/.
