# Imitation vs imitation + PPO

Teacher (physics striker with simulator aim correction), validation mean: **98.8%**.

| Group | Pure imitation | + DAgger (200-shot val) | + PPO (best val) | PPO change | Labelled steps |
|---|---|---|---|---|---|
| H1: hints, 1M examples per loop, 512 net | 18.5% | 61.2% | 85.8% | +24.5 pts | 10M |
| H2: hints, 0.5M examples per loop, 512 net | 17.8% | 60.4% | 88.5% | +28.1 pts | 10M |
| H3: hints, 2M examples per loop, 512 net | 27.5% | 70.4% | 90.8% | +20.4 pts | 20M |
| H4: hints, 1M per loop, PPO on 2-3 blocks | 8.8% | 67.4% | 91.8% | +24.4 pts | 10M |

## Test shots (seed 30000, chosen on validation)

- **Physics striker**: mean 97.1% (1 block 99.5%, 2 blocks 97.0%, 3 blocks 96.0%, block on straight line 96.0%)
- **Imitation**: mean 68.2% (1 block 81.5%, 2 blocks 68.5%, 3 blocks 60.0%, block on straight line 63.0%)
- **Imitation + PPO**: mean 87.4% (1 block 91.0%, 2 blocks 91.0%, 3 blocks 80.0%, block on straight line 87.5%)

## What this means

- **Imitation alone** reaches 70.4% at best, which is 28.4 points below the teacher it copies. The gap is the part of the teacher's skill the network has not learned to copy.
- **DAgger vs plain copying:** plain copying (iteration 0) scored 27.5% at best; letting the network play and correcting its mistakes raised that to 70.4%.
- **PPO:** the best PPO gain over its starting imitation network is +28.1 points (mean +24.3). Verdict: PPO clearly helps.

Figures: imitation_curve.png, ppo_curve.png, comparison.png, by_condition.png; GIFs in gifs/.
