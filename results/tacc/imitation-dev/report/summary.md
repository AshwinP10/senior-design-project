# Imitation vs imitation + PPO

Teacher (physics striker with simulator aim correction), validation mean: **98.8%**.

| Group | Pure imitation | + DAgger (200-shot val) | + PPO (best val) | PPO change | Labelled steps |
|---|---|---|---|---|---|
| D1: 1M steps/iter, 256 net | 10.2% | 33.8% | 16.8% | -17.0 pts | 5M |
| D2: 1M steps/iter, 512 net | 15.2% | 36.0% | 10.8% | -25.2 pts | 5M |
| D3: 2M steps/iter, 512 net | 20.0% | 42.9% | 16.8% | -26.1 pts | 10M |
| D4: 1M steps/iter, 256 net, 50% blocked | 16.5% | 31.2% | 16.0% | -15.2 pts | 5M |

## Test shots (seed 30000, chosen on validation)

- **Physics striker**: mean 97.1% (1 block 99.5%, 2 blocks 97.0%, 3 blocks 96.0%, block on straight line 96.0%)
- **Imitation**: mean 37.9% (1 block 57.0%, 2 blocks 40.0%, 3 blocks 23.5%, block on straight line 31.0%)
- **Imitation + PPO**: mean 16.5% (1 block 26.5%, 2 blocks 14.5%, 3 blocks 10.0%, block on straight line 15.0%)

## What this means

- **Imitation alone** reaches 42.9% at best, which is 55.9 points below the teacher it copies. The gap is the part of the teacher's skill the network has not learned to copy.
- **DAgger vs plain copying:** plain copying (iteration 0) scored 20.0% at best; letting the network play and correcting its mistakes raised that to 42.9%.
- **PPO:** the best PPO gain over its starting imitation network is -15.2 points (mean -20.9). Verdict: PPO does not add accuracy on top of imitation here.

Figures: imitation_curve.png, ppo_curve.png, comparison.png, by_condition.png; GIFs in gifs/.
