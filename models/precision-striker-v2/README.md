# Precision Striker V2 model card

Experimental state-based PPO controller, trained September 24, 2026.

- Task: intercept an incoming puck and score around one static obstacle.
- Training: 1,048,576 transitions, one training seed, local CPU.
- Architecture: separate 20-128-128-2 actor and 20-128-128-1 critic; Tanh hidden units; 38,789 total parameters.
- Nominal incoming speeds: 0.3 to 1.5 m/s. One square obstacle, randomized position.
- Final test: 18/200 goals; 37/200 obstacle-collision episodes; design acceptance NOT met.
- Goals with a detected paddle contact: 9/200. Other counted goals occurred without detected paddle contact and are not evidence of learned striking.
- Intended use: inspect, reproduce, and extend a simulated learning experiment.
- Limitations: exact simulator state, static obstacles, no camera pipeline or robot dynamics; no physical deployment validation. Multi-obstacle and higher-speed behavior is out of distribution.
- Runtime used: Python 3.11.5, PyTorch 2.11.0+cpu, Stable-Baselines3 2.2.1. The proposed older CUDA cluster environment has not been tested with this bundle.

Run from the repository root with a graphical OpenCV installation:

```sh
python -m scripts.watch_precision_striker --run models/precision-striker-v2
```

Keep policy.zip and vecnormalize.pkl together. Evaluation freezes normalization and uses deterministic mean actions. Load only model/pickle files from a trusted source.

Policy SHA-256: ff1a86bb9236f6839d397b94cc81ee4e6ad98032eb6b808654d5c737fbd69a49

Normalizer SHA-256: c856a3be5d07dc16c51fe68a8891f84b669d6fb34667d081a90851a59ce88639

See [full technical report](../../docs/TECHNICAL_REPORT.md), [runbook](../../docs/PRECISION_STRIKER_PLAN.md), and manifest.json for exact source hashes and experiment settings. The source commit in the historical manifest predates the v2 working-tree edits; the recorded hashes identify the code actually used.
