"""Matched-scenario sanity controls; these are not the learned policy."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.precision_striker_env import PrecisionStrikerEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.evaluation.read_text())
    env = PrecisionStrikerEnv(obstacles=1, randomize=False)
    results = {}
    try:
        for controller in ['zero_action', 'random_action']:
            records = []
            for trial in data['records']:
                env.reset(seed=trial['seed'], options={'blocks': trial['blocks']})
                rng = np.random.RandomState(trial['seed'] + 700000)
                for step in range(200):
                    action = np.zeros(2) if controller == 'zero_action' else rng.uniform(-1, 1, 2)
                    _, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                records.append({'layout': trial['layout'], 'shot': trial['shot'],
                                'seed': trial['seed'], 'success': info['is_success'],
                                'outcome': info['outcome'], 'hit': info['hit'],
                                'obstacle_collision': info['obstacle_collision'], 'steps': step+1})
            results[controller] = {'episodes': len(records),
                                   'successes': sum(x['success'] for x in records),
                                   'collisions': sum(x['obstacle_collision'] for x in records),
                                   'hits': sum(x['hit'] for x in records),
                                   'outcomes': dict(Counter(x['outcome'] for x in records)),
                                   'records': records}
            print(controller, {k: v for k, v in results[controller].items() if k != 'records'}, flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
