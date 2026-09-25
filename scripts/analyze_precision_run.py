"""Build measured tables, figures, and Markdown from completed experiment files."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs'
FIG = DOC / 'figures'
DATA = DOC / 'experiment_data'
FINAL = ROOT / 'runs/incoming-v2-final'


def read(path):
    return json.loads(path.read_text())


def summary(data):
    r = data['records']
    return {'n':len(r), 'goals':sum(x['success'] for x in r),
            'contact_goals':sum(x['success'] and x['hit'] for x in r),
            'hits':sum(x['hit'] for x in r),
            'collisions':sum(x['obstacle_collision'] for x in r),
            'duration':float(np.mean([x['steps']*.05 for x in r])),
            'outcomes':dict(Counter(x['outcome'] for x in r))}


def savefig(name):
    plt.savefig(FIG/name, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    manifest = read(FINAL/'manifest.json')
    assert manifest['completed_timesteps'] == 1048576
    cases = [('Initial, new final scenarios', ROOT/'runs/incoming-v2/final-comparison'),
             ('Final, nominal', FINAL/'evaluation-200'),
             ('Slow incoming 0.3-0.7 m/s', FINAL/'speed-slow'),
             ('Fast incoming 1.1-1.5 m/s', FINAL/'speed-fast'),
             ('Unseen speed 1.5-2.0 m/s', FINAL/'speed-unseen'),
             ('Two obstacles (unseen)', FINAL/'two-obstacles'),
             ('Three obstacles (unseen)', FINAL/'three-obstacles')]
    evals = [(name, folder, read(folder/'evaluation.json')) for name,folder in cases]
    nominal = evals[1][2]
    records = nominal['records']
    stats = summary(nominal)
    initial_records = evals[0][2]['records']
    assert [r['seed'] for r in initial_records] == [r['seed'] for r in records]
    initial_stats = summary(evals[0][2])
    gained = sum(b['success'] and not a['success'] for a,b in zip(initial_records,records))
    lost = sum(a['success'] and not b['success'] for a,b in zip(initial_records,records))
    baselines = read(FINAL/'baselines.json')
    for name, folder, data in evals:
        shutil.copy2(folder/'evaluation.json', DATA/(folder.parent.name+'-'+folder.name+'.json'))
    shutil.copy2(FINAL/'baselines.json', DATA/'baselines.json')
    for run in ['incoming-v2', 'incoming-v2-final']:
        shutil.copy2(ROOT/'runs'/run/'manifest.json', DATA/(run+'-manifest.json'))
    layout = []
    for j in range(10):
        row = [r for r in records if r['layout']==j]
        layout.append({'layout':j, **summary({'records':row})})
    rates = np.array([r['goals']/r['n'] for r in layout])
    boot = np.random.RandomState(42).choice(rates,(10000,10),replace=True).mean(axis=1)
    cluster_ci = np.percentile(boot,[2.5,97.5]).tolist()
    traces = [np.load(FINAL/'evaluation-200'/r['trajectory']) for r in records]
    latency = np.concatenate([x['inference_ms'] for x in traces])
    actions = np.concatenate([x['actions'] for x in traces])
    obs = np.concatenate([x['observations'] for x in traces])
    latency_stats = {k:float(v) for k,v in zip(['mean','median','p95','p99','max'],
                    [latency.mean(),*np.percentile(latency,[50,95,99]),latency.max()])}
    saturation = float(np.any(np.abs(actions)>=.9999, axis=1).mean())
    stage_data = []
    all_rows = []
    offset = 0
    for run, budget in [('incoming-v2',262144),('incoming-v2-final',786432)]:
        rows=[]
        for path in (ROOT/'runs'/run).glob('worker_*.monitor.csv'):
            with path.open() as f:
                meta=json.loads(f.readline()[1:])
                for row in csv.DictReader(f):
                    row['absolute_time']=meta['t_start']+float(row['t'])
                    row['r']=float(row['r']); row['l']=int(row['l'])
                    rows.append(row)
        rows.sort(key=lambda r:r['absolute_time'])
        for row in rows:
            row['success']=row['is_success']=='True'
            row['collision']=row['obstacle_collision']=='True'
        logs=(ROOT/'runs'/ (run+'.log')).read_text(errors='replace')
        elapsed=re.findall(r'\|\s*time_elapsed\s*\|\s*(\d+)', logs)
        outcomes=dict(Counter(r['outcome'] for r in rows))
        stage_data.append({'run':run,'transitions':budget,'episodes':len(rows),
                           'successes':sum(r['success'] for r in rows),'outcomes':outcomes,
                           'logged_seconds':int(elapsed[-1]) if elapsed else None})
        x=offset+np.cumsum([r['l'] for r in rows])
        all_rows.extend([{**r,'step':int(t)} for t,r in zip(x,rows)])
        offset+=budget
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
    window=100
    xx=np.array([r['step'] for r in all_rows])[window-1:]
    for field,label,color in [('success','Scoring','#157c91'),('collision','Obstacle contact','#d36348')]:
        yy=np.convolve([r[field] for r in all_rows],np.ones(window)/window,mode='valid')
        axs[0].plot(xx/1e6,100*yy,label=label,color=color,lw=1.3)
    axs[0].set_ylabel('Episodes (%)'); axs[0].legend(loc='upper right')
    axs[1].plot(xx/1e6,np.convolve([r['r'] for r in all_rows],np.ones(window)/window,mode='valid'),color='#314d71')
    axs[1].set_ylabel('Shaped return'); axs[1].set_xlabel('Approximate completed-episode transitions (millions)')
    for ax in axs:
        ax.axvline(.262144,color='#888888',ls='--',lw=1); ax.grid(alpha=.15)
    fig.suptitle('Training behavior: rolling 100 completed episodes')
    fig.tight_layout(); savefig('training_curves.png')
    fig,ax=plt.subplots(figsize=(9,3.5))
    labels=['Initial PPO','Final PPO','Zero action','Random action']
    vals=[summary(evals[0][2])['goals'],stats['goals'],baselines['zero_action']['successes'],baselines['random_action']['successes']]
    ax.bar(labels,np.array(vals)/2,color=['#9fb9c8','#157c91','#8d969b','#c1c7cc'])
    ax.axhline(80,color='#bf694e',ls='--',label='Design target: 80%')
    for j,v in enumerate(vals): ax.text(j,v/2+2,f'{v}/200',ha='center')
    ax.set_ylim(0,100); ax.set_ylabel('Successful episodes (%)'); ax.legend(); ax.set_title('Matched final scenarios, seed 30000')
    fig.tight_layout(); savefig('final_comparison.png')
    fig,ax=plt.subplots(figsize=(9,3.6))
    x=np.arange(10); w=.36
    ax.bar(x-w/2,[r['goals'] for r in layout],w,label='Goals',color='#157c91')
    ax.bar(x+w/2,[r['collisions'] for r in layout],w,label='Collision episodes',color='#d36348')
    ax.set_xticks(x,[str(j+1) for j in x]);ax.set_xlabel('Held-out layout');ax.set_ylabel('Episodes out of 20');ax.set_ylim(0,20);ax.legend()
    fig.tight_layout();savefig('per_layout.png')
    fig,axs=plt.subplots(2,2,figsize=(9,8.5))
    chosen=[]
    for ax,outcome in zip(axs.flat,['goal','obstacle_collision','conceded','timeout']):
        trial=next((r for r in records if r['outcome']==outcome),None)
        if trial is None:
            ax.text(.5,.5,'No '+outcome+' in test',ha='center',transform=ax.transAxes);ax.axis('off');continue
        chosen.append(trial)
        tr=np.load(FINAL/'evaluation-200'/trial['trajectory'])['observations']
        ax.add_patch(Rectangle((-.4318,-.9652),.8636,1.9304,fill=False,edgecolor='#80929c'))
        for bx,by in trial['blocks']:ax.add_patch(Rectangle((by-.035,bx-.035),.07,.07,color='#d36348'))
        ax.plot(tr[:,5],tr[:,4],label='Puck',color='#c38a23',lw=1.3)
        ax.plot(tr[:,1],tr[:,0],label='Paddle',color='#157c91',lw=1,alpha=.8)
        ax.scatter(tr[0,5],tr[0,4],s=25,color='#c38a23',marker='o')
        ax.scatter(tr[-1,5],tr[-1,4],s=35,color='#c38a23',marker='x')
        for gx in [-.9652,.9652]:ax.plot([-.12,.12],[gx,gx],color='#56a36d',lw=4)
        ax.axhline(0,color='#b8c4ca',lw=.5)
        ax.set_xlim(-.51,.51);ax.set_ylim(1.16,-1.16);ax.set_aspect('equal')
        ax.set_title(f"{outcome.replace('_',' ').title()} | seed {trial['seed']}",fontsize=10)
        ax.set_xlabel('Lateral y (m)');ax.set_ylabel('Longitudinal x (m)')
    handles,labels=next(ax for ax in axs.flat if ax.lines).get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.tight_layout(rect=[0,.025,1,1]);savefig('example_trajectories.png')
    hashes={name:hashlib.sha256((FINAL/name).read_bytes()).hexdigest() for name in ['policy.zip','vecnormalize.pkl']}
    analysis={'final':stats,'matched_gained_goals':gained,'matched_lost_goals':lost,
              'layout_bootstrap_95':cluster_ci,'latency_ms':latency_stats,
              'action_saturation_fraction':saturation,'per_layout':layout,'training':stage_data,
              'max_observed_paddle_speed':float(np.linalg.norm(obs[:,2:4],axis=1).max()),
              'max_observed_puck_speed':float(np.linalg.norm(obs[:,6:8],axis=1).max()),
              'model_sha256':hashes,'selected_trajectories':chosen}
    (DATA/'analysis.json').write_text(json.dumps(analysis,indent=2))
    example = next(r for r in records if r['seed'] == 31090 and r['success'] and r['hit'])
    (DATA/'example_contact_goal.json').write_text(json.dumps(example,indent=2))
    bundle = ROOT/'models/precision-striker-v2'
    bundle.mkdir(parents=True, exist_ok=True)
    for filename in ['policy.zip','vecnormalize.pkl','manifest.json']:
        shutil.copy2(FINAL/filename,bundle/filename)
    card = f'''# Precision Striker V2 model card

Experimental state-based PPO controller, trained September 24, 2026.

- Task: intercept an incoming puck and score around one static obstacle.
- Training: 1,048,576 transitions, one training seed, local CPU.
- Architecture: separate 20-128-128-2 actor and 20-128-128-1 critic; Tanh hidden units; 38,789 total parameters.
- Nominal incoming speeds: 0.3 to 1.5 m/s. One square obstacle, randomized position.
- Final test: {stats['goals']}/200 goals; {stats['collisions']}/200 obstacle-collision episodes; design acceptance {'met' if nominal['meets_report_target'] else 'NOT met'}.
- Goals with a detected paddle contact: {stats['contact_goals']}/200. Other counted goals occurred without detected paddle contact and are not evidence of learned striking.
- Intended use: inspect, reproduce, and extend a simulated learning experiment.
- Limitations: exact simulator state, static obstacles, no camera pipeline or robot dynamics; no physical deployment validation. Multi-obstacle and higher-speed behavior is out of distribution.
- Runtime used: Python 3.11.5, PyTorch 2.11.0+cpu, Stable-Baselines3 2.2.1. The proposed older CUDA cluster environment has not been tested with this bundle.

Run from the repository root with a graphical OpenCV installation:

```sh
python -m scripts.watch_precision_striker --run models/precision-striker-v2
```

Keep policy.zip and vecnormalize.pkl together. Evaluation freezes normalization and uses deterministic mean actions. Load only model/pickle files from a trusted source.

Policy SHA-256: {hashes['policy.zip']}

Normalizer SHA-256: {hashes['vecnormalize.pkl']}

See [full technical report](../../docs/TECHNICAL_REPORT.md), [runbook](../../docs/PRECISION_STRIKER_PLAN.md), and manifest.json for exact source hashes and experiment settings. The source commit in the historical manifest predates the v2 working-tree edits; the recorded hashes identify the code actually used.
'''
    (bundle/'README.md').write_text(card,encoding='utf-8')
    pct=lambda n,d=200:f'{100*n/d:.1f}%'
    ci=nominal['wilson_95_interval']
    lines=['# Precision Striker: RL Model, Training and Results',
           '', 'Technical experiment report | September 24, 2026 | Task version: precision-striker-v2-incoming',
           '', '**Completed training budget: 1,048,576 environment transitions.** One PPO policy was trained in two stages on local CPU, then evaluated in the actual Box2D environment. This was not a TACC GPU run.',
           '',f"**Final nominal result: {stats['goals']}/200 successful shots ({pct(stats['goals'])}), with {stats['collisions']}/200 obstacle-collision episodes ({pct(stats['collisions'])}).** The System Design target of at least 80% scoring and zero obstacle collisions was {'met in this simulation test' if nominal['meets_report_target'] else 'not met'}. Finishing the training budget does not mean the control problem has been solved.",
           '', 'The deliverables are a real trained checkpoint and its normalization, an interactive simulator viewer, reproducible evaluation traces, measured failure analysis, and a complete technical specification. The numerical results below come from saved files; unperformed hardware and remote-compute tests remain explicitly identified.',
           '', '## Reading guide', '', 'Part A presents measured training and test results, uncertainty, failure examples, and artifact provenance. Part B specifies the task, physics, neural policy, control interface, reward, PPO update, validation protocol, TACC execution model, and remaining experiments. The separate PRECISION_STRIKER_PLAN.md contains copyable training and demonstration commands.',
           '', '<!-- pagebreak -->', '', '## Part A. Measured results', '', '### A1. Experiment budget and execution',
           '', '| Stage | Transitions | Completed episodes | Training goals | Logged seconds |', '|---|---|---|---|---|']
    for s in stage_data:lines.append(f"| {s['run']} | {s['transitions']:,} | {s['episodes']:,} | {s['successes']} | {s['logged_seconds']} |")
    lines += ['', 'Training wall times are the final SB3 time_elapsed values, excluding process startup and later evaluation. The second stage resumed weights and optimizer but changed from four sequential environments to eight worker processes. Concurrent desktop work and evaluation prevent a controlled throughput comparison. Training outcomes use stochastic exploration and are not held-out performance.',
              '', '![Figure 1. Rolling scoring/contact rates and shaped return. The dashed line marks the change to eight worker processes. Steps are approximated from completed episode lengths; incomplete episodes are omitted.](figures/training_curves.png)',
              '', '### A2. Final unseen-layout test and simple controls',
              '', '| Controller | Goals / attempts | Scoring | Collision episodes | Paddle contacts |', '|---|---|---|---|---|']
    for name,data in [('Initial PPO, 262,144 steps',summary(evals[0][2])),('Final PPO, 1,048,576 steps',stats)]:
        lines.append(f"| {name} | {data['goals']}/200 | {pct(data['goals'])} | {data['collisions']} | {data['hits']} |")
    for name,key in [('Zero action','zero_action'),('Seeded random action','random_action')]:
        b=baselines[key];lines.append(f"| {name} | {b['successes']}/200 | {pct(b['successes'])} | {b['collisions']} | {b['hits']} |")
    lines += ['', 'All four controllers use the same 200 reset seeds and obstacle layouts (evaluation seed 30000). Zero action holds the velocity target at zero; random action samples uniformly in the action box each decision. They are simple sanity controls, not strong planning baselines. The initial checkpoint was re-evaluated on these scenarios for a matched comparison; its earlier development evaluation used seed 10000 and scored 12/200 with 38 collision episodes.',
              '', f"On the matched scenarios, continued training changed the scoring rate by {(stats['goals']-initial_stats['goals'])/2:+.1f} percentage points and collision count by {stats['collisions']-initial_stats['collisions']:+d} episodes. The final policy gained success on {gained} scenarios that the initial policy failed, and lost success on {lost} scenarios that the initial policy scored. This is a within-run comparison; additional transitions, a changed worker count, and the resumed process state are not independently controlled factors.",
              '', f"Conceded goals changed from {initial_stats['outcomes'].get('conceded',0)} to {stats['outcomes'].get('conceded',0)}, while timeouts changed from {initial_stats['outcomes'].get('timeout',0)} to {stats['outcomes'].get('timeout',0)}. The outcome distribution therefore shifted substantially toward episodes that remain in play without scoring. The final policy's small scoring margin over the random-action control ({pct(stats['goals'])} versus {pct(baselines['random_action']['successes'])}) does not establish robust superiority from a single trained seed and this test alone. The evidence supports an experimental baseline, not a reliable obstacle-avoiding striker.",
              '', f"The final score-rate Wilson 95% interval is [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]. Resampling the ten layout success rates 10,000 times gives a layout-bootstrap interval of [{100*cluster_ci[0]:.1f}%, {100*cluster_ci[1]:.1f}%]. These describe finite-test uncertainty, not variability across independently trained policies. Only one training seed was used.",
              '', '![Figure 2. Matched final test scoring rates. The target line is an acceptance requirement, not achieved performance.](figures/final_comparison.png)',
              '', '| Final outcome | Count | Fraction |', '|---|---|---|']
    for k,v in stats['outcomes'].items():lines.append(f"| {k.replace('_',' ')} | {v} | {pct(v)} |")
    lines += ['',f"Paddle contact occurred in {stats['hits']}/200 episodes ({pct(stats['hits'])}); mean episode duration was {stats['duration']:.2f} simulated seconds. Contact alone is insufficient: the task also requires a goal and no obstacle collision. Timeouts can represent missed interception, weak returns, or a puck remaining in play; the outcome label alone cannot distinguish these mechanisms.",
              '', f"Of the final policy's {stats['goals']} counted goals, {stats['contact_goals']} included a detected paddle contact and {stats['goals']-stats['contact_goals']} did not. Thus the contact-qualified scoring rate is {pct(stats['contact_goals'])}. The zero-action control scored {baselines['zero_action']['successes']} times while commanding no motion; a stationary paddle can still be hit, and wall rebounds can produce goals. Raw goal count therefore does not by itself establish learned striking skill. A revised task should require an appropriate striker contact and report contact-qualified results, with a new task version and new training/evaluation rather than retroactively changing this experiment's labels.",
              '', f"For comparison, zero action produced {sum(r['success'] and r['hit'] for r in baselines['zero_action']['records'])} contact-qualified goals and random action produced {sum(r['success'] and r['hit'] for r in baselines['random_action']['records'])}. Even the contact-qualified metric does not require an intentional or beneficial commanded strike; the baseline controls remain necessary.",
              '', '### A3. Layout variability', '', '| Layout | Goals / 20 | Collisions | Contacts |', '|---|---|---|---|']
    for r in layout:lines.append(f"| {r['layout']+1} | {r['goals']} | {r['collisions']} | {r['hits']} |")
    lines += ['', '![Figure 3. Goals and obstacle-collision episodes per held-out layout. Each layout has twenty randomized incoming shots.](figures/per_layout.png)',
              '', '### A4. Speed and obstacle-count stress tests',
              '', '| Test distribution | Attempts | Goals | Scoring | Collisions |', '|---|---|---|---|---|']
    for name,folder,data in evals[2:]:
        s=summary(data);lines.append(f"| {name} | {s['n']} | {s['goals']} | {pct(s['goals'],s['n'])} | {s['collisions']} |")
    lines += ['', 'Each stress test uses six layouts with ten shots each and a separate documented seed. These small exploratory sets have different scenarios, so differences cannot be attributed solely to speed/count without paired tests. Two and three obstacles populate slots that were absent throughout one-obstacle training; poor results would not be evidence that the trained model had already learned variable-count avoidance. High-speed results above 1.5 m/s are extrapolation. No stress test was used to select a different checkpoint.',
              '', '### A5. Measured inference and action behavior', '', '| Measurement | Value |', '|---|---|']
    for k,v in latency_stats.items():lines.append(f'| Normalization + actor prediction, {k} | {v:.4f} ms |')
    lines += [f'| Recorded policy decisions | {len(latency):,} |',f'| Decisions at an action-box bound | {100*saturation:.1f}% |',
              f"| Maximum sampled paddle speed | {analysis['max_observed_paddle_speed']:.6f} m/s |",
              f"| Maximum sampled puck speed | {analysis['max_observed_puck_speed']:.6f} m/s |", '',
              'Timings use perf_counter around observation normalization and deterministic model.predict on CPU, including the first call. They exclude Box2D stepping, rendering, camera exposure/transfer, state estimation, communication, and physical actuation. Concurrent training/evaluation can affect outliers. The action-bound fraction counts steps with either returned action coordinate at magnitude at least 0.9999; it is not the fraction of time the paddle physically travels at its speed cap. Sampled speed checks are at policy boundaries, not a proof over every contact instant.',
              '', '### A6. Representative success and failure traces',
              '', '![Figure 4. First occurrence of each shown outcome in the final test, selected by category rather than visual quality. Lines are sampled center trajectories; circles mark puck starts and crosses mark endpoints. The opposing goal is at negative x.](figures/example_trajectories.png)',
              '', 'The red square is the fixed obstacle, gold is puck motion, and blue is paddle motion. A path intersecting an obstacle is a failed trial even if the puck could otherwise reach the goal. Center traces omit body radii and intermediate 200 Hz states, so they supplement contact-event labels rather than replacing the collision detector. A visually attractive goal demonstrates one successful rollout, not a reliable policy.',
              '', 'The baseline and final counts above measure whether more training improved this run. They do not isolate the reason. Reward design favors avoiding a costly collision over a risky attempt that may time out cheaply; delayed goal rewards, incoming-state variability, and exploration at contact are plausible contributors. The report proposes explicit ablations instead of asserting that these explanations were experimentally proven.',
              '', '### A7. Delivered artifacts and provenance', '', 'The feature branch contains the v2 source, these report sources, compact evaluation JSON, figures, and a model bundle under models/precision-striker-v2. Full logs, periodic checkpoints, TensorBoard data, and NPZ traces remain in the local runs directories. The interactive launcher selects the local final run, or the bundled model on a fresh checkout. The saved controller is experimental and the documented failure rate applies.',
              '', '| Artifact | SHA-256 |', '|---|---|']
    for name,h in hashes.items():lines.append(f'| {name} | {h} |')
    lines += ['', 'The environment source hash recorded during training is '+manifest['source_sha256']['scripts/precision_striker_env.py']+'. The four source hashes and library versions are preserved in docs/experiment_data/incoming-v2-final-manifest.json. The historical training commit predates uncommitted v2 edits, so hashes are necessary to identify the actual implementation.',
              '', '### A8. Investigation of apparent centerline resets',
              '', 'The project folder contains five inherited GIFs per repository copy: assets/puck_vel.gif, assets/puck_goal_pos.gif, and results/sac/puck_touch/eval_0.gif through eval_2.gif. These were already tracked in the repository and are not recordings of this PPO experiment. Their task goals and episode boundaries differ. Because the specific animation was not identified, the precise cause of the reported blank/reset in that old animation remains unconfirmed.',
              '', 'The current v2 environment has no puck-centerline termination condition. A new regression test drives a puck continuously across x=0 in both directions and verifies that the same body remains alive, the step count advances, and the outcome stays running. All eight environment tests pass. The saved 200-shot final test contains 145 positive-to-negative centerline crossings and 197 negative-to-positive crossings; multiple crossings may occur in an episode, so these are events rather than episode counts.',
              '', 'For a concrete learned-policy example, final-test seed 31090 with its recorded obstacle layout crosses toward the opponent half at policy step 21 and scores at step 38, ending at x=-1.00136 m. This is saved as docs/experiment_data/example_contact_goal.json and can be replayed through the actual model using the viewer --scenario option. It is deliberately selected to demonstrate continuity, not to imply a high success rate.',
              '', 'The viewer now holds a terminal frame and displays its outcome until N is pressed. A toggles automatic advance after two seconds; pausing also suspends that advance. This makes a real episode boundary visible rather than immediately replacing the puck with the next reset. These are presentation and regression-test changes; the trained dynamics, rewards, weights, and measured evaluation results were not altered.',
              '', '<!-- pagebreak -->', '', '## Part B. Model and approach', '']
    method=(DOC/'MODEL_AND_APPROACH.md').read_text(encoding='utf-8').splitlines()
    method_start = next(i for i,line in enumerate(method) if line.startswith('## 1. '))
    lines.extend(method[method_start:])
    (DOC/'TECHNICAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
