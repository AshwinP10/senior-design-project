"""Charts, GIFs and a plain-language summary for an imitation_sweep.slurm output directory.

    python -m scripts.analysis.imitation_report --sweep $WORK/airhockey-runs/imitation-long

Reads, per group (D1-D4): the DAgger history (manifest.json, or the JSON lines in <group>-imitation.log
if the run was cut short), the best imitation network's strict-suite score, and the PPO checkpoint scores.
Missing scores are computed first (validation seed 50000). Models are chosen on validation; the chosen
imitation and PPO networks are then scored once on the test shots (seed 30000).
Writes <sweep>/report/: imitation_curve.png, ppo_curve.png, comparison.png, by_condition.png,
gifs/*.gif, summary.json, summary.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = ["random_1", "random_2", "random_3", "blocked_1"]
COND_LABEL = {"random_1": "1 block", "random_2": "2 blocks", "random_3": "3 blocks", "blocked_1": "block on\nstraight line"}
GROUP_TEXT = {"D1": "D1: 1M examples per loop, 256 net", "D2": "D2: 1M examples per loop, 512 net",
              "D3": "D3: 2M examples per loop, 512 net", "D4": "D4: 1M examples per loop, 256 net, 50% blocked",
              "H1": "H1: hints, 1M examples per loop, 512 net", "H2": "H2: hints, 0.5M examples per loop, 512 net",
              "H3": "H3: hints, 2M examples per loop, 512 net", "H4": "H4: hints, 1M per loop, PPO on 2-3 blocks"}
GROUP_TEXT = type("Labels", (dict,), {"__missing__": lambda self, k: k})(GROUP_TEXT)
COL = {"D1": "#2a78d6", "D2": "#eb6834", "D3": "#1baf7a", "D4": "#6250d6",
       "H1": "#2a78d6", "H2": "#eb6834", "H3": "#1baf7a", "H4": "#6250d6"}
GREY, INK, MUTED = "#a9a8a2", "#0b0b0b", "#52514e"


def _suite(job):
    run, controller, seed, shots, out = job
    from scripts.eval_suite import run_suite
    res = run_suite(run, controller, seed, shots, workers=4, overrides={"sim": 1} if controller else None,
                    conditions=CONDS)
    Path(out).write_text(json.dumps(res, indent=1))
    return out


def ensure_scores(sweep, groups, parallel):
    """Score whatever the training job did not get to (checkpoint curves, finals, the teacher)."""
    jobs = []
    teacher = sweep / "teacher_suite_val50000.json"
    if not teacher.exists():
        jobs.append((None, True, 50000, 200, str(teacher)))
    for g in groups:
        im = sweep / f"{g}-imitation"
        if (im / "policy.zip").exists() and not (im / "suite_val50000.json").exists():
            jobs.append((str(im), False, 50000, 200, str(im / "suite_val50000.json")))
        for run in sorted(sweep.glob(f"{g}-ppo-seed*")):
            if not run.is_dir():
                continue
            if (run / "policy.zip").exists() and not (run / "suite_val50000.json").exists():
                jobs.append((str(run), False, 50000, 200, str(run / "suite_val50000.json")))
            ck = run / "checkpoints"
            steps = []
            if ck.exists():
                for f in ck.glob("policy_*_steps.zip"):
                    m = re.search(r"policy_(\d+)_steps\.zip", f.name)
                    if m:
                        steps.append(int(m.group(1)))
                steps.sort()
            pick = steps[3::4] + ([steps[-1]] if steps and steps[-1] not in steps[3::4] else [])
            for n in pick:
                d = run / "curve" / f"ckpt-{n}"
                if (d / "suite_val50000.json").exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                for src, dst in ((f"policy_{n}_steps.zip", "policy.zip"), (f"policy_vecnormalize_{n}_steps.pkl", "vecnormalize.pkl")):
                    if (ck / src).exists() and not (d / dst).exists():
                        (d / dst).write_bytes((ck / src).read_bytes())
                if not (d / "manifest.json").exists() and (run / "manifest.json").exists():
                    (d / "manifest.json").write_bytes((run / "manifest.json").read_bytes())
                if (d / "policy.zip").exists() and (d / "vecnormalize.pkl").exists():
                    jobs.append((str(d), False, 50000, 100, str(d / "suite_val50000.json")))
    if jobs:
        print(f"scoring {len(jobs)} missing evaluations", flush=True)
        with ProcessPoolExecutor(parallel) as pool:
            for out in pool.map(_suite, jobs):
                print("scored", out, flush=True)


def load(sweep, groups):
    data = {}
    for g in groups:
        im = sweep / f"{g}-imitation"
        hist = []
        if (im / "manifest.json").exists():
            hist = json.loads((im / "manifest.json").read_text()).get("history", [])
        if not hist and (sweep / f"{g}-imitation.log").exists():
            hist = [json.loads(l) for l in (sweep / f"{g}-imitation.log").read_text().splitlines() if l.startswith("{")]
        entry = {"history": hist, "imitation_suite": None, "ppo": {}}
        if (im / "suite_val50000.json").exists():
            entry["imitation_suite"] = json.loads((im / "suite_val50000.json").read_text())
        for run in sorted(sweep.glob(f"{g}-ppo-seed*")):
            if not run.is_dir():
                continue
            pts = []
            for f in run.glob("curve/ckpt-*/suite_val50000.json"):
                n = int(re.search(r"ckpt-(\d+)", str(f)).group(1))
                pts.append((n, json.loads(f.read_text())))
            final = json.loads((run / "suite_val50000.json").read_text()) if (run / "suite_val50000.json").exists() else None
            entry["ppo"][run.name] = {"curve": sorted(pts, key=lambda p: p[0]), "final": final, "dir": str(run)}
        data[g] = entry
    return data


def _style(ax, ylabel):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ebeae6")
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def charts(data, teacher, out):
    tmean = 100 * teacher["mean_accuracy"] if teacher else None
    # 1. imitation curve
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for g, e in data.items():
        if not e["history"]:
            continue
        # x = DAgger round, so setups with different data per round line up round for round.
        x = [h["iteration"] for h in e["history"]]
        y = [100 * h["validation_accuracy"] for h in e["history"]]
        ax.plot(x, y, marker="o", ms=4, lw=1.8, color=COL.get(g, GREY), label=GROUP_TEXT[g])
    if tmean:
        ax.axhline(tmean, ls="--", color=INK, lw=1)
        ax.text(ax.get_xlim()[0], tmean + 1.5, f"teacher (physics striker) {tmean:.1f}%", fontsize=8, color=INK)
    _style(ax, "Validation goals (mean of 4 cases)")
    ax.set_xlabel("Training loop number (loop 0 = the striker plays and the network only copies)", color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Imitation only. One loop = play new shots, the striker says the right move at every step, retrain",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "imitation_curve.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # 2. PPO curve
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for g, e in data.items():
        start = 100 * e["imitation_suite"]["mean_accuracy"] if e["imitation_suite"] else None
        first = True
        for name, r in e["ppo"].items():
            xs = [0.0] if start is not None else []
            ys = [start] if start is not None else []
            xs += [n / 1e6 for n, _ in r["curve"]]
            ys += [100 * d["mean_accuracy"] for _, d in r["curve"]]
            if len(xs) > 1:
                ax.plot(xs, ys, color=COL.get(g, GREY), lw=1.4, alpha=0.85, marker="o", ms=3, label=GROUP_TEXT[g] if first else None)
                first = False
    if tmean:
        ax.axhline(tmean, ls="--", color=INK, lw=1)
        ax.text(0, tmean + 1.5, f"teacher {tmean:.1f}%", fontsize=8, color=INK)
    _style(ax, "Validation goals (mean of 4 cases)")
    ax.set_xlabel("PPO training steps after imitation (millions); 0 = the imitation network", color=MUTED, fontsize=9)
    ax.set_title("Imitation + PPO: does reinforcement learning add accuracy? (3 seeds per group)", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "ppo_curve.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def summarize(data, teacher):
    rows = {}
    for g, e in data.items():
        h = e["history"]
        bc = 100 * h[0]["validation_accuracy"] if h else None
        dag = 100 * max(x["validation_accuracy"] for x in h) if h else None
        im = 100 * e["imitation_suite"]["mean_accuracy"] if e["imitation_suite"] else None
        best_ppo, best_src = None, None
        for name, r in e["ppo"].items():
            cands = [(100 * d["mean_accuracy"], f"{name} @ {n / 1e6:.1f}M") for n, d in r["curve"]]
            if r["final"]:
                cands.append((100 * r["final"]["mean_accuracy"], f"{name} final"))
            for v, src in cands:
                if best_ppo is None or v > best_ppo:
                    best_ppo, best_src = v, src
        rows[g] = {"pure_imitation_iter0": bc, "best_dagger_iteration_val100": dag, "imitation_suite_val200": im,
                   "best_ppo_val": best_ppo, "best_ppo_source": best_src,
                   "ppo_gain_points": None if best_ppo is None or im is None else best_ppo - im,
                   "labelled_steps_millions": h[-1]["total_steps"] / 1e6 if h else None}
    return rows


def comparison_chart(rows, teacher, out):
    groups = list(rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    stages = [("pure_imitation_iter0", "Pure imitation (copying)", GREY), ("imitation_suite_val200", "Imitation + DAgger", "#2a78d6"),
              ("best_ppo_val", "Imitation + PPO", "#eb6834")]
    x = np.arange(len(groups))
    for i, (k, lab, c) in enumerate(stages):
        vals = [rows[g][k] if rows[g][k] is not None else 0 for g in groups]
        bars = ax.bar(x + (i - 1) * 0.26, vals, 0.24, color=c, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=MUTED)
    if teacher:
        ax.axhline(100 * teacher["mean_accuracy"], ls="--", color=INK, lw=1, label=f"Teacher {100 * teacher['mean_accuracy']:.1f}%")
    ax.set_xticks(x, [GROUP_TEXT[g].replace(": ", "\n", 1) for g in groups], fontsize=8)
    _style(ax, "Validation goals (mean of 4 cases)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.14))
    fig.savefig(out / "comparison.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def by_condition_chart(results, out):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    names = list(results)
    cols = [GREY, "#2a78d6", "#eb6834"]
    x = np.arange(len(CONDS))
    w = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        vals = [100 * results[n][c]["accuracy"] for c in CONDS]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w * 0.95, color=cols[i % 3], label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x, [COND_LABEL[c] for c in CONDS], fontsize=8.5)
    _style(ax, "Test goals (200 shots each)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
    fig.savefig(out / "by_condition.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def gifs(models, out, per_kind=3):
    """Successful and failed test shots for each model, as GIFs (one per shot plus a grid per model)."""
    from scripts.analysis.render_episode_gifs import Canvas, montage, record, save_gif
    from scripts.eval_suite import CONDITIONS, RULES, _policy
    from scripts.precision_striker_env import PrecisionStrikerEnv
    gdir = out / "gifs"
    gdir.mkdir(exist_ok=True)
    canvas = Canvas(width=720)
    for label, run, controller in models:
        act, reset, opts = _policy(run, controller, {"sim": 1} if controller else None)
        clips, found = [], {"goal": 0, "miss": 0}
        for cond in CONDS:
            env = PrecisionStrikerEnv(seed=30000, randomize=False, **{**RULES, **opts, **CONDITIONS[cond]})
            for i in range(60):
                kind_left = [k for k, v in found.items() if v < per_kind * (CONDS.index(cond) + 1) / len(CONDS) + 0.99]
                if not kind_left:
                    break
                reset()
                frames, info = record(env, act, {"seed": 31000 + i}, f"{label} | {COND_LABEL[cond].replace(chr(10), ' ')}",
                                      canvas=canvas)
                kind = "goal" if info.get("is_success") else "miss"
                if found[kind] >= per_kind or kind not in kind_left:
                    continue
                found[kind] += 1
                name = f"{label.replace(' ', '_').replace('+', 'plus')}_{cond}_{kind}{found[kind]}.gif"
                save_gif(frames, gdir / name)
                clips.append(frames)
            env.close()
        if clips:
            montage(clips[:6], 3, gdir / f"{label.replace(' ', '_').replace('+', 'plus')}_grid.gif")
        print("gifs for", label, found, flush=True)


def write_summary(rows, teacher, test, out):
    t = 100 * teacher["mean_accuracy"] if teacher else float("nan")
    lines = ["# Imitation vs imitation + PPO", "",
             f"Teacher (physics striker with simulator aim correction), validation mean: **{t:.1f}%**.", "",
             "| Group | Pure imitation | + DAgger (200-shot val) | + PPO (best val) | PPO change | Labelled steps |",
             "|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.1f}%"
    for g, r in rows.items():
        gain = "-" if r["ppo_gain_points"] is None else f"{r['ppo_gain_points']:+.1f} pts"
        steps = "-" if r["labelled_steps_millions"] is None else f"{r['labelled_steps_millions']:.0f}M"
        lines.append(f"| {GROUP_TEXT[g]} | {fmt(r['pure_imitation_iter0'])} | {fmt(r['imitation_suite_val200'])} | "
                     f"{fmt(r['best_ppo_val'])} | {gain} | {steps} |")
    lines += ["", "## Test shots (seed 30000, chosen on validation)", ""]
    for name, res in test.items():
        lines.append(f"- **{name}**: mean {100 * res['mean_accuracy']:.1f}% (" +
                     ", ".join(f"{COND_LABEL[c].replace(chr(10), ' ')} {100 * res[c]['accuracy']:.1f}%" for c in CONDS) + ")")
    gains = [r["ppo_gain_points"] for r in rows.values() if r["ppo_gain_points"] is not None]
    ims = [r["imitation_suite_val200"] for r in rows.values() if r["imitation_suite_val200"] is not None]
    lines += ["", "## What this means", ""]
    if ims:
        best_im = max(ims)
        lines.append(f"- **Imitation alone** reaches {best_im:.1f}% at best, which is {t - best_im:.1f} points below the teacher it copies. "
                     + ("The network copies the teacher closely." if t - best_im < 5 else
                        "The gap is the part of the teacher's skill the network has not learned to copy."))
        bc = [r["pure_imitation_iter0"] for r in rows.values() if r["pure_imitation_iter0"] is not None]
        if bc:
            lines.append(f"- **DAgger vs plain copying:** plain copying (iteration 0) scored {max(bc):.1f}% at best; "
                         f"letting the network play and correcting its mistakes raised that to {best_im:.1f}%.")
    if gains:
        g = max(gains)
        verdict = ("PPO clearly helps" if g >= 3 else "PPO helps a little" if g >= 1 else
                   "PPO does not add accuracy on top of imitation here")
        lines.append(f"- **PPO:** the best PPO gain over its starting imitation network is {g:+.1f} points (mean {np.mean(gains):+.1f}). "
                     f"Verdict: {verdict}.")
    lines += ["", "Figures: imitation_curve.png, ppo_curve.png, comparison.png, by_condition.png; GIFs in gifs/."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--groups", nargs="*", help="Default: every <group>-imitation in the sweep")
    p.add_argument("--parallel", type=int, default=24, help="Evaluations at once (each uses 4 processes)")
    p.add_argument("--no-gifs", action="store_true")
    args = p.parse_args()
    if not args.groups:
        args.groups = sorted({p.name.split("-imitation")[0] for p in args.sweep.glob("*-imitation*")})
    groups = [g for g in args.groups if (args.sweep / f"{g}-imitation").exists() or (args.sweep / f"{g}-imitation.log").exists()]
    out = args.sweep / "report"
    out.mkdir(exist_ok=True)
    ensure_scores(args.sweep, groups, args.parallel)
    data = load(args.sweep, groups)
    tf = args.sweep / "teacher_suite_val50000.json"
    teacher = json.loads(tf.read_text()) if tf.exists() else None
    rows = summarize(data, teacher)
    charts(data, teacher, out)
    comparison_chart(rows, teacher, out)
    # Chosen on validation: best imitation network and best PPO network (final policy or checkpoint).
    best_im = max(((r["imitation_suite_val200"], g) for g, r in rows.items() if r["imitation_suite_val200"] is not None), default=None)
    best_pp = None
    for g, e in data.items():
        for name, r in e["ppo"].items():
            if r["final"]:
                cand = (100 * r["final"]["mean_accuracy"], r["dir"])
                best_pp = cand if best_pp is None or cand > best_pp else best_pp
            for n, d in r["curve"]:
                cand = (100 * d["mean_accuracy"], str(Path(r["dir"]) / "curve" / f"ckpt-{n}"))
                best_pp = cand if best_pp is None or cand > best_pp else best_pp
    chosen = [("Physics striker", None, True)]
    if best_im:
        chosen.append(("Imitation", str(args.sweep / f"{best_im[1]}-imitation"), False))
    if best_pp:
        chosen.append(("Imitation + PPO", best_pp[1], False))
    test = {}
    for label, run, controller in chosen:
        f = out / f"test30000_{label.replace(' ', '_').replace('+', 'plus')}.json"
        if not f.exists():
            _suite((run, controller, 30000, 200, str(f)))
        test[label] = json.loads(f.read_text())
    by_condition_chart(test, out)
    summary = {"groups": rows, "teacher_val": teacher and teacher["mean_accuracy"],
               "chosen": {l: r for l, r, _ in chosen}, "test": {k: v["mean_accuracy"] for k, v in test.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_summary(rows, teacher, test, out)
    if not args.no_gifs:
        gifs(chosen, out)
    print("report written to", out, flush=True)


if __name__ == "__main__":
    main()
