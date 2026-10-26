"""Build the dated progress-report PDF (figures + plain-language explanations) from saved results.

    python -m scripts.analysis.build_report --out output/report/Precision_Striker_Progress_2026-10-21.pdf

Inputs (all produced earlier and kept on disk):
  runs/eval_suite/*.json                               strict-suite results (validation 50000, test 30000)
  docs/experiment_data/train_curves_strict_aim.csv     training success per 0.5M steps (from TACC monitors)
  output/report/shot_classes_AB-seed3-bank_test30000.json   per-shot classes from shot_compilation.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

DATE = "October 21, 2026"
COND = ["random_1", "blocked_1", "random_1to3", "blocked_1to3"]
COND_LABEL = {"random_1": "1 block,\nrandom spot", "blocked_1": "1 block on the\nstraight line",
              "random_1to3": "1-3 blocks,\nrandom spots", "blocked_1to3": "1-3 blocks,\nline blocked"}
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a9a8a2", "#0b0b0b", "#52514e"
FIG = Path("output/report/figures")


def load(p):
    return json.loads(Path(p).read_text())


def style_axes(ax, ylabel=None, pct=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if pct:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------- figures
def fig_progress():
    labels = ["First PPO\n(Sept)", "Imitation\n+ PPO 4M", "Imitation\n+ PPO 12M", "DAgger + PPO\n(W1-seed2)"]
    vals = [9.0, 60.5, 79.5, 84.5]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(labels, vals, color=[GREY, GREY, GREY, BLUE], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", color=INK, fontsize=9)
    style_axes(ax, "Goals on 200 test shots")
    ax.set_title("Original rules, one block: accuracy over the project", loc="left", fontsize=11, color=INK)
    return save(fig, "01_progress_original_rules")


def fig_strict(results):
    names = [("Old RL model (W1-seed2)", GREY), ("Hand-coded controller", ORANGE), ("New RL model (AB-seed3-bank)", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(COND) + 1)
    w = 0.26
    for i, (n, c) in enumerate(names):
        r = results[n]
        vals = [100 * r[k]["accuracy"] for k in COND] + [100 * r["mean_accuracy"]]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=c, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", color=MUTED, fontsize=7)
    ax.set_xticks(x, [COND_LABEL[k] for k in COND] + ["MEAN"], fontsize=8)
    style_axes(ax, "Goals (200 test shots each)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13))
    return save(fig, "02_strict_suite_test")


def _curves():
    rows = defaultdict(list)
    with open("docs/experiment_data/train_curves_strict_aim.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["sweep"], r["run"])].append((int(r["bin_end_steps"]) / 1e6, 100 * float(r["success"]),
                                                  100 * float(r["missed"]), 100 * float(r["double_hit"])))
    return rows


def fig_training_curves(curves):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for (sweep, run), pts in sorted(curves.items()):
        if sweep != "aim-long":
            continue
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        x, y = x[:-1], y[:-1]                      # last bin is a partial one (training stopped mid-bin)
        ys = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        if run == "AB-seed3":
            ax.plot(x, ys, color=BLUE, lw=2.2, label="AB-seed3 (the run that learned bank shots)", zorder=5)
        elif run.startswith("AR"):
            ax.plot(x, ys, color=ORANGE, lw=1.2, ls="--", alpha=0.9,
                    label="From scratch (no imitation)" if run == "AR-seed1" else None)
        else:
            ax.plot(x, ys, color=GREY, lw=1.1, alpha=0.9, label="Other fine-tuned runs" if run == "AA-seed1" else None)
    style_axes(ax, "Training success rate")
    ax.set_xlabel("Training steps in this run (millions)", color=MUTED, fontsize=9)
    ax.set_title("Training curves, strict rules + aim reward (12 runs, 12-hour job)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, "03_training_curves")


def fig_aim_vs_none(curves):
    """Same starting network (W1-seed2), same seeds 1-3, same task: without (SA) vs with (AB) the aim reward."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    groups = (("strict-dev", "SA", "No aim reward (3 seeds)", GREY), ("aim-long", "AB", "With aim reward (3 seeds)", BLUE))
    means = {}
    for sweep, prefix, label, c in groups:
        runs = [pts for (s, r), pts in curves.items() if s == sweep and r.startswith(prefix)]
        for i, pts in enumerate(runs):
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            m = x <= 3.0
            ax.plot(x[m], y[m], color=c, lw=1.6, alpha=0.9, marker="o", ms=3, label=label if i == 0 else None)
        means[prefix] = np.mean([[p[1] for p in pts if p[0] == 3.0][0] for pts in runs])
    style_axes(ax, "Training success rate")
    ax.set_ylim(0, 50)
    ax.set_xlabel("Training steps (millions), first 3M steps from the same starting network", color=MUTED, fontsize=9)
    ax.set_title("Same starting network, with vs. without the aim reward", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, "04_aim_vs_no_aim"), means


def fig_checkpoints():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, run, title in ((axes[0], "AB-seed3", "AB-seed3 (learned bank shots)"), (axes[1], "AA-seed1", "AA-seed1 (typical run)")):
        pts = []
        for f in glob.glob(f"runs/eval_suite/ckpt_{run}_ckpt-*_val50000.json"):
            n = int(re.search(r"ckpt-(\d+)_", f).group(1))
            pts.append((n, load(f)))
        if not pts:
            continue
        pts.sort()
        x = np.array([(n - pts[0][0]) / 1e6 + 0.25 for n, _ in pts])
        for key, c, lab in (("blocked_1", BLUE, "1 block on the straight line"), ("random_1", AQUA, "1 random block"),
                            ("mean_accuracy", INK, "Mean of all 4 cases")):
            y = [100 * (d[key] if key == "mean_accuracy" else d[key]["accuracy"]) for _, d in pts]
            ax.plot(x, y, color=c, lw=2 if key == "mean_accuracy" else 1.4, marker="o", ms=3, label=lab)
        style_axes(ax, "Validation goals" if ax is axes[0] else None)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Steps in this run (millions)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Saved checkpoints scored on validation shots (100 per case)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, "05_checkpoint_curves")


def fig_seeds():
    rows = []
    for f in glob.glob("runs/eval_suite/aimrun_*_val50000.json"):
        name = Path(f).stem.replace("aimrun_", "").replace("_val50000", "").replace("long-", "12 h job: ").replace("dev-", "2 h job: ")
        rows.append((name, 100 * load(f)["mean_accuracy"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    cols = [BLUE if n == "12 h job: AB-seed3" else ORANGE if "AR" in n else GREY for n, _ in rows]
    bars = ax.barh([n for n, _ in rows], [v for _, v in rows], color=cols, height=0.65)
    for b, (_, v) in zip(bars, rows):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.set_xlabel("Mean validation accuracy, strict suite", color=MUTED, fontsize=9)
    ax.set_title("Every aim-reward run (blue = chosen model, orange = from scratch)", loc="left", fontsize=11)
    return save(fig, "06_all_runs")


def fig_failures(classes):
    fails = Counter(r["kind"] for r in classes if r["outcome"] != "goal")
    names = {"clipped_block": "Clipped a block", "bank_wide": "Bank shot off target", "wide": "Just wide of the goal",
             "double_touch": "Touched the puck twice", "no_shot": "No clean hit in time", "other": "Other"}
    items = fails.most_common()
    total = sum(fails.values())
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bars = ax.barh([names[k] for k, _ in items][::-1], [100 * c / total for _, c in items][::-1], color=ORANGE, height=0.6)
    for b, (k, c) in zip(bars, items[::-1]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{c} ({100 * c / total:.0f}%)", va="center",
                fontsize=8, color=MUTED)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.set_title(f"Why shots fail: all {total} misses out of {len(classes)} test shots", loc="left", fontsize=11)
    return save(fig, "07_failure_types"), fails, total


def fig_shot_types(classes):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    kinds = [("straight", "Straight shot", AQUA), ("bank", "Bank shot (off the side rail)", BLUE), ("ricochet", "Ricochet off a block", YELLOW)]
    bottom = np.zeros(len(COND))
    for k, lab, c in kinds:
        vals = np.array([100 * sum(1 for r in classes if r["condition"] == cnd and r["kind"] == k) /
                         sum(1 for r in classes if r["condition"] == cnd) for cnd in COND])
        ax.bar([COND_LABEL[c2] for c2 in COND], vals, bottom=bottom, color=c, label=lab, width=0.55,
               edgecolor="white", linewidth=1)
        bottom += vals
    for i, v in enumerate(bottom):
        ax.text(i, v + 2, f"{v:.0f}% goals", ha="center", fontsize=8, color=INK)
    style_axes(ax, "Share of test shots")
    ax.set_xticks(range(len(COND)), [COND_LABEL[c] for c in COND], fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.12))
    return save(fig, "08_goal_shot_types")


def fig_blocks(classes):
    by = defaultdict(lambda: [0, 0])
    for r in classes:
        if r["condition"] in ("random_1", "random_1to3"):
            by[r["blocks"]][0] += r["outcome"] == "goal"
            by[r["blocks"]][1] += 1
    ks = sorted(by)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    vals = [100 * by[k][0] / by[k][1] for k in ks]
    bars = ax.bar([f"{k} block{'s' * (k > 1)}" for k in ks], vals, color=BLUE, width=0.5)
    for b, k, v in zip(bars, ks, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%  (n={by[k][1]})", ha="center", fontsize=8, color=MUTED)
    style_axes(ax, "Goals, random layouts")
    ax.set_title("New model: accuracy by number of blocks (test shots)", loc="left", fontsize=10.5)
    return save(fig, "09_by_block_count"), {k: (by[k][0], by[k][1]) for k in ks}


# --------------------------------------------------------------------------------------------- PDF
def build(out):
    results = {"Old RL model (W1-seed2)": load("runs/eval_suite/W1-seed2_test30000.json"),
               "Hand-coded controller": load("runs/eval_suite/controller_test30000.json"),
               "New RL model (AB-seed3-bank)": load("runs/eval_suite/aimlong_AB-seed3_test30000.json")}
    classes = load("output/report/shot_classes_AB-seed3-bank_test30000.json")
    curves = _curves()
    f1, f2 = fig_progress(), fig_strict(results)
    (f4, aim_means), f3, f5, f6 = fig_aim_vs_none(curves), fig_training_curves(curves), fig_checkpoints(), fig_seeds()
    (f7, fails, nfail), f8 = fig_failures(classes), fig_shot_types(classes)
    f9, by_blocks = fig_blocks(classes)
    new, old, ctl = (results[k] for k in ("New RL model (AB-seed3-bank)", "Old RL model (W1-seed2)", "Hand-coded controller"))
    pct = lambda r, k: f"{100 * (r[k] if k == 'mean_accuracy' else r[k]['accuracy']):.1f}%"
    goals = [r for r in classes if r["outcome"] == "goal"]
    bank_share = 100 * sum(r["kind"] == "bank" for r in goals) / len(goals)
    open_bank = [r for r in goals if r["condition"] == "random_1"]
    open_bank_share = 100 * sum(r["kind"] == "bank" for r in open_bank) / len(open_bank)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor(MUTED))
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16, spaceBefore=6, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=10, spaceAfter=4)
    title = ParagraphStyle("t", parent=h1, fontSize=24, leading=29, alignment=TA_CENTER)
    sub = ParagraphStyle("st", parent=body, alignment=TA_CENTER, fontSize=11.5, textColor=colors.HexColor(MUTED))
    box = ParagraphStyle("box", parent=body, fontSize=10, leading=14, leftIndent=4)
    P = lambda t, s=body: Paragraph(t, s)

    def img(path, width=6.6):
        im = Image(str(path))
        im.drawHeight = width * inch * im.imageHeight / im.imageWidth
        im.drawWidth = width * inch
        return im

    def figure(path, caption, interp, width=6.6):
        return KeepTogether([img(path, width), P(caption, small), P("<b>What this means:</b> " + interp)])

    def table(data, widths, header=True):
        t = Table(data, colWidths=[w * inch for w in widths])
        st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9a9890")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f2")]),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.8))
        t.setStyle(TableStyle(st))
        return t

    def callout(lines, colour="#eef4fc"):
        t = Table([[P(l, box)] for l in lines], colWidths=[6.6 * inch])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(colour)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (0, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
        return t

    s = []
    # ---------------------------------------------------------------- title + summary
    s += [Spacer(1, 1.2 * inch), P("Precision Striker", title), P("Reinforcement-learning air hockey: training progress report", sub),
          Spacer(1, 6), P(f"<b>{DATE}</b>", sub), Spacer(1, 4),
          P("Akshat Mehta, Ashwin Prakash, Siddhartha Palle, Charan Pradeep - UT Austin senior design", sub),
          Spacer(1, 0.45 * inch)]
    s.append(callout([
        "<b>Bottom line.</b> A pure reinforcement-learning (RL) policy now scores around blocks under realistic rules "
        f"(one paddle touch, no artificial speed-up). On 800 held-out test shots it scores <b>{pct(new, 'mean_accuracy')}</b> "
        f"on average, up from {pct(old, 'mean_accuracy')} for our previous model, and slightly ahead of our hand-coded "
        f"controller ({pct(ctl, 'mean_accuracy')}).",
        f"<b>The breakthrough:</b> it learned bank shots. With a block sitting on the straight line to the goal it scores "
        f"<b>{pct(new, 'blocked_1')}</b> (previous model: {pct(old, 'blocked_1')}; controller: {pct(ctl, 'blocked_1')}).",
        f"<b>What is left:</b> it now over-uses bank shots ({open_bank_share:.0f}% of its goals with one random block are banks, "
        "even when the straight shot is open), and most misses are aim errors of a few degrees that clip a block. "
        "Both are specific, measurable problems, and training aimed at them is already submitted to TACC.",
        "<b>Goal for the next round:</b> 80% or better on the strict test suite.",
    ]))
    s.append(PageBreak())

    # ---------------------------------------------------------------- explainer
    s += [P("1. What we are building", h1), P(
        "A robot paddle has to hit an incoming air-hockey puck so that it goes into the far goal, even when square blocks "
        "are placed in the way. Everything here runs in a physics simulator (Box2D) of a real table: the puck slides with "
        "light friction, bounces off rails and blocks, and the paddle has realistic speed and acceleration limits. The "
        "controller decides 20 times per second; the physics runs 200 times per second."),
          P("2. How the policy learns (plain language)", h1),
          P("The <b>policy</b> is a small neural network (two layers of 128 numbers). Twenty times a second it looks at where "
            "the paddle, the puck and the blocks are, and outputs how fast to move the paddle. No hand-written rules run "
            "during a shot: everything it does was learned."),
          table([["Step", "What it is", "Analogy"],
                 ["1. Imitation (DAgger)", "The network first copies a hand-coded controller that\ncomputes the puck's path and aims.", "Watching a coach, then\nbeing corrected"],
                 ["2. Reinforcement\nlearning (PPO)", "It then plays millions of shots and is rewarded for goals,\nso it improves beyond what it copied.", "Practice with a scoreboard"],
                 ["3. Aim reward (new)", "The instant the puck leaves the paddle, a trajectory\ncalculator checks whether it is heading along a clear\nstraight or bank line, and gives credit right away.", "A coach saying \"good angle\"\nbefore the puck lands"]],
                [1.5, 3.4, 1.7]),
          Spacer(1, 6),
          P("Why step 3 mattered: a bank shot (bouncing the puck off the side rail) almost never goes in by accident, so a "
            "network that only gets credit for goals almost never discovers it. The aim reward gives it a signal the moment "
            "it tries a good angle. Checking it on the controller's shots: goals left the paddle within 0.2 degrees of a clear "
            "line, misses about 4.5 degrees off, so the signal is accurate."),
          P("<b>Trajectory calculator.</b> Predicts the puck's path (sliding friction and rail bounces) and finds every clear "
            "straight or bank line into the goal. We use it in three places: to make sure every random block layout can "
            "actually be scored, to compute the aim reward, and inside the hand-coded controller we compare against."),
          P("3. The rules and how we test", h1),
          table([["Rule", "Setting"],
                 ["Paddle contacts per shot", "One. A second touch, or the puck coming back, ends the shot as a miss"],
                 ["Paddle bounce", "Restitution 0.8 (the original simulator's 1.0 made the puck speed up)"],
                 ["Blocks", "1-3, anywhere on the far half; half of training layouts block the straight line"],
                 ["Fair layouts", "Every random layout is verified scorable by the trajectory calculator"],
                 ["Incoming puck", "Random position, 0.3-1.5 m/s, random angle"]], [2.0, 4.6]),
          Spacer(1, 6),
          P("<b>Strict test suite:</b> 4 cases x 200 shots = 800 shots: one random block; one block deliberately on the straight "
            "line; 1-3 random blocks; 1-3 blocks with the straight line blocked. Models are <b>chosen on validation shots</b> "
            "(seed 50000) and <b>reported once on separate test shots</b> (seed 30000) that were never used for any decision, "
            "so the numbers are not cherry-picked."),
          PageBreak()]

    # ---------------------------------------------------------------- results
    s += [P("4. Results", h1),
          figure(f2, "Figure 1. Strict test suite, same 800 test shots for all three. Bars are % of 200 shots that scored.",
                 f"The new RL policy is slightly ahead of the hand-coded controller on average ({pct(new, 'mean_accuracy')} vs "
                 f"{pct(ctl, 'mean_accuracy')}) and is far better when the straight line is blocked "
                 f"({pct(new, 'blocked_1')} vs {pct(ctl, 'blocked_1')}). The old model could essentially only shoot straight "
                 f"({pct(old, 'blocked_1')} when that was blocked). The new model's weak spot is the opposite: layouts where "
                 f"a straight shot is open but it banks anyway ({pct(new, 'random_1')} with one random block)."),
          table([["Case (200 test shots each)", "Old RL model", "Controller", "New RL model"]] +
                [[COND_LABEL[k].replace("\n", " "), pct(old, k), pct(ctl, k), pct(new, k)] for k in COND] +
                [["Mean", pct(old, "mean_accuracy"), pct(ctl, "mean_accuracy"), pct(new, "mean_accuracy")]],
                [2.6, 1.3, 1.3, 1.4]),
          Spacer(1, 4),
          P("Consistency check: on two independent validation sets the new model scored 67.6% and 65.6%, close to its 69.5% "
            "test score, so the result is stable and not luck of the test draw.", small),
          PageBreak(),
          figure(f8, "Figure 2. What kind of shot scored, per case (test shots).",
                 f"{bank_share:.0f}% of all goals are bank shots. That is why the blocked-line cases are now the strongest: "
                 "the policy found one reliable strategy (go around via the rail) and uses it almost everywhere. Using banks "
                 "when a straight shot is open costs accuracy, because a bank travels further and a small angle error grows."),
          figure(f9, "Figure 3. Random layouts only, grouped by how many blocks were on the table.",
                 "More blocks means narrower clear lanes, so accuracy drops with block count. For comparison, the old "
                 "model fell from 91.5% (1 block) to 12.5% (2) and 6% (3) on validation shots even under the easier "
                 "original rules; the new model holds up much better under the stricter rules.", 5.6),
          PageBreak(),
          figure(f1, "Figure 4. History under the original, easier rules (multiple touches allowed, paddle bounce adds speed).",
                 "This is how we got here. These numbers are not comparable with Figure 1: once we enforced one touch and "
                 "realistic bounce, the 84.5% model dropped to 27.4% on the strict test suite. That was the honest reset "
                 "that led to the current work.")]
    s.append(PageBreak())

    # ---------------------------------------------------------------- training curves
    s += [P("5. Training curves", h1),
          figure(f3, "Figure 5. Success rate during training, 12 runs of the 12-hour TACC job (30M steps each). "
                     "Success here includes random exploration and a 50% share of blocked-line layouts.",
                 "Most fine-tuned runs climb and then flatten around 40-50%. One run, AB-seed3, keeps climbing after the "
                 "others flatten: that is where it discovered bank shots. Runs trained from scratch (orange) stay near "
                 "10-20%: without the head start from imitation, the network never learns to make a clean strike under "
                 "the one-touch rule. So the head start plus the aim reward together are what work."),
          figure(f4, "Figure 6. The same starting network (W1-seed2) and the same three seeds, trained without and with "
                     "the aim reward (first 3M steps).",
                 f"With the aim reward, training success at 3M steps averages {aim_means['AB']:.0f}% versus "
                 f"{aim_means['SA']:.0f}% without it. The gap is real but modest, and one no-aim seed does as well. The "
                 "clearer evidence is what the runs end up learning: the best run without the aim reward reached 32% on "
                 "blocked-line shots (and lost accuracy elsewhere), while an aim-reward run reached 85%. Caveat: the runs "
                 "without the aim reward were stopped after about 2.5M steps, so a full-length comparison run is on the "
                 "next-steps list to confirm how much of the gain is due to the aim reward."),
          PageBreak(),
          figure(f5, "Figure 7. Saved checkpoints scored on validation shots during training.",
                 "For the successful run, blocked-line accuracy climbs steadily from about 0% to 81%. Early on, the "
                 "one-random-block score falls from 78% (the starting network) to 45% as the network starts trying bank "
                 "shots, then both climb together. The typical run never makes that switch: its blocked-line score stays "
                 "below 15%. The successful run's mean is still rising at the end, so longer training should keep helping."),
          figure(f6, "Figure 8. Final validation accuracy of every aim-reward run.",
                 "Results vary a lot between runs that differ only in their random seed. One run found the "
                 "bank-shot strategy (one of the eleven runs that finished). This is common in RL, and it tells us the next round should run more seeds in "
                 "parallel and continue from the best one, which is what is running now.")]
    s.append(PageBreak())

    # ---------------------------------------------------------------- failures
    order = [k for k, _ in fails.most_common()]
    label = {"clipped_block": "Clipped a block", "bank_wide": "Bank shot off target", "wide": "Just wide",
             "double_touch": "Touched twice", "no_shot": "No clean hit", "other": "Other"}
    explain = {"clipped_block": "Aim a few degrees off; the puck hit a block and bounced back.",
               "bank_wide": "Bounced off the side rail at the wrong angle; missed the goal.",
               "wide": "Reached the far wall just beside the goal opening.",
               "double_touch": "The paddle touched the puck a second time (not allowed).",
               "no_shot": "Slow or awkward incoming puck; no clean strike before time ran out.", "other": "Other."}
    s += [P("6. Failure analysis", h1),
          figure(f7, "Figure 9. Every miss on the 800 test shots, sorted by cause (from replaying each shot).",
                 "Nearly all misses are the right idea executed slightly off: the policy chooses a sensible shot, but the "
                 "angle is a few degrees wrong, which over 1.5 m of travel is enough to clip a 7 cm block or miss a 24 cm "
                 "goal mouth. Rule violations (double touches) and no-shot failures are rare. Aim precision is therefore "
                 "the single biggest lever for the next accuracy gain."),
          table([["Failure type", "Count", "Plain explanation"]] +
                [[label[k], f"{fails[k]} ({100 * fails[k] / nfail:.0f}%)", explain[k]] for k in order],
                [1.5, 0.9, 4.2]),
          Spacer(1, 8),
          P("<b>Video.</b> <i>strike_compilation_2026-10-21.mp4</i> (about 2 minutes) shows 10 successful and 9 failed test "
            "shots, each labelled with its case and an explanation, at half speed. Every clip is a real test shot; the "
            "success clips cover straight shots, bank shots around a blocked line and multi-block layouts, and the failure "
            "clips cover each failure type above.")]
    s.append(PageBreak())

    # ---------------------------------------------------------------- direction
    s += [P("7. Where we stand", h1),
          table([["We know", "Evidence"],
                 ["The physics and rules are realistic and fair", "One touch, no speed-up, every layout verified scorable"],
                 ["Pure RL can solve the hard case", "85.5% with the straight line blocked (was 7%)"],
                 ["Why it works", "Aim reward + imitation head start (Figures 5-7); from-scratch runs fail"],
                 ["What limits it now", "Over-use of bank shots; few-degree aim errors (Figures 2, 9)"],
                 ["Results are honest", "Chosen on validation, reported once on unseen test shots"]], [2.7, 3.9]),
          P("8. Next steps", h1),
          P("<b>A. Training (submitted to TACC, jobs 3526475 and 3526476, waiting for nodes).</b> Continue from the bank-shot model so it "
            "keeps the new skill while getting straight shots back:"),
          table([["Variant", "Change", "Why"],
                 ["BA", "Lower learning rate (1e-4, decaying)", "Small careful updates; protect the bank-shot skill"],
                 ["BB", "Same learning rate as before", "Baseline for the continuation"],
                 ["BC", "Straight line blocked in 30% (was 50%)", "More open layouts, so straight shots pay off again"],
                 ["BD", "Score every checkpoint of AB-seed3", "Exact learning curve; best stopping point"]], [0.7, 2.6, 3.3]),
          Spacer(1, 6),
          P("<b>B. If the continuation stalls below 80%,</b> in order:"),
          P("1. <b>Reward the safest shot, not just any clear shot.</b> Give the aim reward more weight for the straight line "
            "when it is open and for lines with more clearance from blocks, so bank shots are used only when needed.<br/>"
            "2. <b>Sharpen the aim reward over time</b> (a curriculum): reward angles within 3 degrees now, then 2, then 1, "
            "since few-degree errors cause most misses.<br/>"
            "3. <b>More seeds, keep the best.</b> One run in twelve found bank shots; run 24+ seeds from the best model and "
            "continue the top few (a simple population-based search).<br/>"
            "4. <b>Give the network the geometry.</b> Add the trajectory calculator's clear-shot directions as extra inputs, "
            "so it chooses among good lines instead of having to work them out.<br/>"
            "5. <b>Bigger network and longer runs</b> where the curves are still rising (Figure 7).<br/>"
            "6. <b>Control experiment:</b> a full-length run without the aim reward from the same start, to measure "
            "exactly how much the aim reward contributes (Figure 6 caveat)."),
          P("<b>C. Toward the real table.</b> Measure the real paddle and puck (bounce, friction, servo speed) and match the "
            "simulator; randomize these values during training so the policy tolerates the difference; then connect camera "
            "tracking of the puck (streamed at 20 Hz)."),
          P("<b>Success criteria for the next report:</b> at least 80% mean on the strict test suite, at least 75% on every "
            "single case, and no more than 2% rule violations, measured exactly as in this report."),
          PageBreak(),
          P("Appendix: run settings and files", h1),
          table([["Item", "Value"],
                 ["Chosen model", "models/tacc-2026-10/AB-seed3-bank (on GitHub)"],
                 ["Algorithm", "PPO (Stable-Baselines3 2.2.1), 32 parallel simulators, 128-128 MLP"],
                 ["Training history", "DAgger imitation -> 4M PPO -> 30M PPO with strict rules + aim reward"],
                 ["Compute", "TACC Lonestar6, 4 nodes x 12 h for the main job"],
                 ["Evaluation", "scripts/eval_suite.py; validation seeds 50000/60000, test seed 30000"],
                 ["Video", "scripts/analysis/shot_compilation.py"], ["This report", "scripts/analysis/build_report.py"],
                 ["Demo", "demo_rl.bat (live, click to place blocks), demo_showcase.bat"]], [1.6, 5.0])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(0.9 * inch, 0.55 * inch, f"Precision Striker progress report - {DATE}")
        canvas.drawRightString(7.6 * inch, 0.55 * inch, f"Page {doc.page}")
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.85 * inch,
                            title=f"Precision Striker progress report {DATE}", author="Precision Striker team")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("output/report/Precision_Striker_Progress_2026-10-21.pdf"))
    build(p.parse_args().out)
