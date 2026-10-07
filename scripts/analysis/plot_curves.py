"""Plot learning curves (test accuracy vs training steps) from curve_eval.py output.

Usage: python plot_curves.py OUT_PREFIX RULES curves.json [curves2.json ...]
  RULES: "trained" (each checkpoint scored with the rules it was trained on) or "bounce".
Accuracy = goals after a paddle hit / 200 test shots. Line = mean of the 3 training runs,
band = min-max across runs, dashed line = 80% target.
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe hues, fixed per version (color follows the version, never its rank).
COLORS = {"A": "#7F7F7F", "B": "#0072B2", "C": "#E69F00", "E": "#009E73", "F": "#CC79A7",
          "G": "#56B4E9", "H": "#D55E00", "I": "#000000",
          # Sweep 3 is always drawn on its own chart, so it reuses the first hues in fixed order.
          "J": "#0072B2", "K": "#E69F00", "L": "#009E73", "M": "#D55E00"}
LABELS = {"A": "A: old reward", "B": "B: goal needs hit", "C": "C: B + easy stage",
          "E": "E: bounce + easy stage", "F": "F: C fine-tuned (bounce)",
          "G": "G: E + hit/time inputs + aim reward", "H": "H: G + tuned PPO",
          "I": "I: H + 16 simulations", "J": "Shot choice (PPO)", "K": "Shot choice (SAC)",
          "L": "Residual PPO on controller", "M": "Imitation then PPO"}
INK, MUTED, GRID = "#1F1F1F", "#6B6B6B", "#E3E3E3"


def main():
    args = sys.argv[1:]
    refs, xunit = [], "steps"
    while "--ref" in args:      # --ref "Label=0.93": dotted horizontal reference line
        i = args.index("--ref")
        label, value = args[i + 1].rsplit("=", 1)
        refs.append((label, float(value)))
        del args[i:i + 2]
    if "--x" in args:           # --x shots: x axis in thousands of shots (shot-choice runs)
        i = args.index("--x")
        xunit = args[i + 1]
        del args[i:i + 2]
    out, rules, files = args[0], args[1], args[2:]
    rows = [r for f in files for r in json.load(open(f)) if r["rules"] == rules]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        # Round to the nearest 0.25M so the 3 seeds' checkpoints line up.
        x = (round(r["steps"] / 250_000) * 0.25 if xunit == "steps"
             else round(r["steps"] / 25_000) * 25)      # thousands of shots
        by[r["variant"]][x].append(100 * r["goals_after_hit"] / r["episodes"])
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    ax.axhline(80, color=INK, lw=1.2, ls=(0, (5, 4)))
    ax.text(0.995, 77, "Target 80%", color=INK, fontsize=11, ha="right", va="top", transform=ax.get_yaxis_transform())
    for label, value in refs:
        ax.axhline(100 * value, color=MUTED, lw=1.2, ls=(0, (1, 2)))
        ax.text(0.995, 100 * value + 1.2, f"{label} {100 * value:.0f}%", color=MUTED, fontsize=10,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
    table = []
    for v in sorted(by, key=lambda k: list(COLORS).index(k)):
        xs = sorted(by[v])
        means = [sum(by[v][x]) / len(by[v][x]) for x in xs]
        lo = [min(by[v][x]) for x in xs]
        hi = [max(by[v][x]) for x in xs]
        c = COLORS[v]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(xs, means, color=c, lw=2.2, marker="o", ms=4.5, label=LABELS[v],
                markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"{means[-1]:.0f}%", (xs[-1], means[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=11, color=INK)
        table += [(v, x, m, a, b) for x, m, a, b in zip(xs, means, lo, hi)]
    ax.set_xlabel("Training steps (millions)" if xunit == "steps" else "Training shots (thousands)",
                  color=MUTED, fontsize=11)
    ax.set_ylabel("Goals after a paddle hit (% of 200 test shots)", color=MUTED, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    rule_text = ("collisions end the shot" if rules == "trained" else "puck bounces off obstacles")
    ax.set_title(f"Test accuracy during training ({rule_text})", loc="left", fontsize=14, color=INK, pad=12)
    # Legend below the plot: the area near the target line is where good results land.
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, labelcolor=INK)
    fig.text(0.01, 0.01, "Line: average of 3 training runs. Band: lowest to highest run. "
             "Same 200 test shots at every point.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{out}.png")
    with open(f"{out}.csv", "w") as f:
        f.write("variant,{},mean_pct,min_pct,max_pct\n".format(
            "steps_millions" if xunit == "steps" else "shots_thousands"))
        for row in table:
            f.write("{},{:.2f},{:.1f},{:.1f},{:.1f}\n".format(*row))
    print("wrote", f"{out}.png", f"{out}.csv")


if __name__ == "__main__":
    main()
