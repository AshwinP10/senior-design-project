"""Final comparison: goals after a paddle hit on the 200-shot test (seed 30000), bounce rule.

Usage: python plot_comparison.py results.json out_prefix
results.json: [{"label": ..., "runs": [goals, ...], "group": "rl" | "controller" | "hybrid"}, ...]
Bar = mean of runs; thin line = lowest to highest run; dashed line = 80% target.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1F1F1F", "#6B6B6B", "#E3E3E3"
GROUP_COLORS = {"rl": "#56B4E9", "hybrid": "#0072B2", "controller": "#E69F00"}
GROUP_LABELS = {"rl": "Learned policy only", "hybrid": "RL + strike controller",
                "controller": "Hand-coded strike controller"}
VALUE_X = 103          # values sit in a fixed column right of the 100% gridline


def main():
    rows, out = json.load(open(sys.argv[1])), sys.argv[2]
    for r in rows:
        r["pcts"] = [100 * g / 200 for g in r["runs"]]
        r["mean"] = sum(r["pcts"]) / len(r["pcts"])
    rows.sort(key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8), dpi=200)
    for i, r in enumerate(rows):
        ax.barh(i, r["mean"], height=0.62, color=GROUP_COLORS[r["group"]], edgecolor="white", linewidth=2)
        if len(r["pcts"]) > 1:
            ax.plot([min(r["pcts"]), max(r["pcts"])], [i, i], color=INK, lw=1.4, solid_capstyle="butt")
        runs = f"  (runs {min(r['pcts']):.0f}–{max(r['pcts']):.0f}%)" if len(r["pcts"]) > 1 else ""
        ax.text(VALUE_X, i, f"{r['mean']:.1f}%{runs}", va="center", fontsize=10, color=INK)
    ax.axvline(80, color=INK, lw=1.2, ls=(0, (5, 4)))
    ax.text(80.8, len(rows) - 0.45, "Target 80%", fontsize=10, color=INK, va="bottom")
    ax.axvline(100, color=GRID, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10, color=INK)
    ax.set_xlim(0, 140)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Goals after a paddle hit (% of 200 test shots, puck bounces off obstacles)", color=MUTED, fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_LABELS
               if any(r["group"] == g for r in rows)]
    labels = [GROUP_LABELS[g] for g in GROUP_LABELS if any(r["group"] == g for r in rows)]
    ax.legend(handles, labels, frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, labelcolor=INK)
    ax.set_title("Scoring accuracy by approach (same 200 test shots)", loc="left", fontsize=13, color=INK, pad=30)
    fig.tight_layout()
    fig.savefig(f"{out}.png")
    with open(f"{out}.csv", "w") as f:
        f.write("approach,group,mean_pct,min_pct,max_pct,runs_goals\n")
        for r in rows:
            f.write(f"\"{r['label']}\",{r['group']},{r['mean']:.1f},{min(r['pcts']):.1f},{max(r['pcts']):.1f},"
                    f"\"{' '.join(map(str, r['runs']))}\"\n")
    print("wrote", f"{out}.png", f"{out}.csv")


if __name__ == "__main__":
    main()
