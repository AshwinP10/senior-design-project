"""Plot precision-recall curves from tmp/midreview/pr_curves.py output.

Usage: python plot_pr.py pr_curves.json out_prefix
Solid step line = precision at each recall; dotted line of the same colour = chance (base rate).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1F1F1F", "#6B6B6B", "#E3E3E3"
STYLE = {"controller": ("Strike controller", "#E69F00"), "original_rl": ("Original RL model", "#0072B2")}


def main():
    res, out = json.load(open(sys.argv[1])), sys.argv[2]
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=200)
    for key, (label, colour) in STYLE.items():
        r = res[key]
        # scikit-learn appends a fixed (recall 0, precision 1) end point that is not a real threshold.
        ax.step(r["recall"][:-1], r["precision"][:-1], where="post", color=colour, lw=2.4,
                label=f"{label}: AP {r['ap']:.2f}")
        ax.axhline(r["base_rate"], color=colour, lw=1.2, ls=(0, (1, 2)),
                   label=f"{label}: chance ({100 * r['base_rate']:.0f}% of shots score)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall: share of all goals the confidence score catches", color=MUTED, fontsize=11)
    ax.set_ylabel("Precision: share of confident shots that score", color=MUTED, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_title("Can it tell, before the shot, whether it will score?", loc="left", fontsize=13, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, labelcolor=INK)
    fig.text(0.01, 0.01, "200 test shots. Controller score fitted on validation shots only; RL score = its critic's value estimate.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{out}.png")
    print("wrote", f"{out}.png")


if __name__ == "__main__":
    main()
