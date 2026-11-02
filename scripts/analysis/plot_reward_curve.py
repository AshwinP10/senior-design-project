"""Plot the bundled bank policy's recorded training rewards, without synthetic smoothing.

The source contains episode means in 500,000-transition bins. The final 30.5M bin
is incomplete, so the default plot ends at the last complete bin (30M).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/experiment_data/train_curves_strict_aim.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/figures/current_policy_reward_curve")
    args = parser.parse_args()
    with args.source.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["sweep"] == "aim-long" and row["run"] == "AB-seed3"
                and int(row["bin_end_steps"]) <= 30_000_000]
    rows.sort(key=lambda row: int(row["bin_end_steps"]))
    if not rows:
        parser.error("No complete aim-long / AB-seed3 reward bins found")
    xs = [int(row["bin_end_steps"]) / 1e6 for row in rows]
    ys = [float(row["mean_reward"]) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.23, top=0.80)
    fig.text(0.10, 0.94, "Training reward · current bundled policy", fontsize=18, weight="bold", color="#172B4D")
    fig.text(0.10, 0.885, "AB-seed3-bank  |  Historical training with obstacle rebounds allowed",
             fontsize=11, color="#526174")
    ax.axhline(0, color="#A8B2BF", linewidth=1, linestyle="--", zorder=1)
    ax.plot(xs, ys, color="#1565C0", linewidth=2.2, marker="o", markersize=2.8,
            label="Mean episode return per 500,000-step bin", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], color="#1565C0", s=35, zorder=4)
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(-9, 13),
                textcoords="offset points", ha="right", color="#1565C0", weight="bold")
    ax.set_xlabel("Training steps in this fine-tuning run (millions)", labelpad=10)
    ax.set_ylabel("Mean episode return", labelpad=10)
    ax.set_xlim(0, 30.8)
    ax.grid(axis="y", color="#E4E9F0")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCD4DF")
    fig.text(0.10, 0.09, "Recorded bin means; no additional smoothing. Incomplete final bin omitted.",
             fontsize=9, color="#526174")
    fig.text(0.10, 0.045, "Reward is not scoring accuracy. This history predates the clean-scoring reward changes.",
             fontsize=9, color="#526174")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix("." + suffix), facecolor="white")
    plt.close(fig)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"source": str(args.source), "sweep": "aim-long", "run": "AB-seed3",
                "checkpoint": "models/tacc-2026-10/AB-seed3-bank", "bins": len(rows),
                "bin_width_steps": 500000, "additional_smoothing": False,
                "rules": "Historical training; obstacle rebounds permitted",
                "omitted": "Incomplete final 30.5M-step bin",
                "first_mean_return": ys[0], "last_mean_return": ys[-1]}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Saved PNG, PDF, CSV and metadata: {args.output}")


if __name__ == "__main__":
    main()
