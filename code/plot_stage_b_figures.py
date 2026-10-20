"""Paper figures for the supervised stage B comparison, from report_stage_b_paper_tables.py outputs. Run from code/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = Path("../experiments/qwen3vl_journal_v2_training/paper_tables")
OUT = T / "figures"
# fixed categorical order from the validated palette: slot 1 blue, 2 orange, 3 aqua
METHODS = [("stage_b", "Stage B fine-tuned", "#2a78d6"), ("fixed", "Fixed DINOv2 residual", "#eb6834"),
           ("raw", "Raw Qwen3-VL-32B", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
BINS = [(0, 100, "<100"), (100, 200, "100–199"), (200, 300, "200–299"), (300, 450, "300–449"), (450, np.inf, "≥450")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titlecolor": INK, "savefig.dpi": 300})


def scatter(frame, name):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, (key, label, color) in zip(axes, reversed(METHODS)):
        mae = np.mean(np.abs(frame[key] - frame.true))
        ax.plot([5, 2000], [5, 2000], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(frame.true, frame[key], s=9, color=color, alpha=0.55, linewidths=0, zorder=2)
        ax.set(xscale="log", yscale="log", xlim=(5, 1500), ylim=(5, 1500), xlabel="Measured mass (g)")
        ax.set_title(f"{label}\nMAE {mae:.1f} g", loc="left")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Predicted mass (g)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_scatter_{name}.{ext}")
    plt.close(fig)


def by_mass(frames):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    width, x = 0.26, np.arange(len(BINS))
    for ax, (name, frame) in zip(axes, frames.items()):
        for j, (key, label, color) in enumerate(reversed(METHODS)):
            vals = [np.mean(np.abs(frame[key] - frame.true)[(frame.true >= lo) & (frame.true < hi)]) for lo, hi, _ in BINS]
            ax.bar(x + (j - 1) * width, vals, width - 0.03, color=color, label=label, zorder=2)
        counts = [int(((frame.true >= lo) & (frame.true < hi)).sum()) for lo, hi, _ in BINS]
        ax.set_xticks(x, [f"{b[2]}\nn={c}" for b, c in zip(BINS, counts)])
        ax.set_title(f"{ {'development': 'Validation set', 'confirmation': 'Test set'}.get(name, name)} (n = {len(frame)})", loc="left")
        ax.set_xlabel("Measured mass (g)")
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE (g)")
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_mae_by_mass.{ext}")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {n: pd.read_csv(T / f"{n}_row_predictions.csv") for n in ("development", "confirmation")}
    scatter(frames["confirmation"], "confirmation")
    by_mass(frames)
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
