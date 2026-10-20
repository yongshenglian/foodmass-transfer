"""Scatter of predicted vs measured mass on the Nutrition5k test set, photograph-only setting (paper Table III).
Loads predictions exactly as score_n5k_single_prompt.py and score_confirmation_photo_only.py do, so panel MAEs match the table. Run from code/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = Path("../experiments/confirmation_photo_only")
S = Path("../experiments/n5k_single_prompt")
OUT = C / "figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10.5, "axes.titlecolor": INK, "savefig.dpi": 300})


def main():
    sample = pd.read_csv(C / "confirmation_498_sample.csv").set_index("dish_id")
    ids = sample.index; t = sample.true_weight.to_numpy(float)
    load = lambda f, col="mass": pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[col].to_numpy(float).copy()
    q = load(S / "qwen_blind_test_fp8.csv", "predicted_weight"); q[~(q > 0)] = 186.0  # single prompt, photograph only
    delta = load(C / "residual_transfer_delta_saved_embeddings.csv", "neighbor_log_residual")
    panels = [("Qwen3-VL, single prompt", q, "#1baf7a"),
              ("Qwen3-VL + kNN", np.clip(q * np.exp(0.75 * delta), 10, 1500), "#eb6834"),
              ("DINOv2-base", load(C / "dinov2_photo_only_n5k.csv"), "#2a78d6"),
              ("DINOv2-large, multi-domain", load(C / "dinov2_large_multidomain.csv"), "#1c4f99")]
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0), sharex=True, sharey=True)
    for ax, (label, pred, color) in zip(axes, panels):
        mae = np.mean(np.abs(pred - t))
        ax.plot([5, 2000], [5, 2000], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(t, pred, s=9, color=color, alpha=0.55, linewidths=0, zorder=2)
        ax.set(xscale="log", yscale="log", xlim=(5, 1500), ylim=(5, 1500), xlabel="Measured mass (g)")
        ax.set_title(f"{label}\nMAE {mae:.2f} g", loc="left")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        print(f"{label.replace(chr(10), ' '):40s} MAE {mae:.2f}")
    axes[0].set_ylabel("Predicted mass (g)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_scatter_test_photo_only.{ext}")


if __name__ == "__main__":
    main()
