"""Figure: MAE of the main methods on the four datasets (one panel per dataset; scales differ). Run from code/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clinical_endpoints import n5k_sets, SP, DN, DL
import numpy as np
validation = n5k_sets()["validation"]

OUT = Path("../experiments/paper_v3_figures"); OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
# one colour per model: Qwen blue, Nutrition5k-only orange, Nutrition5k + ASA24 aqua, three-domain purple
QWEN, N5K, N5K_ASA, THREE = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0"
LEGEND = [("Qwen3-VL-32B, zero-shot", QWEN), ("DINOv2-base, trained on Nutrition5k", N5K),
          ("DINOv2-large, trained on Nutrition5k + ASA24", N5K_ASA), ("DINOv2-base, trained on Nutrition5k + ASA24 + FPB", THREE)]
DATA = {  # dataset: [(MAE, colour)]; the third bar is the best model that was trained with that dataset's training photographs, if any
    "Nutrition5k validation\n(in domain)": list(zip((float(np.mean(np.abs(validation["preds"][k] - validation["t"]))) for k in (SP, DN, DL)), (QWEN, N5K, N5K_ASA))),
    "ASA24 test\n(unseen foods)": [(72.9, QWEN), (67.9, N5K), (42.0, N5K_ASA)],
    "ECUSTFD\n(never trained on)": [(40.5, QWEN), (44.7, N5K), (36.0, N5K_ASA)],
    "FPB test\n(unseen trays)": [(248.5, QWEN), (467.5, N5K), (161.7, THREE)],
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titlecolor": INK, "savefig.dpi": 300})
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.8))
for ax, (name, bars_) in zip(axes, DATA.items()):
    vals = [v for v, _ in bars_]
    bars = ax.bar(range(3), vals, color=[c for _, c in bars_], width=0.72, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title(name, loc="left"); ax.set_xticks([]); ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals) * 1.18)
axes[0].set_ylabel("MAE (g)", color=INK)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in LEGEND]
fig.legend(handles, [m for m, _ in LEGEND], loc="lower center", ncol=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_cross_dataset_mae.{ext}", bbox_inches="tight")
print("saved", OUT)
