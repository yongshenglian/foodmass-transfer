"""External test MAE with the DINOv2 variant held fixed within each panel. Run from code/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("../paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
QWEN, N5K, ADDED = "#2a78d6", "#eb6834", "#1baf7a"
LEGEND = [("Qwen3-VL-32B, zero-shot", QWEN), ("DINOv2, Nutrition5k training", N5K),
          ("DINOv2, additional training data", ADDED)]
base = json.loads(Path("../experiments/asa24_base_training/results.json").read_text())["mae_g"]
large = json.loads(Path("../experiments/ecustfd_backbone_control/results.json").read_text())["top"]["mae"]
DATA = {  # DINOv2 variant appears in the panel title; training additions are specified in the manuscript caption.
    "ASA24 test\nDINOv2-base": [(72.9, QWEN), (base["nutrition5k_only"], N5K), (base["nutrition5k_plus_asa24"], ADDED)],
    "ECUSTFD test\nDINOv2-large": [(40.5, QWEN), (large["large, Nutrition5k only"], N5K), (large["large, two-domain"], ADDED)],
    "FPB test\nDINOv2-base": [(248.5, QWEN), (467.5, N5K), (161.7, ADDED)],
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titlecolor": INK, "savefig.dpi": 300})
fig, axes = plt.subplots(1, len(DATA), figsize=(7.4, 2.8))
for ax, (name, bars_) in zip(axes, DATA.items()):
    vals = [v for v, _ in bars_]
    bars = ax.bar(range(3), vals, color=[c for _, c in bars_], width=0.72, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title(name, loc="left"); ax.set_xticks([]); ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals) * 1.18)
axes[0].set_ylabel("MAE (g)", color=INK)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in LEGEND]
fig.legend(handles, [m for m, _ in LEGEND], loc="lower center", ncol=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(OUT / "fig_cross_dataset_mae.pdf", bbox_inches="tight")
print("saved", OUT)
