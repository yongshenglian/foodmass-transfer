"""Plot saved, matched-label training comparisons; run from code/ without GPUs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    source = Path("../paper/reproducibility/results/learning_curve.json")
    results = json.loads(source.read_text())["datasets"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(5.85, 2.2))
    counts = ["5", "10", "25", "50", "100", "250", "all"]
    for ax, (dataset, title, maximum, unit) in zip(
        axes, [("asa24", "ASA24", 575, "foods"), ("fpb", "FPB", 768, "trays")]
    ):
        values = results[dataset]["by_k"]
        x = [int(k) if k != "all" else maximum for k in counts]
        for key, label, color, marker in [
            ("scalar calibration", "Scale factor", "#666666", "^"),
            ("ridge, pooled", "Ridge regression", "#0072B2", "o"),
            ("fine-tuned, pooled", "Fine-tuning", "#D55E00", "s"),
        ]:
            y = [values[k]["mae"][key] for k in counts]
            ax.plot(x, y, label=label, color=color, marker=marker, markersize=4, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel(f"Labeled {unit}")
        ax.set_ylabel("MAE (g)")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.32, top=0.89, wspace=0.30)
    out = Path("../paper/figures")
    out.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"fig_training_choices.{extension}", dpi=200)
    print("Plotted scalar calibration and both pooled training methods at all seven labeled-group counts.")


if __name__ == "__main__":
    main()
