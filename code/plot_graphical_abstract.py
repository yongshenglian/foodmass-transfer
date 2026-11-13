"""Draw the J-BHI graphical abstract; run from code/ without GPUs.

MAEs are those in Tables II and III of paper/main_jbhi.tex; learning-curve values are read
from the saved results. Output is a 2.24:1 image (2640 x 1180 px PNG, plus PDF).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#999999", "#009E73"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def label_bars(ax, bars, size=8):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=size)


def main():
    curve = json.loads(Path("../paper/reproducibility/results/learning_curve.json").read_text())
    fpb = curve["datasets"]["fpb"]["by_k"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(8.8, 3.933), dpi=300)
    fig.suptitle("Image-based food mass estimation: lost in transfer, recovered with local data",
                 fontsize=12.5, fontweight="bold", y=0.965)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.9, 1.3], left=0.065, right=0.985,
                            bottom=0.33, top=0.80, wspace=0.38)

    # 1. Accurate within Nutrition5k (plate-grouped CV)
    ax = fig.add_subplot(grid[0])
    bars = ax.bar(["DINOv2\nbase", "DINOv2\nlarge", "Qwen\n(zero-shot)"], [28.0, 25.9, 62.5],
                  color=[BLUE, BLUE, ORANGE], width=0.65)
    label_bars(ax, bars)
    ax.set_title("1  Nutrition5k", fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel("MAE (g)")
    ax.tick_params(axis="x", labelsize=7.5)
    style(ax)

    # 2. Nutrition5k-only model on other datasets
    ax = fig.add_subplot(grid[1])
    names = ["ASA24", "ECUSTFD\noverhead", "ECUSTFD\nside", "FPB"]
    series = [("DINOv2-base, Nutrition5k only", BLUE, [67.9, 44.7, 55.0, 467.5]),
              ("Qwen3-VL-32B, zero-shot", ORANGE, [72.9, 40.5, 40.2, 248.5]),
              ("Constant prediction", GREY, [121.2, 69.2, 69.4, 373.3])]
    width = 0.26
    for i, (name, color, values) in enumerate(series):
        bars = ax.bar([x + (i - 1) * width for x in range(4)], values, width, color=color, label=name)
        label_bars(ax, bars, size=6.5)
    ax.set_xticks(range(4), names, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(20, 900)
    ax.set_yticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.set_ylabel("MAE (g, log scale)")
    ax.set_title("2  Other datasets", fontsize=10, loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.8, frameon=False)
    style(ax)

    # 3. Fine-tuning with labeled FPB trays versus one scale factor
    ax = fig.add_subplot(grid[2])
    counts = ["5", "10", "25", "50", "100", "250", "all"]
    x = [int(k) if k != "all" else 768 for k in counts]
    for key, name, color, marker in [("scalar calibration", "Scale factor", GREY, "^"),
                                     ("fine-tuned, pooled", "Fine-tuning", GREEN, "s")]:
        ax.plot(x, [fpb[k]["mae"][key] for k in counts], color=color, marker=marker,
                markersize=4, linewidth=1.6, label=name)
    ax.set_xscale("log")
    ax.set_xticks([5, 25, 100, 768], ["5", "25", "100", "768"])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of trays")
    ax.set_ylabel("MAE (g)")
    ax.set_title("3  Labeled FPB trays", fontsize=10, loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    style(ax)

    notes = [
        "Mass-trained DINOv2 beat a\nzero-shot vision-language\nmodel on Nutrition5k.",
        "Trained on Nutrition5k only, it ordered ASA24 portions\nwell but lost the gram scale: "
        "on FPB its estimates were\nabout one-seventh of the true weight.",
        "Learning from labeled local photographs\nbeat one scale factor. Added training data\n"
        "helped on its own dataset, not on all.",
    ]
    for ax, text in zip(fig.axes, notes):
        box = ax.get_position()
        fig.text(box.x0 - 0.04, 0.03, text, fontsize=8, va="bottom", ha="left", color="#333333")

    out = Path("../paper/figures/graphical_abstract")
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
