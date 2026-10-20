"""Figure: per-food calibration on one ASA24 test series (illustration of Section sec:fewshot). Run from code/.

Series: among test series with at least six photographs, with the middle portion as the labelled photograph, those whose
MAE reduction is closest to the median reduction over series; brown rice (third closest) is shown for a clear single
food. Model: multi-domain DINOv2-large; correction as in the paper, prediction x exp(-0.75 log(pred/true)) of the
labelled photograph.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

E = Path("../experiments/asa24_external"); OUT = Path("../paper/figures"); LAM = 0.75
SERIES = "Rice, brown, cooked, regular, fat not added in cooking | Aerial"


def main():
    a = pd.read_csv(E / "asa24_aerial_sample.csv").set_index("dish_id")
    a["pred"] = pd.read_csv(E / "runs/joint_large_consensus/asa24_test_predictions.csv").drop_duplicates("dish_id").set_index("dish_id").mass.reindex(a.index)
    g = a[a.series == SERIES].sort_values("true_weight"); lab = len(g) // 2
    f = np.exp(-LAM * np.log(g.pred.iloc[lab] / g.true_weight.iloc[lab])); g["cal"] = g.pred * f
    rest = g.drop(g.index[lab]); before, after = np.abs(rest.pred - rest.true_weight).mean(), np.abs(rest.cal - rest.true_weight).mean()
    fig, axes = plt.subplots(1, len(g), figsize=(10, 2.75))
    for k, (ax, r) in enumerate(zip(axes, g.itertuples())):
        ax.imshow(Image.open(r.image).convert("RGB")); ax.set_axis_off()
        if k == lab:
            ax.add_patch(Rectangle((1, 1), 157, 157, fill=False, lw=2.5, ec="tab:blue"))
            txt = f"LABELLED\nlabel {r.true_weight:.0f} g\nmodel {r.pred:.0f} g\nfactor {f:.2f}"
        else:
            txt = f"label {r.true_weight:.0f} g\nmodel {r.pred:.0f} g\ncorrected {r.cal:.0f} g"
        ax.set_title(r.Portion + "\n" + txt, fontsize=7.6, color="tab:blue" if k == lab else "black")
    fig.suptitle(f"Brown rice, ASA24 test: MAE on the other {len(rest)} photographs {before:.1f} g uncorrected, {after:.1f} g corrected", fontsize=8.5, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "fig_calibration_example.pdf"); fig.savefig(OUT / "fig_calibration_example.png", dpi=200)
    print(g[["Portion", "true_weight", "pred", "cal"]].round(1).to_string(), f"\nfactor {f:.3f} before {before:.1f} after {after:.1f}")


if __name__ == "__main__":
    main()
