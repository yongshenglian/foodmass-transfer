"""Two figures. Run from code/.

fig_dataset_examples: on each dataset, the photograph whose log ratio (predicted/true) is closest to that dataset's median
log ratio for the Nutrition5k-only DINOv2-base, i.e. a typical photograph, not a failure.
fig_asa24_series: one ASA24 portion series (angel food cake, aerial) at its native 160x160 pixels with its standard-portion labels.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

E = Path("../experiments"); OUT = Path("../paper/figures")
col = lambda f, c: pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id")[c]
SETS = [  # name, sample, model predictions, Qwen predictions, label kind, subset
    ("Nutrition5k validation", E / "qwen3vl_journal_v2_training/qwen_local/development_502_sample.csv",
     col(E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10/development_predictions.csv", "mass"),
     col(E / "n5k_single_prompt/qwen_blind_val_fp8.csv", "predicted_weight"), "weighed"),
    ("ASA24 test", E / "asa24_external/asa24_aerial_sample.csv", col(E / "asa24_external/finetuned_stage_b_imageonly_aux10.csv", "mass"),
     col(E / "asa24_external/qwen_blind_fp8_lenient_imputed.csv", "predicted_weight"), "standard portion"),
    ("ECUSTFD", E / "external_ecustfd/ecustfd_top_sample.csv", col(E / "external_ecustfd/pred_n5k_imageonly.csv", "mass"),
     col(E / "external_ecustfd/qwen_blind_fp8.csv", "predicted_weight"), "weighed"),
    ("FPB test", E / "external_fpb/fpb_val_eval.csv", col(E / "external_fpb/pred_n5k_imageonly.csv", "mass"),
     col(E / "external_fpb/qwen_blind_fp8.csv", "predicted_weight"), "weighed"),
]
SERIES = "Cake, angel food, without icing | Aerial"


def main():
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.1), gridspec_kw={"width_ratios": [4 / 3, 1, 4 / 3, 16 / 9]})
    fpb_train = set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion)
    for k, (ax, (name, sample, model, qwen, kind)) in enumerate(zip(axes, SETS)):
        s = pd.read_csv(sample).drop_duplicates("dish_id").set_index("dish_id")
        if name == "FPB test":
            s = s[~s.portion.isin(fpb_train)]
        p = model.reindex(s.index).to_numpy(float); t = s.true_weight.to_numpy(float); lr = np.log(p / t)
        i = int(np.nanargmin(np.abs(lr - np.nanmedian(lr)))); r = s.iloc[i]; im = Image.open(r.image)
        ax.imshow(im.convert("RGB")); ax.set_axis_off()
        q = qwen.get(s.index[i], np.nan)
        short = {"weighed": "weighed", "standard portion": "standard weight"}[kind]
        ax.set_title(f"({'abcd'[k]}) {name.replace(' validation', '').replace(' test', '')}, {short} label\ntrue {t[i]:.0f} g\n"
                     f"DINOv2 {p[i]:.0f} g, Qwen {q:.0f} g\nmedian DINOv2 ratio {np.exp(np.nanmedian(lr)):.2f}", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_dataset_examples.pdf"); fig.savefig(OUT / "fig_dataset_examples.png", dpi=200)

    a = pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv"); g = a[a.series == SERIES].sort_values("true_weight").head(5)
    fig, axes = plt.subplots(1, 5, figsize=(7, 2.1))
    for j, (ax, r) in enumerate(zip(axes, g.itertuples())):
        ax.imshow(Image.open(r.image).convert("RGB"), interpolation="nearest"); ax.set_axis_off()
        desc = r.ImgTitle.split("), ", 1)[-1]
        if "wedge" in desc:  # "Tube wedge, Small, 1/4 piece" -> "1/4 small wedge"
            size, frac = desc.replace("Tube wedge, ", "").replace(" piece", "").split(", ")
            desc = f"{frac} {size.lower()} wedge"
        ax.set_title(f"({'abcde'[j]})\n{desc}\nlabel {r.true_weight:.1f} g", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_asa24_series.pdf"); fig.savefig(OUT / "fig_asa24_series.png", dpi=200)


if __name__ == "__main__":
    main()
