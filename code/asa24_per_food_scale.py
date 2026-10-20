"""Per-food scale ratio of the Nutrition5k-only DINOv2 on the ASA24 test set (Table tab:perfood). Run from code/.

Scale ratio = geometric mean of predicted / true mass, over each food's series of portions (predictions clipped to 10-1,500 g).
Descriptive only: uses the existing test predictions, no model is run.
"""
from pathlib import Path

import numpy as np
import pandas as pd

A = Path("../experiments/asa24_external")
d = pd.read_csv(A / "asa24_aerial_sample.csv").merge(pd.read_csv(A / "finetuned_stage_b_imageonly_aux10.csv"), on="dish_id")
pred = np.clip(d.mass, 10, 1500)
d["log_ratio"] = np.log(pred / d.true_weight)
print(f"MAE {np.abs(pred - d.true_weight).mean():.1f} g; scale ratio {np.exp(d.log_ratio.mean()):.3f}; "
      f"after one global correction MAE {np.abs(pred / np.exp(d.log_ratio.mean()) - d.true_weight).mean():.1f} g")
f = d.groupby("series").agg(n=("log_ratio", "size"), ratio=("log_ratio", lambda x: np.exp(x.mean())), title=("ImgTitle", "first"))
print(f"{len(f)} foods; median ratio {f.ratio.median():.2f}; below 0.67: {(f.ratio < 2 / 3).mean():.0%}; above 1.5: {(f.ratio > 1.5).mean():.0%}")
print(f.sort_values("ratio").round(2).to_string())
