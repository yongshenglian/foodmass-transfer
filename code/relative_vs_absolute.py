"""Relative ordering versus absolute gram scale on every test set (review: make this separation the paper's center).
For each photograph-only method: Spearman rank correlation of predicted with true mass (ordering), the geometric-mean
ratio of predicted to true mass (systematic scale error; 1 = no bias), MAE, and MAE after dividing every prediction by
that set's geometric-mean ratio. The last is an oracle (it uses the test labels) and shows how much of the error a single
per-dataset scale correction could remove, i.e. how much is scale rather than ordering. Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_endpoints import DL, DN, SP, external_sets, load, n5k_sets, qwen

E = Path("../experiments"); OUT = E / "relative_vs_absolute"; OUT.mkdir(exist_ok=True)


def metrics(p, t):
    lr = np.log(p / t); g = float(np.exp(lr.mean()))
    return {"n": int(len(t)), "spearman": float(pd.Series(p).rank().corr(pd.Series(t).rank())), "scale_ratio": g,
            "median_abs_log_ratio": float(np.median(np.abs(lr))), "mae": float(np.mean(np.abs(p - t))),
            "mae_after_global_rescale_oracle": float(np.mean(np.abs(p / g - t)))}


def main():
    sets = {"Nutrition5k test": n5k_sets()["test"], **external_sets()}
    V = E / "ecustfd_side_view"; side = pd.read_csv(V / "ecustfd_side_sample.csv").set_index("dish_id"); ids = side.index
    sets["ECUSTFD side view"] = {"t": side.true_weight.to_numpy(float), "preds": {
        SP: qwen(V / "qwen_blind_fp8.csv", ids), DN: load(V / "pred_n5k_imageonly.csv", "mass", ids),
        DL: load(V / "pred_joint_large_consensus.csv", "mass", ids)}}
    res = {}
    for name, d in sets.items():
        res[name] = {k: metrics(p, d["t"]) for k, p in d["preds"].items() if k != "Qwen, single prompt + kNN"}
    (OUT / "relative_vs_absolute.json").write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        print(f"== {name}")
        for k, m in r.items():
            print(f"  {k:30s} rho {m['spearman']:.2f}  scale x{m['scale_ratio']:.2f}  MAE {m['mae']:6.1f}  -> rescaled {m['mae_after_global_rescale_oracle']:6.1f}")


if __name__ == "__main__":
    main()
