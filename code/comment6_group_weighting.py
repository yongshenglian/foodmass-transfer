"""Reviewer comment 6: MAE with photographs weighted equally (as reported) and with groups weighted equally (each food
series, object or tray counts once), with group sizes, from existing predictions. Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_endpoints import DL, DN, SP, external_sets, load, qwen

E = Path("../experiments"); OUT = E / "comment6_statistics"; OUT.mkdir(exist_ok=True); B = 10000
D3 = "DINOv2-base, three-domain"


def sets():
    x = external_sets(); out = {}
    a = pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv")
    x["ASA24"]["preds"][D3] = load(E / "asa24_external/runs/joint3_base/asa24_test_predictions.csv", "mass", a.dish_id)
    out["ASA24 test (food series)"] = (x["ASA24"], a.series.to_numpy())
    e = pd.read_csv(E / "external_ecustfd/ecustfd_all_predictions.csv")
    top = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").set_index("dish_id").loc[e.dish_id]
    out["ECUSTFD top (objects)"] = (x["ECUSTFD"], top["id"].to_numpy())
    s = pd.read_csv(E / "ecustfd_side_view/ecustfd_side_sample.csv").set_index("dish_id"); V = E / "ecustfd_side_view"
    out["ECUSTFD side (objects)"] = ({"t": s.true_weight.to_numpy(float), "preds": {SP: qwen(V / "qwen_blind_fp8.csv", s.index),
                                     DN: load(V / "pred_n5k_imageonly.csv", "mass", s.index), DL: load(V / "pred_joint_large_consensus.csv", "mass", s.index)}},
                                     s["id"].to_numpy())
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    f = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))]
    x["FPB"]["preds"][DL] = load(E / "external_fpb/pred_joint_large_consensus.csv", "mass", f.index)
    out["FPB test (trays)"] = (x["FPB"], f.portion.to_numpy())
    return out


def main():
    rng = np.random.default_rng(0); res = {}
    for name, (d, g) in sets().items():
        t = d["t"]; u, inv = np.unique(g.astype(str), return_inverse=True); n = np.bincount(inv)
        idx = rng.integers(0, len(u), (B, len(u))); r = {"photos": int(len(t)), "groups": int(len(u)),
                                                         "photos_per_group": {"median": float(np.median(n)), "min": int(n.min()), "max": int(n.max())}}
        for k, p in d["preds"].items():
            e = np.abs(p - t); s = np.bincount(inv, e); gm = s / n  # each group's mean error
            r[k] = {"mae_photo_weighted": float(e.mean()), "ci_photo_weighted": np.quantile(s[idx].sum(1) / n[idx].sum(1), [0.025, 0.975]).tolist(),
                    "mae_group_weighted": float(gm.mean()), "ci_group_weighted": np.quantile(gm[idx].mean(1), [0.025, 0.975]).tolist()}
        res[name] = r
    (OUT / "group_weighting.json").write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        print(f"== {name}: {r['photos']} photos, {r['groups']} groups, photos per group median {r['photos_per_group']['median']:.0f} "
              f"({r['photos_per_group']['min']}-{r['photos_per_group']['max']})")
        for k, v in r.items():
            if isinstance(v, dict) and "mae_photo_weighted" in v:
                print(f"  {k:32s} photo-weighted {v['mae_photo_weighted']:6.1f} [{v['ci_photo_weighted'][0]:.1f}, {v['ci_photo_weighted'][1]:.1f}]  "
                      f"group-weighted {v['mae_group_weighted']:6.1f} [{v['ci_group_weighted'][0]:.1f}, {v['ci_group_weighted'][1]:.1f}]")


if __name__ == "__main__":
    main()
