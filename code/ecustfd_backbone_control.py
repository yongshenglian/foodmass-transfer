"""Backbone or ASA24 training: which lowered the ECUSTFD error? (experiments/ecustfd_backbone_control/PLAN.md)
Run from code/:  python ecustfd_backbone_control.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); D = E / "ecustfd_backbone_control"
VIEWS = {"top": (E / "external_ecustfd/ecustfd_top_sample.csv", E / "external_ecustfd"),
         "side": (E / "ecustfd_side_view/ecustfd_side_sample.csv", E / "ecustfd_side_view")}


def main():
    res = {}; rng = np.random.default_rng(0)
    for v, (sample, folder) in VIEWS.items():
        s = pd.read_csv(sample).drop_duplicates("dish_id").set_index("dish_id"); t = s.true_weight.to_numpy(float)
        get = lambda f: pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id").reindex(s.index).mass.to_numpy(float)
        p = {"base, Nutrition5k only": get(folder / "pred_n5k_imageonly.csv"), "large, Nutrition5k only": get(D / f"pred_n5k_large_{v}.csv"),
             "large, two-domain": get(folder / "pred_joint_large_consensus.csv")}
        assert all(np.isfinite(x).all() for x in p.values())
        u, inv = np.unique(s["id"].to_numpy(), return_inverse=True); n = np.bincount(inv); idx = rng.integers(0, len(u), (10000, len(u)))

        def diff(a, b):
            d = np.abs(p[a] - t) - np.abs(p[b] - t); sm = np.bincount(inv, d); m = sm[idx].sum(1) / n[idx].sum(1)
            return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]
        res[v] = {"n": int(len(t)), "objects": int(len(u)),
                  "mae": {k: float(np.abs(x - t).mean()) for k, x in p.items()},
                  "scale_ratio": {k: float(np.exp(np.mean(np.log(x / t)))) for k, x in p.items()},
                  "backbone: large minus base, Nutrition5k only": diff("large, Nutrition5k only", "base, Nutrition5k only"),
                  "ASA24 training: two-domain minus large Nutrition5k only": diff("large, two-domain", "large, Nutrition5k only"),
                  "total: two-domain minus base Nutrition5k only": diff("large, two-domain", "base, Nutrition5k only")}
    (D / "results.json").write_text(json.dumps(res, indent=2))
    for v, r in res.items():
        print(f"== {v} views, {r['n']} photographs of {r['objects']} objects")
        for k in r["mae"]:
            print(f"  {k:26s} MAE {r['mae'][k]:5.1f} g  scale {r['scale_ratio'][k]:.2f}")
        for k, x in r.items():
            if isinstance(x, list):
                print(f"  {k:55s} {x[0]:+5.1f} g [{x[1]:+.1f}, {x[2]:+.1f}]")


if __name__ == "__main__":
    main()
