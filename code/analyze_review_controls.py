"""Seed, backbone and FPB-only controls (post hoc; experiments/review_controls/PLAN.md). No test set is read. Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("../experiments/review_controls"); T = Path("../experiments/qwen3vl_journal_v2_training"); A = Path("../experiments/asa24_external")
SEEDS = (20261004, 20261011, 20261012, 20261013); B = 10000
OLD = {"depth": "stage_b_imageonly_aux10", "nodepth": "stage_b_imageonly_nodepth", "frozen": "stage_b_imageonly_frozen"}
rank = lambda x: pd.Series(x).rank().to_numpy()


def transfer(p, t, series=None):
    ok = np.isfinite(p) & (p > 0); p, t = p[ok], t[ok]; s = float(np.exp(np.mean(np.log(p / t))))
    r = {"mae": float(np.abs(p - t).mean()), "pooled_spearman": float(np.corrcoef(rank(p), rank(t))[0, 1]), "scale_ratio": s,
         "rescaled_mae": float(np.abs(p / s - t).mean())}
    if series is not None:
        g = series[ok]; rho = []
        for k in np.unique(g):
            m = g == k
            if m.sum() >= 3 and np.ptp(t[m]) > 0:
                c = np.corrcoef(rank(p[m]), rank(t[m]))[0, 1]; rho.append(c if np.isfinite(c) else 0.0)
        r["within_series_spearman"] = float(np.mean(rho))
    return r


def run_metrics(run_dir, name, pool, fpb):
    r = {"validation_mae": json.loads((run_dir / "summary.json").read_text())["development_mae_g"],
         "reference_cv_mae": json.loads((run_dir / "summary.json").read_text())["reference_cv_mae_g"]}
    a = pd.read_csv(D / f"{name}_asa24pool.csv").set_index("dish_id").reindex(pool.index).ensemble.to_numpy(float)
    f = pd.read_csv(D / f"{name}_fpbtrain.csv").set_index("dish_id").reindex(fpb.index).ensemble.to_numpy(float)
    r["asa24_pool"] = transfer(a, pool.grams.to_numpy(float), pool.series.to_numpy()); r["fpb_train"] = transfer(f, fpb.true_weight.to_numpy(float))
    return r


def flat(r):
    return {"validation_mae": r["validation_mae"], **{f"{d}.{k}": v for d in ("asa24_pool", "fpb_train") for k, v in r[d].items()}}


def main():
    pool = pd.read_csv(A / "asa24_train_pool.csv").set_index("image_file"); fpb = pd.read_csv("../experiments/external_fpb/fpb_train_1024.csv").set_index("dish_id")
    res = {"seeds": {}, "contrasts": {}}; tab = {}
    for v in OLD:
        runs = {s: run_metrics(T / OLD[v] if s == SEEDS[0] else D / f"runs/{v}_s{s}", f"{v}_s{s}", pool, fpb) for s in SEEDS}
        tab[v] = pd.DataFrame({s: flat(r) for s, r in runs.items()}).T
        res["seeds"][v] = {"per_seed": {str(s): r for s, r in runs.items()},
                           "summary": {c: {"mean": float(tab[v][c].mean()), "sd": float(tab[v][c].std(ddof=1)), "min": float(tab[v][c].min()), "max": float(tab[v][c].max())} for c in tab[v]}}
    for a, b in (("nodepth", "depth"), ("frozen", "depth")):  # paired by seed, a minus b
        d = tab[a] - tab[b]
        res["contrasts"][f"{a}_minus_{b}"] = {c: {"mean": float(d[c].mean()), "per_seed": d[c].round(4).tolist(), "same_sign": bool((np.sign(d[c]) == np.sign(d[c].iloc[0])).all()),
                                                  "ranges_disjoint": bool(tab[a][c].min() > tab[b][c].max() or tab[a][c].max() < tab[b][c].min())} for c in d}
    res["backbone"] = {"base_s20261004": res["seeds"]["depth"]["per_seed"]["20261004"],
                       "large_s20261004": run_metrics(D / "runs/large_depth_s20261004", "large_depth_s20261004", pool, fpb),
                       "base_seed_range": {c: [float(tab["depth"][c].min()), float(tab["depth"][c].max())] for c in tab["depth"]}}
    # C. FPB-only against three-domain training: grouped out-of-fold on the FPB training split, bootstrap over annotation signatures
    j = pd.read_csv(A / "runs/joint3_base/fpb_train_oof.csv").set_index("dish_id"); rng = np.random.default_rng(0); res["fpb_only"] = {}
    for ep in (15, 29):
        o = pd.read_csv(D / f"runs/fpb_only_base_ep{ep}/fpb_train_oof.csv").set_index("dish_id").reindex(j.index)
        assert (o.portion == j.portion).all() and np.allclose(o.true, j.true)
        g = pd.DataFrame({"d": np.abs(o.oof - o.true) - np.abs(j.oof - j.true), "n": 1, "portion": j.portion}).groupby("portion").sum()
        i = rng.integers(0, len(g), (B, len(g))); bs = g.d.to_numpy()[i].sum(1) / g.n.to_numpy()[i].sum(1)
        res["fpb_only"][f"epochs={ep}"] = {"fpb_only_mae": float(np.abs(o.oof - o.true).mean()), "three_domain_mae": float(np.abs(j.oof - j.true).mean()),
                                           "difference": [float(g.d.sum() / g.n.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_signatures": len(g)}
    (D / "review_controls.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
    for v in tab:
        print(f"== {v}"); print(tab[v].T.assign(mean=tab[v].mean(), sd=tab[v].std(ddof=1)))
    for k, c in res["contrasts"].items():
        print(f"== {k}")
        for m, x in c.items():
            print(f"  {m:38s} {x['mean']:+8.3f}  {x['per_seed']}  same sign {x['same_sign']}  disjoint {x['ranges_disjoint']}")
    print("== backbone (seed 20261004)"); print(pd.DataFrame({"base": flat(res["backbone"]["base_s20261004"]), "large": flat(res["backbone"]["large_s20261004"])}))
    print("== FPB-only"); print(json.dumps(res["fpb_only"], indent=1))


if __name__ == "__main__":
    main()
