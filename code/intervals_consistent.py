"""One interval procedure for all models (experiments/intervals_consistent/PLAN.md). Existing predictions only.
Run from code/:  python intervals_consistent.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aspect_control_analyze import view

E = Path("../experiments"); OUT = E / "intervals_consistent"; FM = E / "fold_members"; A = E / "asa24_external"
LO, HI, ALPHA, B = 10.0, 1500.0, 0.9, 10000
BINS = [(0, 100), (100, 300), (300, 600), (600, np.inf)]


def truth_sets():
    """Evaluation sets: truth and bootstrap groups, indexed by dish_id."""
    a = pd.read_csv(A / "asa24_aerial_sample.csv").set_index("dish_id")
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    f = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))]
    c = pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv").set_index("dish_id")
    e = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").drop_duplicates("dish_id").set_index("dish_id")
    pool = pd.read_csv(E / "n5k_grouped_cv/pool.csv").set_index("dish_id")
    return {"Nutrition5k internal test": (c.true_weight, pool.group.reindex(c.index).astype(str), c.index.to_series()),
            "ASA24 test": (a.true_weight, a.series, None), "ECUSTFD top": (e.true_weight, e["id"], None),
            "FPB test": (f.true_weight, f.portion, f)}


def val_ok():
    """Validation dishes without a sibling in the Nutrition5k training set (the paper's sibling rule)."""
    p = pd.read_csv(E / "n5k_grouped_cv/pool.csv").set_index("dish_id")
    train_groups = set(p.sibling_group[p.set == "train"])
    return p.index[(p.set == "validation") & ~p.sibling_group.isin(train_groups)]


def folds_csv(name):
    """Per-fold mass predictions from experiments/fold_members, indexed by dish_id (5 columns)."""
    d = pd.read_csv(FM / name).drop_duplicates("dish_id").set_index("dish_id"); return d[[f"fold{i}" for i in range(5)]]


def folds_npz(run, key, ids):
    m = np.stack([np.clip(np.exp(np.load(A / "runs" / run / f"fold{i}.npz")[key]), LO, HI) for i in range(5)], 1)
    return pd.DataFrame(m, index=ids, columns=[f"fold{i}" for i in range(5)])


def oof_errors(run, key, true):
    """Out-of-fold |log(pred/true)| of a train_multidomain run: grouped folds, one error per training photograph."""
    out = np.full(len(true), np.nan)
    for i in range(5):
        z = np.load(A / "runs" / run / f"fold{i}.npz"); idx = z[f"{key}_index"]
        out[idx] = np.abs(np.log(np.clip(np.exp(z[key]), LO, HI) / true[idx]))
    assert np.isfinite(out).all(); return out


def models():
    dev_ids = [str(i) for i in np.load(E / "qwen3vl_journal_v1/development_502_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    dev_t = pd.read_csv(E / "qwen3vl_journal_v1/dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id").true_weight
    ok = val_ok(); asa_t = pd.read_csv(A / "asa24_train_pool_consensus.csv").grams.to_numpy(float)
    fpb_t = pd.read_csv(E / "external_fpb/fpb_train_1024.csv").true_weight.to_numpy(float)
    a_ids = pd.read_csv(A / "asa24_aerial_sample.csv").dish_id; f_ids = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").dish_id

    def val_err(fm):  # fold models on validation dishes without a training sibling; one error per (dish, fold)
        v = fm.loc[fm.index.intersection(ok)]; return np.abs(np.log(v.to_numpy() / dev_t.loc[v.index].to_numpy()[:, None])).ravel()
    base_val, large_val = folds_csv("base_n5k_n5k_val.csv"), folds_csv("large_multidomain_n5k_val.csv")
    three_val = folds_npz("joint3_base", "dev", dev_ids)
    return {
        "Nutrition5k-only (base)": {"calibration": {"n5k": val_err(base_val)}, "test": {
            "Nutrition5k internal test": folds_csv("base_n5k_n5k_test.csv"), "ASA24 test": folds_csv("base_n5k_asa24.csv"),
            "ECUSTFD top": folds_csv("base_n5k_ecustfd.csv"), "FPB test": folds_csv("base_n5k_fpb.csv")}},
        "Two-domain (large)": {"calibration": {"n5k": val_err(large_val), "asa24": oof_errors("joint_large_consensus", "asa_oof", asa_t)}, "test": {
            "Nutrition5k internal test": folds_csv("large_multidomain_n5k_test.csv"), "ASA24 test": folds_csv("large_multidomain_asa24.csv"),
            "ECUSTFD top": folds_csv("large_multidomain_ecustfd.csv"), "FPB test": folds_csv("large_multidomain_fpb.csv")}},
        "Three-domain (base)": {"calibration": {"n5k": val_err(three_val), "asa24": oof_errors("joint3_base", "asa_oof", asa_t),
                                                "fpb": oof_errors("joint3_base", "fpb_oof", fpb_t)}, "test": {
            "ASA24 test": folds_npz("joint3_base", "test", a_ids), "FPB test": folds_npz("joint3_base", "fpbval", f_ids)}}}


def k_factor(cal):
    """exp of the weighted 90% quantile of |log error|; every domain gets equal total weight."""
    e = np.concatenate(list(cal.values())); w = np.concatenate([np.full(len(v), 1 / len(v)) for v in cal.values()])
    o = np.argsort(e); cw = np.cumsum(w[o]) / w.sum()
    return float(np.exp(e[o][np.searchsorted(cw, ALPHA)]))


def summarize(pred, t, g, k, rng):
    """pred: photographs x folds. Coverage (mean over fold models), its group-bootstrap interval, median width."""
    inside = ((t[:, None] >= pred / k) & (t[:, None] <= pred * k)).mean(1)  # share of fold models covering each photograph
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True); s = np.bincount(inv, inside); n = np.bincount(inv)
    idx = rng.integers(0, len(u), (B, len(u))); bs = s[idx].sum(1) / n[idx].sum(1)
    return {"n": int(len(t)), "coverage": float(inside.mean() * 100), "ci": (np.quantile(bs, [0.025, 0.975]) * 100).tolist(),
            "median_width_g": float(np.median(pred * k - pred / k))}


def main():
    rng = np.random.default_rng(0); T = truth_sets(); res = {}
    for name, m in models().items():
        k = k_factor(m["calibration"]); r = {"k": k, "k_by_domain": {d: k_factor({d: v}) for d, v in m["calibration"].items()},
                                             "n_calibration": {d: int(len(v)) for d, v in m["calibration"].items()}, "sets": {}}
        for s, fm in m["test"].items():
            t_s, g_s, meta = T[s]; ids = t_s.index; pred = fm.reindex(ids).to_numpy(float); t = t_s.to_numpy(float)
            assert np.isfinite(pred).all(), (name, s)
            ens = np.exp(np.log(pred).mean(1, keepdims=True))
            x = {"fold_models": summarize(pred, t, g_s.reindex(ids), k, rng), "ensemble": summarize(ens, t, g_s.reindex(ids), k, rng),
                 "by_mass": {f"{lo}-{hi}" if np.isfinite(hi) else f">={lo}": summarize(pred[mm], t[mm], g_s.reindex(ids)[mm], k, rng)
                             for lo, hi in BINS if (mm := (t >= lo) & (t < hi)).sum() >= 20}}
            if s == "FPB test":
                v = np.array([view(i) for i in ids]); one = meta.loc[ids].n_items.to_numpy() == 1
                x["by_view"] = {f"{vv} / {'single item' if o else 'several items'}": summarize(pred[mm], t[mm], g_s.reindex(ids)[mm], k, rng)
                                for vv in ("lab overhead", "lab oblique", "smartphone") for o in (True, False) if (mm := (v == vv) & (one == o)).sum() >= 20}
            r["sets"][s] = x
        res[name] = r
    OUT.mkdir(exist_ok=True); (OUT / "results.json").write_text(json.dumps(res, indent=2)); report(res)


def report(res):
    for name, r in res.items():
        print(f"== {name}: k {r['k']:.2f} (by domain " + ", ".join(f"{d} {v:.2f}" for d, v in r["k_by_domain"].items()) +
              f"; calibration errors " + ", ".join(f"{d} {v}" for d, v in r["n_calibration"].items()) + ")")
        for s, x in r["sets"].items():
            f, e = x["fold_models"], x["ensemble"]
            print(f"  {s:26s} fold models {f['coverage']:5.1f}% [{f['ci'][0]:.1f}, {f['ci'][1]:.1f}] width {f['median_width_g']:5.0f} g   "
                  f"ensemble {e['coverage']:5.1f}% width {e['median_width_g']:5.0f} g")
            for sub in ("by_mass", "by_view"):
                for b, y in x.get(sub, {}).items():
                    print(f"      {b:32s} n {y['n']:5d} coverage {y['coverage']:5.1f}% width {y['median_width_g']:5.0f} g")


if __name__ == "__main__":
    main()
