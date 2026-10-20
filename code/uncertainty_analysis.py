"""Uncertainty and calibration (review request: the system should know when it does not know). Run from code/.

1. Empirical residual-based prediction intervals for the fine-tuned DINOv2 models (not conformal: the calibration
   scores mostly come from cross-validation fold models rather than the deployed five-model ensemble, datasets are
   pooled, and clustered photographs are not exchangeable). Score: absolute log ratio |log(prediction / truth)|.
   Quantile: the ceil((n + 1) * level)-th smallest calibration score, giving the multiplicative interval
   [m / f, m * f]. We report observed coverage, with 95% CIs from a bootstrap over clusters (dish, ASA24 food series,
   ECUSTFD object, FPB tray), and the median width. A sensitivity analysis calibrates the Nutrition5k-only model on the
   deployed ensemble's predictions for the 502 validation dishes.
2. Calibration of Qwen's self-reported confidence, which the prompt defines as the probability that its total is
   within 50 g of the truth: stated versus observed share within 50 g, expected calibration error, and AUROC.
3. Out-of-distribution signal: maximum cosine similarity of a photograph's frozen DINOv2-base embedding to the 2,253
   Nutrition5k training dishes, how well it separates each external set from the internal test set, and how often
   a threshold set on the Nutrition5k validation set (5th percentile) flags external photographs. Similarities for
   the validation set, ECUSTFD and FPB come from experiments/uncertainty/similarity_*.csv (workstation job).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); OUT = E / "uncertainty"; OUT.mkdir(exist_ok=True)
C, A, V1, V2 = E / "confirmation_photo_only", E / "asa24_external", E / "qwen3vl_journal_v1", E / "qwen3vl_journal_v2_training"
LEVELS = (0.8, 0.9, 0.95)


def get(path, col, ids):
    return pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[col].to_numpy(float)


def conformal_q(resid, level):
    r = np.sort(np.abs(resid)); n = len(r); k = int(np.ceil((n + 1) * level)) - 1
    return float(r[min(k, n - 1)])


def coverage(pred, true, q, clusters=None):
    lo, hi = pred * np.exp(-q), pred * np.exp(q); inside = ((true >= lo) & (true <= hi)).astype(float)
    cl = np.arange(len(true)) if clusters is None else pd.factorize(np.asarray(clusters))[0]
    sums, counts = np.bincount(cl, inside), np.bincount(cl)
    idx = np.random.default_rng(0).integers(0, len(sums), (2000, len(sums)))
    boot = sums[idx].sum(1) / counts[idx].sum(1) * 100
    return {"coverage_pct": float(inside.mean() * 100), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "clusters": int(len(sums)), "median_width_g": float(np.median(hi - lo))}


def auroc(score, label):
    """Probability that a positive scores higher than a negative (ties count half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def test_sets():
    """True mass and predictions of each model on every test set (photograph only)."""
    s = pd.read_csv(C / "confirmation_498_sample.csv").set_index("dish_id"); n5k = s.index
    a = pd.read_csv(A / "asa24_aerial_sample.csv").set_index("dish_id"); asa = a.index
    e = pd.read_csv(E / "external_ecustfd/ecustfd_all_predictions.csv")
    et = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").set_index("dish_id").loc[e.dish_id]
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    f = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))]
    base = pd.read_csv(V1 / "dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id")
    return {
        "Nutrition5k validation": dict(t=base.true_weight.to_numpy(float), ids=base.index, cl=None,
                                       base=get(V2 / "stage_b_imageonly_aux10/development_predictions.csv", "mass", base.index)),
        "Nutrition5k test": dict(t=s.true_weight.to_numpy(float), ids=n5k, cl=None, base=get(C / "dinov2_photo_only_n5k.csv", "mass", n5k),
                                 large=get(C / "dinov2_large_multidomain.csv", "mass", n5k)),
        "ASA24": dict(t=a.true_weight.to_numpy(float), ids=asa, cl=a.series.to_numpy(), base=get(A / "finetuned_stage_b_imageonly_aux10.csv", "mass", asa),
                      large=get(A / "runs/joint_large_consensus/asa24_test_predictions.csv", "mass", asa)),
        "ECUSTFD": dict(t=et.true_weight.to_numpy(float), ids=et.index, cl=et.id.to_numpy(), base=e["Nutrition5k-only photo model"].to_numpy(float),
                        large=e["Joint large + consensus"].to_numpy(float)),
        "FPB": dict(t=f.true_weight.to_numpy(float), ids=f.index, cl=f.portion.to_numpy(), base=get(E / "external_fpb/pred_n5k_imageonly.csv", "mass", f.index),
                    large=get(E / "external_fpb/pred_joint_large_consensus.csv", "mass", f.index),
                    three=get(A / "runs/joint3_base/fpb_val_predictions.csv", "mass", f.index)),
    }


def main():
    res = {}
    sets = test_sets()
    # ---- 1. conformal intervals
    ref = pd.read_csv(V2 / "stage_b_imageonly_aux10/reference_oof.csv")
    cal_base = np.log(np.clip(np.exp(ref.oof_prediction), 10, 1500) / ref.true)
    base_val = pd.read_csv(V1 / "dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id")
    lv = pd.read_csv(A / "runs/joint_large_consensus/n5k_development_predictions.csv").set_index("dish_id").reindex(base_val.index)
    pool = set(pd.read_csv(A / "asa24_train_pool.csv").image_file)
    ao = pd.read_csv(A / "runs/joint_large_consensus/asa24_pool_oof.csv").query("dish_id in @pool")
    cal_large = np.concatenate([np.log(lv.mass / base_val.true_weight).to_numpy(), np.log(ao.oof / ao.true).to_numpy()])
    fo = pd.read_csv(A / "runs/joint3_base/fpb_train_oof.csv"); cal_three = np.log(fo.oof / fo.true).to_numpy()
    dv = pd.read_csv(V2 / "stage_b_imageonly_aux10/development_predictions.csv")
    cal_base_val = np.log(dv.mass / dv.true).to_numpy()  # deployed ensemble on the 502 validation dishes
    models = {"DINOv2-base, Nutrition5k only (calibrated on Nutrition5k out-of-fold)": ("base", cal_base),
              "DINOv2-base, Nutrition5k only (sensitivity: calibrated on the deployed ensemble, validation set)": ("base", cal_base_val),
              "DINOv2-large, multi-domain (calibrated on Nutrition5k validation + ASA24 out-of-fold)": ("large", cal_large),
              "DINOv2-base, three-domain (calibrated on FPB out-of-fold)": ("three", cal_three)}
    res["conformal"] = {}  # key kept for compatibility; these are empirical residual-based intervals
    for mname, (key, cal) in models.items():
        qs = {lv_: conformal_q(cal, lv_) for lv_ in LEVELS}
        rows = {}
        for sname, d in sets.items():
            if key in d and not ("validation set)" in mname and sname == "Nutrition5k validation"):
                rows[sname] = {f"{int(l * 100)}%": coverage(d[key], d["t"], q, d["cl"]) for l, q in qs.items()}
        res["conformal"][mname] = {"n_calibration": int(len(cal)), "factor": {f"{int(l * 100)}%": float(np.exp(q)) for l, q in qs.items()}, "sets": rows}
    # ---- 2. Qwen confidence
    qfiles = {"Nutrition5k test": E / "n5k_single_prompt/qwen_blind_test_fp8.csv", "ASA24": A / "qwen_blind_fp8_lenient.csv",
              "ECUSTFD": E / "external_ecustfd/qwen_blind_fp8.csv", "FPB": E / "external_fpb/qwen_blind_fp8.csv"}
    res["qwen_confidence"] = {}
    for sname, path in qfiles.items():
        d = sets[sname]; q = pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(d["ids"])
        ok = (q.predicted_weight > 0) & q.confidence.notna()
        conf = q.confidence[ok].to_numpy(float) / 100; within = np.abs(q.predicted_weight[ok].to_numpy(float) - d["t"][ok.to_numpy()]) <= 50
        bins = [(0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]; ece = 0; table = []
        for lo, hi in bins:
            m = (conf >= lo) & (conf < hi)
            if m.any():
                table.append({"stated": f"{int(lo * 100)}-{min(int(hi * 100), 100) - 1 if hi < 1 else 100}%", "n": int(m.sum()),
                              "mean_stated_pct": float(conf[m].mean() * 100), "observed_within_50g_pct": float(within[m].mean() * 100)})
                ece += m.mean() * abs(conf[m].mean() - within[m].mean())
        res["qwen_confidence"][sname] = {"n": int(ok.sum()), "mean_stated_pct": float(conf.mean() * 100),
                                         "observed_within_50g_pct": float(within.mean() * 100), "ece_pct": float(ece * 100),
                                         "auroc_within_50g": auroc(conf, within), "bins": table}
    # ---- 3. similarity to the Nutrition5k training bank
    sim = {"Nutrition5k test": (C / "residual_transfer_delta.csv", sets["Nutrition5k test"]["ids"]),
           "ASA24": (A / "residual_transfer_delta.csv", sets["ASA24"]["ids"])}
    for sname, fname in [("Nutrition5k validation", "similarity_n5k_validation.csv"), ("ECUSTFD", "similarity_ecustfd.csv"),
                         ("FPB", "similarity_fpb.csv"), ("ECUSTFD side view", "similarity_ecustfd_side.csv")]:
        if (OUT / fname).is_file():
            ids = sets[sname]["ids"] if sname in sets else pd.read_csv(OUT / fname).dish_id
            sim[sname] = (OUT / fname, ids)
    S = {k: get(p, "max_similarity", ids) for k, (p, ids) in sim.items()}
    res["similarity"] = {"median": {k: float(np.nanmedian(v)) for k, v in S.items()}}
    ref_key = "Nutrition5k validation" if "Nutrition5k validation" in S else None
    thr = float(np.nanpercentile(S[ref_key], 5)) if ref_key else None
    res["similarity"]["threshold_5th_pct_validation"] = thr
    res["similarity"]["vs_internal_test"] = {}
    for k, v in S.items():
        if k.startswith("Nutrition5k"):
            continue
        r = {"auroc_internal_vs_external": auroc(-np.concatenate([S["Nutrition5k test"], v]),
                                                 np.r_[np.zeros(len(S["Nutrition5k test"])), np.ones(len(v))])}
        if thr is not None:
            r["flagged_pct"] = float(np.mean(v < thr) * 100)
        res["similarity"]["vs_internal_test"][k] = r
    if thr is not None:
        res["similarity"]["internal_test_flagged_pct"] = float(np.mean(S["Nutrition5k test"] < thr) * 100)
    d = sets["ASA24"]; lerr = np.abs(np.log(d["base"] / d["t"])); s_ = S["ASA24"]
    res["similarity"]["asa24_error_vs_similarity"] = {"spearman_similarity_abs_log_error": float(pd.Series(s_).rank().corr(pd.Series(lerr).rank()))}
    (OUT / "uncertainty.json").write_text(json.dumps(res, indent=2))
    # ---- print
    for mname, r in res["conformal"].items():
        print(f"\n== Residual-based intervals: {mname}; n = {r['n_calibration']}; factors " + ", ".join(f"{k} x{v:.2f}" for k, v in r["factor"].items()))
        for sname, c in r["sets"].items():
            print(f"  {sname:24s} " + "  ".join(f"{k}: {v['coverage_pct']:5.1f}% [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}] w {v['median_width_g']:4.0f} g" for k, v in c.items()) + f"  ({c['90%']['clusters']} clusters)")
    print("\n== Qwen self-reported confidence (probability within 50 g)")
    for sname, r in res["qwen_confidence"].items():
        print(f"  {sname:18s} stated {r['mean_stated_pct']:5.1f}%  observed {r['observed_within_50g_pct']:5.1f}%  ECE {r['ece_pct']:5.1f}  AUROC {r['auroc_within_50g']:.2f}")
    print("\n== Similarity to the Nutrition5k training bank:", json.dumps(res["similarity"], indent=1))


if __name__ == "__main__":
    main()
