"""NutritionVerse-Real analysis, exactly as pre-registered (experiments/nutritionverse_real/PLAN.md). Photograph is the
unit; every CI is a cluster bootstrap over dishes (10,000 resamples, seed 0). Interval factors and the similarity
threshold are read from experiments/uncertainty/uncertainty.json (fixed earlier; nothing is recalibrated). Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_endpoints import CARB_TOL, DL, DN, SP, SPK, composition_endpoints, load, mass_endpoints, qwen
from uncertainty_analysis import auroc

E = Path("../experiments"); D = E / "nutritionverse_real"; D3 = "DINOv2-base, three-domain"; CONST = "Constant (186 g)"
U = json.loads((E / "uncertainty/uncertainty.json").read_text())
FACTORS = {DN: U["conformal"]["DINOv2-base, Nutrition5k only (calibrated on Nutrition5k out-of-fold)"]["factor"],
           DL: U["conformal"]["DINOv2-large, multi-domain (calibrated on Nutrition5k validation + ASA24 out-of-fold)"]["factor"],
           D3: U["conformal"]["DINOv2-base, three-domain (calibrated on FPB out-of-fold)"]["factor"]}
THR = U["similarity"]["threshold_5th_pct_validation"]
RNG = np.random.default_rng(0); B = 10000


def boot(values, cl):
    """Cluster-bootstrap 95% CI of a per-photograph mean."""
    sums, counts = np.bincount(cl, values), np.bincount(cl)
    idx = RNG.integers(0, len(sums), (B, len(sums))); b = sums[idx].sum(1) / counts[idx].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def opt(path, col, ids):
    return load(path, col, ids) if path.is_file() else None


def mass_block(p, t, cl, dish):
    out = mass_endpoints(p, t); ae = np.abs(p - t); g = float(np.exp(np.mean(np.log(p / t))))
    out.update(mae_ci=boot(ae, cl), median_ae_g=float(np.median(ae)), spearman=float(pd.Series(p).rank().corr(pd.Series(t).rank())),
               scale_ratio=g, mae_after_global_rescale_oracle=float(np.mean(np.abs(p / g - t))))
    sd = pd.Series(np.log(p)).groupby(dish).std(ddof=1).dropna()  # angle sensitivity: spread across a dish's photographs
    out["within_dish_sd_log"] = float(sd.mean()); out["within_dish_cv_pct"] = float((np.exp(sd) - 1).mean() * 100)
    return out


def main():
    s = pd.read_csv(D / "nvr_sample.csv").set_index("dish_id")
    if "dish" not in s:
        s["dish"] = s.dish_group
        s["n_items"] = s.food_item_count
        s["portrait"] = s.source_height > s.source_width
    ids = s.index; t = s.true_weight.to_numpy(float)
    cl = pd.factorize(s.dish)[0]; dish = s.dish.to_numpy()
    preds = {DN: opt(D / "pred_n5k_imageonly.csv", "mass", ids), DL: opt(D / "pred_joint_large_consensus.csv", "mass", ids),
             D3: opt(D / "pred_joint3_base.csv", "mass", ids)}
    if (D / "qwen_blind_fp8.csv").is_file():
        q = pd.read_csv(D / "qwen_blind_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(ids)
        preds[SP] = qwen(D / "qwen_blind_fp8.csv", ids)
        if (D / "similarity_knn.csv").is_file():
            preds[SPK] = np.clip(preds[SP] * np.exp(0.75 * load(D / "similarity_knn.csv", "neighbor_log_residual", ids)), 10, 1500)
    preds[CONST] = np.full(len(t), 186.0); preds = {k: v for k, v in preds.items() if v is not None}
    res = {"n_photographs": int(len(t)), "n_dishes": int(s.dish.nunique()), "n_portrait": int(s.portrait.sum()),
           "true_mass_g": {"median": float(np.median(t)), "min": float(t.min()), "max": float(t.max())}}
    if SP in preds:
        res["qwen_no_usable_estimate"] = int((~(q.predicted_weight > 0)).sum())
    res["mass"] = {k: mass_block(p, t, cl, dish) for k, p in preds.items()}
    res["paired_vs_qwen"] = {}
    if SP in preds:
        for k in (DN, DL, D3, SPK):
            if k in preds:
                diff = np.abs(preds[k] - t) - np.abs(preds[SP] - t); res["paired_vs_qwen"][k] = [float(diff.mean()), *boot(diff, cl)]
    # landscape sensitivity (amendment 1): portrait photographs turned to landscape before the DINOv2 resize
    res["landscape_sensitivity"] = {}
    for k, f in [(DN, "pred_n5k_imageonly"), (DL, "pred_joint_large_consensus"), (D3, "pred_joint3_base")]:
        pl = opt(D / f"{f}_landscape.csv", "mass", ids)
        if pl is not None and k in preds:
            diff = np.abs(pl - t) - np.abs(preds[k] - t); m = s.portrait.to_numpy()
            res["landscape_sensitivity"][k] = {"mae": float(np.mean(np.abs(pl - t))), "minus_primary": [float(diff.mean()), *boot(diff, cl)],
                                              "primary_mae_portrait": float(np.mean(np.abs(preds[k] - t)[m])),
                                              "primary_mae_landscape_photos": float(np.mean(np.abs(preds[k] - t)[~m])),
                                              "landscape_mae_portrait": float(np.mean(np.abs(pl - t)[m]))}
    # intervals (fixed factors) and familiarity flag
    res["intervals"] = {}
    for k, fac in FACTORS.items():
        if k in preds:
            res["intervals"][k] = {}
            for lv, f in fac.items():
                inside = ((t >= preds[k] / f) & (t <= preds[k] * f)).astype(float)
                res["intervals"][k][lv] = {"factor": f, "coverage_pct": float(inside.mean() * 100), "ci": [100 * x for x in boot(inside, cl)],
                                           "median_width_g": float(np.median(preds[k] * (f - 1 / f)))}
    if (D / "similarity_knn.csv").is_file():
        sim = load(D / "similarity_knn.csv", "max_similarity", ids); flag = sim < THR
        res["similarity"] = {"threshold": THR, "median": float(np.median(sim)), "flagged_pct": float(flag.mean() * 100)}
        if DN in preds and 0 < flag.sum() < len(flag):
            ae = np.abs(preds[DN] - t); res["similarity"].update(mae_flagged=float(ae[flag].mean()), mae_unflagged=float(ae[~flag].mean()))
    if SP in preds:
        ok = ((q.predicted_weight > 0) & q.confidence.notna()).to_numpy()
        conf = q.confidence.to_numpy(float)[ok] / 100; within = np.abs(q.predicted_weight.to_numpy(float)[ok] - t[ok]) <= 50
        ece = sum((m := (conf >= lo) & (conf < hi)).mean() * abs(conf[m].mean() - within[m].mean())
                  for lo, hi in [(0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)] if ((conf >= lo) & (conf < hi)).any())
        res["qwen_confidence"] = {"n": int(ok.sum()), "mean_stated_pct": float(conf.mean() * 100),
                                  "observed_within_50g_pct": float(within.mean() * 100), "ece_pct": float(ece * 100), "auroc": auroc(conf, within)}
    # nutrients
    truth = pd.DataFrame({"true_mass": t, "true_calories": s.true_calories.to_numpy(float), "true_carb": s.true_carb.to_numpy(float)})
    res["energy_at_true_composition"] = {k: composition_endpoints(p, truth) for k, p in preds.items() if k != CONST}
    h = D / "pred_nutrient_head.csv"
    if h.is_file():
        hd = pd.read_csv(h).drop_duplicates("dish_id").set_index("dish_id").reindex(ids); res["nutrient_head"] = {}
        for name, pc, tc in [("energy_kcal", "pred_calories", "true_calories"), ("carbohydrate_g", "pred_carb", "true_carb"),
                             ("protein_g", "pred_protein", "true_protein"), ("fat_g", "pred_fat", "true_fat")]:
            p, tt = hd[pc].to_numpy(float), s[tc].to_numpy(float); ae = np.abs(p - tt)
            res["nutrient_head"][name] = {"mae": float(ae.mean()), "ci": boot(ae, cl), "wape_pct": float(ae.sum() / tt.sum() * 100),
                                          "mean_true": float(tt.mean()), "scale_ratio": float(np.exp(np.mean(np.log((p + 1) / (tt + 1)))))}
        res["nutrient_head"]["carb_within_10g_pct"] = float(np.mean(np.abs(hd.pred_carb.to_numpy(float) - s.true_carb.to_numpy(float)) <= CARB_TOL) * 100)
        res["nutrient_head"]["energy_within_20pct"] = float(np.mean(np.abs(hd.pred_calories.to_numpy(float) - s.true_calories.to_numpy(float))
                                                                    <= 0.2 * s.true_calories.to_numpy(float)) * 100)
    # pre-declared subgroup: number of food items
    grp = np.where(s.n_items >= 3, "3+", s.n_items.astype(str)); res["by_items"] = {}
    for g in sorted(set(grp)):
        m = grp == g; res["by_items"][g] = {"n": int(m.sum()), **{k: float(np.mean(np.abs(p - t)[m])) for k, p in preds.items()}}
    (D / "nvr_results.json").write_text(json.dumps(res, indent=2))
    print(f"{res['n_photographs']} photographs, {res['n_dishes']} dishes, {res['n_portrait']} portrait; true mass median "
          f"{res['true_mass_g']['median']:.0f} g ({res['true_mass_g']['min']:.0f}-{res['true_mass_g']['max']:.0f})")
    for k, m in res["mass"].items():
        print(f"  {k:32s} MAE {m['mae_g']:6.1f} [{m['mae_ci'][0]:.1f},{m['mae_ci'][1]:.1f}] WAPE {m['wape_pct']:5.1f}% w20 {m['within_20pct']:4.1f}% "
              f"rho {m['spearman']:.2f} x{m['scale_ratio']:.2f} resc {m['mae_after_global_rescale_oracle']:.1f} dishCV {m['within_dish_cv_pct']:.0f}%")
    for k, v in res["paired_vs_qwen"].items():
        print(f"  {k} minus Qwen: {v[0]:+.1f} [{v[1]:+.1f},{v[2]:+.1f}]")
    print(json.dumps({k: res[k] for k in ("landscape_sensitivity", "similarity", "qwen_confidence", "nutrient_head", "by_items") if k in res}, indent=1))


if __name__ == "__main__":
    main()
