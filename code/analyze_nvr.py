"""Analyze all predeclared NutritionVerse-Real endpoints without selecting models.
Photograph-weighted endpoints; 95% CIs resample whole dishes, 10,000 times, seed 0.
Fixed interval factors come from the manifest written before inference.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("../experiments/nutritionverse_real")
BOOT = 10000


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class DishBootstrap:
    def __init__(self, groups):
        self.cl, self.names = pd.factorize(np.asarray(groups), sort=True)
        self.n = len(self.names)
        draws = np.random.default_rng(0).integers(0, self.n, (BOOT, self.n))
        self.weights = np.zeros((BOOT, self.n), dtype=np.int32)
        np.add.at(self.weights, (np.arange(BOOT)[:, None], draws), 1)
        self.counts = np.bincount(self.cl, minlength=self.n)
        self.den = self.weights @ self.counts
        self.photo_weights = self.weights[:, self.cl]

    def record(self, value, bootstrap):
        b = np.asarray(bootstrap, float)
        b = b[np.isfinite(b)]
        return {"estimate": float(value), "ci95": np.percentile(b, [2.5, 97.5]).tolist() if len(b) else None}

    def sums(self, values):
        return self.weights @ np.bincount(self.cl, np.asarray(values, float), minlength=self.n)

    def mean(self, values, mask=None):
        values = np.asarray(values, float)
        mask = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)
        if not mask.any():
            return {"estimate": None, "ci95": None, "n": 0}
        numerator = self.sums(np.where(mask, values, 0))
        denominator = self.sums(mask.astype(float))
        b = np.divide(numerator, denominator, out=np.full(BOOT, np.nan), where=denominator > 0)
        return self.record(values[mask].mean(), b)

    def ratio(self, numerator, denominator, factor=1.0):
        n, d = self.sums(numerator), self.sums(denominator)
        b = np.divide(n, d, out=np.full(BOOT, np.nan), where=d != 0) * factor
        return self.record(np.sum(numerator) / np.sum(denominator) * factor, b)

    def median(self, values):
        values = np.asarray(values, float)
        order = np.argsort(values, kind="stable")
        v = values[order]
        cw = np.cumsum(self.photo_weights[:, order], axis=1)
        n = cw[:, -1]
        # Exactly the median of the bootstrap sample with replicated photographs.
        l = np.argmax(cw > ((n - 1) // 2)[:, None], axis=1)
        r = np.argmax(cw > (n // 2)[:, None], axis=1)
        return self.record(np.median(values), (v[l] + v[r]) / 2)

    def dish_mean(self, values):
        values = np.asarray(values, float)
        return self.record(values.mean(), (self.weights @ values) / self.n)

    def ordering(self, pred, true):
        # Recompute weighted midranks in each replicate, including ties and replicated dishes.
        _, ip = np.unique(pred, return_inverse=True)
        _, it = np.unique(true, return_inverse=True)

        def rho(w):
            def ranks(inv):
                counts = np.bincount(inv, weights=w)
                return (np.cumsum(counts) - (counts - 1) / 2)[inv]
            p, t = ranks(ip), ranks(it)
            p -= np.average(p, weights=w)
            t -= np.average(t, weights=w)
            denominator = np.sqrt(np.sum(w * p * p) * np.sum(w * t * t))
            return float(np.sum(w * p * t) / denominator) if denominator else np.nan

        if np.ptp(pred) == 0 or np.ptp(true) == 0:
            return {"estimate": None, "ci95": None, "reason": "constant prediction or truth"}
        point = rho(np.ones(len(pred)))
        b = np.array([rho(w) for w in self.photo_weights])
        return self.record(point, b)


def load(path, sample, columns):
    d = pd.read_csv(path)
    assert d.dish_id.is_unique, f"duplicate IDs: {path}"
    assert set(d.dish_id) == set(sample.dish_id), f"incomplete/extra rows: {path}"
    d = d.set_index("dish_id").loc[sample.dish_id]
    assert np.isfinite(d[columns].to_numpy(float)).all(), f"nonfinite predictions: {path}"
    return d


def mass_metrics(pred, true, boot):
    ae = np.abs(pred - true)
    lr = np.log(pred / true)
    ratio = np.exp(lr.mean())
    scale_boot = np.exp(boot.sums(lr) / boot.den)
    scaled_error_boot = np.sum(boot.photo_weights * np.abs(pred[None, :] / scale_boot[:, None] - true[None, :]), axis=1) / boot.den
    sd = np.array([np.std(np.log(pred[boot.cl == g]), ddof=0) for g in range(boot.n)])
    return {
        "mae_g": boot.mean(ae), "median_absolute_error_g": boot.median(ae),
        "wape_pct": boot.ratio(ae, true, 100),
        "mean_absolute_percentage_error_pct": boot.mean(ae / true * 100),
        "median_absolute_percentage_error_pct": boot.median(ae / true * 100),
        "within_20pct_pct": boot.mean((ae <= 0.2 * true) * 100),
        "within_50g_pct": boot.mean((ae <= 50) * 100),
        "spearman": boot.ordering(pred, true),
        "geometric_mean_scale_ratio": boot.record(ratio, scale_boot),
        "mae_after_oracle_global_rescale_g": boot.record(np.mean(np.abs(pred / ratio - true)), scaled_error_boot),
        "mean_within_dish_sd_log_prediction": boot.dish_mean(sd),
        "consistency_sd_definition": "population SD of log predictions within each dish; all dishes, singletons contribute zero",
        "n_dishes_with_multiple_photos": int((boot.counts > 1).sum()),
    }


def main():
    s = pd.read_csv(D / "nvr_sample.csv")
    manifest = json.loads((D / "freeze_manifest.json").read_text())
    assert hashlib.sha256((D / "nvr_sample.csv").read_bytes()).hexdigest() == manifest["sample_sha256"]
    assert s.dish_id.is_unique and len(s) == manifest["n_photographs"]
    assert s.groupby("dish_group").true_weight.nunique().max() == 1
    t = s.true_weight.to_numpy(float)
    assert np.isfinite(t).all() and (t > 0).all()
    boot = DishBootstrap(s.dish_group)
    preds = {name: load(D / filename, s, ["mass"]).mass.to_numpy(float)
             for name, filename in {"n5k": "pred_n5k.csv", "large": "pred_large.csv", "three": "pred_three.csv"}.items()}
    nutrient = load(D / "pred_nutrients.csv", s, ["pred_calories", "pred_fat", "pred_carb", "pred_protein"])
    q = pd.read_csv(D / "qwen_blind_fp8.csv")
    assert q.dish_id.is_unique and set(q.dish_id) == set(s.dish_id), "Qwen rows missing/duplicated"
    q = q.set_index("dish_id").loc[s.dish_id]
    raw = pd.to_numeric(q.predicted_weight, errors="coerce").to_numpy(float)
    bad = ~np.isfinite(raw) | (raw <= 0)
    qmass = raw.copy()
    qmass[bad] = manifest["qwen_fallback_g"]
    preds["qwen"] = np.clip(qmass, manifest["qwen_min_g"], None)
    similarity = load(D / "residual_transfer_delta.csv", s, ["max_similarity", "neighbor_log_residual"])
    delta = similarity.neighbor_log_residual.to_numpy(float)
    preds["knn"] = np.clip(preds["qwen"] * np.exp(manifest["knn"]["log_residual_weight"] * delta), *manifest["knn"]["clip_g"])
    preds["constant"] = np.full(len(s), manifest["constant_g"], float)
    assert all(np.isfinite(p).all() and (p > 0).all() for p in preds.values())
    out = s[["dish_id", "dish_group", "true_weight", "food_item_count"]].copy()
    for key, value in preds.items():
        out[f"mass_{key}"] = value
    out["qwen_imputed"] = bad
    out["max_similarity"] = similarity.max_similarity.to_numpy(float)
    flagged = out.max_similarity.to_numpy() < manifest["similarity_threshold"]
    out["similarity_flag"] = flagged
    for key in ["calories", "fat", "carb", "protein"]:
        out[f"true_{key}"] = s[f"true_{key}"].to_numpy(float)
        out[f"pred_{key}"] = nutrient[f"pred_{key}"].to_numpy(float)
    out.to_csv(D / "all_predictions.csv", index=False)
    result = {
        "dataset": "NutritionVerse-Real, curated smartphone dataset; not participant validation",
        "n_photographs": len(s), "n_dishes": boot.n, "inclusion": manifest["excluded"],
        "bootstrap": manifest["bootstrap"], "qwen_invalid_or_nonpositive_imputed": int(bad.sum()),
        "qwen_transport_error_rows": int(q.get("error", pd.Series(index=q.index, dtype=object)).notna().sum()),
        "mass": {}, "comparisons_vs_qwen": {}, "uncertainty": {}, "nutrients": {},
        "energy_at_true_composition": {}, "subgroups": {},
        "similarity": {"threshold": manifest["similarity_threshold"],
                       "flagged_pct": boot.mean(flagged * 100), "median_max_similarity": boot.median(out.max_similarity)},
    }
    for name, p in preds.items():
        print(f"Analyzing {name}", flush=True)
        result["mass"][name] = mass_metrics(p, t, boot)
        ae = np.abs(p - t)
        result["mass"][name]["flagged_mae_g"] = boot.mean(ae, flagged)
        result["mass"][name]["unflagged_mae_g"] = boot.mean(ae, ~flagged)
        if name in {"n5k", "large", "three", "knn"}:
            result["comparisons_vs_qwen"][name] = {
                "role": "primary" if name == "n5k" else "secondary",
                "mae_difference_g": boot.mean(ae - np.abs(preds["qwen"] - t)),
            }
        result["subgroups"][name] = {
            str(int(n)): {"n_photographs": int((s.food_item_count == n).sum()),
                          "mae_g": boot.mean(ae, s.food_item_count == n)}
            for n in sorted(s.food_item_count.unique())
        }
    for name, factors in manifest["interval_factors"].items():
        p = preds[name]
        result["uncertainty"][name] = {}
        for level, factor in factors.items():
            lo, hi = p / factor, p * factor
            result["uncertainty"][name][level] = {
                "factor": factor, "coverage_pct": boot.mean(((t >= lo) & (t <= hi)) * 100),
                "median_width_g": boot.median(hi - lo),
                "primary_level": level == "90%",
            }
    for key in ["calories", "fat", "carb", "protein"]:
        truth = s[f"true_{key}"].to_numpy(float)
        p = nutrient[f"pred_{key}"].to_numpy(float)
        valid = np.isfinite(truth) & (truth >= 0)
        ae = np.abs(p - np.where(valid, truth, 0))
        r = {"n_valid_labels": int(valid.sum()), "n_missing_or_negative_labels": int((~valid).sum()),
             "mae": boot.mean(ae, valid),
             "wape_pct": boot.ratio(np.where(valid, ae, 0), np.where(valid, truth, 0), 100)}
        if key == "carb":
            r["within_10g_pct"] = boot.mean((ae <= 10) * 100, valid)
        result["nutrients"][key] = r
    energy = s.true_calories.to_numpy(float)
    valid = np.isfinite(energy) & (energy >= 0)
    for name in ["n5k", "large", "three", "qwen"]:
        pred_energy = preds[name] / t * np.where(valid, energy, 0)
        ae = np.abs(pred_energy - np.where(valid, energy, 0))
        result["energy_at_true_composition"][name] = {
            "n_valid_labels": int(valid.sum()), "mae_kcal": boot.mean(ae, valid),
            "wape_pct": boot.ratio(np.where(valid, ae, 0), np.where(valid, energy, 0), 100),
        }
    result["subgroup_metadata"] = {"food_item_count": "recorded ingredient type fields", "camera_angle": "not provided"}
    result = clean(result)
    (D / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = [f"NutritionVerse-Real: {len(s)} photographs, {boot.n} dishes; no tuning.",
             "All intervals: 95% dish-cluster bootstrap, 10,000 resamples, seed 0.",
             f"Qwen fallback: {int(bad.sum())} photographs retained and imputed to 186 g."]
    for name, r in result["mass"].items():
        def show(k):
            v = r[k]
            point = v["estimate"]
            return "undefined" if point is None else f"{point:.3f}"
        lines.append(f"{name}: MAE {show('mae_g')} g; WAPE {show('wape_pct')}%; within20% {show('within_20pct_pct')}%; Spearman {show('spearman')}; scale {show('geometric_mean_scale_ratio')}")
    for name, r in result["comparisons_vs_qwen"].items():
        v = r["mae_difference_g"]
        lines.append(f"{r['role']} {name} minus Qwen MAE: {v['estimate']:+.3f} g, 95% CI {v['ci95']}")
    for key, r in result["nutrients"].items():
        lines.append(f"Direct {key}: MAE {r['mae']['estimate']:.3f}; WAPE {r['wape_pct']['estimate']:.3f}%")
    lines.append("Every predeclared endpoint is recorded in results.json; rescale is an oracle diagnostic only.")
    text = "\n".join(lines) + "\n"
    (D / "results.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
