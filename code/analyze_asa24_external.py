"""External evaluation on ASA24 portion images (descriptive; no model was fitted or selected on ASA24).

Metrics per method: MAE, median AE, median |log ratio| (scale-free), signed bias, share within 25%/50% of truth,
log-log slope of prediction on truth (1 = tracks portion size, 0 = constant), and the mean within-series
Spearman correlation (does the method order the portions of one food correctly?).
Run from code/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

A = Path("../experiments/asa24_external")
LO, HI = 10.0, 1500.0


def load(path, col, ids):
    return pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[col].to_numpy(float).copy()


def metrics(pred, t, series):
    lr = np.log(pred / t)
    rho = []
    for s in np.unique(series):
        m = series == s
        if m.sum() >= 3 and np.ptp(t[m]) > 0:
            rho.append(np.corrcoef(pd.Series(pred[m]).rank(), pd.Series(t[m]).rank())[0, 1])
    rho = np.array(rho); rho = rho[np.isfinite(rho)]
    return {"mae_g": float(np.mean(np.abs(pred - t))), "median_ae_g": float(np.median(np.abs(pred - t))),
            "median_abs_log_ratio": float(np.median(np.abs(lr))), "bias_g": float(np.mean(pred - t)),
            "within_25pct": float(np.mean(np.abs(pred / t - 1) <= 0.25) * 100), "within_50pct": float(np.mean(np.abs(pred / t - 1) <= 0.5) * 100),
            "loglog_slope": float(np.polyfit(np.log(t), np.log(pred), 1)[0]),
            "within_series_spearman": float(rho.mean()), "n_series": int(len(rho))}


def main():
    s = pd.read_csv(A / "asa24_aerial_sample.csv").set_index("dish_id")
    ids, t, series = s.index, s.true_weight.to_numpy(float), s.series.to_numpy()
    med = float(pd.read_csv("../experiments/qwen3vl_journal_v1/reference_train2253_qwen3vl_results.csv").true_weight.median())
    delta = load(A / "residual_transfer_delta.csv", "neighbor_log_residual", ids)
    rows, invalid = {}, {}
    for mode in ("oracle", "blind"):
        f = A / f"qwen_{mode}_fp8_lenient.csv"  # strict parse + lenient sum for comma-split names
        if not f.is_file():
            continue
        q = load(f, "predicted_weight", ids); bad = ~(q > 0); invalid[mode] = int(bad.sum()); q[bad] = med
        label = "true food name" if mode == "oracle" else "no name (blind)"
        rows[f"Raw Qwen, {label}"] = np.clip(q, 1, None)
        rows[f"Residual transfer, {label}"] = np.clip(q * np.exp(0.75 * delta), LO, HI)
    for name, f in [("Fine-tuned image-only", "finetuned_stage_b_imageonly_aux10.csv"),
                    ("Fine-tuned image-only + nutrient head", "finetuned_stage_b_imageonly_aux10_nutr.csv"),
                    ("Fine-tuned with Qwen inputs (true name)", "finetuned_noingr_qwen_oracle.csv"),
                    ("Fine-tuned with Qwen inputs (blind)", "finetuned_noingr_qwen_blind.csv")]:
        if (A / f).is_file():
            rows[name] = load(A / f, "mass", ids)
    rows["Constant: Nutrition5k reference median"] = np.full(len(t), med)
    out = {"n": len(t), "true_median_g": float(np.median(t)), "qwen_invalid_imputed": invalid,
           "median_similarity_to_bank": float(pd.read_csv(A / "residual_transfer_delta.csv").max_similarity.median()),
           "mass": {k: metrics(v, t, series) for k, v in rows.items()}}
    # calories: true kcal; direct head; mass x true FNDDS energy density (uses the true food identity)
    kcal = s.kcal.to_numpy(float); dens = kcal / t; ok = np.isfinite(kcal)
    cal = {}
    nutr = A / "finetuned_stage_b_imageonly_aux10_nutr.csv"
    if nutr.is_file():
        cal["Fine-tuned image-only, direct nutrient head"] = load(nutr, "pred_calories", ids)
    for k, v in rows.items():
        if not k.startswith("Constant"):
            cal[f"{k} mass x true food energy density"] = v * dens
    out["calories"] = {"true_mean_kcal": float(np.nanmean(kcal)),
                       "mae_kcal": {k: float(np.mean(np.abs(v[ok] - kcal[ok]))) for k, v in cal.items()}}
    idx = np.random.default_rng(0).integers(0, len(t), (10000, len(t)))
    pairs = {}
    names = list(rows)
    ref = "Fine-tuned image-only"
    for k in names:
        if k != ref and ref in rows:
            d = np.abs(rows[k] - t) - np.abs(rows[ref] - t); b = d[idx].mean(1)
            pairs[f"{k} minus {ref}"] = [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out["paired_mae_vs_finetuned_image_only"] = pairs
    (A / "asa24_external_results.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(out["mass"]).T.round(2).to_string())
    print(json.dumps({k: out[k] for k in ("n", "true_median_g", "qwen_invalid_imputed", "median_similarity_to_bank", "calories",
                                          "paired_mae_vs_finetuned_image_only")}, indent=1))


if __name__ == "__main__":
    main()
