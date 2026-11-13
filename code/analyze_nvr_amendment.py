"""Complete Amendment 1 endpoints; primary predictions are never replaced."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_nvr import D, DishBootstrap, clean, load
from uncertainty_analysis import auroc


def main():
    s = pd.read_csv(D / "nvr_sample.csv")
    primary = pd.read_csv(D / "all_predictions.csv")
    assert list(s.dish_id) == list(primary.dish_id)
    result = json.loads((D / "results.json").read_text())
    boot = DishBootstrap(s.dish_group)
    portrait = (s.source_height > s.source_width).to_numpy()
    t = s.true_weight.to_numpy(float)
    result["landscape_sensitivity"] = {}
    for name, filename in {
        "n5k": "pred_n5k_imageonly_landscape.csv",
        "large": "pred_joint_large_consensus_landscape.csv",
        "three": "pred_joint3_base_landscape.csv",
    }.items():
        p = primary[f"mass_{name}"].to_numpy(float)
        rotated = load(D / filename, s, ["mass"]).mass.to_numpy(float)
        assert np.allclose(rotated[~portrait], p[~portrait]), f"{name}: nonportrait predictions changed"
        ae, re = np.abs(p - t), np.abs(rotated - t)
        result["landscape_sensitivity"][name] = {
            "primary_mae_g": boot.mean(ae), "rotated_mae_g": boot.mean(re),
            "rotated_minus_primary_mae_g": boot.mean(re - ae),
            "n_portrait": int(portrait.sum()), "n_landscape": int((~portrait).sum()),
            "primary_portrait_mae_g": boot.mean(ae, portrait),
            "primary_landscape_mae_g": boot.mean(ae, ~portrait),
            "rotated_portrait_mae_g": boot.mean(re, portrait),
            "rotated_landscape_mae_g": boot.mean(re, ~portrait),
        }
    nu = load(D / "pred_nutrient_head_landscape.csv", s, ["pred_calories", "pred_fat", "pred_carb", "pred_protein"])
    result["landscape_nutrient_head"] = {}
    for name in ["calories", "fat", "carb", "protein"]:
        truth = s[f"true_{name}"].to_numpy(float)
        p = nu[f"pred_{name}"].to_numpy(float)
        ae = np.abs(p - truth)
        base_error = np.abs(primary[f"pred_{name}"].to_numpy(float) - truth)
        result["landscape_nutrient_head"][name] = {
            "mae": boot.mean(ae), "wape_pct": boot.ratio(ae, truth, 100),
            "rotated_minus_primary_mae": boot.mean(ae - base_error),
            "portrait_mae": boot.mean(ae, portrait), "landscape_mae": boot.mean(ae, ~portrait),
        }
    result["landscape_nutrient_head"]["carb_within_10g_pct"] = boot.mean(
        (np.abs(nu.pred_carb.to_numpy(float) - s.true_carb.to_numpy(float)) <= 10) * 100)
    # Report the sample SD used by the concurrently authored analysis, with a dish-bootstrap CI.
    for name in result["mass"]:
        p = primary[f"mass_{name}"].to_numpy(float)
        sd = pd.Series(np.log(p)).groupby(boot.cl).std(ddof=1).reindex(np.arange(boot.n)).to_numpy()
        valid = np.isfinite(sd)
        den = boot.weights @ valid.astype(float)
        numer = boot.weights @ np.where(valid, sd, 0)
        samples = np.divide(numer, den, out=np.full(len(den), np.nan), where=den > 0)
        result["mass"][name]["mean_within_dish_sample_sd_log_prediction"] = boot.record(np.mean(sd[valid]), samples)
        result["mass"][name]["n_dishes_for_sample_sd"] = int(valid.sum())
    q = pd.read_csv(D / "qwen_blind_fp8.csv").set_index("dish_id").loc[s.dish_id]
    conf = pd.to_numeric(q.confidence, errors="coerce").to_numpy(float) / 100
    raw = pd.to_numeric(q.predicted_weight, errors="coerce").to_numpy(float)
    ok = np.isfinite(conf) & (conf >= 0) & (conf <= 1) & np.isfinite(raw) & (raw > 0)
    within = np.abs(raw - t) <= 50
    valid_total = boot.sums(ok.astype(float))
    ece, ece_boot = 0.0, np.zeros(len(boot.den))
    bins = []
    for lo, hi in [(0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
        mask = ok & (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        ece += mask.sum() / ok.sum() * abs(conf[mask].mean() - within[mask].mean())
        difference = boot.sums(np.where(mask, np.nan_to_num(conf) - within, 0))
        ece_boot += np.divide(np.abs(difference), valid_total, out=np.zeros(len(valid_total)), where=valid_total > 0)
        bins.append({"low_pct": lo * 100, "high_exclusive_pct": hi * 100,
                     "n_photographs": int(mask.sum()),
                     "mean_stated_pct": boot.mean(np.nan_to_num(conf) * 100, mask),
                     "observed_within_50g_pct": boot.mean(within * 100, mask)})
    _, inverse = np.unique(conf[ok], return_inverse=True)

    def weighted_auc(w):
        total = np.bincount(inverse, weights=w)
        ranks = (np.cumsum(total) - (total - 1) / 2)[inverse]
        yes, no = np.sum(w * within[ok]), np.sum(w * (~within[ok]))
        return (np.sum(w * within[ok] * ranks) - yes * (yes + 1) / 2) / (yes * no) if yes > 0 and no > 0 else np.nan

    auc_boot = np.array([weighted_auc(w[ok]) for w in boot.photo_weights])
    result["qwen_confidence"] = {
        "n_valid_photographs": int(ok.sum()),
        "mean_stated_pct": boot.mean(np.nan_to_num(conf) * 100, ok),
        "observed_within_50g_pct": boot.mean(within * 100, ok),
        "ece_pct": boot.record(ece * 100, ece_boot * 100),
        "auroc_within_50g": boot.record(auroc(conf[ok], within[ok]), auc_boot),
        "bins": bins,
    }
    items = s.food_item_count.to_numpy()
    result["by_items"] = {}
    for name in result["mass"]:
        ae = np.abs(primary[f"mass_{name}"].to_numpy() - t)
        result["by_items"][name] = {}
        for label, mask in [("1", items == 1), ("2", items == 2), ("3+", items >= 3)]:
            result["by_items"][name][label] = {"n_photographs": int(mask.sum()), "mae_g": boot.mean(ae, mask)}
    result["orientation_counts"] = {"portrait": int(portrait.sum()), "landscape": int((~portrait).sum())}
    result["amendment_implementation"] = {
        "source_commit": "1a311fc",
        "original_predictions_retained": True,
        "primary_preprocessing": "LANCZOS thumbnail; interpolation was unspecified in original plan",
        "sensitivity": "counter-clockwise 90-degree rotation of original processed portrait PNG, before unchanged model resize",
        "consistency": "both population SD across all dishes and sample SD excluding singleton dishes explicitly reported",
    }
    result = clean(result)
    (D / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["Amendment 1: portrait rotation sensitivity; primary pipeline remains primary."]
    for name, r in result["landscape_sensitivity"].items():
        point = r["rotated_mae_g"]["estimate"]
        diff = r["rotated_minus_primary_mae_g"]
        lines.append(f"{name}: rotated MAE {point:.3f} g; minus primary {diff['estimate']:+.3f} g, 95% CI {diff['ci95']}")
    c = result["qwen_confidence"]
    lines.append(f"Qwen confidence: mean stated {c['mean_stated_pct']['estimate']:.3f}%; observed within 50 g {c['observed_within_50g_pct']['estimate']:.3f}%; ECE {c['ece_pct']['estimate']:.3f}%; AUROC {c['auroc_within_50g']['estimate']:.3f}")
    text = "\n".join(lines) + "\n"
    # Retain original summary and replace any previous amendment block on CPU-only reruns.
    base = (D / "results.txt").read_text().split("Amendment 1:")[0].rstrip() + "\n\n"
    (D / "results.txt").write_text(base + text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
