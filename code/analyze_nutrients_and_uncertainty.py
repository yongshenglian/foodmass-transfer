"""Descriptive analyses on frozen predictions (no model selection): nutrient conversion and uncertainty.

1. Nutrients: Qwen's per-ingredient gram estimates x Nutrition5k per-gram nutrient densities give
   calories/fat/carbs/protein. A corrected total mass M is turned into nutrients by scaling Qwen's
   ingredient split to M. Two analysis-only bounds separate total-mass error from composition error:
   true total mass with Qwen's split, and the predicted total mass with the true ingredient split.
2. Uncertainty: split-conformal intervals on the log scale, calibrated on reference out-of-fold errors
   only, evaluated for coverage and width on development and confirmation.
Run from code/.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

N5K = Path.home() / "data/nutrition5k_dataset/metadata"
V1 = Path("../experiments/qwen3vl_journal_v1")
V2 = Path("../experiments/qwen3vl_journal_v2_training")
CONF = Path.home() / "data/confirmation_2026-10-02/experiments/qwen3vl_journal_v1"
OUT = V2 / "analysis_nutrients_uncertainty"
NUTR = ["calories", "fat", "carb", "protein"]


def truth_table():
    rows = {}
    for cafe in (1, 2):
        for line in (N5K / f"dish_metadata_cafe{cafe}.csv").read_text().splitlines():
            r = line.split(",")
            ingr = [(r[i + 1], float(r[i + 2]), *map(float, r[i + 3:i + 7])) for i in range(6, len(r) - 6, 7) if r[i].startswith("ingr_")]
            rows[r[0]] = {"calories": float(r[1]), "mass": float(r[2]), "fat": float(r[3]), "carb": float(r[4]),
                          "protein": float(r[5]), "ingr": ingr}
    return rows


def densities():
    d = pd.read_csv(N5K / "ingredients_metadata.csv")
    return {n.strip().lower(): np.array([c, f, cb, p]) for n, c, f, cb, p in
            zip(d.ingr, d["cal/g"], d["fat(g)"], d["carb(g)"], d["protein(g)"])}


def qwen_split(raw, dens):
    """Per-gram nutrient density of Qwen's estimated dish, and Qwen's parsed total."""
    total, nutr, missing = 0.0, np.zeros(4), 0
    for name, grams in re.findall(r"^\s*(.+?):\s*([\d.]+)\s*g\s*$", str(raw), flags=re.M):
        g = float(grams); total += g
        key = name.strip().lower()
        if key in dens:
            nutr += g * dens[key]
        else:
            missing += 1
    return (nutr / total if total > 0 else np.full(4, np.nan)), total, missing


def nutrient_block(name, frame, truth, dens):
    """frame: dish_id, raw_response, and one column per mass method."""
    methods = [c for c in frame.columns if c not in ("dish_id", "raw_response")]
    split = [qwen_split(r, dens) for r in frame.raw_response]
    q_density = np.array([s[0] for s in split]); missing = sum(s[2] for s in split)
    t = np.array([[truth[d][k] for k in NUTR] for d in frame.dish_id])
    true_mass = np.array([truth[d]["mass"] for d in frame.dish_id])
    true_density = np.array([np.sum([np.array(i[2:6]) for i in truth[d]["ingr"]], 0) / max(truth[d]["mass"], 1e-9)
                             for d in frame.dish_id])
    rows = []
    for m in methods:
        mass = frame[m].to_numpy()
        pred = q_density * mass[:, None]
        rows.append({"cohort": name, "method": m, **{f"{k}_mae": float(np.nanmean(np.abs(pred[:, j] - t[:, j]))) for j, k in enumerate(NUTR)},
                     **{f"{k}_mape_pct": float(np.nanmean(np.abs(pred[:, j] - t[:, j])) / np.nanmean(t[:, j]) * 100) for j, k in enumerate(NUTR)}})
    for label, pred in [("bound: true mass x Qwen split", q_density * true_mass[:, None]),
                        ("bound: " + methods[-1] + " mass x true split", true_density * frame[methods[-1]].to_numpy()[:, None])]:
        rows.append({"cohort": name, "method": label, **{f"{k}_mae": float(np.nanmean(np.abs(pred[:, j] - t[:, j]))) for j, k in enumerate(NUTR)},
                     **{f"{k}_mape_pct": float(np.nanmean(np.abs(pred[:, j] - t[:, j])) / np.nanmean(t[:, j]) * 100) for j, k in enumerate(NUTR)}})
    return rows, {"cohort": name, "ingredient_lines_without_density": missing, "n": len(frame),
                  "true_mean": {k: float(t[:, j].mean()) for j, k in enumerate(NUTR)}}


def conformal(ref_err, pred, true, levels=(0.8, 0.9, 0.95)):
    out = []
    for a in levels:
        q = np.quantile(ref_err, np.ceil((len(ref_err) + 1) * a) / len(ref_err))
        lo, hi = pred * np.exp(-q), pred * np.exp(q)
        out.append({"level": a, "log_halfwidth": float(q), "coverage_pct": float(np.mean((true >= lo) & (true <= hi)) * 100),
                    "median_width_g": float(np.median(hi - lo))})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth, dens = truth_table(), densities()

    # reference: OOF stage B (primary) and raw Qwen
    ref_q = pd.read_csv(V1 / "reference_train2253_qwen3vl_results.csv").drop_duplicates("dish_id")
    ref_b = pd.read_csv(V2 / "stage_b_logmass_aux10_seed2/reference_oof.csv")
    ref = ref_q[["dish_id", "raw_response", "predicted_weight"]].merge(ref_b[["dish_id", "oof_prediction"]], on="dish_id")
    ref = pd.DataFrame({"dish_id": ref.dish_id, "raw_response": ref.raw_response, "raw_qwen": np.clip(ref.predicted_weight, 1, None),
                        "stage_b_oof": np.clip(np.exp(ref.oof_prediction), 10, 1500)})
    # confirmation: raw, fixed residual, stage B primary
    cq = pd.read_csv(CONF / "extension_498_qwen3vl_results.csv")[["dish_id", "raw_response"]]
    cp = pd.read_csv(V2 / "paper_tables/confirmation_row_predictions.csv")
    conf = cq.merge(cp, on="dish_id")
    conf = pd.DataFrame({"dish_id": conf.dish_id, "raw_response": conf.raw_response, "raw_qwen": conf.raw,
                         "fixed_residual": conf.fixed, "stage_b": conf.stage_b})
    rows, info = [], []
    for name, f in [("reference_oof", ref), ("confirmation", conf)]:
        r, i = nutrient_block(name, f, truth, dens); rows += r; info.append(i)
    nut = pd.DataFrame(rows); nut.to_csv(OUT / "nutrients.csv", index=False)

    # uncertainty
    ref_err = np.abs(np.log(ref.stage_b_oof / ref_q.set_index("dish_id").loc[ref.dish_id, "true_weight"].to_numpy()))
    ref_raw_err = np.abs(np.log(ref.raw_qwen / ref_q.set_index("dish_id").loc[ref.dish_id, "true_weight"].to_numpy()))
    unc = {}
    for name in ("development", "confirmation"):
        f = pd.read_csv(V2 / f"paper_tables/{name}_row_predictions.csv")
        unc[name] = {"stage_b": conformal(ref_err, f.stage_b.to_numpy(), f.true.to_numpy()),
                     "raw_qwen": conformal(ref_raw_err, f.raw.to_numpy(), f.true.to_numpy())}
    # does fold disagreement flag large errors? (development: per-fold predictions are saved)
    folds = [np.load(V2 / f"stage_b_logmass_aux10_seed2/fold{k}.npz")["dev_residual"] for k in range(5)]
    spread = np.std(folds, 0)
    dev = pd.read_csv(V2 / "stage_b_logmass_aux10_seed2/development_predictions.csv")
    abs_log_err = np.abs(np.log(dev.mass / dev.true))
    rank_corr = float(np.corrcoef(pd.Series(spread).rank(), pd.Series(abs_log_err).rank())[0, 1])
    top = spread >= np.quantile(spread, 0.8)
    unc["development_fold_spread"] = {"spearman_spread_vs_abs_log_error": rank_corr,
                                      "mae_top20pct_spread_g": float(np.mean(np.abs(dev.mass - dev.true)[top])),
                                      "mae_rest_g": float(np.mean(np.abs(dev.mass - dev.true)[~top]))}
    (OUT / "summary.json").write_text(json.dumps({"nutrient_info": info, "uncertainty": unc}, indent=2))
    pd.set_option("display.width", 200)
    print(nut[["cohort", "method"] + [f"{k}_mae" for k in NUTR] + [f"{k}_mape_pct" for k in NUTR]].round(1).to_string(index=False))
    print(json.dumps({"nutrient_info": info, "uncertainty": unc}, indent=1))


if __name__ == "__main__":
    main()
