"""Development-only analysis of the local Qwen3-VL-32B FP8 runs (descriptive; no model selection).

A. Quantization check: FP8 with the original oracle-name prompt vs the paper's bf16 outputs.
B. Recognition: Qwen's closed-vocabulary ingredient lists vs ground truth.
C. End-to-end mass: every method fed by Qwen with recognized (not ground-truth) names.
D. Nutrients through Qwen's ingredient split, oracle vs recognized names.
Stage B rows come from score_stage_b_checkpoints.py outputs (see the paths below). Run from code/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_nutrients_and_uncertainty import NUTR, densities, qwen_split, truth_table

V1 = Path("../experiments/qwen3vl_journal_v1")
V2 = Path("../experiments/qwen3vl_journal_v2_training")
Q = V2 / "qwen_local"
LO, HI = 10.0, 1500.0


def mae(x, t):
    return float(np.mean(np.abs(x - t)))


def boot(x, y, t, idx):
    d = np.abs(x - t) - np.abs(y - t); s = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def full_coverage(pred, fallback):
    """Invalid Qwen outputs get the reference-bank median mass, so every dish keeps a prediction."""
    pred = pred.copy(); bad = ~np.isfinite(pred) | (pred <= 0)
    pred[bad] = fallback
    return np.clip(pred, 1, None), int(bad.sum())


def main():
    out = {}
    base = pd.read_csv(V1 / "dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id")
    ids = base.index; t = base.true_weight.to_numpy()
    idx = np.random.default_rng(0).integers(0, len(t), (10000, len(t)))
    ref_median = float(pd.read_csv(V1 / "reference_train2253_qwen3vl_results.csv").true_weight.median())
    oracle = pd.read_csv(Q / "dev_oracle_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(ids)
    rec = pd.read_csv(Q / "dev_recognize_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(ids)
    pred = pd.read_csv(Q / "dev_predicted_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(ids)

    # A. quantization
    q16 = base.predicted_weight.to_numpy(); q8, bad8 = full_coverage(oracle.predicted_weight.to_numpy(), ref_median)
    out["A_quantization"] = {
        "fp8_valid": int(oracle.format_valid.fillna(False).sum()), "fp8_invalid_imputed": bad8,
        "mae_bf16": mae(q16, t), "mae_fp8": mae(q8, t), "fp8_minus_bf16": boot(q8, q16, t, idx),
        "exact_same_total_pct": float(np.mean(q8 == q16) * 100),
        "log_corr": float(np.corrcoef(np.log(q8), np.log(np.clip(q16, 1, None)))[0, 1]),
        "median_abs_log_ratio": float(np.median(np.abs(np.log(q8 / np.clip(q16, 1, None))))),
        "confidence_exact_match_pct": float(np.mean(oracle.confidence.to_numpy() == base.confidence.to_numpy()) * 100)}

    # B. recognition
    truth = truth_table()
    tp = fp = fn = 0; f1s, mass_recall, empty = [], [], 0
    for d in ids:
        true_names = {i[0].strip().lower(): i[1] for i in truth[d]["ingr"]}
        p = {n.strip().lower() for n in str(rec.loc[d, "predicted_ingredients"]).split(" | ") if n.strip() and n != "nan"}
        empty += not p
        hit = set(true_names) & p
        tp += len(hit); fp += len(p - hit); fn += len(set(true_names) - hit)
        pr = len(hit) / len(p) if p else 0.0; rc = len(hit) / len(true_names)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        mass_recall.append(sum(true_names[n] for n in hit) / max(sum(true_names.values()), 1e-9))
    pred_count = pred.ingredient_count.to_numpy(float); true_count = base.ingredient_count.to_numpy(float)
    out["B_recognition"] = {"micro_precision": tp / (tp + fp), "micro_recall": tp / (tp + fn), "mean_dish_f1": float(np.mean(f1s)),
                            "mean_true_mass_recalled_pct": float(np.mean(mass_recall) * 100), "dishes_with_no_names": empty,
                            "mean_true_count": float(true_count.mean()), "mean_predicted_count": float(np.nanmean(pred_count))}

    # C. end-to-end mass
    qp, badp = full_coverage(pred.predicted_weight.to_numpy(), ref_median)
    delta = base.neighbor_log_residual.to_numpy()
    rows = {"Raw Qwen, oracle names (bf16, paper)": q16, "Raw Qwen, oracle names (FP8)": q8,
            "Raw Qwen, recognized names (FP8)": qp,
            "Residual transfer, oracle names (bf16, paper)": base.fixed_dinov2_visual_residual.to_numpy(),
            "Residual transfer, oracle names (FP8)": np.clip(q8 * np.exp(0.75 * delta), LO, HI),
            "Residual transfer, recognized names (FP8)": np.clip(qp * np.exp(0.75 * delta), LO, HI)}
    stage_b = {"Fine-tuned primary, oracle inputs (bf16 Qwen)": V2 / "stage_b_logmass_aux10_seed2/development_predictions.csv",
               "Fine-tuned primary, recognized inputs (FP8 Qwen)": Q / "score_primary_recognized_fp8.csv",
               "Fine-tuned no multi-hot, recognized inputs (FP8 Qwen)": Q / "score_noingr_recognized_fp8.csv",
               "Fine-tuned image-only (no Qwen, no names)": V2 / "stage_b_imageonly_aux10/development_predictions.csv"}
    for k, path in stage_b.items():
        if path.is_file():
            rows[k] = pd.read_csv(path).set_index("dish_id").reindex(ids).mass.to_numpy()
    ref_key = "Fine-tuned image-only (no Qwen, no names)"
    out["C_end_to_end_mass"] = {"recognized_invalid_imputed": badp, "rows": {
        k: {"mae": mae(v, t), "median_ae": float(np.median(np.abs(v - t))), "within_50g_pct": float(np.mean(np.abs(v - t) <= 50) * 100),
            "minus_raw_qwen_oracle_bf16": boot(v, q16, t, idx),
            **({"minus_image_only": boot(v, rows[ref_key], t, idx)} if ref_key in rows and k != ref_key else {})}
        for k, v in rows.items()}}

    # D. nutrients through Qwen's split (oracle FP8 vs recognized FP8), development
    dens = densities(); tn = np.array([[truth[d][k] for k in NUTR] for d in ids])
    split_o = np.array([qwen_split(r, dens)[0] for r in oracle.raw_response])
    split_p = np.array([qwen_split(r, dens)[0] for r in pred.raw_response])
    nut = {}
    for label, split, masses in [("oracle names", split_o, {"raw Qwen": q8, "residual transfer": rows["Residual transfer, oracle names (FP8)"],
                                                            "fine-tuned (oracle inputs)": rows.get("Fine-tuned primary, oracle inputs (bf16 Qwen)"),
                                                            "true mass (bound)": t}),
                                 ("recognized names", split_p, {"raw Qwen": qp, "residual transfer": rows["Residual transfer, recognized names (FP8)"],
                                                                "fine-tuned image-only": rows.get(ref_key), "true mass (bound)": t})]:
        for m, v in masses.items():
            if v is None:
                continue
            p_ = split * v[:, None]
            nut[f"{label} | {m}"] = {k: float(np.nanmean(np.abs(p_[:, j] - tn[:, j]))) for j, k in enumerate(NUTR)}
    out["D_nutrients_dev"] = {"true_mean": {k: float(tn[:, j].mean()) for j, k in enumerate(NUTR)}, "mae": nut}
    (Q / "analysis_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
