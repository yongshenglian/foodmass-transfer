"""One-time scoring of single-prompt (open-vocabulary) Qwen on the Nutrition5k validation and test sets
(declared in experiments/n5k_single_prompt/PLAN.md before scoring). Run from code/."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("../experiments/n5k_single_prompt")
V1 = Path("../experiments/qwen3vl_journal_v1")
V2 = Path("../experiments/qwen3vl_journal_v2_training")
C = Path("../experiments/confirmation_photo_only")
FALLBACK, LO, HI = 186.0, 10.0, 1500.0


def load(path, col, ids):
    return pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[col].to_numpy(float).copy()


def qwen(path, ids):
    q = load(path, "predicted_weight", ids); bad = ~np.isfinite(q) | (q <= 0); q[bad] = FALLBACK
    return np.clip(q, 1, None), int(bad.sum())


def score(name, t, rows, pairs):
    e = {k: v - t for k, v in rows.items()}
    metrics = {k: dict(mae=float(np.mean(np.abs(x))), median_ae=float(np.median(np.abs(x))), rmse=float(np.sqrt(np.mean(x ** 2))),
                       bias=float(np.mean(x)), within_50g_pct=float(np.mean(np.abs(x) <= 50) * 100)) for k, x in e.items()}
    idx = np.random.default_rng(0).integers(0, len(t), (10000, len(t))); diffs = {}
    for a, b in pairs:
        d = np.abs(e[a]) - np.abs(e[b]); s = d[idx].mean(1)
        diffs[f"{a} minus {b}"] = [float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    print(f"== {name} (n = {len(t)})")
    for k, m in metrics.items():
        print(f"  {k:42s} MAE {m['mae']:6.2f}  MedAE {m['median_ae']:6.2f}  RMSE {m['rmse']:6.2f}  bias {m['bias']:+7.2f}  <=50g {m['within_50g_pct']:5.1f}%")
    for k, (d, lo, hi) in diffs.items():
        print(f"  {k}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    return {"n": len(t), "metrics": metrics, "paired": diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=str(D / "qwen_blind_val_fp8.csv"))
    ap.add_argument("--test", default=str(D / "qwen_blind_test_fp8.csv"))
    ap.add_argument("--out", default=str(D / "results.json"))
    a = ap.parse_args()
    SP, SPK, TS, TSK, DN = ("Qwen single prompt", "Qwen single prompt + kNN", "Qwen two-step", "Qwen two-step + kNN", "DINOv2 photo only")
    pairs = [(SP, TS), (SPK, SP), (SPK, TSK), (DN, SP), (DN, SPK)]
    out = {}
    # validation: same files as analyze_qwen_local.py
    base = pd.read_csv(V1 / "dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id"); ids = base.index
    t = base.true_weight.to_numpy(float); delta = base.neighbor_log_residual.to_numpy(float)
    sp, bad_sp = qwen(a.val, ids); ts, _ = qwen(V2 / "qwen_local/dev_predicted_fp8.csv", ids)
    rows = {SP: sp, SPK: np.clip(sp * np.exp(0.75 * delta), LO, HI), TS: ts, TSK: np.clip(ts * np.exp(0.75 * delta), LO, HI),
            DN: load(V2 / "stage_b_imageonly_aux10/development_predictions.csv", "mass", ids)}
    out["validation"] = {"single_prompt_imputed": bad_sp, **score("validation", t, rows, pairs)}
    # test: same files as score_confirmation_photo_only.py
    s = pd.read_csv(C / "confirmation_498_sample.csv").set_index("dish_id"); ids = s.index; t = s.true_weight.to_numpy(float)
    delta = load(C / "residual_transfer_delta_saved_embeddings.csv", "neighbor_log_residual", ids)
    sp, bad_sp = qwen(a.test, ids); ts, _ = qwen(C / "qwen_predicted_fp8.csv", ids)
    rows = {SP: sp, SPK: np.clip(sp * np.exp(0.75 * delta), LO, HI), TS: ts, TSK: np.clip(ts * np.exp(0.75 * delta), LO, HI),
            DN: load(C / "dinov2_photo_only_n5k.csv", "mass", ids)}
    out["test"] = {"single_prompt_imputed": bad_sp, **score("test", t, rows, pairs)}
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
