"""One-time scoring of the photograph-only methods on the Nutrition5k confirmation cohort (third, later use; declared in
STAGE_B_FREEZE_AND_CONFIRMATION_PLAN_2026-10-02.md before scoring). Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("../experiments/confirmation_photo_only")
H = Path.home() / "data/confirmation_2026-10-02/experiments/qwen3vl_journal_v1"


def main():
    paper = pd.read_csv(H / "extension_498_dinov2_gate/confirmation_predictions.csv").set_index("dish_id")
    ids = paper.index; t = paper.true_weight.to_numpy(float)
    load = lambda f, col="mass": pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[col].to_numpy(float).copy()
    q = load(C / "qwen_predicted_fp8.csv", "predicted_weight"); bad = ~(q > 0); q[bad] = 186.0
    delta = load(C / "residual_transfer_delta_saved_embeddings.csv", "neighbor_log_residual")
    rows = {
        "Qwen, photo only (two-step)": q,
        "Residual transfer on Qwen, photo only": np.clip(q * np.exp(0.75 * delta), 10, 1500),
        "DINOv2, photo only (Nutrition5k-trained)": load(C / "dinov2_photo_only_n5k.csv"),
        "DINOv2-large, photo only (multi-domain)": load(C / "dinov2_large_multidomain.csv"),
        "Reference: raw Qwen with ground-truth names": paper.predicted_weight.to_numpy(float),
        "Reference: residual transfer with names": paper.fixed_dinov2_visual_residual.to_numpy(float),
        "Reference: name-conditioned DINOv2": load(Path("../experiments/qwen3vl_journal_v2_training/confirmation_498/stage_b_logmass_aux10_seed2_confirmation_predictions.csv")),
    }
    print(f"n = {len(t)}; Qwen invalid/zero imputed: {int(bad.sum())}")
    print(f"{'Method':45s} {'MAE':>7s} {'MedAE':>7s} {'RMSE':>7s} {'Bias':>7s} {'<=50g':>6s}")
    out = {}
    for k, v in rows.items():
        e = v - t
        m = dict(mae=np.mean(np.abs(e)), medae=np.median(np.abs(e)), rmse=np.sqrt(np.mean(e ** 2)), bias=e.mean(), w50=np.mean(np.abs(e) <= 50) * 100)
        out[k] = m
        print(f"{k:45s} {m['mae']:7.2f} {m['medae']:7.2f} {m['rmse']:7.2f} {m['bias']:+7.2f} {m['w50']:5.1f}%")
    idx = np.random.default_rng(0).integers(0, len(t), (10000, len(t))); pairs = {}
    for a, b in [("DINOv2, photo only (Nutrition5k-trained)", "Qwen, photo only (two-step)"),
                 ("DINOv2, photo only (Nutrition5k-trained)", "Residual transfer on Qwen, photo only"),
                 ("DINOv2, photo only (Nutrition5k-trained)", "Reference: name-conditioned DINOv2"),
                 ("DINOv2, photo only (Nutrition5k-trained)", "Reference: residual transfer with names"),
                 ("DINOv2-large, photo only (multi-domain)", "DINOv2, photo only (Nutrition5k-trained)"),
                 ("Residual transfer on Qwen, photo only", "Qwen, photo only (two-step)"),
                 ("Qwen, photo only (two-step)", "Reference: raw Qwen with ground-truth names")]:
        d = np.abs(rows[a] - t) - np.abs(rows[b] - t); s = d[idx].mean(1)
        pairs[f"{a} minus {b}"] = [float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
        print(f"  {a} minus {b}: {d.mean():+.2f} [{np.percentile(s, 2.5):+.2f}, {np.percentile(s, 97.5):+.2f}]")
    (C / "results.json").write_text(json.dumps({"n": len(t), "qwen_imputed": int(bad.sum()), "metrics": out, "paired": pairs}, indent=2, default=float))


if __name__ == "__main__":
    main()
