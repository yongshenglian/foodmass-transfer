"""Stage F: blend a stage B DINOv2 model with Qwen's raw mass estimate (log space).

Inputs are a stage B run folder (reference_oof.csv, development_predictions.csv,
config.json). Candidate blends are fitted on the reference out-of-fold predictions
only and scored by 5-fold reference CV; the blend with the lowest reference CV MAE
is selected, refitted on all reference rows and applied once to development.
Development MAE of every candidate is also written, but is not used for selection.
The confirmation cohort is never loaded. Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LO, HI = 10.0, 1500.0
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
ALPHAS = [0.01, 0.1, 1.0, 10.0]


def log_prediction(frame, pred_column, target):
    pred = frame[pred_column].to_numpy(float)
    return np.log(frame.qwen.to_numpy(float)) + pred if target == "residual" else pred


def to_mass(logm):
    return np.clip(np.exp(logm), LO, HI)


def mae(logm, true):
    return float(np.mean(np.abs(to_mass(logm) - true)))


def fit_weight(lb, lq, true):
    """Weight on the stage B log prediction that minimises gram MAE."""
    errors = [mae(w * lb + (1 - w) * lq, true) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmin(errors))])


def fit_ridge(lb, lq, true, alpha):
    x = np.column_stack([lb, lq])
    mu = x.mean(0)
    xc, y = x - mu, np.log(true)
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(2), xc.T @ (y - y.mean()))
    return {"coef": coef.tolist(), "intercept": float(y.mean() - mu @ coef)}


def apply_ridge(model, lb, lq):
    return np.column_stack([lb, lq]) @ np.asarray(model["coef"]) + model["intercept"]


CANDIDATES = {
    "stage_b_only": (lambda lb, lq, t: None, lambda m, lb, lq: lb),
    "qwen_only": (lambda lb, lq, t: None, lambda m, lb, lq: lq),
    "weighted_average": (lambda lb, lq, t: {"w": fit_weight(lb, lq, t)},
                         lambda m, lb, lq: m["w"] * lb + (1 - m["w"]) * lq),
    **{f"ridge_alpha{a:g}": ((lambda a: lambda lb, lq, t: fit_ridge(lb, lq, t, a))(a), apply_ridge)
       for a in ALPHAS},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage-b-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261003)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; use a new folder")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = json.loads((args.stage_b_dir / "config.json").read_text())["target"]
    ref = pd.read_csv(args.stage_b_dir / "reference_oof.csv")
    dev = pd.read_csv(args.stage_b_dir / "development_predictions.csv")
    rb, rq, rt = log_prediction(ref, "oof_prediction", target), np.log(ref.qwen.to_numpy(float)), ref.true.to_numpy(float)
    db, dq, dt = log_prediction(dev, "prediction", target), np.log(dev.qwen.to_numpy(float)), dev.true.to_numpy(float)

    fold = np.random.default_rng(args.seed).permutation(np.arange(len(ref)) % args.folds)
    rows, fitted = [], {}
    for name, (fit, apply) in CANDIDATES.items():
        cv = np.empty(len(ref))
        for f in range(args.folds):
            tr, te = fold != f, fold == f
            cv[te] = apply(fit(rb[tr], rq[tr], rt[tr]), rb[te], rq[te])
        model = fit(rb, rq, rt)
        fitted[name] = model
        rows.append({"method": name, "reference_cv_mae_g": mae(cv, rt),
                     "development_mae_g": mae(apply(model, db, dq), dt), "params": json.dumps(model)})
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "blend_results.csv", index=False)
    print(results.to_string(index=False))

    best = results.loc[results.reference_cv_mae_g.idxmin(), "method"]
    dev_log = CANDIDATES[best][1](fitted[best], db, dq)
    pd.DataFrame({"dish_id": dev.dish_id, "true": dt, "qwen": dev.qwen, "stage_b_mass": to_mass(db),
                  "blend_mass": to_mass(dev_log)}).to_csv(args.output_dir / "development_predictions.csv", index=False)
    summary = {"stage_b_dir": str(args.stage_b_dir), "stage_b_target": target,
               "selected_by_reference_cv": best, "params": fitted[best],
               "reference_cv_mae_g": float(results.set_index("method").loc[best, "reference_cv_mae_g"]),
               "development_mae_g": float(results.set_index("method").loc[best, "development_mae_g"]),
               "note": "reference OOF comes from single fold models; development uses the 5-fold ensemble",
               "confirmation_cohort_used": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
