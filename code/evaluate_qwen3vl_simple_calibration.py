"""Fit simple Qwen mass calibrators on training predictions and evaluate development data.

This script never reads the locked final-holdout ID file. It is intentionally
dependency-light and writes row-level predictions plus paired-bootstrap CIs.
Run from code/.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def valid_rows(path):
    frame = pd.read_csv(path).copy()
    valid_column = "format_valid" if "format_valid" in frame else "complete"
    valid = frame[valid_column].astype(str).str.lower().isin({"true", "1"})
    for column in ("predicted_weight", "true_weight", "confidence", "ingredient_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[valid].dropna(subset=["dish_id", "predicted_weight", "true_weight"]).copy()


def metrics(truth, prediction):
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    error = prediction - truth
    absolute = np.abs(error)
    return {
        "n": len(truth),
        "mae_g": float(absolute.mean()),
        "median_ae_g": float(np.median(absolute)),
        "rmse_g": float(np.sqrt(np.mean(error ** 2))),
        "bias_g": float(error.mean()),
        "wape": float(absolute.sum() / truth.sum()),
        "within_25g": float((absolute <= 25).mean()),
        "within_50g": float((absolute <= 50).mean()),
        "within_100g": float((absolute <= 100).mean()),
    }


def pava_fit(x, y):
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = np.asarray(x)[order], np.asarray(y)[order]
    blocks = []
    for index, value in enumerate(sorted_y):
        blocks.append([index, index, float(value), 1])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            count = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / count
            blocks.append([left[0], right[1], mean, count])
    fitted = np.empty(len(sorted_y))
    for start, stop, value, _ in blocks:
        fitted[start:stop + 1] = value
    # Collapse tied x values so interpolation is deterministic.
    table = pd.DataFrame({"x": sorted_x, "fit": fitted}).groupby("x", as_index=False).fit.mean()
    return table.x.to_numpy(float), table.fit.to_numpy(float)


def feature_matrix(frame):
    pred = np.maximum(frame.predicted_weight.to_numpy(float), 1)
    confidence = frame.confidence.fillna(frame.confidence.median()).to_numpy(float) / 100
    count = frame.ingredient_count.fillna(frame.ingredient_count.median()).to_numpy(float)
    return np.column_stack([
        np.log(pred), confidence, np.log1p(count),
        np.maximum(pred - 150, 0) / 100,
        np.maximum(pred - 300, 0) / 100,
        np.maximum(pred - 500, 0) / 100,
        confidence * np.log(pred),
    ])


def ridge_fit(x, y, alpha):
    center = np.median(x, axis=0)
    scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    scale[scale < 1e-8] = 1
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.diag([0] + [alpha] * z.shape[1])
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return center, scale, coefficient


def ridge_predict(model, x):
    center, scale, coefficient = model
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])
    return design @ coefficient


def select_ridge_alpha(frame, seed):
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    rng = np.random.default_rng(seed)
    fold = np.arange(len(frame)) % 5
    rng.shuffle(fold)
    x = feature_matrix(frame)
    pred = np.maximum(frame.predicted_weight.to_numpy(float), 1)
    target = np.log(np.maximum(frame.true_weight.to_numpy(float), 1)) - np.log(pred)
    scores = {}
    for alpha in alphas:
        fold_errors = []
        for held_out in range(5):
            train = fold != held_out
            test = ~train
            model = ridge_fit(x[train], target[train], alpha)
            calibrated = pred[test] * np.exp(ridge_predict(model, x[test]))
            fold_errors.extend(np.abs(calibrated - frame.true_weight.to_numpy(float)[test]))
        scores[alpha] = float(np.mean(fold_errors))
    return min(scores, key=scores.get), scores


def bootstrap_difference(truth, baseline, candidate, iterations, seed):
    truth, baseline, candidate = map(lambda value: np.asarray(value, float), (truth, baseline, candidate))
    base_error = np.abs(baseline - truth)
    candidate_error = np.abs(candidate - truth)
    difference = candidate_error - base_error
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(truth), len(truth))
        draws[index] = difference[sample].mean()
    return {
        "mae_difference_g": float(difference.mean()),
        "ci95_low_g": float(np.percentile(draws, 2.5)),
        "ci95_high_g": float(np.percentile(draws, 97.5)),
        "probability_better_than_raw": float((draws < 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    train, development = valid_rows(args.train), valid_rows(args.development)
    overlap = set(train.dish_id.astype(str)) & set(development.dish_id.astype(str))
    if overlap:
        raise RuntimeError(f"Training/development leakage: {len(overlap)} overlapping dish IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_pred = train.predicted_weight.to_numpy(float)
    train_true = train.true_weight.to_numpy(float)
    dev_pred = development.predicted_weight.to_numpy(float)
    dev_true = development.true_weight.to_numpy(float)

    predictions = {"E0_raw_qwen": dev_pred}
    additive = float(np.median(train_true - train_pred))
    predictions["E1_median_bias"] = np.clip(dev_pred + additive, 10, 1500)
    slope, intercept = np.polyfit(train_pred, train_true, 1)
    predictions["E2_affine"] = np.clip(slope * dev_pred + intercept, 10, 1500)
    log_slope, log_intercept = np.polyfit(np.log(np.maximum(train_pred, 1)), np.log(np.maximum(train_true, 1)), 1)
    predictions["E3_log_affine"] = np.clip(np.exp(log_intercept) * np.maximum(dev_pred, 1) ** log_slope, 10, 1500)
    iso_x, iso_y = pava_fit(train_pred, train_true)
    predictions["E4_isotonic"] = np.clip(np.interp(dev_pred, iso_x, iso_y), 10, 1500)
    alpha, cv_scores = select_ridge_alpha(train, args.seed)
    train_x = feature_matrix(train)
    residual = np.log(np.maximum(train_true, 1)) - np.log(np.maximum(train_pred, 1))
    ridge = ridge_fit(train_x, residual, alpha)
    predictions["E5_feature_log_residual"] = np.clip(
        dev_pred * np.exp(ridge_predict(ridge, feature_matrix(development))), 10, 1500
    )

    row_output = development[["dish_id", "true_weight", "predicted_weight", "confidence", "ingredient_count"]].copy()
    summary = []
    bootstrap = {}
    for name, values in predictions.items():
        row_output[name] = values
        result = {"method": name, **metrics(dev_true, values)}
        paired = bootstrap_difference(dev_true, dev_pred, values, args.bootstrap, args.seed)
        result.update(paired)
        summary.append(result)
        bootstrap[name] = paired
    summary_frame = pd.DataFrame(summary).sort_values("mae_g")
    row_output.to_csv(args.output_dir / "development_predictions.csv", index=False)
    summary_frame.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    configuration = {
        "train_rows": len(train), "development_rows": len(development), "overlap": len(overlap),
        "median_additive_correction_g": additive,
        "affine_slope": float(slope), "affine_intercept": float(intercept),
        "log_affine_slope": float(log_slope), "log_affine_intercept": float(log_intercept),
        "feature_ridge_selected_alpha": alpha, "feature_ridge_cv_mae_by_alpha": cv_scores,
        "bootstrap_iterations": args.bootstrap, "seed": args.seed,
    }
    (args.output_dir / "calibration_configuration.json").write_text(json.dumps(configuration, indent=2))
    lines = [
        "# Qwen3-VL Simple Calibration: Development Results", "",
        f"Training dishes: {len(train)}  ", f"Development dishes: {len(development)}  ",
        "Training/development overlap: 0", "",
        "| Method | MAE (g) | Difference vs raw (g) | Paired 95% CI (g) | Median AE (g) | Bias (g) | Within 50 g |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['mae_g']:.2f} | {row['mae_difference_g']:+.2f} | "
            f"[{row['ci95_low_g']:+.2f}, {row['ci95_high_g']:+.2f}] | {row['median_ae_g']:.2f} | "
            f"{row['bias_g']:+.2f} | {row['within_50g']:.1%} |"
        )
    lines.extend(["", "These are development results. The locked final holdout was not read or evaluated.", ""])
    (args.output_dir / "CALIBRATION_RESULTS.md").write_text("\n".join(lines))
    print(summary_frame.to_string(index=False))
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    main()
