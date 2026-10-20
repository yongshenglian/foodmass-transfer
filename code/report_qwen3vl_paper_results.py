"""Consolidate Qwen3-VL development results into paper-ready metrics.

The script reads only completed development artifacts. Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_column(path, column):
    frame = pd.read_csv(path, usecols=["dish_id", "true_weight", column]).copy()
    frame = frame.rename(columns={column: "prediction"})
    return frame


def metrics(frame):
    truth = frame.true_weight.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    error = prediction - truth
    absolute = np.abs(error)
    return {
        "n": len(frame), "coverage": 1.0,
        "mae_g": float(absolute.mean()), "median_ae_g": float(np.median(absolute)),
        "rmse_g": float(np.sqrt(np.mean(error ** 2))), "bias_g": float(error.mean()),
        "wape": float(absolute.sum() / truth.sum()),
        "within_25g": float((absolute <= 25).mean()),
        "within_50g": float((absolute <= 50).mean()),
        "within_100g": float((absolute <= 100).mean()),
    }


def paired_bootstrap(frame, baseline, iterations, seed):
    merged = frame.merge(
        baseline[["dish_id", "true_weight", "prediction"]], on=["dish_id", "true_weight"],
        suffixes=("_candidate", "_baseline"), validate="one_to_one",
    )
    truth = merged.true_weight.to_numpy(float)
    difference = np.abs(merged.prediction_candidate - truth) - np.abs(merged.prediction_baseline - truth)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(merged), len(merged))
        draws[index] = difference.to_numpy()[sample].mean()
    return {
        "mae_difference_vs_qwen_g": float(difference.mean()),
        "difference_ci95_low_g": float(np.percentile(draws, 2.5)),
        "difference_ci95_high_g": float(np.percentile(draws, 97.5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--dinov2-retrieval", type=Path, required=True)
    parser.add_argument("--dinov2-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    sources = {
        "Raw Qwen3-VL": (args.simple, "E0_raw_qwen", "baseline"),
        "Median-bias calibration": (args.simple, "E1_median_bias", "calibration"),
        "Handcrafted multimodal residual": (args.handcrafted, "fixed_multimodal_residual", "local residual"),
        "Handcrafted bounded gate": (args.handcrafted, "learned_bounded_gate", "gated residual"),
        "DINOv2 reference-only": (args.dinov2_retrieval, "R1_dinov2_visual_reference_only", "reference-only"),
        "DINOv2 visual residual": (args.dinov2_gate, "fixed_dinov2_visual_residual", "local residual"),
        "DINOv2 bounded gate": (args.dinov2_gate, "learned_bounded_gate", "gated residual"),
    }
    frames = {name: load_column(path, column) for name, (path, column, _) in sources.items()}
    baseline = frames["Raw Qwen3-VL"]
    baseline_ids = set(baseline.dish_id)
    rows = []
    for name, frame in frames.items():
        if len(frame) != 502 or set(frame.dish_id) != baseline_ids:
            raise RuntimeError(f"Cohort mismatch for {name}: {len(frame)} rows")
        result = {"method": name, "family": sources[name][2], **metrics(frame)}
        result.update(paired_bootstrap(frame, baseline, args.bootstrap, args.seed))
        rows.append(result)
    summary = pd.DataFrame(rows).sort_values("mae_g")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "paper_development_metrics.csv", index=False)
    (args.output_dir / "paper_development_metrics.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2)
    )
    lines = [
        "# Qwen3-VL Journal Study: Development Results", "",
        "Cohort: 502 usable dishes from the official Nutrition5k depth-test split.  ",
        "Evaluation role: development/legacy benchmark; not the locked final holdout.  ",
        "Coverage: 100% for every method.", "",
        "| Method | MAE | Change vs Qwen (95% CI) | Median AE | RMSE | Bias | WAPE | <=25 g | <=50 g | <=100 g |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_name = {row["method"]: row for row in rows}
    for name in sources:
        row = by_name[name]
        change = f"{row['mae_difference_vs_qwen_g']:+.2f} [{row['difference_ci95_low_g']:+.2f}, {row['difference_ci95_high_g']:+.2f}]"
        lines.append(
            f"| {name} | {row['mae_g']:.2f} | {change} | {row['median_ae_g']:.2f} | "
            f"{row['rmse_g']:.2f} | {row['bias_g']:+.2f} | {row['wape']:.1%} | "
            f"{row['within_25g']:.1%} | {row['within_50g']:.1%} | {row['within_100g']:.1%} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "Simple calibration does not improve Qwen. Transferring Qwen log-residuals from visually similar measured training dishes is the main source of improvement. DINOv2 is substantially more effective than handcrafted RGB features. The bounded gate provides a smaller refinement whose incremental gain over fixed DINOv2 residual transfer is not statistically established on development data.",
        "", "The locked final holdout was not read or evaluated.", "",
    ])
    (args.output_dir / "PAPER_DEVELOPMENT_RESULTS.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
