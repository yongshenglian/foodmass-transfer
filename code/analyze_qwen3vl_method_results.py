"""Generate subgroup and robustness summaries for the Qwen retrieval method.

This analysis reads development predictions only. Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(frame, prediction):
    truth = frame.true_weight.to_numpy(float)
    pred = frame[prediction].to_numpy(float)
    error = pred - truth
    return {
        "mae_g": float(np.abs(error).mean()),
        "median_ae_g": float(np.median(np.abs(error))),
        "rmse_g": float(np.sqrt(np.mean(error ** 2))),
        "bias_g": float(error.mean()),
        "within_50g": float((np.abs(error) <= 50).mean()),
    }


def paired_ci(frame, first, second, seed, iterations=10000):
    truth = frame.true_weight.to_numpy(float)
    difference = np.abs(frame[first].to_numpy(float) - truth) - np.abs(frame[second].to_numpy(float) - truth)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(frame), len(frame))
        draws[index] = difference[sample].mean()
    return {
        "mae_difference_g": float(difference.mean()),
        "ci95_low_g": float(np.percentile(draws, 2.5)),
        "ci95_high_g": float(np.percentile(draws, 97.5)),
    }


def add_groups(frame):
    frame = frame.copy()
    frame["complexity"] = np.where(frame.ingredient_count <= 1, "single", "multiple")
    frame["ingredient_count_group"] = pd.cut(
        frame.ingredient_count, bins=[0, 1, 2, 4, np.inf],
        labels=["1", "2", "3-4", "5+"], include_lowest=True,
    ).astype(str)
    frame["confidence_group"] = pd.cut(
        frame.confidence, bins=[-np.inf, 65, 79, np.inf],
        labels=["<=65", "70-79", ">=80"], include_lowest=True,
    ).astype(str)
    frame["mass_group"] = pd.cut(
        frame.true_weight, bins=[0, 100, 200, 300, 450, np.inf],
        labels=["<100", "100-199", "200-299", "300-449", ">=450"],
        right=False, include_lowest=True,
    ).astype(str)
    frame["raw_error_direction"] = np.where(
        frame.predicted_weight < frame.true_weight, "Qwen underestimates", "Qwen overestimates"
    )
    return frame


def subgroup_rows(frame):
    rows = []
    for group_type in ("complexity", "ingredient_count_group", "confidence_group", "mass_group", "raw_error_direction"):
        for group, subset in frame.groupby(group_type, observed=True):
            if not len(subset):
                continue
            raw = metrics(subset, "predicted_weight")
            fixed = metrics(subset, "fixed_multimodal_residual")
            gated = metrics(subset, "learned_continuous_gate")
            truth = subset.true_weight.to_numpy(float)
            raw_error = np.abs(subset.predicted_weight.to_numpy(float) - truth)
            gate_error = np.abs(subset.learned_continuous_gate.to_numpy(float) - truth)
            rows.append({
                "group_type": group_type, "group": str(group), "n": len(subset),
                "raw_mae_g": raw["mae_g"], "fixed_mae_g": fixed["mae_g"],
                "gated_mae_g": gated["mae_g"],
                "gated_change_vs_raw_g": gated["mae_g"] - raw["mae_g"],
                "gated_relative_change": (gated["mae_g"] - raw["mae_g"]) / raw["mae_g"],
                "fraction_dishes_improved": float((gate_error < raw_error).mean()),
                "raw_bias_g": raw["bias_g"], "gated_bias_g": gated["bias_g"],
                "raw_within_50g": raw["within_50g"], "gated_within_50g": gated["within_50g"],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-column", default="fixed_multimodal_residual")
    parser.add_argument("--gate-column", default="learned_continuous_gate")
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    full = add_groups(pd.read_csv(args.full))
    small = pd.read_csv(args.small)
    if args.fixed_column not in full or args.fixed_column not in small:
        raise RuntimeError(f"Missing fixed method column: {args.fixed_column}")
    if args.gate_column not in full or args.gate_column not in small:
        raise RuntimeError(f"Missing gate method column: {args.gate_column}")
    full["fixed_multimodal_residual"] = full[args.fixed_column]
    small["fixed_multimodal_residual"] = small[args.fixed_column]
    full["learned_continuous_gate"] = full[args.gate_column]
    small["learned_continuous_gate"] = small[args.gate_column]
    required = {"dish_id", "true_weight", "predicted_weight", "fixed_multimodal_residual", "learned_continuous_gate"}
    if not required.issubset(full) or not required.issubset(small):
        raise RuntimeError("Prediction files do not contain the required method columns")
    subgroups = subgroup_rows(full)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subgroups.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)

    overall_rows = []
    for training_size, frame in ((600, small), (2253, full)):
        for method in ("predicted_weight", "fixed_multimodal_residual", "learned_continuous_gate"):
            overall_rows.append({"residual_training_size": training_size, "method": method, **metrics(frame, method)})
    learning = pd.DataFrame(overall_rows)
    learning.to_csv(args.output_dir / "learning_curve.csv", index=False)

    comparison = paired_ci(full, "learned_continuous_gate", "fixed_multimodal_residual", args.seed)
    raw_error = np.abs(full.predicted_weight - full.true_weight)
    gated_error = np.abs(full.learned_continuous_gate - full.true_weight)
    multiplier = full.learned_continuous_gate / full.predicted_weight.clip(lower=1)
    robustness = {
        "full_development_rows": len(full),
        "fraction_dishes_improved": float((gated_error < raw_error).mean()),
        "fraction_dishes_unchanged": float(np.isclose(gated_error, raw_error).mean()),
        "gate_vs_fixed_paired_mae": comparison,
        "gated_mass_multiplier_quantiles": {
            str(quantile): float(multiplier.quantile(quantile))
            for quantile in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
        },
    }
    (args.output_dir / "robustness.json").write_text(json.dumps(robustness, indent=2))

    lines = [
        "# Qwen3-VL Retrieval Method: Subgroup and Robustness Results", "",
        "## Learning curve", "",
        "| Residual-training dishes | Method | MAE (g) | Median AE (g) | Bias (g) | Within 50 g |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    labels = {"predicted_weight": "Raw Qwen", "fixed_multimodal_residual": "Fixed residual", "learned_continuous_gate": "Learned gate"}
    for row in overall_rows:
        lines.append(
            f"| {row['residual_training_size']} | {labels[row['method']]} | {row['mae_g']:.2f} | "
            f"{row['median_ae_g']:.2f} | {row['bias_g']:+.2f} | {row['within_50g']:.1%} |"
        )
    lines.extend([
        "", "## Subgroups for the 2,253-dish residual bank", "",
        "| Group type | Group | n | Raw MAE | Gated MAE | Change | Dishes improved |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in subgroups.itertuples(index=False):
        lines.append(
            f"| {row.group_type} | {row.group} | {row.n} | {row.raw_mae_g:.2f} | "
            f"{row.gated_mae_g:.2f} | {row.gated_change_vs_raw_g:+.2f} | {row.fraction_dishes_improved:.1%} |"
        )
    lines.extend([
        "", "## Gate contribution", "",
        f"Learned gate minus fixed-residual MAE: {comparison['mae_difference_g']:+.2f} g "
        f"(paired 95% CI [{comparison['ci95_low_g']:+.2f}, {comparison['ci95_high_g']:+.2f}] g).",
        "", "The locked final holdout was not read or evaluated.", "",
    ])
    (args.output_dir / "SUBGROUP_RESULTS.md").write_text("\n".join(lines))
    print(learning.to_string(index=False))
    print(subgroups.to_string(index=False))
    print(json.dumps(robustness, indent=2))


if __name__ == "__main__":
    main()
