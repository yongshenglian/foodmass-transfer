"""Audit the frozen Qwen3-VL Nutrition5k baseline and cohort relationships.

Run from ``code/`` so the repository-level defaults resolve as documented in
CLAUDE.md. The script is read-only with respect to source data; all generated
artifacts are written under the requested output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path("/mnt/sda/data/data/Nutrition5k/nutrition5k_dataset")
DEFAULT_COHORTS = {
    "development_legacy_502": "../eval_nutrition5k_openknead_reconstructed_502_sample.csv",
    "calibration_train300": "../eval_nutrition5k_weight_calibration_train300.csv",
    "calibration_test100": "../eval_nutrition5k_weight_calibration_test100.csv",
    "calibration_confirmation100": "../eval_nutrition5k_weight_calibration_confirmation100.csv",
    "calibration_fresh100": "../eval_nutrition5k_weight_calibration_fresh100.csv",
    "calibration_final_holdout100": "../eval_nutrition5k_weight_calibration_final_holdout100.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def normalized_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def metric_payload(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = pd.to_numeric(frame["true_weight"], errors="coerce")
    pred = pd.to_numeric(frame["predicted_weight"], errors="coerce")
    keep = truth.notna() & pred.notna() & np.isfinite(truth) & np.isfinite(pred)
    truth = truth[keep].astype(float)
    pred = pred[keep].astype(float)
    error = pred - truth
    absolute = error.abs()
    return {
        "n": int(len(truth)),
        "mae_g": float(absolute.mean()),
        "median_ae_g": float(absolute.median()),
        "rmse_g": float(math.sqrt(float(np.mean(np.square(error))))),
        "bias_g": float(error.mean()),
        "wape": float(absolute.sum() / truth.sum()),
        "within_25g": float((absolute <= 25).mean()),
        "within_50g": float((absolute <= 50).mean()),
        "within_100g": float((absolute <= 100).mean()),
        "mean_true_weight_g": float(truth.mean()),
        "mean_predicted_weight_g": float(pred.mean()),
    }


def load_cohort(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "dish_id" not in frame:
        raise ValueError(f"Missing dish_id column: {path}")
    frame = frame.copy()
    frame["dish_id"] = frame["dish_id"].astype(str).str.strip()
    return frame


def markdown_report(payload: dict) -> str:
    metric = payload["baseline_metrics"]
    membership = payload["development_membership"]
    quality = payload["baseline_quality"]
    overlap = payload["cohort_overlap_with_development"]
    lines = [
        "# Qwen3-VL Phase-0 Reproducibility and Leakage Audit",
        "",
        f"Generated: {payload['generated_date']}",
        "",
        "## Baseline result",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Evaluated dishes | {metric['n']} |",
        f"| MAE | {metric['mae_g']:.2f} g |",
        f"| Median absolute error | {metric['median_ae_g']:.2f} g |",
        f"| RMSE | {metric['rmse_g']:.2f} g |",
        f"| Signed bias | {metric['bias_g']:+.2f} g |",
        f"| WAPE | {metric['wape']:.2%} |",
        f"| Within 25 g | {metric['within_25g']:.2%} |",
        f"| Within 50 g | {metric['within_50g']:.2%} |",
        f"| Within 100 g | {metric['within_100g']:.2%} |",
        "",
        "The result is computed from row-level predictions, not copied from a prior report.",
        "",
        "## Cohort integrity",
        "",
        f"- Development sample rows: {quality['sample_rows']}",
        f"- Unique development dish IDs: {quality['sample_unique_dish_ids']}",
        f"- Baseline prediction rows: {quality['prediction_rows']}",
        f"- Unique baseline dish IDs: {quality['prediction_unique_dish_ids']}",
        f"- Valid matched predictions: {quality['valid_matched_predictions']}",
        f"- Missing predictions: {quality['missing_predictions']}",
        f"- Duplicate sample IDs: {quality['duplicate_sample_ids']}",
        f"- Duplicate prediction IDs: {quality['duplicate_prediction_ids']}",
        f"- Missing RGB images: {quality['missing_rgb_images']}",
        f"- Missing depth images: {quality['missing_depth_images']}",
        "",
        "## Official split membership",
        "",
        f"- Development dishes in official depth test: {membership['in_depth_test']}/{membership['development_count']}",
        f"- Development dishes in official RGB test: {membership['in_rgb_test']}/{membership['development_count']}",
        f"- Official depth-test IDs absent from development cohort: {membership['depth_test_not_in_development']}",
        f"- Development IDs outside official depth test: {membership['development_not_in_depth_test']}",
        "",
        "The current 502-dish cohort is retained as a development/legacy benchmark because its labels and errors have already influenced method selection.",
        "",
        "## Overlap with existing calibration cohorts",
        "",
        "| Cohort | Rows | Unique IDs | Overlap with development 502 |",
        "|---|---:|---:|---:|",
    ]
    for row in overlap:
        lines.append(
            f"| {row['cohort']} | {row['rows']} | {row['unique_dish_ids']} | {row['overlap_with_development']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen baseline configuration",
            "",
            "- Model path: `/mnt/sdb/models/Qwen3-VL-32B-Instruct`",
            "- Served name: `Qwen/Qwen3-VL-32B-Instruct`",
            "- Prompt implementation: `code/run_nutrition5k_confidence_selective.py`",
            "- Temperature: 0.01",
            "- Maximum output tokens: 650",
            "- Thinking: disabled",
            "- Target ingredients: known Nutrition5k ingredient names",
            "- Dish prediction: sum of the model's per-ingredient gram estimates",
            "",
            "File hashes and the complete machine-readable audit are stored in `phase0_audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        default="../eval_nutrition5k_openknead_reconstructed_502_sample.csv",
    )
    parser.add_argument(
        "--predictions",
        default="../eval_nutrition5k_qwen3vl32b_502_confidence_results.csv",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("../results/qwen3vl_phase0"))
    args = parser.parse_args()

    sample_path = Path(args.sample).resolve()
    prediction_path = Path(args.predictions).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_cohort(sample_path)
    predictions = load_cohort(prediction_path)
    sample_duplicates = int(sample["dish_id"].duplicated().sum())
    prediction_duplicates = int(predictions["dish_id"].duplicated().sum())
    if sample_duplicates or prediction_duplicates:
        raise RuntimeError(
            f"Dish IDs must be unique; sample duplicates={sample_duplicates}, "
            f"prediction duplicates={prediction_duplicates}"
        )

    predictions["format_valid_normalized"] = normalized_bool(predictions["format_valid"])
    merged = sample[["dish_id", "image", "true_weight"]].merge(
        predictions[["dish_id", "predicted_weight", "format_valid_normalized"]],
        on="dish_id",
        how="left",
        validate="one_to_one",
    )
    valid = merged[
        merged["format_valid_normalized"].fillna(False)
        & pd.to_numeric(merged["predicted_weight"], errors="coerce").notna()
    ].copy()

    split_dir = args.dataset_root / "dish_ids" / "splits"
    depth_train = set(read_ids(split_dir / "depth_train_ids.txt"))
    depth_test = set(read_ids(split_dir / "depth_test_ids.txt"))
    rgb_train = set(read_ids(split_dir / "rgb_train_ids.txt"))
    rgb_test = set(read_ids(split_dir / "rgb_test_ids.txt"))
    development_ids = set(sample["dish_id"])

    manifest = sample[["dish_id", "image"]].copy()
    manifest["role"] = "development_legacy"
    manifest["in_depth_train"] = manifest["dish_id"].isin(depth_train)
    manifest["in_depth_test"] = manifest["dish_id"].isin(depth_test)
    manifest["in_rgb_train"] = manifest["dish_id"].isin(rgb_train)
    manifest["in_rgb_test"] = manifest["dish_id"].isin(rgb_test)
    manifest["rgb_exists"] = manifest["image"].map(lambda value: Path(value).is_file())
    manifest["depth_exists"] = manifest["dish_id"].map(
        lambda dish: (args.dataset_root / "imagery" / "realsense_overhead" / dish / "depth_raw.png").is_file()
    )
    manifest.to_csv(output_dir / "development_502_manifest.csv", index=False)

    overlap_rows = []
    for name, relative_path in DEFAULT_COHORTS.items():
        path = Path(relative_path).resolve()
        if not path.is_file():
            continue
        cohort = load_cohort(path)
        ids = set(cohort["dish_id"])
        overlap_rows.append(
            {
                "cohort": name,
                "path": str(path),
                "rows": int(len(cohort)),
                "unique_dish_ids": int(len(ids)),
                "overlap_with_development": int(len(ids & development_ids)),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(output_dir / "cohort_overlap.csv", index=False)

    hashes = {
        "sample_sha256": sha256(sample_path),
        "predictions_sha256": sha256(prediction_path),
        "prompt_script_sha256": sha256(Path("run_nutrition5k_confidence_selective.py").resolve()),
        "depth_train_ids_sha256": sha256(split_dir / "depth_train_ids.txt"),
        "depth_test_ids_sha256": sha256(split_dir / "depth_test_ids.txt"),
    }
    payload = {
        "generated_date": "2026-09-12",
        "baseline_metrics": metric_payload(valid),
        "baseline_quality": {
            "sample_rows": int(len(sample)),
            "sample_unique_dish_ids": int(sample["dish_id"].nunique()),
            "prediction_rows": int(len(predictions)),
            "prediction_unique_dish_ids": int(predictions["dish_id"].nunique()),
            "valid_matched_predictions": int(len(valid)),
            "missing_predictions": int(merged["predicted_weight"].isna().sum()),
            "duplicate_sample_ids": sample_duplicates,
            "duplicate_prediction_ids": prediction_duplicates,
            "missing_rgb_images": int((~manifest["rgb_exists"]).sum()),
            "missing_depth_images": int((~manifest["depth_exists"]).sum()),
        },
        "official_split_sizes": {
            "depth_train": len(depth_train),
            "depth_test": len(depth_test),
            "rgb_train": len(rgb_train),
            "rgb_test": len(rgb_test),
        },
        "development_membership": {
            "development_count": len(development_ids),
            "in_depth_train": len(development_ids & depth_train),
            "in_depth_test": len(development_ids & depth_test),
            "in_rgb_train": len(development_ids & rgb_train),
            "in_rgb_test": len(development_ids & rgb_test),
            "depth_test_not_in_development": len(depth_test - development_ids),
            "development_not_in_depth_test": len(development_ids - depth_test),
        },
        "cohort_overlap_with_development": overlap_rows,
        "file_hashes": hashes,
    }
    with (output_dir / "phase0_audit.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    (output_dir / "PHASE0_AUDIT.md").write_text(markdown_report(payload))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
