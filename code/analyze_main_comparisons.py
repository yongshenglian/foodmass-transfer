"""Paired group bootstrap for the revised main tables, using saved predictions only.

Run from code/. No inference, fitting, model selection or changes to predictions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dinov2_dataset_matrix import E, DATASETS, prediction_file, sample
from evaluate_fpb import qwen as parse_fpb_qwen

OUT = E / "main_comparisons"
GROUPS = {"asa24": "series", "ecustfd_top": "id", "ecustfd_side": "id", "fpb": "portion", "nvr": "dish_group"}
QWEN = {
    "asa24": "asa24_external/qwen_blind_fp8_lenient.csv",
    "ecustfd_top": "external_ecustfd/qwen_blind_fp8.csv",
    "ecustfd_side": "ecustfd_side_view/qwen_blind_fp8.csv",
    "fpb": "external_fpb/qwen_blind_fp8.csv",
    "nvr": "nutritionverse_real/qwen_blind_fp8.csv",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(dataset, first, second):
    frame = sample(dataset)
    truth = frame.true_weight.to_numpy(float)
    files = {"sample": E / DATASETS[dataset][1]}
    if dataset == "fpb":
        files["training_tray_exclusion"] = E / "external_fpb/fpb_train.csv"

    def predictions(model):
        path = E / QWEN[dataset] if model == "qwen" else prediction_file(model, dataset)
        files[model] = path
        if model == "qwen" and dataset == "fpb":
            pool = E / "external_fpb/fpb_train_1024.csv"
            files["fpb_fallback_pool"] = pool
            files["fpb_response_parser"] = Path("evaluate_fpb.py")
            fallback = float(pd.read_csv(pool).true_weight.median())
            value = np.clip(parse_fpb_qwen("blind", frame, fallback), 1, None)
            assert np.isfinite(value).all() and (value > 0).all()
            return value
        table = pd.read_csv(path).set_index("dish_id")
        assert table.index.is_unique
        column = "predicted_weight" if model == "qwen" else "mass"
        value = table.reindex(frame.index)[column].to_numpy(float).copy()
        if model == "qwen":
            value[~(value > 0)] = 186.0
            value = np.clip(value, 1, None)
        assert np.isfinite(value).all() and (value > 0).all()
        return value

    errors = {name: np.abs(predictions(name) - truth) for name in [first, second]}
    if dataset == "fpb" and "qwen" in errors:
        original_file = E / "external_fpb/fpb_results.json"
        original_mae = json.loads(original_file.read_text())["portion_disjoint"]["Raw Qwen, blind"]["mae_g"]
        assert np.isclose(errors["qwen"].mean(), original_mae), "FPB parsing differs from the reported evaluation"
        files["original_fpb_result"] = original_file
    delta = errors[first] - errors[second]
    _, inverse = np.unique(frame[GROUPS[dataset]].astype(str), return_inverse=True)
    totals = np.bincount(inverse, delta)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(0).integers(0, len(counts), (10000, len(counts)))
    boot = totals[draws].sum(1) / counts[draws].sum(1)
    return {
        "first": first, "second": second, "dataset": dataset,
        "n_photographs": len(frame), "n_groups": len(counts), "group_column": GROUPS[dataset],
        "mae_g": {k: float(v.mean()) for k, v in errors.items()},
        "difference_g": float(delta.mean()), "ci95_g": np.percentile(boot, [2.5, 97.5]).tolist(),
        "inputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in files.items()},
    }


def main():
    comparisons = {}
    for dataset in ["asa24", "ecustfd_top", "ecustfd_side", "fpb", "nvr"]:
        comparisons[f"base_vs_qwen_{dataset}"] = comparison(dataset, "base_n5k", "qwen")
    comparisons["added_asa24_base"] = comparison("asa24", "base_n5k_asa24", "base_n5k")
    for dataset in ["asa24", "ecustfd_top", "ecustfd_side", "fpb", "nvr"]:
        comparisons[f"three_vs_two_base_{dataset}"] = comparison(dataset, "base_n5k_asa24_fpb", "base_n5k_asa24")
    result = {
        "scope": "Retrospective paired uncertainty summaries requested for main-text reporting; saved predictions only.",
        "bootstrap": {"resamples": 10000, "seed_per_comparison": 0, "method": "paired percentile; entire groups with replacement; photograph-weighted MAE; fixed predictions"},
        "comparisons": comparisons,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for key, row in comparisons.items():
        print(key, f"{row['difference_g']:+.1f}", [round(v, 1) for v in row["ci95_g"]])


if __name__ == "__main__":
    main()
