"""Compare saved ASA24 test predictions from the two DINOv2-base training runs.

Run from code/. No fitting or inference: paired bootstrap of food series.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    folder = Path("../experiments/asa24_external")
    files = {
        "test": folder / "asa24_aerial_sample.csv",
        "nutrition5k_only": folder / "finetuned_stage_b_imageonly_aux10.csv",
        "nutrition5k_plus_asa24": folder / "runs/joint_base_stretch/asa24_test_predictions.csv",
    }
    sample = pd.read_csv(files["test"]).set_index("dish_id")
    assert sample.index.is_unique and sample.series.notna().all()
    truth = sample.true_weight.to_numpy(float)
    errors = {}
    for name in ("nutrition5k_only", "nutrition5k_plus_asa24"):
        predictions = pd.read_csv(files[name]).set_index("dish_id")
        assert predictions.index.is_unique
        mass = predictions.reindex(sample.index).mass.to_numpy(float)
        assert np.isfinite(mass).all() and (mass > 0).all()
        errors[name] = np.abs(mass - truth)
    delta = errors["nutrition5k_plus_asa24"] - errors["nutrition5k_only"]
    groups = pd.DataFrame({"series": sample.series.to_numpy(), "delta": delta}).groupby("series").delta.agg(["sum", "count"])
    draws = np.random.default_rng(0).integers(0, len(groups), (10000, len(groups)))
    bootstrap = groups["sum"].to_numpy()[draws].sum(1) / groups["count"].to_numpy()[draws].sum(1)
    result = {
        "comparison": "DINOv2-base trained on Nutrition5k plus strict ASA24 versus Nutrition5k alone",
        "n_photographs": len(sample),
        "n_food_series": len(groups),
        "mae_g": {name: float(error.mean()) for name, error in errors.items()},
        "added_asa24_minus_nutrition5k_only_g": {
            "difference": float(delta.mean()),
            "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        },
        "bootstrap": {"resamples": 10000, "seed": 0, "unit": "food series", "method": "paired percentile; fixed predictions; photograph-weighted MAE"},
        "inputs": {name: {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()},
    }
    output = Path("../experiments/asa24_base_training/results.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
