"""Build a measured Nutrition5k reference bank and lightweight visual features.

The bank contains only IDs supplied by the versioned reference-training split.
No final-holdout prediction or evaluation is performed. Run from code/.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATASET_ROOT = Path(os.environ.get("N5K_ROOT", "/mnt/sda/data/data/Nutrition5k/nutrition5k_dataset"))


def metadata(root):
    records = {}
    for cafe in (1, 2):
        path = root / "metadata" / f"dish_metadata_cafe{cafe}.csv"
        with path.open(newline="") as handle:
            for row in csv.reader(handle):
                ingredients = []
                for index in range(6, len(row), 7):
                    if index + 2 < len(row) and row[index].startswith("ingr_"):
                        ingredients.append({
                            "id": row[index], "name": row[index + 1],
                            "mass_g": float(row[index + 2]),
                        })
                records[row[0]] = {
                    "total_mass_g": float(row[2]), "ingredients": ingredients, "cafe": cafe,
                }
    return records


def visual_features(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32) / 255
        hsv = np.asarray(image.convert("HSV").resize((128, 128)), dtype=np.float32) / 255
    values, names = [], []
    for channel, label in enumerate("rgb"):
        histogram, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0, 1), density=False)
        values.extend((histogram / histogram.sum()).tolist())
        names.extend(f"{label}_hist_{index:02d}" for index in range(16))
    for channel, label, bins in ((0, "h", 16), (1, "s", 8), (2, "v", 8)):
        histogram, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 1), density=False)
        values.extend((histogram / histogram.sum()).tolist())
        names.extend(f"{label}_hist_{index:02d}" for index in range(bins))
    for row in range(4):
        for column in range(4):
            block = rgb[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32]
            values.extend(block.mean(axis=(0, 1)).tolist())
            names.extend(f"grid_{row}_{column}_{label}" for label in "rgb")
    gray = rgb.mean(axis=2)
    spectrum = np.log1p(np.abs(np.fft.rfft2(gray)))[:8, :8].reshape(-1)
    spectrum = spectrum[1:] / max(float(np.linalg.norm(spectrum[1:])), 1e-8)
    values.extend(spectrum.tolist())
    names.extend(f"fft_{index:02d}" for index in range(len(spectrum)))
    gradient = np.hypot(np.diff(gray, axis=0, append=gray[-1:]), np.diff(gray, axis=1, append=gray[:, -1:]))
    stats = [gray.mean(), gray.std(), gradient.mean(), gradient.std(), *np.quantile(gradient, [0.5, 0.75, 0.9, 0.95])]
    values.extend(float(value) for value in stats)
    names.extend(["gray_mean", "gray_std", "edge_mean", "edge_std", "edge_q50", "edge_q75", "edge_q90", "edge_q95"])
    return np.asarray(values, np.float32), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    records = metadata(args.dataset_root)
    rows, feature_ids, matrices, feature_names, excluded = [], [], [], None, []
    for number, dish_id in enumerate(ids, 1):
        image = args.dataset_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
        depth = args.dataset_root / "imagery" / "realsense_overhead" / dish_id / "depth_raw.png"
        if dish_id not in records or not image.is_file():
            excluded.append(dish_id)
            continue
        info = records[dish_id]
        ingredients = info["ingredients"]
        rows.append({
            "dish_id": dish_id, "image": str(image), "depth": str(depth),
            "depth_exists": depth.is_file(), "total_mass_g": info["total_mass_g"],
            "ingredient_count": len(ingredients),
            "visible_ingredient_count": sum(item["mass_g"] >= 5 for item in ingredients),
            "ingredients_json": json.dumps(ingredients, separators=(",", ":")),
        })
        feature, names = visual_features(image)
        feature_names = names if feature_names is None else feature_names
        matrices.append(feature)
        feature_ids.append(dish_id)
        if number % 250 == 0:
            print(f"{number}/{len(ids)}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    np.savez_compressed(
        args.output_features,
        dish_ids=np.asarray(feature_ids), features=np.stack(matrices),
        feature_names=np.asarray(feature_names),
    )
    print(json.dumps({
        "requested": len(ids), "reference_rows": len(rows), "excluded": len(excluded),
        "visual_feature_dimensions": len(feature_names), "all_have_depth": all(row["depth_exists"] for row in rows),
        "output_csv": str(args.output_csv.resolve()),
        "output_features": str(args.output_features.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
