"""Build the NutritionVerse-Real cohort and freeze record before model inference.
Labels determine positive-mass inclusion only; every included photograph is retained.
Run from code/. Processed photographs stay outside the repository.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

E = Path("../experiments")
D = E / "nutritionverse_real"
RUNS = {
    "n5k": E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10",
    "nutrients": E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10_nutr",
    "large": E / "asa24_external/runs/joint_large_consensus",
    "three": E / "asa24_external/runs/joint3_base",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--processed", type=Path, default=Path.home() / "data/nutritionverse-real_processed")
    a = p.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    sample_path = D / "nvr_sample.csv"
    manifest_path = D / "freeze_manifest.json"
    if sample_path.exists() or manifest_path.exists():
        raise SystemExit("sample/freeze record exists: do not rebuild the scored cohort")
    labels = a.root / "nutritionverse_dish_metadata3.csv"
    meta = pd.read_csv(labels)
    assert meta.dish_id.is_unique, "duplicate dish labels"
    meta = meta.set_index("dish_id")
    required = ["total_food_weight", "total_calories", "total_fats", "total_carbohydrates", "total_protein"]
    assert set(required) <= set(meta.columns)
    images = sorted(a.root.glob("nutritionverse-manual/nutritionverse-manual/images/*.jpg"))
    assert images, "no photographs found"
    a.processed.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    nutrient_map = {"calories": "total_calories", "fat": "total_fats", "carb": "total_carbohydrates", "protein": "total_protein"}
    for image in images:
        match = re.match(r"dish_(\d+)_", image.name)
        if match is None:
            raise ValueError(f"cannot identify dish: {image.name}")
        group = int(match[1])
        if group not in meta.index:
            excluded.append({"image": image.name, "reason": "no dish label"})
            continue
        label = meta.loc[group]
        mass = float(label.total_food_weight)
        if not np.isfinite(mass) or mass <= 0:
            excluded.append({"image": image.name, "reason": "nonpositive/missing dish mass"})
            continue
        dest = a.processed / (image.stem + ".png")
        if dest.exists():
            raise FileExistsError(dest)
        with Image.open(image) as original:
            source_size = original.size
            orientation = original.getexif().get(274, None)
            im = ImageOps.exif_transpose(original).convert("RGB")
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            im.save(dest, format="PNG")
            width, height = im.size
        foods = [str(label[k]).strip() for k in meta.columns if k.startswith("food_item_type_") and pd.notna(label[k]) and str(label[k]).strip()]
        rows.append({
            "dish_id": image.stem, "dish_group": group, "image": str(dest.resolve()),
            "source_image": str(image.resolve()), "true_weight": mass, "ingredients": " | ".join(foods),
            "food_item_count": len(foods), "source_width": source_size[0], "source_height": source_size[1],
            "source_exif_orientation": orientation, "width": width, "height": height,
            **{f"true_{k}": float(label[v]) for k, v in nutrient_map.items()},
            "processed_sha256": sha(dest),
        })
    s = pd.DataFrame(rows)
    assert len(s) and s.dish_id.is_unique
    assert (s[["width", "height"]].max(axis=1) <= 1024).all()
    assert s.groupby("dish_group").true_weight.nunique().max() == 1
    u = json.loads((E / "uncertainty/uncertainty.json").read_text())
    interval_factors = {}
    for key, value in u["conformal"].items():
        if "Nutrition5k only (calibrated on Nutrition5k out-of-fold)" in key:
            interval_factors["n5k"] = value["factor"]
        elif "DINOv2-large, multi-domain" in key:
            interval_factors["large"] = value["factor"]
    assert set(interval_factors) == {"n5k", "large"}
    checkpoint_hashes = {}
    for name, run in RUNS.items():
        paths = sorted(run.glob("fold*.pt"))
        assert len(paths) == 5, f"{name}: expected five checkpoints"
        checkpoint_hashes[name] = {str(path): sha(path) for path in paths}
    s.to_csv(sample_path, index=False)
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(D / "PLAN.md"), "label_file_sha256": sha(labels),
        "sample_sha256": sha(sample_path), "n_photographs": len(s), "n_dishes": int(s.dish_group.nunique()),
        "n_source_photographs": len(images), "excluded": excluded,
        "preprocessing": "EXIF transpose; RGB; aspect-preserving LANCZOS thumbnail, longest side <=1024; PNG",
        "checkpoints_sha256": checkpoint_hashes, "interval_factors": interval_factors,
        "similarity_threshold": u["similarity"]["threshold_5th_pct_validation"],
        "qwen_fallback_g": 186.0, "qwen_min_g": 1.0,
        "knn": {"neighbors": 20, "log_residual_weight": 0.75, "clip_g": [10, 1500]},
        "constant_g": 186.0, "bootstrap": {"clusters": "dish_group", "resamples": 10000, "seed": 0},
        "angle_metadata": "not provided; no angle inferred from photograph or results",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"{len(s)} included photographs of {s.dish_group.nunique()} dishes; {len(excluded)} excluded for label criteria.")
    print("Saved sample and freeze record; no model was evaluated.")


if __name__ == "__main__":
    main()
