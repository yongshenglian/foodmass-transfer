"""Build deterministic Nutrition5k samples for Qwen weight inference.

Run from code/. Every metadata ingredient is retained to match the frozen
502-dish Qwen3-VL baseline prompt.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATASET_ROOT = Path("/mnt/sda/data/data/Nutrition5k/nutrition5k_dataset")


def metadata(root):
    records = {}
    for cafe in (1, 2):
        path = root / "metadata" / f"dish_metadata_cafe{cafe}.csv"
        with path.open(newline="") as handle:
            for row in csv.reader(handle):
                ingredients = []
                for index in range(6, len(row), 7):
                    if index + 2 < len(row) and row[index].startswith("ingr_"):
                        ingredients.append((row[index + 1], float(row[index + 2])))
                records[row[0]] = {
                    "true_weight": float(row[2]),
                    "ingredients": ingredients,
                }
    return records


def mass_bin(value):
    for upper, label in ((100, "lt100"), (200, "100_199"), (300, "200_299"), (450, "300_449")):
        if value < upper:
            return label
    return "ge450"


def complexity_bin(ingredients):
    visible = sum(grams >= 5 for _, grams in ingredients)
    return "single" if visible <= 1 else "two" if visible == 2 else "three" if visible == 3 else "four_plus"


def subset(ids, records, limit, seed):
    if limit <= 0 or limit >= len(ids):
        return sorted(ids)
    groups = defaultdict(list)
    for dish_id in ids:
        info = records[dish_id]
        key = f"{mass_bin(info['true_weight'])}__{complexity_bin(info['ingredients'])}"
        groups[key].append(dish_id)
    exact = {key: limit * len(values) / len(ids) for key, values in groups.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(allocation.values())
    for key in sorted(groups, key=lambda item: (exact[item] - allocation[item], item), reverse=True):
        if not remaining:
            break
        allocation[key] += 1
        remaining -= 1
    rng = random.Random(seed)
    chosen = []
    for key in sorted(groups):
        values = sorted(groups[key])
        rng.shuffle(values)
        chosen.extend(values[:allocation[key]])
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    requested = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if len(requested) != len(set(requested)):
        raise RuntimeError("Input ID file contains duplicates")
    records = metadata(args.dataset_root)
    usable, excluded = [], []
    for dish_id in requested:
        image = args.dataset_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
        if dish_id not in records:
            excluded.append({"dish_id": dish_id, "reason": "missing_metadata"})
        elif not image.is_file():
            excluded.append({"dish_id": dish_id, "reason": "missing_rgb"})
        else:
            usable.append(dish_id)
    selected = subset(usable, records, args.limit, args.seed)
    rows = []
    for dish_id in selected:
        info = records[dish_id]
        names = [name for name, _ in info["ingredients"]]
        rows.append({
            "dish_id": dish_id,
            "image": str(args.dataset_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"),
            "true_weight": info["true_weight"],
            "ingredients": " | ".join(names),
            "ingredient_count": len(names),
            "visible_ingredient_count": sum(grams >= 5 for _, grams in info["ingredients"]),
            "mass_bin": mass_bin(info["true_weight"]),
            "complexity_bin": complexity_bin(info["ingredients"]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if excluded:
        pd.DataFrame(excluded).to_csv(args.output.with_suffix(".excluded.csv"), index=False)
    print(json.dumps({
        "requested_ids": len(requested), "usable_ids": len(usable),
        "selected_ids": len(selected), "excluded_ids": len(excluded),
        "limit": args.limit, "seed": args.seed,
        "ingredient_rule": "all metadata ingredients, including trace ingredients",
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
