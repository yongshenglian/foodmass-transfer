"""Create deterministic, leakage-aware splits for the Qwen3-VL journal study.

The current 502 official-depth-test dishes remain a legacy development cohort.
The final holdout is selected from official depth-train dishes that do not
appear in any existing repository-level ``eval_nutrition5k*.csv`` artifact.
Only dish IDs and split metadata are written; final-holdout outcomes are not
computed or displayed by this script.

Run from ``code/`` to use the repository-relative defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ROOT = Path("/mnt/sda/data/data/Nutrition5k/nutrition5k_dataset")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_metadata(dataset_root: Path) -> dict[str, dict[str, float | int]]:
    records: dict[str, dict[str, float | int]] = {}
    for cafe in (1, 2):
        path = dataset_root / "metadata" / f"dish_metadata_cafe{cafe}.csv"
        with path.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 7:
                    continue
                visible_count = 0
                for index in range(6, len(row), 7):
                    if index + 2 >= len(row) or not row[index].startswith("ingr_"):
                        continue
                    if float(row[index + 2]) >= 5.0:
                        visible_count += 1
                records[row[0]] = {
                    "total_mass_g": float(row[2]),
                    "visible_ingredient_count": max(1, visible_count),
                    "cafe": cafe,
                }
    return records


def mass_bin(value: float) -> str:
    if value < 100:
        return "m00_lt100"
    if value < 200:
        return "m01_100_199"
    if value < 300:
        return "m02_200_299"
    if value < 450:
        return "m03_300_449"
    return "m04_ge450"


def complexity_bin(value: int) -> str:
    if value <= 1:
        return "c01_single"
    if value == 2:
        return "c02_two"
    if value == 3:
        return "c03_three"
    return "c04_four_plus"


def prior_experiment_ids(repo_root: Path) -> tuple[set[str], dict[str, list[str]], list[str]]:
    all_ids: set[str] = set()
    sources: dict[str, list[str]] = defaultdict(list)
    warnings: list[str] = []
    for path in sorted(repo_root.glob("eval_nutrition5k*.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda name: name == "dish_id")
        except Exception as error:  # Preserve the audit trail instead of silently skipping.
            warnings.append(f"{path.name}: {error}")
            continue
        if "dish_id" not in frame:
            continue
        ids = set(frame["dish_id"].dropna().astype(str).str.strip())
        for dish_id in ids:
            sources[dish_id].append(path.name)
        all_ids.update(ids)
    return all_ids, sources, warnings


def allocate_proportionally(groups: dict[str, list[str]], target: int) -> dict[str, int]:
    population = sum(len(values) for values in groups.values())
    if target > population:
        raise ValueError(f"Requested {target} holdout dishes from only {population} eligible dishes")
    exact = {name: target * len(values) / population for name, values in groups.items()}
    allocation = {name: min(len(groups[name]), int(value)) for name, value in exact.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda name: (exact[name] - int(exact[name]), len(groups[name]), name),
        reverse=True,
    )
    for name in order:
        if remaining == 0:
            break
        if allocation[name] < len(groups[name]):
            allocation[name] += 1
            remaining -= 1
    if remaining:
        for name in sorted(groups):
            while remaining and allocation[name] < len(groups[name]):
                allocation[name] += 1
                remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate the requested holdout across strata")
    return allocation


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{dish_id}\n" for dish_id in ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("../eval_nutrition5k_openknead_reconstructed_502_sample.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("../splits/qwen3vl_journal_v1"))
    parser.add_argument("--final-size", type=int, default=502)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    repo_root = Path("..").resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty split directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = args.dataset_root / "dish_ids" / "splits"
    depth_train = read_ids(split_dir / "depth_train_ids.txt")
    depth_test = set(read_ids(split_dir / "depth_test_ids.txt"))
    development = pd.read_csv(args.development)
    development_ids = development["dish_id"].dropna().astype(str).str.strip().tolist()
    if len(development_ids) != len(set(development_ids)):
        raise RuntimeError("Development cohort contains duplicate dish IDs")
    if not set(development_ids).issubset(depth_test):
        raise RuntimeError("Development cohort is not a subset of the official depth-test split")

    metadata = load_metadata(args.dataset_root)
    prior_ids, prior_sources, warnings = prior_experiment_ids(repo_root)
    eligible: list[str] = []
    exclusions = defaultdict(int)
    for dish_id in depth_train:
        rgb = args.dataset_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
        depth = args.dataset_root / "imagery" / "realsense_overhead" / dish_id / "depth_raw.png"
        if dish_id not in metadata:
            exclusions["missing_metadata"] += 1
        elif not rgb.is_file():
            exclusions["missing_rgb"] += 1
        elif not depth.is_file():
            exclusions["missing_depth"] += 1
        elif dish_id in prior_ids:
            exclusions["present_in_prior_experiment"] += 1
        else:
            eligible.append(dish_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for dish_id in eligible:
        row = metadata[dish_id]
        stratum = f"{mass_bin(float(row['total_mass_g']))}__{complexity_bin(int(row['visible_ingredient_count']))}"
        groups[stratum].append(dish_id)
    allocation = allocate_proportionally(groups, args.final_size)
    rng = random.Random(args.seed)
    final_ids: list[str] = []
    stratum_rows = []
    for stratum in sorted(groups):
        values = sorted(groups[stratum])
        rng.shuffle(values)
        selected = values[: allocation[stratum]]
        final_ids.extend(selected)
        stratum_rows.append(
            {
                "stratum": stratum,
                "eligible_count": len(values),
                "final_holdout_count": len(selected),
            }
        )
    final_ids = sorted(final_ids)
    if len(final_ids) != args.final_size:
        raise RuntimeError(f"Expected {args.final_size} final IDs, selected {len(final_ids)}")

    final_set = set(final_ids)
    reference_ids = sorted(dish_id for dish_id in depth_train if dish_id not in final_set)
    if final_set & set(reference_ids):
        raise RuntimeError("Final holdout overlaps the reference-training pool")
    if final_set & prior_ids:
        raise RuntimeError("Final holdout contains previously evaluated dishes")
    if set(development_ids) & set(reference_ids):
        raise RuntimeError("Development and reference-training roles unexpectedly overlap")

    write_ids(output_dir / "development_legacy_502_ids.txt", sorted(development_ids))
    write_ids(output_dir / "final_holdout_502_ids.txt", final_ids)
    write_ids(output_dir / "reference_train_ids.txt", reference_ids)
    pd.DataFrame(stratum_rows).to_csv(output_dir / "final_holdout_strata.csv", index=False)

    manifest_rows = []
    for role, ids in (
        ("development_legacy", sorted(development_ids)),
        ("final_holdout_locked", final_ids),
        ("reference_train", reference_ids),
    ):
        manifest_rows.extend({"dish_id": dish_id, "role": role} for dish_id in ids)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "split_manifest.csv", index=False)

    prior_detail = [
        {"dish_id": dish_id, "source_files": " | ".join(sorted(prior_sources[dish_id]))}
        for dish_id in sorted(set(depth_train) & prior_ids)
    ]
    pd.DataFrame(prior_detail).to_csv(output_dir / "prior_depth_train_experiment_ids.csv", index=False)

    summary = {
        "version": "qwen3vl_journal_v1",
        "created_date": "2026-09-12",
        "seed": args.seed,
        "selection": "proportional fixed-bin stratification by measured mass and count of ingredients >=5 g",
        "official_depth_train_count": len(depth_train),
        "development_legacy_count": len(development_ids),
        "eligible_unseen_depth_train_count": len(eligible),
        "final_holdout_count": len(final_ids),
        "reference_train_count": len(reference_ids),
        "depth_train_exclusions": dict(sorted(exclusions.items())),
        "prior_artifact_parse_warnings": warnings,
        "invariants": {
            "final_vs_reference_overlap": len(final_set & set(reference_ids)),
            "final_vs_prior_experiment_overlap": len(final_set & prior_ids),
            "development_vs_reference_overlap": len(set(development_ids) & set(reference_ids)),
        },
    }
    with (output_dir / "split_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    summary["hashes"] = {
        "development_ids_sha256": file_hash(output_dir / "development_legacy_502_ids.txt"),
        "final_holdout_ids_sha256": file_hash(output_dir / "final_holdout_502_ids.txt"),
        "reference_train_ids_sha256": file_hash(output_dir / "reference_train_ids.txt"),
        "split_manifest_sha256": file_hash(output_dir / "split_manifest.csv"),
    }
    with (output_dir / "split_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
