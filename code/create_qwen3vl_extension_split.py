"""Select a label-independent 498-dish extension from the locked holdout.

Selection uses only dish IDs and a frozen SHA-256 ranking. No metadata labels
or model outcomes are read. Run from code/.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read_ids(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_ids(path, ids):
    path.write_text("".join(f"{dish_id}\n" for dish_id in ids))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=498)
    parser.add_argument("--seed", default="20260913")
    args = parser.parse_args()

    locked = read_ids(args.locked)
    development = set(read_ids(args.development))
    reference = set(read_ids(args.reference))
    if len(locked) != len(set(locked)):
        raise RuntimeError("Locked holdout contains duplicate IDs")
    if set(locked) & development or set(locked) & reference:
        raise RuntimeError("Locked holdout overlaps development or reference training")
    if not 0 < args.size <= len(locked):
        raise ValueError("Extension size must be within the locked holdout size")

    ranked = sorted(
        locked,
        key=lambda dish_id: hashlib.sha256(f"{args.seed}:{dish_id}".encode()).hexdigest(),
    )
    extension = sorted(ranked[:args.size])
    reserved = sorted(ranked[args.size:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension_path = args.output_dir / "evaluation_extension_498_ids.txt"
    reserved_path = args.output_dir / "reserved_4_ids.txt"
    write_ids(extension_path, extension)
    write_ids(reserved_path, reserved)
    summary = {
        "selection": "SHA-256 ranking of seed:dish_id; no labels used",
        "seed": args.seed,
        "locked_source_count": len(locked),
        "extension_count": len(extension),
        "reserved_count": len(reserved),
        "extension_vs_development_overlap": len(set(extension) & development),
        "extension_vs_reference_overlap": len(set(extension) & reference),
        "extension_ids_sha256": digest(extension_path),
        "reserved_ids_sha256": digest(reserved_path),
    }
    (args.output_dir / "extension_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
