"""Prepare Amendment 1 without rebuilding or replacing the frozen primary cohort."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

D = Path("../experiments/nutritionverse_real")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    s = pd.read_csv(D / "nvr_sample.csv")
    m = json.loads((D / "freeze_manifest.json").read_text())
    assert digest(D / "nvr_sample.csv") == m["sample_sha256"]
    assert not (D / "nvr_sample_landscape.csv").exists(), "sensitivity already prepared"
    output = Path.home() / "data/nutritionverse-real_processed_landscape"
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for i, row in s.iterrows():
        original = Path(row.image)
        assert digest(original) == row.processed_sha256
        if row.source_height > row.source_width:
            dest = output / original.name
            assert not dest.exists(), dest
            with Image.open(original) as im:
                rotated = im.rotate(90, expand=True)
                rotated.save(dest, format="PNG")
                s.loc[i, "width"], s.loc[i, "height"] = rotated.size
            s.loc[i, "image"] = str(dest)
            s.loc[i, "processed_sha256"] = digest(dest)
        hashes[row.dish_id] = s.loc[i, "processed_sha256"]
    s.to_csv(D / "nvr_sample_landscape.csv", index=False)
    aliases = {
        "pred_n5k.csv": "pred_n5k_imageonly.csv",
        "pred_nutrients.csv": "pred_nutrient_head.csv",
        "pred_large.csv": "pred_joint_large_consensus.csv",
        "pred_three.csv": "pred_joint3_base.csv",
        "residual_transfer_delta.csv": "similarity_knn.csv",
    }
    for source, target in aliases.items():
        target = D / target
        assert not target.exists() or digest(target) == digest(D / source)
        target.write_bytes((D / source).read_bytes())
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_source_commit": "1a311fc",
        "primary_sample_sha256": m["sample_sha256"],
        "landscape_sample_sha256": digest(D / "nvr_sample_landscape.csv"),
        "preprocessing": "rotate already processed portrait PNG 90 degrees counter-clockwise; landscape PNG unchanged",
        "n_portrait": int((s.source_height > s.source_width).sum()),
        "n_photographs": len(s), "checkpoint_hashes": m["checkpoints_sha256"],
        "processed_hashes": hashes,
        "primary_outputs_retained": {source: digest(D / source) for source in aliases},
    }
    (D / "sensitivity_freeze_manifest.json").write_text(json.dumps(freeze, indent=2))
    print(f"Prepared {len(s)} photos; {freeze['n_portrait']} portraits rotated. Primary images, sample and outputs unchanged.")


if __name__ == "__main__":
    main()
