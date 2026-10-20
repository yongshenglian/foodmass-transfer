"""Export true mass and nutrients (energy, fat, carbohydrate, protein) of the 498 internal-test dishes from the
Nutrition5k metadata, for the clinically interpretable endpoints (experiments/clinical_endpoints/PLAN.md).
No images, no model. Run from code/ on the machine that holds the Nutrition5k dataset (N5K_ROOT)."""
import csv
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("N5K_ROOT", Path.home() / "data/nutrition5k_dataset"))
SAMPLE = Path("../experiments/confirmation_photo_only/confirmation_498_sample.csv")
OUT = Path("../experiments/clinical_endpoints/test_true_nutrients.csv")


def main():
    ids = set(pd.read_csv(SAMPLE).dish_id)
    rows = []
    for cafe in (1, 2):
        with (ROOT / "metadata" / f"dish_metadata_cafe{cafe}.csv").open(newline="") as handle:
            for r in csv.reader(handle):
                if r[0] in ids:  # dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, ...
                    rows.append(dict(dish_id=r[0], true_calories=float(r[1]), true_mass=float(r[2]), true_fat=float(r[3]),
                                     true_carb=float(r[4]), true_protein=float(r[5])))
    out = pd.DataFrame(rows).drop_duplicates("dish_id")
    missing = ids - set(out.dish_id)
    if missing:
        raise SystemExit(f"{len(missing)} test dishes not found in metadata, e.g. {sorted(missing)[:3]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} dishes -> {OUT}")


if __name__ == "__main__":
    main()
