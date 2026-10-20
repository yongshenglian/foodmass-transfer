"""Side-view sample for ECUSTFD: every side-view (S) photograph of the 146 objects in the top-view test sample, with the
object's weighed mass. Run from code/ on the machine holding ECUSTFD (ECUSTFD_ROOT, folder with JPEGImages/)."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ECUSTFD_ROOT", Path.home() / "data/ecustfd"))
TOP = Path("../experiments/external_ecustfd/ecustfd_top_sample.csv")
OUT = Path("../experiments/ecustfd_side_view/ecustfd_side_sample.csv")


def main():
    top = pd.read_csv(TOP).drop_duplicates("id").set_index("id")
    rows = []
    for obj, r in top.iterrows():
        # ECUSTFD uses lowercase s for egg002's side-view filenames.
        for path in sorted((ROOT / "JPEGImages").glob(f"{obj}[Ss](*).JPG")):
            rows.append(dict(dish_id=path.name, id=obj, view="S", image=str(path), type=r.type, grams=r.grams,
                             true_weight=r.true_weight, ingredients=r.ingredients, series=r.series))
    out = pd.DataFrame(rows)
    missing = sorted(set(top.index) - set(out.id))
    print(f"{len(out)} side-view photographs of {out.id.nunique()} objects; objects without a side view: {len(missing)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
