"""Build the NutritionVerse-Real evaluation sample (experiments/nutritionverse_real/PLAN.md). One row per photograph;
dish id parsed from the file name (dish_<id>_IMG_...); true mass and nutrients are the dish totals in
nutritionverse_dish_metadata3.csv. Preprocessing fixed in the plan: EXIF orientation (none present), long side reduced
to at most 1,024 pixels, saved as PNG. A second copy turns portrait photographs 90 degrees to landscape for the
pre-declared sensitivity analysis (PLAN.md, amendment 1). No model is run. Run from code/."""
import argparse
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(os.environ.get("NVR_ROOT", Path.home() / "data/nutritionverse-real"))
OUT = Path("../experiments/nutritionverse_real")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=ROOT); a = p.parse_args()
    meta = pd.read_csv(a.root / "nutritionverse_dish_metadata3.csv")
    meta["dish"] = meta.dish_id.astype(int)
    items = [c for c in meta.columns if c.startswith("food_item_type_")]
    meta["ingredients"] = meta[items].apply(lambda r: " | ".join(str(v) for v in r if isinstance(v, str) and v.strip()), axis=1)
    meta["n_items"] = meta[items].apply(lambda r: sum(isinstance(v, str) and bool(v.strip()) for v in r), axis=1)
    meta = meta.set_index("dish")
    img_dir = a.root / "nutritionverse-manual/nutritionverse-manual/images"
    splits = pd.read_csv(img_dir.parent / "updated-manual-dataset-splits.csv").set_index("file_name").category
    proc, rot = a.root / "processed", a.root / "processed_landscape"; proc.mkdir(exist_ok=True); rot.mkdir(exist_ok=True)
    rows, unmatched = [], []
    for f in sorted(img_dir.glob("*.jpg")):
        m = re.match(r"dish_(\d+)_", f.name)
        if not m or int(m.group(1)) not in meta.index:
            unmatched.append(f.name); continue
        d = int(m.group(1)); r = meta.loc[d]
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB"); w, h = im.size
            im.thumbnail((1024, 1024), Image.BICUBIC)
            im.save(proc / f"{f.stem}.png")
            (im.rotate(90, expand=True) if h > w else im).save(rot / f"{f.stem}.png")
        rows.append(dict(dish_id=f.stem, dish=d, image=str(proc / f"{f.stem}.png"), image_landscape=str(rot / f"{f.stem}.png"),
                         portrait=h > w, true_weight=float(r.total_food_weight), true_calories=float(r.total_calories),
                         true_fat=float(r.total_fats), true_carb=float(r.total_carbohydrates), true_protein=float(r.total_protein),
                         ingredients=r.ingredients, n_items=int(r.n_items), source_split=splits.get(f.name, "")))
    s = pd.DataFrame(rows)
    dropped = int((s.true_weight <= 0).sum()); s = s[s.true_weight > 0]  # inclusion rule: positive measured mass
    OUT.mkdir(parents=True, exist_ok=True); s.to_csv(OUT / "nvr_sample.csv", index=False)
    s.assign(image=s.image_landscape).to_csv(OUT / "nvr_sample_landscape.csv", index=False)
    print(f"{len(s)} photographs of {s.dish.nunique()} dishes; {int(s.portrait.sum())} portrait; "
          f"{len(unmatched)} images without a metadata dish; {dropped} with non-positive mass excluded")
    if unmatched:
        print("unmatched examples:", unmatched[:5])


if __name__ == "__main__":
    main()
