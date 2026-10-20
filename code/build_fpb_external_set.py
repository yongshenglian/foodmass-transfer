"""Build evaluation/training tables from the Food Portion Benchmark (FPB, Hugging Face issai/Food_Portion_Benchmark).

Labels are YOLO lines `class x y w h grams`; dish mass per photo = sum of item grams. File names start with
`<food>_<size>_RGB_...`; `portion_group` = the name stem before `_jpg.rf.` with the trailing image counter removed is
not available, so leakage across views of one portion is checked separately. Writes ../experiments/external_fpb/.
Run from code/.
"""
from pathlib import Path

import pandas as pd
import yaml

FPB = Path.home() / "data/fpb/FPB_Dataset/RGB"
OUT = Path("../experiments/external_fpb")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load((FPB / "data.yaml").read_text())["names"]
    import sys
    for split in (sys.argv[1:] or ["val", "train"]):
        rows = []
        for lab in sorted((FPB / split / "labels").glob("*.txt")):
            img = FPB / split / "images" / (lab.stem + ".jpg")
            if not img.is_file():
                continue
            items = [l.split() for l in lab.read_text().splitlines() if l.strip()]
            if not items or any(len(i) < 6 for i in items):
                continue
            grams = [float(i[5]) for i in items]; cls = [names[int(i[0])] for i in items]
            stem = lab.stem.split("_jpg.rf.")[0].split("_JPG.rf.")[0]
            prefix = stem.split("_RGB_")[0]; parts = prefix.rsplit("_", 1)
            food, size = (parts[0], parts[1]) if len(parts) == 2 and parts[1] in {"small", "average", "big", "medium", "large"} else (prefix, "")
            rows.append({"dish_id": lab.stem, "image": str(img), "true_weight": sum(grams), "n_items": len(items),
                         "classes": " | ".join(cls), "ingredients": " | ".join(c.replace("_", " ") for c in cls),
                         "food": food, "size": size, "series": f"{food}_{size}"})
        d = pd.DataFrame(rows); d.to_csv(OUT / f"fpb_{split}.csv", index=False)
        print(split, len(d), "images;", d.food.nunique(), "foods; mass median", d.true_weight.median(), "| items per photo", d.n_items.value_counts().head(4).to_dict())


if __name__ == "__main__":
    main()
