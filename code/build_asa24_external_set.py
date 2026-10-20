"""Build an external test set from the NCI ASA24 portion-size images (labels from FNDDS 2011-2012).

grams = FNDDS FoodWeights(food code, portion code) x ASA24 multiplier; kcal = grams x FNDDS energy (208)/100.
An image is kept only when every FNDDS code linked to it gives nearly the same weight (max/min <= 1.25),
so its label does not depend on which linked food is meant. Writes ../experiments/asa24_external/.
Run from code/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ASA = Path.home() / "data/asa24"
FN = Path.home() / "data/fndds/ascii_2011"
OUT = Path("../experiments/asa24_external")


def read_fndds(name, cols):
    d = pd.read_csv(FN / name, sep="^", header=None, dtype=str, encoding="latin-1", quoting=3)
    d = d.iloc[:, :len(cols)]; d.columns = cols
    return d.apply(lambda c: c.str.strip("~") if c.dtype == object else c)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = pd.ExcelFile(ASA / "ASA24_ImageMetadata.xlsx")
    inv = x.parse("ImageFileInventory"); ac = x.parse("AssociatedCodes")
    fw = read_fndds("FoodWeights.txt", ["food", "sub", "seq", "portion", "start", "end", "grams"])
    fw = fw.assign(food=fw.food.astype(int), portion=fw.portion.astype(int), grams=fw.grams.astype(float))
    fw = fw[fw.grams > 0].groupby(["food", "portion"]).grams.median()
    nut = read_fndds("FNDDSNutVal.txt", ["food", "nutr", "start", "end", "value"])
    nut = nut.assign(food=nut.food.astype(int), value=nut.value.astype(float))
    per100 = nut.pivot_table(index="food", columns="nutr", values="value", aggfunc="first")
    per100 = per100.rename(columns={"208": "kcal", "203": "protein", "204": "fat", "205": "carb"})[["kcal", "protein", "fat", "carb"]]

    ac = ac.dropna(subset=["FoodCode"]).assign(FoodCode=lambda d: d.FoodCode.astype(int), PortionCode=lambda d: d.PortionCode.astype(int))
    ac["unit_g"] = [fw.get((f, p), np.nan) for f, p in zip(ac.FoodCode, ac.PortionCode)]
    ac["grams"] = ac.unit_g * ac.Multiplier
    ac = ac.dropna(subset=["grams"])
    ac = ac[ac.grams > 0]
    for k in per100.columns:
        ac[k] = ac.grams * ac.FoodCode.map(per100[k]) / 100
    g = ac.groupby("Image")
    lab = pd.DataFrame({"grams": g.grams.median(), "grams_min": g.grams.min(), "grams_max": g.grams.max(),
                        "n_codes": g.FoodCode.nunique(), "food_code": g.FoodCode.first(),
                        "fndds_description": g.FC_Description.first(), "category": g["Category Description"].first(),
                        "image_type": g.ImageType.first(),
                        **{k: g[k].median() for k in per100.columns}})
    lab = lab[(lab.grams_max / lab.grams_min) <= 1.25]
    inv = inv.set_index("FileName")
    lab = lab.join(inv[["ImgDescription", "ImgTitle", "Display", "Portion", "ImgWidth", "ImgHeight"]], how="inner")
    lab["image"] = [str(ASA / "Asa24PortionImages" / f) for f in lab.index]
    lab = lab[[Path(p).is_file() for p in lab.image]]
    lab = lab[(lab.grams >= 5) & (lab.grams <= 1500)]
    lab["series"] = lab.ImgDescription.fillna(lab.fndds_description) + " | " + lab.Display.fillna("")
    lab.index.name = "image_file"
    lab.to_csv(OUT / "asa24_labeled_all.csv")

    # evaluation sample: overhead ('Aerial') photos, whole series kept together, at most 1,000 images,
    # series chosen by a fixed hash so the choice does not depend on any model output
    aerial = lab[lab.Display == "Aerial"].copy()
    series = sorted(aerial.series.unique(), key=lambda s: pd.util.hash_pandas_object(pd.Series([s + "20261002"])).iloc[0])
    keep, n = [], 0
    for s in series:
        m = aerial.series == s
        if n + m.sum() > 1000:
            continue
        keep.append(s); n += m.sum()
    sample = aerial[aerial.series.isin(keep)].reset_index()
    sample = sample.rename(columns={"image_file": "dish_id"})
    sample["true_weight"] = sample.grams
    sample["ingredients"] = sample.ImgDescription.fillna(sample.fndds_description)
    sample.to_csv(OUT / "asa24_aerial_sample.csv", index=False)
    print(f"labeled images {len(lab)} (aerial {len(aerial)}); sample {len(sample)} images in {len(keep)} portion series")
    print(sample.true_weight.describe().round(1).to_dict())
    print(sample.category.value_counts().head(12).to_dict())


if __name__ == "__main__":
    main()
