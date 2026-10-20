"""Expanded ASA24 training pool with consensus labels (training data only; the test sample is unchanged).

An image is kept when >= 60% of its linked FNDDS codes give a weight within x1.25 of their median; label = median.
Test foods are excluded by every linked FNDDS code, by description, and by image. Run from code/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from build_asa24_external_set import read_fndds

A = Path("../experiments/asa24_external")
ASA = Path.home() / "data/asa24"


def main():
    x = pd.ExcelFile(ASA / "ASA24_ImageMetadata.xlsx")
    inv = x.parse("ImageFileInventory").set_index("FileName"); ac = x.parse("AssociatedCodes").dropna(subset=["FoodCode"])
    fw = read_fndds("FoodWeights.txt", ["food", "sub", "seq", "portion", "start", "end", "grams"])
    fw = fw.assign(food=fw.food.astype(int), portion=fw.portion.astype(int), grams=fw.grams.astype(float))
    fw = fw[fw.grams > 0].groupby(["food", "portion"]).grams.median()
    ac = ac.assign(FoodCode=ac.FoodCode.astype(int), PortionCode=ac.PortionCode.astype(int))
    ac["grams"] = [fw.get((f, p), np.nan) for f, p in zip(ac.FoodCode, ac.PortionCode)]
    ac["grams"] = ac.grams * ac.Multiplier
    codes = ac.groupby("Image").FoodCode.apply(set)
    ac = ac.dropna(subset=["grams"]); ac = ac[ac.grams > 0]

    def consensus(g):
        m = g.grams.median(); share = ((g.grams >= m / 1.25) & (g.grams <= m * 1.25)).mean()
        return pd.Series({"grams": m, "share": share, "food_code": g.FoodCode.iloc[0],
                          "fndds_description": g.FC_Description.iloc[0], "category": g["Category Description"].iloc[0]})
    c = ac.groupby("Image").apply(consensus)
    c = c[(c.share >= 0.6) & (c.grams >= 5) & (c.grams <= 1500)]
    c = c.join(inv[["ImgDescription", "Display"]], how="inner")
    test = pd.read_csv(A / "asa24_aerial_sample.csv")
    tc = set().union(*[codes.get(i, set()) for i in test.dish_id]); tn = set(test.ingredients.str.lower().str.strip())
    c["food_group"] = c.ImgDescription.fillna(c.fndds_description).str.lower().str.strip()
    keep = np.array([not (codes.get(i, set()) & tc) for i in c.index]) & ~c.food_group.isin(tn).to_numpy() & ~c.index.isin(test.dish_id)
    c = c[keep]
    out, src = ASA / "rgb_white", ASA / "Asa24PortionImages"
    for n in c.index:
        if not (out / n).exists():
            im = Image.open(src / n).convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            Image.alpha_composite(bg, im).convert("RGB").save(out / n)
    c.index.name = "image_file"; c = c.reset_index()
    c["image"] = [str(out / n) for n in c.image_file]; c["image_pad43"] = c["image"]
    old = pd.read_csv(A / "asa24_train_pool.csv")
    print(f"consensus pool {len(c)} images, {c.food_group.nunique()} foods; old pool {len(old)}; new images {len(set(c.image_file) - set(old.image_file))}")
    c.to_csv(A / "asa24_train_pool_consensus.csv", index=False)


if __name__ == "__main__":
    main()
