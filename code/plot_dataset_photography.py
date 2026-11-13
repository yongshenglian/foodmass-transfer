"""Illustrate photographic conditions with actual dataset images, independent of model errors.

Run from code/. Produces a paper figure and a source-selection record. Images are not cropped.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

E = Path("../experiments")
OUT = Path("../paper/figures")
REP = Path("../paper/reproducibility")


def dimensions(path):
    with Image.open(path) as image:
        return image.size


def main():
    examples = []
    frame = pd.read_csv(E / "learning_curve/frame.csv")
    n5k = frame[frame.domain == "n5k"].sort_values("dish_id").sample(2, random_state=19)
    examples.append(("Nutrition5k", "Training", [(row.image, "Overhead view", row.dish_id) for row in n5k.itertuples()]))

    asa = pd.read_csv(E / "asa24_external/asa24_train_pool.csv")
    rows = []
    for category in ("Vegetables, Dry Beans", "Rice"):
        pool = asa[asa.category == category]
        if category == "Vegetables, Dry Beans":
            pool = pool[pool.ImgDescription.str.startswith("Broccoli,")]
        pool = pool[pool.Display == "Aerial"]
        row = pool.loc[(pool.grams - pool.grams.median()).abs().idxmin()]
        rows.append((row.image, "Overhead portion", row.image_file))
    examples.append(("ASA24", "Training", rows))

    fpb = pd.read_csv(E / "external_fpb/fpb_train_1024.csv")
    pool = fpb[(fpb.classes == "broccoli") & (fpb.n_items == 1)].copy()
    pool["landscape"] = pool.image.map(lambda path: dimensions(path)[0] >= dimensions(path)[1])
    groups = [group for _, group in pool.groupby("portion") if group.landscape.nunique() == 2]
    assert groups, "Need a food-and-weight group with landscape and portrait photographs"
    group = min(groups, key=lambda part: abs(part.true_weight.iloc[0] - pool.true_weight.median()))
    rows = []
    for landscape, caption in ((True, "Landscape"), (False, "Portrait")):
        row = group[group.landscape == landscape].sort_values("dish_id").iloc[0]
        rows.append((row.image, caption, row.dish_id))
    examples.append(("FPB", "Training", rows))

    top = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv")
    side = pd.read_csv(E / "ecustfd_side_view/ecustfd_side_sample.csv")
    candidates = top[(top.type == "banana") & top.id.isin(set(side.id))].drop_duplicates("id")
    obj = candidates.loc[(candidates.true_weight - candidates.true_weight.median()).abs().idxmin(), "id"]
    rows = []
    for pool, caption in ((top, "Overhead"), (side, "Side view")):
        row = pool[pool.id == obj].sort_values("dish_id").iloc[0]
        rows.append((row.image, caption, row.dish_id))
    examples.append(("ECUSTFD", "External test", rows))

    nvr = pd.read_csv(E / "nutritionverse_real/nvr_sample.csv")
    nvr["landscape"] = nvr.width >= nvr.height
    groups = [part for _, part in nvr.groupby("dish_group") if part.landscape.nunique() == 2]
    median = nvr.drop_duplicates("dish_group").true_weight.median()
    group = min(groups, key=lambda part: abs(part.true_weight.iloc[0] - median))
    rows = []
    for landscape, caption in ((True, "Landscape"), (False, "Portrait")):
        row = group[group.landscape == landscape].sort_values("dish_id").iloc[0]
        rows.append((row.image, caption, row.dish_id))
    examples.append(("NutritionVerse-Real", "External test", rows))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 5, figsize=(5.85, 3.0))
    record = {"selection": "Illustrative photographic conditions; no model predictions used. Full image frames and native aspect ratios preserved.", "examples": []}
    for col, (dataset, role, rows) in enumerate(examples):
        for row_number, (path, caption, identifier) in enumerate(rows):
            ax = axes[row_number, col]
            with Image.open(path) as raw:
                rgba = raw.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, "white")
                canvas.alpha_composite(rgba)
                ax.imshow(canvas.convert("RGB"))
                size = raw.size
            ax.set_axis_off()
            ax.set_anchor("S")
            if row_number == 0:
                fig.text((col + 0.5) / 5, 0.985, f"{dataset}\n{role}", ha="center", va="top", fontsize=8)
            ax.text(0.5, -0.05, caption, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
            record["examples"].append({"dataset": dataset, "role": role, "id": identifier, "image": str(path), "caption": caption, "dimensions": list(size), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.07, wspace=0.12, hspace=0.25)
    OUT.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"fig_dataset_photography.{extension}", dpi=200, bbox_inches="tight")
    REP.mkdir(exist_ok=True)
    (REP / "dataset_photo_examples.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Wrote dataset photography figure and 10-image selection record")


if __name__ == "__main__":
    main()
