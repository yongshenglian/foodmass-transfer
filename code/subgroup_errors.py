"""Subgroup error tables (review request): dish mass, single vs mixed, number of ingredients and energy density on the
internal Nutrition5k test set; food group and energy density on ASA24; single items vs trays and number of items on
FPB; top vs side view on ECUSTFD (once experiments/ecustfd_side_view/ has predictions). MAE (g) and WAPE (%) per
subgroup for the photograph-only methods. Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_endpoints import DL, DN, SP, SPK, external_sets, load, n5k_sets, qwen

E = Path("../experiments"); OUT = E / "subgroups"; OUT.mkdir(exist_ok=True)

FOOD_GROUPS = {
    "Fruits and vegetables": ["Vegetables, Dry Beans", "Fruits, Berries", "Sweet Potatoes", "Green Salads", "Other Salads"],
    "Meat, fish and eggs": ["Fish, Shellfish", "Beef, Lamb, Veal, Game Meats", "Poultry", "Sausage", "Bacon", "Ham or Pork",
                            "Deli Meats, Packaged Lunch Meats, Meat Spreads, Potted Meat", "Eggs, Egg Substitutes"],
    "Breads, grains and snacks": ["Bread (Not Sweet)", "Cornbread, Corn Muffins, Corn Pone", "Pancakes/Flapjacks", "Crackers",
                                  "Rolls, Buns, Bagels, English Muffins", "Rice", "Ready-to-Eat Cereals", "Cooked Cereals",
                                  "Breadsticks", "Pretzels", "Popcorn", "Chips, Puffs, Twists, Potato Sticks",
                                  "Tortillas, Taco Shells, Gordita Shells, Sope Shells, Chalupa Shells", "Bread or Rice Stuffing, Dressing"],
    "Nuts and seeds": ["Nuts, Seeds, and Mixtures of Nuts and Seeds"],
    "Sweets, desserts and dairy": ["Candy", "Pies, Tarts", "Cakes, Cupcakes, Snack Cakes", "Cookies, Brownies", "Pudding", "Jello, Gelatin",
                                   "Sweet Breads, Coffee Cakes, Sweet Rolls, Pastries, Muffins (not corn), and Croissants",
                                   "Ice Cream, Sherbet, Tofu Desserts, Frozen Dairy Desserts, Frozen Yogurt, Sundaes",
                                   "Syrups, Pancake Syrup", "Yogurt"],
}


def table(t, preds, groups):
    """groups: ordered dict name -> boolean mask."""
    rows = []
    for name, m in groups.items():
        r = {"subgroup": name, "n": int(m.sum())}
        for k, p in preds.items():
            e = np.abs(p[m] - t[m]); r[f"{k} MAE"] = float(e.mean()); r[f"{k} WAPE"] = float(e.sum() / t[m].sum() * 100)
        rows.append(r)
    return pd.DataFrame(rows)


def show(title, df, preds):
    print(f"\n== {title}")
    head = f"{'subgroup':34s} {'n':>5s} " + " ".join(f"{k[:24]:>24s}" for k in preds)
    print(head)
    for _, r in df.iterrows():
        print(f"{r.subgroup:34s} {r.n:5d} " + " ".join(f"{r[k + ' MAE']:9.1f} g ({r[k + ' WAPE']:5.1f}%)  " for k in preds))


def tertiles(x, unit):
    q = np.quantile(x, [1 / 3, 2 / 3]); lo, hi = x.min(), x.max()
    return {f"Low ({lo:.2f}-{q[0]:.2f} {unit})": x <= q[0], f"Middle ({q[0]:.2f}-{q[1]:.2f} {unit})": (x > q[0]) & (x <= q[1]),
            f"High ({q[1]:.2f}-{hi:.2f} {unit})": x > q[1]}


def main():
    res = {}
    # ---- Nutrition5k internal test set
    d = n5k_sets()["test"]; t = d["t"]; preds = d["preds"]
    s = pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv").set_index("dish_id")
    n_ingr = s.ingredients.str.split(r" \| ").str.len().to_numpy()
    groups = {"Mass <100 g": t < 100, "Mass 100-299 g": (t >= 100) & (t < 300), "Mass >=300 g": t >= 300,
              "Single ingredient": n_ingr == 1, "Mixed (>=2 ingredients)": n_ingr >= 2,
              "2-4 ingredients": (n_ingr >= 2) & (n_ingr <= 4), "5-9 ingredients": (n_ingr >= 5) & (n_ingr <= 9), ">=10 ingredients": n_ingr >= 10}
    if d["truth"] is not None:
        groups.update({f"Energy density {k}": v for k, v in tertiles(d["truth"].true_calories.to_numpy(float) / t, "kcal/g").items()})
    df = table(t, preds, groups); show("Nutrition5k internal test set", df, preds); res["nutrition5k_test"] = df.to_dict("records")
    # ---- external sets
    ext = external_sets()
    d = ext["ASA24"]; t = d["t"]; preds = d["preds"]
    a = pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv").set_index("dish_id")
    cat = a.category.to_numpy(); groups = {g: np.isin(cat, cs) for g, cs in FOOD_GROUPS.items()}
    groups["Mixed dishes, soups and other"] = ~np.any(np.stack(list(groups.values())), 0)
    groups.update({f"Energy density {k}": v for k, v in tertiles(a.kcal.to_numpy(float) / t, "kcal/g").items()})
    groups.update({"Mass <100 g": t < 100, "Mass 100-299 g": (t >= 100) & (t < 300), "Mass >=300 g": t >= 300})
    df = table(t, preds, groups); show("ASA24 test set", df, preds); res["asa24_test"] = df.to_dict("records")
    d = ext["FPB"]; t = d["t"]; preds = d["preds"]
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    f = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))]; ni = f.n_items.to_numpy()
    groups = {"Single item": ni == 1, "Tray, 2-3 items": (ni >= 2) & (ni <= 3), "Tray, >=4 items": ni >= 4,
              "Mass <300 g": t < 300, "Mass 300-599 g": (t >= 300) & (t < 600), "Mass >=600 g": t >= 600}
    df = table(t, preds, groups); show("FPB test set (unseen trays)", df, preds); res["fpb_test"] = df.to_dict("records")
    # ---- ECUSTFD top vs side
    d = ext["ECUSTFD"]; V = E / "ecustfd_side_view"
    if (V / "qwen_blind_fp8.csv").is_file() and (V / "pred_n5k_imageonly.csv").is_file() and (V / "pred_joint_large_consensus.csv").is_file():
        side = pd.read_csv(V / "ecustfd_side_sample.csv").set_index("dish_id"); ids = side.index
        sp = {SP: qwen(V / "qwen_blind_fp8.csv", ids), DN: load(V / "pred_n5k_imageonly.csv", "mass", ids),
              DL: load(V / "pred_joint_large_consensus.csv", "mass", ids)}
        top_ids = pd.read_csv(E / "external_ecustfd/ecustfd_all_predictions.csv").dish_id
        top = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").set_index("dish_id").loc[top_ids]
        rows = []; rng = np.random.default_rng(0)
        for k in (SP, DN, DL):
            et = pd.Series(np.abs(d["preds"][k] - d["t"]), index=top.id.values).groupby(level=0).mean()
            es = pd.Series(np.abs(sp[k] - side.true_weight.to_numpy(float)), index=side.id.values).groupby(level=0).mean()
            objs = et.index.intersection(es.index); diff = (es[objs] - et[objs]).to_numpy()
            b = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
            rows.append({"method": k, "top_n": len(top), "side_n": len(side), "objects": len(objs),
                         "top_MAE": float(np.mean(np.abs(d["preds"][k] - d["t"]))), "side_MAE": float(np.mean(np.abs(sp[k] - side.true_weight.to_numpy(float)))),
                         "top_WAPE": float(np.abs(d["preds"][k] - d["t"]).sum() / d["t"].sum() * 100),
                         "side_WAPE": float(np.abs(sp[k] - side.true_weight.to_numpy(float)).sum() / side.true_weight.sum() * 100),
                         "side_minus_top_object_MAE": [float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]})
        res["ecustfd_view"] = rows
        print("\n== ECUSTFD top vs side view")
        for r in rows:
            c = r["side_minus_top_object_MAE"]
            print(f"  {r['method']:32s} top {r['top_MAE']:6.1f} g ({r['top_WAPE']:.1f}%)  side {r['side_MAE']:6.1f} g ({r['side_WAPE']:.1f}%)  "
                  f"side-top per object {c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]  (n top {r['top_n']}, side {r['side_n']}, objects {r['objects']})")
    else:
        print("\n(ECUSTFD side views not yet scored: run experiments/ecustfd_side_view/run_workstation.sh)")
    (OUT / "subgroup_errors.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
