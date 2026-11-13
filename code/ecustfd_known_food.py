"""Per-food calibration with weighed references on ECUSTFD (experiments/ecustfd_known_food/PLAN.md). Existing predictions
only; nothing is trained or tuned. Run from code/:  python ecustfd_known_food.py
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); OUT = E / "ecustfd_known_food"; DATA = Path.home() / "data/ecustfd"
SAMPLES = {"top": E / "external_ecustfd/ecustfd_top_sample.csv", "side": E / "ecustfd_side_view/ecustfd_side_sample.csv"}
PRED = {"top": E / "external_ecustfd", "side": E / "ecustfd_side_view"}
MODELS = {"DINOv2-base, Nutrition5k only": "pred_n5k_imageonly.csv", "DINOv2-large, two-domain": "pred_joint_large_consensus.csv"}
LAM, DRAWS, B, COIN_MM, NEAR = 0.75, 20, 10000, 25.0, np.log(1.25)
RULES = ("uncalibrated", "calibrated", "constant", "pixel_area", "coin_area")


def boxes(files):
    """Food box area in pixels and in mm² (coin scale) and field of view in mm, from the ECUSTFD annotations."""
    rows = []
    for f in files:
        r = ET.parse(DATA / "Annotations" / (Path(f).stem + ".xml")).getroot(); w, h = int(r.find("size/width").text), int(r.find("size/height").text)
        box = {}
        for o in r.findall("object"):
            x0, y0, x1, y1 = (int(o.find("bndbox/" + k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            box[o.find("name").text.strip().lower() == "coin"] = (x1 - x0, y1 - y0)
        mm = COIN_MM / max(box[True]); a = box[False][0] * box[False][1]
        rows.append((a, a * mm ** 2, mm * max(w, h)))
    return pd.DataFrame(rows, index=list(files), columns=["px_area", "mm2_area", "fov_mm"])


def load(view, model):
    s = pd.read_csv(SAMPLES[view]).drop_duplicates("dish_id").set_index("dish_id")
    p = pd.read_csv(PRED[view] / MODELS[model]).drop_duplicates("dish_id").set_index("dish_id").reindex(s.index)["mass"]
    return s.assign(pred=p.to_numpy(float)).join(boxes(s.index))


def draw(ref_set, tgt_set, rng):
    """One draw: per photograph absolute errors of every rule, for every type with at least 2 objects."""
    out = []
    for typ, r in ref_set.groupby("type"):
        objs = r["id"].unique()
        if len(objs) < 2:
            continue
        ref = r.iloc[rng.integers(len(r))]; tgt = tgt_set[(tgt_set["type"] == typ) & (tgt_set["id"] != ref["id"])]
        t = tgt.true_weight.to_numpy(float); w = float(ref.true_weight)
        p = {"uncalibrated": tgt.pred.to_numpy(),
             "calibrated": tgt.pred.to_numpy() / np.exp(LAM * np.log(ref.pred / w)),
             "constant": np.full(len(t), w),
             "pixel_area": w * tgt.px_area.to_numpy() / ref.px_area,
             "coin_area": w * (tgt.mm2_area.to_numpy() / ref.mm2_area) ** 1.5}
        far = np.abs(np.log(tgt.fov_mm.to_numpy() / ref.fov_mm)) > NEAR
        out.append(pd.DataFrame({"type": typ, "far": far, **{k: np.abs(v - t) for k, v in p.items()}}))
    return pd.concat(out)


def summarize(d, rng):
    """MAE per rule (pooled over draws) and calibrated minus each rule, with a bootstrap over food types."""
    g = d.groupby("type")[list(RULES)].agg(["sum", "count"]); types = g.index; idx = rng.integers(0, len(types), (B, len(types)))
    S = {k: g[(k, "sum")].to_numpy() for k in RULES}; N = g[("uncalibrated", "count")].to_numpy()
    r = {"n_photos_per_draw": float(N.sum() / DRAWS), "n_types": int(len(types)), "mae": {k: float(S[k].sum() / N.sum()) for k in RULES}}
    for k in RULES:
        if k != "calibrated":
            bs = (S["calibrated"] - S[k])[idx].sum(1) / N[idx].sum(1)
            r[f"calibrated_minus_{k}"] = [r["mae"]["calibrated"] - r["mae"][k], float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return r


def main():
    OUT.mkdir(exist_ok=True); res = {}
    s = load("top", list(MODELS)[0]); spread = s.drop_duplicates("id").groupby("type").true_weight.agg(["count", "min", "max", "std"])
    res["mass_spread_by_type"] = spread.round(1).reset_index().to_dict("records")
    for model in MODELS:
        v = {view: load(view, model) for view in SAMPLES}; res[model] = {}
        for ref_view, tgt_view in (("top", "top"), ("side", "side"), ("top", "side"), ("side", "top")):
            rng = np.random.default_rng(0); d = pd.concat([draw(v[ref_view], v[tgt_view], rng) for _ in range(DRAWS)])
            key = f"{ref_view} -> {tgt_view}"; res[model][key] = {"all": summarize(d, np.random.default_rng(1))}
            if ref_view == tgt_view:
                for name, m in (("distance changed >25%", d.far), ("distance within 25%", ~d.far)):
                    res[model][key][name] = summarize(d[m.to_numpy()], np.random.default_rng(1))
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(spread.round(0).to_string())
    for model in MODELS:
        print(f"== {model}")
        for key, x in res[model].items():
            for part, r in x.items():
                m = r["mae"]; c = {k: r[f"calibrated_minus_{k}"] for k in RULES if k != "calibrated"}
                print(f"  {key:13s} {part:22s} n {r['n_photos_per_draw']:5.0f}  " + "  ".join(f"{k} {m[k]:5.1f}" for k in RULES))
                print(" " * 40 + "calibrated minus: " + "  ".join(f"{k} {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]" for k, v in c.items()))


if __name__ == "__main__":
    main()
