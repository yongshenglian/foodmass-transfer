"""ECUSTFD framing diagnostic and field-of-view matching (experiments/ecustfd_scale/PLAN.md). Post hoc, exploratory.
  python ecustfd_scale.py build     # write FOV-matched images and sample CSVs
  python ecustfd_scale.py analyze   # diagnostic slopes; matched-image results once scored
Run from code/."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

E = Path("../experiments"); OUT = E / "ecustfd_scale"; DATA = Path.home() / "data/ecustfd"
SAMPLES = {"top": E / "external_ecustfd/ecustfd_top_sample.csv", "side": E / "ecustfd_side_view/ecustfd_side_sample.csv"}
PRED = {"top": E / "external_ecustfd", "side": E / "ecustfd_side_view"}
MODELS = {"DINOv2-base, Nutrition5k only": "pred_n5k_imageonly.csv", "DINOv2-large, multi-domain": "pred_joint_large_consensus.csv",
          "Qwen, single prompt": "qwen_blind_fp8.csv"}
MATCHED = {"DINOv2-base, Nutrition5k only": "n5k", "DINOv2-large, multi-domain": "multi"}
COIN_MM, CANVAS, SCALES, B = 25.0, (640, 480), {"0.88": 406.9 / 462.0, "0.77": 0.772}, 10000


def geometry(sample):
    rows = []
    for f in sample.dish_id:
        r = ET.parse(DATA / "Annotations" / (Path(f).stem + ".xml")).getroot()
        w, h = int(r.find("size/width").text), int(r.find("size/height").text)
        c = [o for o in r.findall("object") if o.find("name").text.strip().lower() == "coin"][0]
        x0, y0, x1, y1 = (int(c.find("bndbox/" + k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        mm = COIN_MM / max(x1 - x0, y1 - y0); rows.append((mm, mm * max(w, h)))
    return pd.DataFrame(rows, index=sample.dish_id, columns=["mm_per_px", "fov_mm"])


def build():
    for view, path in SAMPLES.items():
        s = pd.read_csv(path); g = geometry(s)
        for tag, target in SCALES.items():
            d = DATA / f"fov_matched_{tag}" ; d.mkdir(exist_ok=True); out = []
            for f, src, mm in zip(s.dish_id, s.image, g.mm_per_px):
                im = Image.open(src).convert("RGB"); k = mm / target
                im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BICUBIC); a = np.asarray(im)
                border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3), a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
                canvas = Image.new("RGB", CANVAS, tuple(int(v) for v in np.median(border, 0)))
                canvas.paste(im, ((CANVAS[0] - im.width) // 2, (CANVAS[1] - im.height) // 2))
                p = d / (Path(f).stem + ".png"); canvas.save(p); out.append(str(p))
            pd.DataFrame({"dish_id": s.dish_id, "image": out}).to_csv(OUT / f"sample_{view}_{tag}.csv", index=False)
        print(view, len(s), "photographs; FOV mm (5/50/95%):", np.round(g.fov_mm.quantile([.05, .5, .95]).to_numpy()))


def within_slope(y, x, obj, rng):
    """Slope of y on x after removing object means; bootstrap over objects."""
    d = pd.DataFrame({"y": y, "x": x, "o": obj}); d = d[np.isfinite(d.y) & np.isfinite(d.x)]
    d["y"] -= d.groupby("o").y.transform("mean"); d["x"] -= d.groupby("o").x.transform("mean")
    g = d.assign(xy=d.x * d.y, xx=d.x * d.x).groupby("o")[["xy", "xx"]].sum().to_numpy()
    idx = rng.integers(0, len(g), (B, len(g))); bs = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return [float(g[:, 0].sum() / g[:, 1].sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def held_out_rescale(x, t, fold):
    """Absolute errors after dividing by the geometric-mean ratio estimated on the other object folds."""
    out = np.empty(len(t))
    for k in np.unique(fold):
        te = fold == k; out[te] = x[te] / np.exp(np.mean(np.log(x[~te] / t[~te])))
    return np.abs(out - t)


def stats(p, t):
    return {"mae": float(np.mean(np.abs(p - t))), "scale_ratio": float(np.exp(np.mean(np.log(p / t)))),
            "spearman": float(pd.Series(p).rank().corr(pd.Series(t).rank()))}


def analyze():
    res = {}; rng = np.random.default_rng(0)
    for view, path in SAMPLES.items():
        s = pd.read_csv(path).set_index("dish_id"); g = geometry(s.reset_index()); t = s.true_weight.to_numpy(float); obj = s["id"].to_numpy()
        r = {"n": len(s), "objects": int(len(np.unique(obj))), "fov_mm_5_50_95": [float(v) for v in g.fov_mm.quantile([.05, .5, .95])], "models": {}}
        cl = pd.factorize(obj)[0]; idx = rng.integers(0, cl.max() + 1, (B, cl.max() + 1)); counts = np.bincount(cl)
        for name, f in MODELS.items():
            d = pd.read_csv(PRED[view] / f).drop_duplicates("dish_id").set_index("dish_id").reindex(s.index)
            p = d["mass" if "mass" in d else "predicted_weight"].to_numpy(float).copy(); p[~(p > 0)] = 186.0
            m = {"original": stats(p, t), "within_object_slope_on_log_fov": within_slope(np.log(p / t), np.log(g.fov_mm.to_numpy()), obj, rng)}
            for tag in SCALES:
                f2 = OUT / f"pred_{MATCHED.get(name, 'none')}_{view}_{tag}.csv"
                if not f2.is_file():
                    continue
                q = pd.read_csv(f2).set_index("dish_id").reindex(s.index).ensemble.to_numpy(float)
                diff = np.bincount(cl, np.abs(q - t) - np.abs(p - t)); bs = diff[idx].sum(1) / counts[idx].sum(1)
                m[f"fov_matched_{tag}"] = {**stats(q, t), "mae_minus_original": [float(diff.sum() / counts.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                           "within_object_slope_on_log_fov": within_slope(np.log(q / t), np.log(g.fov_mm.to_numpy()), obj, rng)}
                # unplanned follow-up (added after the planned outcomes were seen): one global factor estimated on other objects
                fold = np.random.default_rng(1).permutation(cl.max() + 1)[cl] % 5; a, b = held_out_rescale(p, t, fold), held_out_rescale(q, t, fold)
                d2 = np.bincount(cl, b - a); bs2 = d2[idx].sum(1) / counts[idx].sum(1)
                m[f"fov_matched_{tag}"]["followup_object_held_out_global_factor"] = {
                    "mae_original": float(a.mean()), "mae_matched": float(b.mean()),
                    "matched_minus_original": [float(d2.sum() / counts.sum()), float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))]}
            r["models"][name] = m
        res[view] = r
    (OUT / "ecustfd_scale.json").write_text(json.dumps(res, indent=2))
    for view, r in res.items():
        print(f"== {view}: {r['n']} photographs, {r['objects']} objects, FOV {r['fov_mm_5_50_95']}")
        for name, m in r["models"].items():
            o, sl = m["original"], m["within_object_slope_on_log_fov"]
            print(f"  {name:30s} MAE {o['mae']:5.1f} scale {o['scale_ratio']:.2f} rho {o['spearman']:.2f}  within-object slope {sl[0]:+.2f} [{sl[1]:+.2f}, {sl[2]:+.2f}]")
            for tag in SCALES:
                if f"fov_matched_{tag}" in m:
                    k = m[f"fov_matched_{tag}"]; d, sl = k["mae_minus_original"], k["within_object_slope_on_log_fov"]
                    print(f"      matched {tag} mm/px: MAE {k['mae']:5.1f} ({d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]) scale {k['scale_ratio']:.2f} rho {k['spearman']:.2f}  slope {sl[0]:+.2f} [{sl[1]:+.2f}, {sl[2]:+.2f}]")
                    h = k["followup_object_held_out_global_factor"]; e = h["matched_minus_original"]
                    print(f"        follow-up, global factor from other objects: MAE {h['mae_original']:.1f} -> {h['mae_matched']:.1f} ({e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}])")


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
