"""Food-area and naive known-food baselines on ASA24 (post hoc; experiments/area_baseline/PLAN.md).

(1) Does a trivial visible-food-area cue order the portions of a food as well as the models do?
(2) Does per-food calibration of the model beat model-free uses of the same k labelled photographs?
No model is trained or selected; fixed rules are scored on the test sample and on the strict training pool. Run from code/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analyze_asa24_external import metrics

A = Path("../experiments/asa24_external"); OUT = Path("../experiments/area_baseline"); RAW = Path.home() / "data/asa24/Asa24PortionImages"
LAM, DRAWS, B = 0.75, 20, 10000


def areas(files):
    rows = []
    for f in files:
        a = np.asarray(Image.open(RAW / f).convert("RGBA")).astype(float); rgb, fg = a[..., :3], a[..., 3] > 127
        lum = rgb @ np.array([0.299, 0.587, 0.114]); food = fg & ((np.ptp(rgb, axis=2) > 40) | (lum < 140))
        rows.append((food.mean(), ((255 - lum) / 255)[fg].sum() / fg.size, fg.mean()))
    return pd.DataFrame(rows, index=files, columns=["area", "darkness", "foreground"])


def ordering(score, t, series):
    """Mean within-series Spearman (as analyze_asa24_external.metrics) with a series bootstrap CI, and pooled Spearman."""
    rho = {}
    for s in np.unique(series):
        m = series == s
        if m.sum() >= 3 and np.ptp(t[m]) > 0:
            r = np.corrcoef(pd.Series(score[m]).rank(), pd.Series(t[m]).rank())[0, 1]
            rho[s] = 0.0 if not np.isfinite(r) else r  # a constant score orders nothing
    return rho, float(pd.Series(score).rank().corr(pd.Series(t).rank()))


def known_food(pred, area, t, series, k, rng):
    """One draw: per-series absolute errors of the scored photographs for each rule."""
    out = []
    for s in np.unique(series):
        i = np.flatnonzero(series == s)
        if len(i) <= k:
            continue
        lab = rng.choice(i, k, replace=False); rest = np.setdiff1d(i, lab)
        n1 = np.full(len(rest), np.median(t[lab]))
        with np.errstate(divide="ignore", invalid="ignore"):
            n2 = np.median(t[lab][None] * area[rest][:, None] / area[lab][None], axis=1)
        n2 = np.where(np.isfinite(n2) & (n2 > 0), n2, n1)
        cal = pred[rest] / np.exp(LAM * np.median(np.log(pred[lab] / t[lab])))
        for name, p in (("uncalibrated", pred[rest]), ("model_calibrated", cal), ("labelled_weight", n1), ("area_scaled", n2)):
            out.append((s, name, np.abs(p - t[rest]).sum(), len(rest)))
    return pd.DataFrame(out, columns=["series", "rule", "abs_err", "n"])


def run(name, s, preds, cal_model):
    ids, t, series = s.index, s.true_weight.to_numpy(float), s.series.to_numpy()
    ar = areas(ids); ar.assign(true_weight=t, series=series).to_csv(OUT / f"areas_{name}.csv")
    scores = {**preds, "food area (non-plate pixels)": ar.area.to_numpy(), "darkness sum": ar.darkness.to_numpy(),
              "foreground (alpha) area": ar.foreground.to_numpy()}
    rng = np.random.default_rng(0); res = {"n": len(t), "n_series": int(len(np.unique(series))), "ordering": {}, "known_food": {}}
    per = {}
    for k_, sc in scores.items():
        ok = np.isfinite(sc)
        rho, pooled = ordering(sc[ok], t[ok], series[ok]); per[k_] = pd.Series(rho)
        v = per[k_].to_numpy(); bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        res["ordering"][k_] = {"n_photos": int(ok.sum()), "n_series": len(v), "within_series_spearman": float(v.mean()),
                               "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pooled_spearman": pooled}
    ref = "food area (non-plate pixels)"
    for k_ in preds:  # paired difference in within-series Spearman, model minus food area, over common series
        d = (per[k_] - per[ref]).dropna().to_numpy(); bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        res["ordering"][k_]["minus_food_area"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pred = preds[cal_model]
    rng = np.random.default_rng(0)  # one generator across k = 1, 2, 3, as in the declared known-food run (same draws as Table XII)
    for k in (1, 2, 3):
        d = pd.concat([known_food(pred, ar.area.to_numpy(), t, series, k, rng) for _ in range(DRAWS)])
        g = d.groupby(["series", "rule"])[["abs_err", "n"]].sum().unstack("rule")  # summed over draws, per series
        E, N = g["abs_err"], g["n"]; idx = np.random.default_rng(1).integers(0, len(E), (B, len(E)))
        r = {c: float(E[c].sum() / N[c].sum()) for c in E.columns}; r["n_scored_avg"] = float(N.iloc[:, 0].sum() / DRAWS)
        for c in ("labelled_weight", "area_scaled"):
            num = (E[c] - E["model_calibrated"]).to_numpy()[idx].sum(1); den = N[c].to_numpy()[idx].sum(1); bs = num / den
            r[f"{c}_minus_model_calibrated"] = [r[c] - r["model_calibrated"], float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["known_food"][f"k={k}"] = r
    return res


def main():
    OUT.mkdir(exist_ok=True); col = lambda f, c, ids: pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[c].to_numpy(float)
    s = pd.read_csv(A / "asa24_aerial_sample.csv").set_index("dish_id"); ids = s.index
    test = run("test", s, {
        "Qwen, single prompt": col(A / "qwen_blind_fp8_lenient_imputed.csv", "predicted_weight", ids),
        "DINOv2-base, Nutrition5k only": col(A / "finetuned_stage_b_imageonly_aux10.csv", "mass", ids),
        "DINOv2-large, multi-domain": col(A / "runs/joint_large_consensus/asa24_test_predictions.csv", "mass", ids)},
        "DINOv2-large, multi-domain")
    p = pd.read_csv(A / "asa24_train_pool.csv").rename(columns={"image_file": "dish_id", "grams": "true_weight"}).set_index("dish_id"); ids = p.index
    pool = run("pool", p, {
        "Qwen, single prompt": col(A / "qwen_pool_blind_fp8_imputed.csv", "predicted_weight", ids),
        "DINOv2-base, Nutrition5k only": col(A / "validation/zeroshot_imageonly_stretch.csv", "mass", ids),
        "DINOv2-large, multi-domain (out-of-fold)": col(A / "runs/joint_large_consensus/asa24_pool_oof.csv", "oof", ids)},
        "DINOv2-large, multi-domain (out-of-fold)")
    res = {"test (1,000 photos)": test, "strict training pool (5,099 photos)": pool}
    (OUT / "area_baselines.json").write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        print(f"== {name}: {r['n']} photos, {r['n_series']} series")
        for k_, m in r["ordering"].items():
            d = m.get("minus_food_area"); extra = f"   minus food area {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""
            print(f"  {k_:42s} n={m['n_photos']:5d} within-food rho {m['within_series_spearman']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  pooled {m['pooled_spearman']:.3f}{extra}")
        for k, m in r["known_food"].items():
            a, b = m["labelled_weight_minus_model_calibrated"], m["area_scaled_minus_model_calibrated"]
            print(f"  {k}: uncal {m['uncalibrated']:.1f}  model cal {m['model_calibrated']:.1f}  labelled weight {m['labelled_weight']:.1f} "
                  f"({a[0]:+.1f} [{a[1]:+.1f}, {a[2]:+.1f}])  area-scaled {m['area_scaled']:.1f} ({b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}])  n {m['n_scored_avg']:.0f}")


if __name__ == "__main__":
    main()
