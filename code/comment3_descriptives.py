"""Reviewer comment 3, descriptive part (existing predictions only, nothing trained or rescored):
mass distributions of every dataset, labels and predictions at or beyond the 10-1,500 g bounds, accuracy by mass range,
FPB by viewpoint and single item against several items, and a descriptive model of FPB error.
Run from code/:  python comment3_descriptives.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aspect_control_analyze import view
from clinical_endpoints import DL, DN, SP, external_sets, load, n5k_sets, qwen

E = Path("../experiments"); OUT = E / "comment3_descriptives"; OUT.mkdir(exist_ok=True)
LO, HI = 10.0, 1500.0
BINS = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, np.inf)]
D3 = "DINOv2-base, three-domain"


def labels():
    """Every dataset's labels, training and test, as used in the paper."""
    f = E / "external_fpb"; fv = pd.read_csv(f / "fpb_val_eval.csv")
    disjoint = ~fv.portion.isin(set(pd.read_csv(f / "fpb_train.csv").portion))
    ecu = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv")
    return {
        "Nutrition5k training": pd.read_csv(E / "qwen3vl_journal_v1/reference_train2253_qwen3vl_results.csv").drop_duplicates("dish_id").true_weight,
        "Nutrition5k validation": pd.read_csv(E / "qwen3vl_journal_v1/dinov2_gate_v1_full2253/development_predictions.csv").true_weight,
        "Nutrition5k internal test": pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv").true_weight,
        "ASA24 training pool": pd.read_csv(E / "asa24_external/asa24_train_pool_consensus.csv").grams,
        "ASA24 test": pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv").true_weight,
        "FPB training": pd.read_csv(f / "fpb_train_1024.csv").true_weight,
        "FPB test (unseen trays)": fv.true_weight[disjoint],
        "ECUSTFD (photographs)": ecu.true_weight,
        "ECUSTFD (objects)": ecu.drop_duplicates("id").true_weight}


def describe(x):
    x = np.asarray(x, float); q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(x)), "min": float(x.min()), "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4], "max": float(x.max()),
            "below_10": int((x < LO).sum()), "above_1500": int((x > HI).sum()), "at_or_above_1000_pct": float(np.mean(x >= 1000) * 100)}


def test_sets():
    """Truth and predictions per test set, loaded as in the paper's tables."""
    v = n5k_sets()["validation"]; x = external_sets()
    out = {"Nutrition5k validation": {"t": v["t"], "preds": {k: v["preds"][k] for k in (SP, DN, DL)}},
           "ASA24 test": {"t": x["ASA24"]["t"], "preds": x["ASA24"]["preds"]},
           "ECUSTFD top": {"t": x["ECUSTFD"]["t"], "preds": x["ECUSTFD"]["preds"]}}
    fpb = x["FPB"]; f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    ids = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))].index
    fpb["preds"][DL] = load(E / "external_fpb/pred_joint_large_consensus.csv", "mass", ids)
    out["FPB test"] = {"t": fpb["t"], "preds": {k: fpb["preds"][k] for k in (SP, DN, DL, D3)}, "ids": ids, "meta": f.loc[ids]}
    a = E / "asa24_external/runs/joint3_base/asa24_test_predictions.csv"
    out["ASA24 test"]["preds"][D3] = load(a, "mass", pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv").dish_id)
    s = pd.read_csv(E / "ecustfd_side_view/ecustfd_side_sample.csv").set_index("dish_id"); V = E / "ecustfd_side_view"
    out["ECUSTFD side"] = {"t": s.true_weight.to_numpy(float), "preds": {
        SP: qwen(V / "qwen_blind_fp8.csv", s.index), DN: load(V / "pred_n5k_imageonly.csv", "mass", s.index),
        DL: load(V / "pred_joint_large_consensus.csv", "mass", s.index)}}
    return out


def acc(p, t):
    e = np.abs(p - t); r = p / t
    return {"n": int(len(t)), "mae": float(e.mean()) if len(t) else None, "wape": float(e.sum() / t.sum() * 100) if len(t) else None,
            "scale_ratio": float(np.exp(np.mean(np.log(r)))) if len(t) else None}


def fpb_model(p, t, meta, rng, B=2000):
    """Least squares of log(pred/true) on viewpoint, several items and log mass; 95% intervals resample trays.
    Every lab-camera test photograph is a tray, so the reference is a smartphone photograph of one item: the view terms
    compare cameras for trays, and the item term rests on the 95 smartphone photographs of several items."""
    v = np.array([view(i) for i in meta.index]); keep = np.isin(v, ["lab overhead", "lab oblique", "smartphone"])
    X = np.column_stack([np.ones(keep.sum()), v[keep] == "lab overhead", v[keep] == "lab oblique",
                         (meta.n_items.to_numpy()[keep] > 1), np.log(t[keep] / 300)]).astype(float)
    y = np.log(p[keep] / t[keep]); tray = meta.portion.to_numpy()[keep]
    names = ["intercept (smartphone, single item, 300 g)", "lab overhead vs smartphone", "lab oblique vs smartphone",
             "several items vs one", "per doubling of true mass"]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u, inv = np.unique(tray, return_inverse=True); members = [np.flatnonzero(inv == i) for i in range(len(u))]; bs = []
    for _ in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])
        bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
    lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
    conv = np.array([1, 1, 1, 1, np.log(2)])  # report the mass term per doubling; all as multiplicative factors
    return {n: [float(np.exp(b * c)), float(np.exp(l * c)), float(np.exp(h * c))] for n, b, l, h, c in zip(names, beta, lo, hi, conv)} | {
        "n": int(keep.sum()), "cross_tab_n": pd.crosstab(v[keep], meta.n_items.to_numpy()[keep] > 1).to_dict()}


def main():
    res = {}; rng = np.random.default_rng(0)
    L = labels(); res["mass_distribution"] = {k: describe(v) for k, v in L.items()}
    T = test_sets()
    res["bounds"] = {s: {"labels_below_10": int((d["t"] < LO).sum()), "labels_above_1500": int((d["t"] > HI).sum()),
                         **{k: {"at_or_below_10": int((p <= LO + 1e-6).sum()), "at_or_above_1500": int((p >= HI - 1e-6).sum())} for k, p in d["preds"].items()}}
                     for s, d in T.items()}
    res["by_mass_range"] = {s: {f"{lo:.0f}-{hi:.0f} g" if np.isfinite(hi) else f">={lo:.0f} g":
                                {k: acc(p[m], d["t"][m]) for k, p in d["preds"].items()} | {"n": int(m.sum())}
                                for lo, hi in BINS if (m := (d["t"] >= lo) & (d["t"] < hi)).sum() > 0}
                            for s, d in T.items()}
    d = T["FPB test"]; meta = d["meta"]; v = np.array([view(i) for i in meta.index]); one = meta.n_items.to_numpy() == 1
    res["fpb_photo_shapes"] = {"test": pd.Series(v).value_counts().to_dict()}
    res["fpb_by_view"] = {f"{vv} / {'single item' if o else 'several items'}":
                          {"n": int(m.sum()), "median_true_g": float(np.median(d["t"][m]))} | {k: acc(p[m], d["t"][m]) for k, p in d["preds"].items()}
                          for vv in ("lab overhead", "lab oblique", "smartphone", "lab reference", "other") for o in (True, False)
                          if (m := (v == vv) & (one == o)).sum() > 0}
    res["fpb_error_model"] = {k: fpb_model(p, d["t"], meta, rng) for k, p in d["preds"].items()}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    figure(L); report(res)


def figure(L):
    fig, ax = plt.subplots(figsize=(7.6, 3.8)); names = list(L)
    parts = ax.violinplot([np.log10(np.asarray(L[k], float)) for k in names], showmedians=True, widths=0.8, orientation="horizontal")
    for b in parts["bodies"]:
        b.set_facecolor("#2a78d6"); b.set_alpha(0.35)
    for x in (LO, HI):
        ax.axvline(np.log10(x), color="#c0392b", ls="--", lw=0.9)
    ax.set_yticks(range(1, len(names) + 1), [f"{k} (n = {len(L[k]):,})" for k in names], fontsize=7.5)
    ax.set_xticks(np.log10([1, 10, 30, 100, 300, 1000, 3000]), ["1", "10", "30", "100", "300", "1,000", "3,000"])
    ax.set_xlabel("Label mass (g, log scale); dashed lines: output bounds"); ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(OUT / "fig_mass_distributions.pdf"); fig.savefig(OUT / "fig_mass_distributions.png", dpi=160); plt.close(fig)


def report(res):
    print("== label mass (g): n, median [5th-95th], min-max, <10, >1500")
    for k, x in res["mass_distribution"].items():
        print(f"  {k:34s} {x['n']:6d}  {x['median']:6.0f} [{x['p5']:.0f}-{x['p95']:.0f}]  {x['min']:.1f}-{x['max']:.0f}  {x['below_10']:3d} {x['above_1500']:4d}")
    print("== predictions at the bounds (<=10 g, >=1500 g)")
    for s, x in res["bounds"].items():
        print(f"  {s:24s} labels <10 {x['labels_below_10']}, >1500 {x['labels_above_1500']};  " +
              "  ".join(f"{k[:22]}: {v['at_or_below_10']}/{v['at_or_above_1500']}" for k, v in x.items() if isinstance(v, dict)))
    print("== MAE by mass range")
    for s, x in res["by_mass_range"].items():
        for b, y in x.items():
            print(f"  {s:24s} {b:12s} n {y['n']:5d}  " + "  ".join(f"{k[:22]} {v['mae']:6.1f} (x{v['scale_ratio']:.2f})" for k, v in y.items() if isinstance(v, dict)))
    print("== FPB by view")
    for b, y in res["fpb_by_view"].items():
        print(f"  {b:32s} n {y['n']:5d} median {y['median_true_g']:5.0f} g  " + "  ".join(f"{k[:22]} {v['mae']:6.1f} (x{v['scale_ratio']:.2f})" for k, v in y.items() if isinstance(v, dict)))
    print("== FPB: multiplicative effect on predicted/true (95% CI over trays)")
    for k, y in res["fpb_error_model"].items():
        print(f"  {k}: " + "; ".join(f"{n} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for n, v in y.items() if isinstance(v, list)))


if __name__ == "__main__":
    main()
