"""Aspect-ratio control (experiments/aspect_control/PLAN.md): pad minus stretch, per model and test set.
Run from code/ after the runs are copied back:  python aspect_control_analyze.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); D = E / "aspect_control"; RUNS = D / "runs"; B = 10000
MODELS = {"n5k": "Nutrition5k-only (base)", "two_large": "Two-domain (large)", "three_base": "Three-domain (base)"}
# where each run writes its predictions for each set
FILES = {"n5k": {"n5k_val": "development_predictions.csv", "asa24": "pred_asa24_test.csv", "fpb": "pred_fpb.csv",
                 "ecustfd_top": "pred_ecustfd_top.csv", "ecustfd_side": "pred_ecustfd_side.csv"},
         "two_large": {"n5k_val": "n5k_development_predictions.csv", "asa24": "asa24_test_predictions.csv", "fpb": "pred_fpb.csv",
                       "ecustfd_top": "pred_ecustfd_top.csv", "ecustfd_side": "pred_ecustfd_side.csv"},
         "three_base": {"n5k_val": "n5k_development_predictions.csv", "asa24": "asa24_test_predictions.csv", "fpb": "fpb_val_predictions.csv",
                        "ecustfd_top": "pred_ecustfd_top.csv", "ecustfd_side": "pred_ecustfd_side.csv"}}
# the paper's predictions, for the reproducibility check
PAPER = {"n5k": {"n5k_val": E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10/development_predictions.csv",
                 "asa24": E / "asa24_external/finetuned_stage_b_imageonly_aux10.csv", "fpb": E / "external_fpb/pred_n5k_imageonly.csv",
                 "ecustfd_top": E / "external_ecustfd/pred_n5k_imageonly.csv", "ecustfd_side": E / "ecustfd_side_view/pred_n5k_imageonly.csv"},
         "two_large": {"n5k_val": E / "asa24_external/runs/joint_large_consensus/n5k_development_predictions.csv",
                       "asa24": E / "asa24_external/runs/joint_large_consensus/asa24_test_predictions.csv",
                       "fpb": E / "external_fpb/pred_joint_large_consensus.csv",
                       "ecustfd_top": E / "external_ecustfd/pred_joint_large_consensus.csv",
                       "ecustfd_side": E / "ecustfd_side_view/pred_joint_large_consensus.csv"},
         "three_base": {"n5k_val": E / "asa24_external/runs/joint3_base/n5k_development_predictions.csv",
                        "asa24": E / "asa24_external/runs/joint3_base/asa24_test_predictions.csv",
                        "fpb": E / "asa24_external/runs/joint3_base/fpb_val_predictions.csv"}}


def view(dish_id):
    """FPB viewpoint from the file name: lab camera frames 01-02 overhead, 05-08 distant oblique; smartphone oblique."""
    if "_Color_png" in dish_id:
        k = dish_id.split("_Color_png")[0][-2:]
        return "lab overhead" if k in ("01", "02") else "lab oblique" if k in ("05", "06", "07", "08") else "lab reference"
    return "smartphone" if ("RGB_IMG" in dish_id or "_IMG_" in dish_id) else "other"


def sets():
    """Truth, bootstrap cluster and subsets for every test set, indexed by dish_id."""
    pool = pd.read_csv(E / "n5k_grouped_cv/pool.csv").set_index("dish_id")
    dev = pd.read_csv(E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10/development_predictions.csv").set_index("dish_id")
    out = {"n5k_val": pd.DataFrame({"true": dev.true, "cluster": pool.group.reindex(dev.index).to_numpy()}, index=dev.index)}
    a = pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv").set_index("dish_id")
    out["asa24"] = pd.DataFrame({"true": a.true_weight, "cluster": a.food_code.astype(str)})
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    disjoint = ~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))
    out["fpb"] = pd.DataFrame({"true": f.true_weight, "cluster": f.portion, "disjoint": disjoint,
                               "view": [view(i) for i in f.index], "single": f.n_items == 1})
    for v, path in (("ecustfd_top", E / "external_ecustfd/ecustfd_top_sample.csv"), ("ecustfd_side", E / "ecustfd_side_view/ecustfd_side_sample.csv")):
        s = pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id")
        out[v] = pd.DataFrame({"true": s.true_weight, "cluster": s["id"]})
    return out


def pred(path, index):
    p = pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(index).mass.to_numpy(float)
    return p if np.isfinite(p).all() else None


def boot(d, cluster, rng):
    """Mean of d and 95% interval, resampling clusters."""
    u, inv = np.unique(cluster.astype(str), return_inverse=True); s = np.bincount(inv, d); n = np.bincount(inv)
    idx = rng.integers(0, len(u), (B, len(u))); m = s[idx].sum(1) / n[idx].sum(1)
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def metrics(p, t):
    r = p / t
    return {"n": int(len(t)), "mae": float(np.abs(p - t).mean()), "scale_ratio": float(np.exp(np.mean(np.log(r)))),
            "within_25pct": float(np.mean(np.abs(r - 1) <= 0.25) * 100)}


def compare(st, pa, s, rng):
    t = s.true.to_numpy(float); d = np.abs(pa - t) - np.abs(st - t)
    return {"stretch": metrics(st, t), "pad": metrics(pa, t), "pad_minus_stretch_mae": boot(d, s.cluster.to_numpy(), rng)}


def main():
    S = sets(); rng = np.random.default_rng(0); res = {"comparison": {}, "fpb_by_view": {}, "reproducibility": {}, "missing": []}
    P = {}
    for m in MODELS:
        for name, s in S.items():
            for rule in ("stretch", "pad"):
                f = RUNS / f"{m}_{rule}" / FILES[m][name]
                P[(m, name, rule)] = pred(f, s.index) if f.is_file() else None
                if P[(m, name, rule)] is None:
                    res["missing"].append(str(f))
    for m in MODELS:
        res["comparison"][m] = {}
        for name, s in S.items():
            st, pa = P[(m, name, "stretch")], P[(m, name, "pad")]
            if st is None or pa is None:
                continue
            if name == "fpb":  # primary: trays not in FPB training; also all photographs
                k = s.disjoint.to_numpy()
                res["comparison"][m]["fpb"] = compare(st[k], pa[k], s[k], rng)
                res["comparison"][m]["fpb_all"] = compare(st, pa, s, rng)
                res["fpb_by_view"][m] = {f"{v} / {'single item' if one else 'several items'}": compare(st[g], pa[g], s[g], rng)
                                         for (v, one), g in ((key, (k & (s.view == key[0]) & (s.single == key[1])).to_numpy())
                                                             for key in sorted(set(zip(s.view, s.single)))) if g.sum() >= 20}
            else:
                res["comparison"][m][name] = compare(st, pa, s, rng)
    # share of the transfer gap: Nutrition5k-only minus the model trained on that dataset, both stretch
    c = res["comparison"]
    for name, home in (("asa24", "two_large"), ("fpb", "three_base")):
        if name in c.get("n5k", {}) and name in c.get(home, {}):
            gap = c["n5k"][name]["stretch"]["mae"] - c[home][name]["stretch"]["mae"]
            gain = -c["n5k"][name]["pad_minus_stretch_mae"][0]
            c["n5k"][name]["transfer_gap_g"] = gap; c["n5k"][name]["share_of_gap_closed_by_pad"] = gain / gap
    for m, files in PAPER.items():  # rerun stretch models against the paper's predictions
        for name, f in files.items():
            s = S[name]; k = s.disjoint.to_numpy() if name == "fpb" else np.ones(len(s), bool)
            st, old = P[(m, name, "stretch")], pred(f, s.index)
            if st is None or old is None:
                continue
            t = s.true.to_numpy(float)[k]
            res["reproducibility"].setdefault(m, {})[name] = {"paper_mae": float(np.abs(old[k] - t).mean()), "rerun_mae": float(np.abs(st[k] - t).mean()),
                                                              "median_abs_log_ratio": float(np.median(np.abs(np.log(st[k] / old[k]))))}
    (D / "results.json").write_text(json.dumps(res, indent=2))
    report(res)


def report(res):
    for m, x in res["comparison"].items():
        print(f"== {MODELS[m]}")
        for name, r in x.items():
            d = r["pad_minus_stretch_mae"]
            print(f"  {name:13s} n {r['stretch']['n']:5d}  MAE stretch {r['stretch']['mae']:6.1f}  pad {r['pad']['mae']:6.1f}  "
                  f"pad - stretch {d[0]:+6.1f} [{d[1]:+.1f}, {d[2]:+.1f}]  scale {r['stretch']['scale_ratio']:.2f} -> {r['pad']['scale_ratio']:.2f}"
                  + (f"  gap {r['transfer_gap_g']:.1f} g, closed {100 * r['share_of_gap_closed_by_pad']:.0f}%" if "transfer_gap_g" in r else ""))
    for m, x in res["fpb_by_view"].items():
        print(f"== FPB by view, trays not in training: {MODELS[m]}")
        for k, r in x.items():
            d = r["pad_minus_stretch_mae"]
            print(f"  {k:32s} n {r['stretch']['n']:5d}  stretch {r['stretch']['mae']:6.1f}  pad {r['pad']['mae']:6.1f}  {d[0]:+6.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
    for m, x in res["reproducibility"].items():
        print(f"== rerun against the paper: {MODELS[m]}  " + "  ".join(f"{k} {v['paper_mae']:.1f}->{v['rerun_mae']:.1f}" for k, v in x.items()))
    if res["missing"]:
        print(f"missing {len(res['missing'])} prediction files, e.g. {res['missing'][0]}")


if __name__ == "__main__":
    main()
