"""FPB validation-split evaluation (one-time per model; models chosen on FPB-train CV / other validation only).

Primary subset: validation photos whose exact portion (item classes + weights) does not occur in FPB train.
Reports MAE, median AE, bias, share within 25%, and paired bootstrap differences vs raw Qwen (given names).
Run from code/.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_asa24_external import metrics

E = Path("../experiments/external_fpb")
R = Path("../experiments/asa24_external/runs")


def qwen(mode, s, fallback):
    q = pd.read_csv(E / f"qwen_{mode}_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(s.index); out = []
    for r, v, pw in zip(q.raw_response.fillna(""), q.format_valid.fillna(False), q.predicted_weight):
        if v and pw > 0:
            out.append(pw); continue
        g = [float(x) for x in re.findall(r"^(?!\s*confidence).*?:\s*([\d.]+)\s*g\s*$", r, flags=re.M | re.I)]
        out.append(sum(g) if g and sum(g) > 0 else fallback)
    return np.array(out)


def main():
    s = pd.read_csv(E / "fpb_val_eval.csv").set_index("dish_id")
    train_portions = set(pd.read_csv(E / "fpb_train.csv").portion)
    s["portion_disjoint"] = ~s.portion.isin(train_portions)
    fallback = float(pd.read_csv(E / "fpb_train_1024.csv").true_weight.median())
    rows = {"Raw Qwen, given item names": qwen("oracle", s, fallback), "Raw Qwen, blind": qwen("blind", s, fallback),
            "Photo model, Nutrition5k only (zero-shot)": pd.read_csv(E / "pred_n5k_imageonly.csv").set_index("dish_id").loc[s.index].mass.to_numpy(float),
            "Photo model, N5K+ASA24 large (zero-shot)": pd.read_csv(E / "pred_joint_large_consensus.csv").set_index("dish_id").loc[s.index].mass.to_numpy(float)}
    for run, label in [("joint3_base", "Photo model, N5K+ASA24+FPB-train, base"), ("joint3_large", "Photo model, N5K+ASA24+FPB-train, large")]:
        f = R / run / "fpb_val_predictions.csv"
        if f.is_file():
            rows[label] = pd.read_csv(f).set_index("dish_id").loc[s.index].mass.to_numpy(float)
    rows["Constant: FPB-train median"] = np.full(len(s), fallback)
    out = {}
    for subset, mask in [("portion_disjoint", s.portion_disjoint.to_numpy()), ("all_val", np.ones(len(s), bool))]:
        t = s.true_weight.to_numpy(float)[mask]; ser = s.series.to_numpy()[mask]
        idx = np.random.default_rng(0).integers(0, len(t), (10000, len(t))); ref = rows["Raw Qwen, given item names"][mask]
        out[subset] = {"n": int(mask.sum())}
        for k, v in rows.items():
            m = metrics(np.clip(v[mask], 1, None), t, ser)
            d = np.abs(v[mask] - t) - np.abs(ref - t); b = d[idx].mean(1)
            m["minus_qwen_given_names"] = [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            out[subset][k] = m
    (E / "fpb_results.json").write_text(json.dumps(out, indent=2))
    for subset in out:
        print(f"== {subset} (n = {out[subset]['n']})")
        for k, m in out[subset].items():
            if k == "n":
                continue
            c = m["minus_qwen_given_names"]
            print(f"  {k:45s} MAE {m['mae_g']:6.1f}  medAE {m['median_ae_g']:6.1f}  bias {m['bias_g']:+7.1f}  within25% {m['within_25pct']:5.1f}  vs Qwen {c[0]:+7.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")


if __name__ == "__main__":
    main()
