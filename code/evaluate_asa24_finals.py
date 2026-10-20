"""One-time ASA24 test evaluation of the pre-declared final models (see experiments/asa24_external/FINALS_DECLARED.md).

The ensemble is chosen by validation only (original 5,099-image pool, food-disjoint OOF): the subset of candidate runs
whose geometric-mean prediction has the lowest validation MAE. Test labels are read only after that choice.
Run from code/.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_asa24_external import metrics

A = Path("../experiments/asa24_external")
FINALS = ["joint_base_stretch", "joint_large_stretch", "joint_large_consensus"]
CANDIDATES = ["joint_base_stretch", "joint_base_stretch_qwen", "joint_large_stretch", "joint_base_consensus",
              "joint_base_text", "joint_large_consensus"]


def main():
    # 1. choose the ensemble on validation (no test data touched)
    base = set(pd.read_csv(A / "asa24_train_pool.csv").image_file)
    P = {r: pd.read_csv(A / "runs" / r / "asa24_pool_oof.csv").query("dish_id in @base").set_index("dish_id").sort_index()
         for r in CANDIDATES}
    t = P[CANDIDATES[0]].true.to_numpy(); best = None
    for k in range(1, len(CANDIDATES) + 1):
        for c in itertools.combinations(CANDIDATES, k):
            m = float(np.mean(np.abs(np.exp(np.mean([np.log(P[r].oof.to_numpy()) for r in c], 0)) - t)))
            if best is None or m < best[0]:
                best = (m, c)
    choice = {"validation_mae": best[0], "runs": list(best[1])}
    (A / "ensemble_choice.json").write_text(json.dumps(choice, indent=2))
    print("ensemble chosen on validation:", choice)

    # 2. test evaluation, once
    s = pd.read_csv(A / "asa24_aerial_sample.csv").set_index("dish_id")
    ids, tt, series = s.index, s.true_weight.to_numpy(float), s.series.to_numpy()
    load = lambda r: pd.read_csv(A / "runs" / r / "asa24_test_predictions.csv").set_index("dish_id").reindex(ids).mass.to_numpy(float)
    rows = {"Nutrition5k-only photo model (reference)":
            pd.read_csv(A / "finetuned_stage_b_imageonly_aux10.csv").set_index("dish_id").reindex(ids).mass.to_numpy(float)}
    for r in FINALS:
        rows[f"{r}"] = load(r)
    rows["validation-chosen ensemble: " + " + ".join(best[1])] = np.exp(np.mean([np.log(load(r)) for r in best[1]], 0))
    out = {k: metrics(v, tt, series) for k, v in rows.items()}
    ref = rows["Nutrition5k-only photo model (reference)"]
    idx = np.random.default_rng(0).integers(0, len(tt), (10000, len(tt)))
    for k, v in rows.items():
        d = np.abs(v - tt) - np.abs(ref - tt); b = d[idx].mean(1)
        out[k]["minus_n5k_only"] = [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    (A / "asa24_finals_test_results.json").write_text(json.dumps({"ensemble_choice": choice, "test": out}, indent=2))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(out).T.drop(columns=["minus_n5k_only"]).round(2).to_string())
    for k, v in out.items():
        print(f"  {k}: vs Nutrition5k-only {v['minus_n5k_only'][0]:+.2f} [{v['minus_n5k_only'][1]:+.2f}, {v['minus_n5k_only'][2]:+.2f}]")


if __name__ == "__main__":
    main()
