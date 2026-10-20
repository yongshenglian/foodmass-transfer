"""Single fold models versus the five-model ensemble (experiments/fold_members/PLAN.md). Run from code/ after
experiments/fold_members/run_workstation.sh. Checks that the recomputed ensemble reproduces the paper's predictions,
then reports each fold model's MAE, their mean and range, the ensemble MAE, and the paired difference
ensemble minus average single model with a bootstrap 95% CI over dishes or clusters."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments")


def truth_and_reference():
    """True mass, cluster id and the paper's saved ensemble predictions for each photograph set."""
    v1 = pd.read_csv(E / "qwen3vl_journal_v1/dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id")
    s = pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv").set_index("dish_id")
    a = pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv").set_index("dish_id")
    e = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").set_index("dish_id")
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    f = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))]  # primary subset: unseen trays
    sets = {"n5k_val": (v1.true_weight, None), "n5k_test": (s.true_weight, None), "asa24": (a.true_weight, a.series),
            "ecustfd": (e.true_weight, e.id), "fpb": (f.true_weight, f.portion)}
    ref = {("base_n5k", "n5k_val"): E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10/development_predictions.csv",
           ("base_n5k", "n5k_test"): E / "confirmation_photo_only/dinov2_photo_only_n5k.csv",
           ("base_n5k", "asa24"): E / "asa24_external/finetuned_stage_b_imageonly_aux10.csv",
           ("base_n5k", "ecustfd"): E / "external_ecustfd/pred_n5k_imageonly.csv",
           ("base_n5k", "fpb"): E / "external_fpb/pred_n5k_imageonly.csv",
           ("large_multidomain", "n5k_val"): E / "asa24_external/runs/joint_large_consensus/n5k_development_predictions.csv",
           ("large_multidomain", "n5k_test"): E / "confirmation_photo_only/dinov2_large_multidomain.csv",
           ("large_multidomain", "asa24"): E / "asa24_external/runs/joint_large_consensus/asa24_test_predictions.csv",
           ("large_multidomain", "ecustfd"): E / "external_ecustfd/pred_joint_large_consensus.csv",
           ("large_multidomain", "fpb"): E / "external_fpb/pred_joint_large_consensus.csv",
           ("base_threedomain", "fpb"): E / "asa24_external/runs/joint3_base/fpb_val_predictions.csv"}
    return sets, ref


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=E / "fold_members"); a = ap.parse_args()
    sets, ref = truth_and_reference(); res = []
    for (model, sname), refpath in ref.items():
        path = a.dir / f"{model}_{sname}.csv"
        if not path.is_file():
            continue
        t_series, cl_series = sets[sname]; ids = t_series.index; t = t_series.to_numpy(float)
        d = pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)
        folds = [c for c in d.columns if c.startswith("fold")]
        saved = pd.read_csv(refpath).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)["mass"].to_numpy(float)
        repro = float(np.nanmax(np.abs(d.ensemble.to_numpy(float) - saved)))
        ens_err = np.abs(d.ensemble.to_numpy(float) - t)
        fold_err = np.stack([np.abs(d[c].to_numpy(float) - t) for c in folds])
        diff = ens_err - fold_err.mean(0)  # per photograph: ensemble error minus the average single-model error
        cl = np.arange(len(t)) if cl_series is None else pd.factorize(cl_series.to_numpy())[0]
        sums, counts = np.bincount(cl, diff), np.bincount(cl)
        idx = np.random.default_rng(0).integers(0, len(sums), (10000, len(sums))); boot = sums[idx].sum(1) / counts[idx].sum(1)
        fm = fold_err.mean(1)
        res.append({"model": model, "set": sname, "n": int(len(t)), "ensemble_mae": float(ens_err.mean()),
                    "fold_mae": [float(x) for x in fm], "fold_mae_mean": float(fm.mean()), "fold_mae_min": float(fm.min()),
                    "fold_mae_max": float(fm.max()), "ensemble_minus_single": [float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "max_abs_diff_vs_paper_g": repro})
    (a.dir / "fold_vs_ensemble.json").write_text(json.dumps(res, indent=2))
    for r in res:
        c = r["ensemble_minus_single"]
        print(f"{r['model']:18s} {r['set']:9s} ensemble {r['ensemble_mae']:6.2f} g | single folds {r['fold_mae_mean']:6.2f} g "
              f"(range {r['fold_mae_min']:.2f}-{r['fold_mae_max']:.2f}) | ensemble - single {c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] "
              f"| reproduces paper to {r['max_abs_diff_vs_paper_g']:.3f} g")


if __name__ == "__main__":
    main()
