"""Matched supervised baselines (experiments/matched_baselines/PLAN.md): what fine-tuning, the depth target and the
backbone add under identical data, head and training budget. For each run: Nutrition5k cross-validation and validation
MAE (from training), MAE, Spearman ordering and scale ratio on the validation set and the three external sets, and
the paired difference against the paper's model (fine-tuned DINOv2-base with depth), bootstrapped over dishes or
clusters. Run from code/."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fold_vs_ensemble import truth_and_reference

warnings.filterwarnings("ignore", category=RuntimeWarning)  # constant predictions within a food give NaN rho (dropped)

E = Path("../experiments"); T = E / "qwen3vl_journal_v2_training"; D = E / "matched_baselines"
RUNS = {"stage_b_imageonly_aux10": "DINOv2-base, fine-tuned, depth (paper)",
        "stage_b_imageonly_nodepth": "DINOv2-base, fine-tuned, no depth",
        "stage_b_imageonly_frozen": "DINOv2-base, frozen, depth",
        "stage_b_imageonly_frozen_nodepth": "DINOv2-base, frozen, no depth",
        "stage_b_imageonly_resnet50": "ResNet-50, fine-tuned, depth (lr 2e-5)",
        "stage_b_imageonly_resnet50_lr1e4": "ResNet-50, fine-tuned, depth (lr 1e-4)",
        "stage_b_imageonly_resnet50_lr3e4": "ResNet-50, fine-tuned, depth (lr 3e-4)",
        "stage_b_imageonly_resnet50_lr1e3": "ResNet-50, fine-tuned, depth (lr 1e-3)"}
SETS = ["n5k_val", "asa24", "ecustfd", "fpb", "n5k_test"]
# n5k_val comes from each run's own validation predictions. The internal test set was not meant to be scored again, but
# the first job (started before that change) scored it; those scores are reported separately and disclosed.


def predictions(run, sname):
    if sname == "n5k_val":  # five-fold ensemble on the validation set, written by the training script
        d = pd.read_csv(T / run / "development_predictions.csv").rename(columns={"mass": "ensemble"})
        return d.drop_duplicates("dish_id").set_index("dish_id")
    if run == "stage_b_imageonly_aux10":  # the paper's model, scored per fold in experiments/fold_members
        path = E / "fold_members" / f"base_n5k_{sname}.csv"
    else:
        path = D / f"{run}_{sname}.csv"
    return pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id") if path.is_file() else None


def main():
    truth, _ = truth_and_reference(); res = {}
    for run, label in RUNS.items():
        summ = T / run / "summary.json"
        if not summ.is_file():
            continue
        sm = json.loads(summ.read_text())
        r = {"label": label, "cv_mae": sm["reference_cv_mae_g"], "validation_mae": sm["development_mae_g"], "sets": {}}
        for sname in SETS:
            d = predictions(run, sname); ref = predictions("stage_b_imageonly_aux10", sname)
            if d is None or ref is None:
                continue
            t_series, cl_series = truth[sname]; ids = t_series.index; t = t_series.to_numpy(float)
            p = d.reindex(ids).ensemble.to_numpy(float); q = ref.reindex(ids).ensemble.to_numpy(float)
            diff = np.abs(p - t) - np.abs(q - t)
            cl = np.arange(len(t)) if cl_series is None else pd.factorize(cl_series.to_numpy())[0]
            sums, counts = np.bincount(cl, diff), np.bincount(cl)
            idx = np.random.default_rng(0).integers(0, len(sums), (10000, len(sums))); b = sums[idx].sum(1) / counts[idx].sum(1)
            g = float(np.exp(np.mean(np.log(p / t))))
            r["sets"][sname] = {"mae": float(np.mean(np.abs(p - t))), "spearman": float(pd.Series(p).rank().corr(pd.Series(t).rank())),
                                "scale_ratio": g, "mae_after_global_rescale_oracle": float(np.mean(np.abs(p / g - t))),
                                "minus_paper_model": [float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
            if sname == "asa24":  # mean Spearman within each food's portion series (as in analyze_asa24_external.py)
                cs = cl_series.to_numpy(); rho = []
                for sv in np.unique(cs):
                    m = cs == sv
                    if m.sum() >= 3 and np.ptp(t[m]) > 0:
                        rho.append(np.corrcoef(pd.Series(p[m]).rank(), pd.Series(t[m]).rank())[0, 1])
                rho = np.array(rho); r["sets"][sname]["within_food_spearman"] = float(rho[np.isfinite(rho)].mean())
        res[run] = r
    D.mkdir(exist_ok=True); (D / "matched_baselines.json").write_text(json.dumps(res, indent=2))
    for run, r in res.items():
        line = f"{r['label']:42s} CV {r['cv_mae']:6.2f}  val {r['validation_mae']:6.2f}"
        for sname, m in r["sets"].items():
            c = m["minus_paper_model"]
            wf = f", wf {m['within_food_spearman']:.2f}" if "within_food_spearman" in m else ""
            line += (f" | {sname} {m['mae']:6.1f} (rho {m['spearman']:.2f}{wf}, x{m['scale_ratio']:.2f},"
                     f" resc {m['mae_after_global_rescale_oracle']:.1f}, {c[0]:+.1f} [{c[1]:+.1f},{c[2]:+.1f}])")
        print(line)


if __name__ == "__main__":
    main()
