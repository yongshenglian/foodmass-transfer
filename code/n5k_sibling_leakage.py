"""Incremental-scan siblings between the Nutrition5k training set and our validation/internal test sets
(experiments/n5k_sibling_leakage/PLAN.md). Fixed-prediction subset analysis; nothing is retrained. Run from code/."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); OUT = E / "n5k_sibling_leakage"
ROOT = Path(os.environ.get("N5K_ROOT", Path.home() / "data/nutrition5k_dataset"))


def ingredient_sets():
    sets = {}
    for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        for line in (ROOT / "metadata" / f).read_text().splitlines():
            p = line.split(","); sets[p[0]] = frozenset(p[7::7])  # dish fields (6), then 7 fields per ingredient: id, name, ...
    return sets


def has_sibling(ids, train, sets, gap, need_subset):
    tt = np.array(sorted(int(d.split("_")[1]) for d in train)); by_time = {int(d.split("_")[1]): d for d in train}
    out = []
    for d in ids:
        t = int(d.split("_")[1]); near = tt[(tt >= t - gap) & (tt <= t + gap)]
        out.append(any((not need_subset) or sets[d] <= sets[by_time[n]] or sets[by_time[n]] <= sets[d] for n in near))
    return np.array(out)


def main():
    sets = ingredient_sets()
    train = list(pd.read_csv(E / "qwen3vl_journal_v1/reference_train2253_qwen3vl_results.csv").dish_id)
    cohorts = {"internal test": (pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv"), E / "fold_members/base_n5k_n5k_test.csv",
                                 E / "n5k_single_prompt/qwen_blind_test_fp8.csv"),
               "validation": (pd.read_csv(E / "qwen3vl_journal_v2_training/qwen_local/development_502_sample.csv"), E / "fold_members/base_n5k_n5k_val.csv", None)}
    res = {}
    for name, (s, dino, qw) in cohorts.items():
        s = s.drop_duplicates("dish_id").set_index("dish_id"); t = s.true_weight.to_numpy(float)
        assert not set(s.index) & set(train)
        preds = {"DINOv2-base": pd.read_csv(dino).drop_duplicates("dish_id").set_index("dish_id").reindex(s.index).ensemble.to_numpy(float)}
        if qw is not None:
            q = pd.read_csv(qw).drop_duplicates("dish_id").set_index("dish_id").reindex(s.index).predicted_weight.to_numpy(float).copy()
            q[~(q > 0)] = 186.0; preds["Qwen, single prompt"] = q
        res[name] = {}
        for label, gap, sub in (("120 s + subset (primary)", 120, True), ("60 s + subset", 60, True), ("300 s + subset", 300, True), ("120 s, time only", 120, False)):
            m = has_sibling(s.index, train, sets, gap, sub)
            r = {"n": int(len(m)), "with_sibling": int(m.sum()), "share_pct": float(m.mean() * 100),
                 "mean_mass_with": float(t[m].mean()) if m.any() else None, "mean_mass_without": float(t[~m].mean())}
            for k, p in preds.items():
                e = np.abs(p - t)
                r[k] = {"mae_all": float(e.mean()), "mae_with": float(e[m].mean()) if m.any() else None, "mae_without": float(e[~m].mean()),
                        "wape_with": float(e[m].sum() / t[m].sum() * 100) if m.any() else None, "wape_without": float(e[~m].sum() / t[~m].sum() * 100)}
            res[name][label] = r
            print(f"{name:14s} {label:26s} siblings {m.sum():3d}/{len(m)} ({m.mean()*100:4.1f}%)  mean mass {r['mean_mass_with'] or 0:.0f}/{r['mean_mass_without']:.0f}  " +
                  "  ".join(f"{k}: MAE with {v['mae_with'] or 0:.1f} without {v['mae_without']:.1f} (WAPE {v['wape_with'] or 0:.1f}/{v['wape_without']:.1f})" for k, v in r.items() if isinstance(v, dict)))
    (OUT / "n5k_sibling_leakage.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
