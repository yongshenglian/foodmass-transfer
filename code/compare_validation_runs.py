"""Compare train_multidomain.py runs on the same validation set: the original 5,099-image ASA24 pool
(out-of-fold, food-disjoint) plus Nutrition5k reference CV and development. No test metric is read. Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

A = Path("../experiments/asa24_external")


def main():
    base = set(pd.read_csv(A / "asa24_train_pool.csv").image_file)
    rows = []
    for run in sorted((A / "runs").iterdir()):
        oof, summ = run / "asa24_pool_oof.csv", run / "summary.json"
        if not (oof.is_file() and summ.is_file()):
            continue
        d = pd.read_csv(oof); d = d[d.dish_id.isin(base)]
        s = json.loads(summ.read_text()); cfg = json.loads((run / "config.json").read_text())
        e = d.oof - d.true
        rows.append({"run": run.name, "backbone": Path(cfg["backbone"]).name, "pool": cfg.get("pool", "asa24_train_pool.csv"),
                     "asa24_val_mae": e.abs().mean(), "asa24_val_median_ae": e.abs().median(), "asa24_val_bias": e.mean(),
                     "n_val": len(d), "n5k_ref_cv": s["validation_n5k_ref_cv_mae"], "n5k_dev": s["n5k_development_mae"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
