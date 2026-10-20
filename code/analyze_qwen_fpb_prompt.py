"""Qwen prompt sensitivity on FPB training photographs (post hoc; experiments/qwen_fpb_prompt/PLAN.md). Run from code/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments/qwen_fpb_prompt"); B = 10000
rank = lambda x: pd.Series(x).rank().to_numpy()


def summary(p, t):
    return {"n": int(len(t)), "mae": float(np.abs(p - t).mean()), "scale_ratio": float(np.exp(np.mean(np.log(p / t)))),
            "spearman": float(np.corrcoef(rank(p), rank(t))[0, 1]), "within_20pct": float((np.abs(p - t) <= 0.2 * t).mean())}


def main():
    s = pd.read_csv(E / "sample.csv").set_index("dish_id"); arms = {}
    for m in ("blind", "blind_all"):
        q = pd.read_csv(E / f"qwen_{m}_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(s.index)
        arms[m] = q.predicted_weight.where(q.format_valid.astype("boolean").fillna(False).astype(bool) & (q.predicted_weight > 0)).to_numpy(float)
    t = s.true_weight.to_numpy(float); ok = np.isfinite(arms["blind"]) & np.isfinite(arms["blind_all"]); rng = np.random.default_rng(0)
    res = {"invalid": {m: int((~np.isfinite(a)).sum()) for m, a in arms.items()}}
    for name, g in (("all", ok), ("single item", ok & (s.n_items == 1).to_numpy()), ("several items", ok & (s.n_items > 1).to_numpy())):
        r = {m: summary(a[g], t[g]) for m, a in arms.items()}
        d = pd.DataFrame({"d": np.abs(arms["blind_all"][g] - t[g]) - np.abs(arms["blind"][g] - t[g]), "n": 1, "portion": s.portion.to_numpy()[g]}).groupby("portion").sum()
        i = rng.integers(0, len(d), (B, len(d))); bs = d.d.to_numpy()[i].sum(1) / d.n.to_numpy()[i].sum(1)
        r["mae_neutral_minus_original"] = [float(d.d.sum() / d.n.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        r["median_ratio_neutral_to_original"] = float(np.median(arms["blind_all"][g] / arms["blind"][g])); res[name] = r
        print(f"== {name} (n={g.sum()})")
        for m in arms:
            print(f"  {m:10s} MAE {r[m]['mae']:6.1f}  scale {r[m]['scale_ratio']:.2f}  rho {r[m]['spearman']:.2f}  within 20% {100 * r[m]['within_20pct']:.1f}%")
        print("  neutral minus original MAE {:+.1f} [{:+.1f}, {:+.1f}]; median ratio {:.2f}".format(*r["mae_neutral_minus_original"], r["median_ratio_neutral_to_original"]))
    print("invalid", res["invalid"]); (E / "qwen_fpb_prompt.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
