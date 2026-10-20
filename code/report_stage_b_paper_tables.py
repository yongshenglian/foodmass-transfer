"""Paper tables for the supervised stage B comparison (descriptive; no model choices are made here).

Joins the frozen stage B predictions with the paper's row-level predictions (raw Qwen,
fixed DINOv2 residual) for development, confirmation and the pooled 1,000 dishes, and
writes metric tables with paired bootstrap CIs plus development/confirmation subgroups.
Run from code/. Reads the unpacked confirmation files from --confirmation-root.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

V1 = Path("../experiments/qwen3vl_journal_v1")
V2 = Path("../experiments/qwen3vl_journal_v2_training")
PRIMARY, SECONDARY = "stage_b_logmass_aux10_seed2", "stage_b_logmass_aux10_noingr"
METHODS = {"raw": "Raw Qwen3-VL-32B", "fixed": "Fixed DINOv2 residual", "stage_b": "Stage B fine-tuned (primary)",
           "stage_b_noingr": "Stage B, no ingredient multi-hot"}


def cohort_frame(paper_csv, primary_csv, secondary_csv):
    p = pd.read_csv(paper_csv).set_index("dish_id")
    a = pd.read_csv(primary_csv).set_index("dish_id").loc[p.index]
    b = pd.read_csv(secondary_csv).set_index("dish_id").loc[p.index]
    assert np.allclose(a.true, p.true_weight) and np.allclose(b.true, p.true_weight)
    return pd.DataFrame({"true": p.true_weight, "confidence": p.confidence, "ingredient_count": p.ingredient_count,
                         "raw": p.predicted_weight, "fixed": p.fixed_dinov2_visual_residual,
                         "stage_b": a.mass, "stage_b_noingr": b.mass})


def metrics(pred, true):
    e = pred - true
    return {"n": len(e), "mae": np.mean(np.abs(e)), "median_ae": np.median(np.abs(e)), "rmse": np.sqrt(np.mean(e ** 2)),
            "bias": e.mean(), "wape_pct": np.abs(e).sum() / true.sum() * 100,
            **{f"within_{k}g_pct": np.mean(np.abs(e) <= k) * 100 for k in (25, 50, 100)}}


def paired(x, y, true, idx):
    d = np.abs(x - true) - np.abs(y - true)
    s = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def subgroups(f):
    groups = {"Single ingredient": f.ingredient_count <= 1, "Multiple ingredients": f.ingredient_count > 1,
              "Qwen confidence <=65": f.confidence <= 65, "Qwen confidence 70-79": f.confidence.between(66, 79.99),
              "Qwen confidence >=80": f.confidence >= 80}
    bins = [(0, 100, "<100 g"), (100, 200, "100-199 g"), (200, 300, "200-299 g"), (300, 450, "300-449 g"), (450, np.inf, ">=450 g")]
    groups.update({f"True mass {name}": (f.true >= lo) & (f.true < hi) for lo, hi, name in bins})
    rows = []
    for name, mask in groups.items():
        g = f[mask]
        rows.append({"subgroup": name, "n": int(mask.sum()),
                     **{k: float(np.mean(np.abs(g[k] - g.true))) for k in METHODS}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation-root", type=Path, default=Path.home() / "data/confirmation_2026-10-02")
    ap.add_argument("--output-dir", type=Path, default=V2 / "paper_tables")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conf_v1 = args.confirmation_root / "experiments/qwen3vl_journal_v1"
    dev = cohort_frame(V1 / "dinov2_gate_v1_full2253/development_predictions.csv",
                       V2 / PRIMARY / "development_predictions.csv", V2 / SECONDARY / "development_predictions.csv")
    conf = cohort_frame(conf_v1 / "extension_498_dinov2_gate/confirmation_predictions.csv",
                        V2 / "confirmation_498" / f"{PRIMARY}_confirmation_predictions.csv",
                        V2 / "confirmation_498" / f"{SECONDARY}_confirmation_predictions.csv")
    cohorts = {"development": dev, "confirmation": conf, "pooled": pd.concat([dev, conf])}
    assert cohorts["pooled"].index.is_unique and len(cohorts["pooled"]) == 1000

    out, lines = {}, []
    for name, f in cohorts.items():
        t = f.true.to_numpy()
        idx = np.random.default_rng(0).integers(0, len(t), (10000, len(t)))
        res = {k: metrics(f[k].to_numpy(), t) for k in METHODS}
        comps = {f"{a} - {b}": paired(f[a].to_numpy(), f[b].to_numpy(), t, idx)
                 for a, b in [("fixed", "raw"), ("stage_b", "raw"), ("stage_b_noingr", "raw"), ("stage_b", "fixed"),
                              ("stage_b_noingr", "fixed"), ("stage_b", "stage_b_noingr")]}
        out[name] = {"metrics": res, "paired_mae_differences": comps}
        lines += [f"### {name.capitalize()} (n = {len(t)})", "",
                  "| Method | MAE | Change vs raw Qwen (95% CI) | Median AE | RMSE | Bias | WAPE | Within 25 g | Within 50 g | Within 100 g |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, label in METHODS.items():
            r = res[k]
            ch = "—" if k == "raw" else "{:+.2f} [{:+.2f}, {:+.2f}] g".format(*comps[f"{k} - raw"])
            lines.append(f"| {label} | {r['mae']:.2f} g | {ch} | {r['median_ae']:.2f} g | {r['rmse']:.2f} g | {r['bias']:+.2f} g | "
                         f"{r['wape_pct']:.1f}% | {r['within_25g_pct']:.1f}% | {r['within_50g_pct']:.1f}% | {r['within_100g_pct']:.1f}% |")
        lines += ["", "Paired MAE differences (95% CI): " + "; ".join(
            "{} {:+.2f} [{:+.2f}, {:+.2f}] g".format(k, *v) for k, v in comps.items() if not k.endswith("raw")), ""]

    for name in ("development", "confirmation"):
        rows = subgroups(cohorts[name]); out[f"{name}_subgroups"] = rows
        lines += [f"### {name.capitalize()} subgroups (MAE)", "",
                  "| Subgroup | N | Raw Qwen | Fixed residual | Stage B | Stage B no multi-hot |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {r['subgroup']} | {r['n']} | {r['raw']:.2f} g | {r['fixed']:.2f} g | {r['stage_b']:.2f} g | {r['stage_b_noingr']:.2f} g |" for r in rows]
        lines.append("")

    (args.output_dir / "paper_tables.json").write_text(json.dumps(out, indent=2, default=float))
    (args.output_dir / "paper_tables.md").write_text("\n".join(lines))
    for name, f in cohorts.items():
        f.reset_index().to_csv(args.output_dir / f"{name}_row_predictions.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
