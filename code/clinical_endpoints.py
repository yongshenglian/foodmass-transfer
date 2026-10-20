"""Clinically interpretable endpoints for the photograph-only mass estimators (experiments/clinical_endpoints/PLAN.md).

Mass endpoints (within 10%/20%/25 g/50 g, WAPE by mass band, Bland-Altman) for the internal Nutrition5k test and
validation sets and the three external test sets; energy and carbohydrate errors implied by each mass error at the
dish's true composition, and the nutrient-head DINOv2's direct errors, where nutrient labels exist (Nutrition5k,
ASA24). Test-set nutrient endpoints need test_true_nutrients.csv and nutrient_head_test.csv from run_workstation.sh;
they are skipped until those files exist. Run from code/.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = Path("../experiments"); OUT = E / "clinical_endpoints"
C, S, V1, V2, A = E / "confirmation_photo_only", E / "n5k_single_prompt", E / "qwen3vl_journal_v1", E / "qwen3vl_journal_v2_training", E / "asa24_external"
BANDS = [(0, 100), (100, 200), (200, 300), (300, 450), (450, np.inf)]
BAND_NAMES = ["<100", "100-199", "200-299", "300-449", ">=450"]
CARB_TOL = 10.0
SP, SPK, DN, DL = "Qwen, single prompt", "Qwen, single prompt + kNN", "DINOv2-base, Nutrition5k only", "DINOv2-large, multi-domain"


def load(path, col, ids):
    return pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(ids)[col].to_numpy(float).copy()


def qwen(path, ids, fallback=186.0):
    q = load(path, "predicted_weight", ids); q[~(q > 0)] = fallback
    return np.clip(q, 1, None)


def mass_endpoints(pred, t):
    e = pred - t; r = np.abs(e) / t; lr = np.log(pred / t)
    out = {"n": int(len(t)), "mae_g": float(np.mean(np.abs(e))), "wape_pct": float(np.abs(e).sum() / t.sum() * 100),
           "within_10pct": float(np.mean(r <= 0.10) * 100), "within_20pct": float(np.mean(r <= 0.20) * 100),
           "within_25g": float(np.mean(np.abs(e) <= 25) * 100), "within_50g": float(np.mean(np.abs(e) <= 50) * 100),
           "ba_bias_g": float(e.mean()), "ba_loa_g": [float(e.mean() - 1.96 * e.std(ddof=1)), float(e.mean() + 1.96 * e.std(ddof=1))],
           "ba_ratio_bias": float(np.exp(lr.mean())),
           "ba_ratio_loa": [float(np.exp(lr.mean() - 1.96 * lr.std(ddof=1))), float(np.exp(lr.mean() + 1.96 * lr.std(ddof=1)))]}
    out["wape_by_band_pct"] = {}
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        m = (t >= lo) & (t < hi)
        out["wape_by_band_pct"][name] = {"n": int(m.sum()), "wape": float(np.abs(e[m]).sum() / t[m].sum() * 100) if m.any() else None}
    return out


def composition_endpoints(pred, truth):
    """Energy/carbohydrate error implied by the mass error if the dish's composition were known exactly."""
    t = truth.true_mass.to_numpy(float); dens = truth.true_calories.to_numpy(float) / t; carb = truth.true_carb.to_numpy(float) / t
    ek = np.abs(pred - t) * dens; ec = np.abs(pred - t) * carb
    terc = np.quantile(dens, [1 / 3, 2 / 3]); groups = np.digitize(dens, terc)
    return {"kcal_mae": float(ek.mean()), "kcal_wape_pct": float(ek.sum() / truth.true_calories.sum() * 100),
            "carb_mae_g": float(ec.mean()), "carb_within_10g": float(np.mean(ec <= CARB_TOL) * 100),
            "kcal_mae_by_energy_density_tertile": {f"T{k + 1} ({lo:.2f}-{hi:.2f} kcal/g)": float(ek[groups == k].mean())
                                                   for k, (lo, hi) in enumerate(zip([dens.min(), *terc], [*terc, dens.max()]))}}


def direct_nutrients(pred_kcal, pred_carb, truth):
    ek = np.abs(pred_kcal - truth.true_calories.to_numpy(float)); ec = np.abs(pred_carb - truth.true_carb.to_numpy(float))
    return {"kcal_mae": float(ek.mean()), "kcal_wape_pct": float(ek.sum() / truth.true_calories.sum() * 100),
            "kcal_within_20pct": float(np.mean(ek <= 0.2 * truth.true_calories.to_numpy(float)) * 100),
            "carb_mae_g": float(ec.mean()), "carb_within_10g": float(np.mean(ec <= CARB_TOL) * 100)}


def n5k_sets():
    sets = {}
    s = pd.read_csv(C / "confirmation_498_sample.csv").set_index("dish_id"); ids = s.index
    delta = load(C / "residual_transfer_delta_saved_embeddings.csv", "neighbor_log_residual", ids); sp = qwen(S / "qwen_blind_test_fp8.csv", ids)
    truth = None
    if (OUT / "test_true_nutrients.csv").is_file():
        truth = pd.read_csv(OUT / "test_true_nutrients.csv").set_index("dish_id").reindex(ids)
        assert np.allclose(truth.true_mass, s.true_weight), "metadata mass differs from the sample's mass"
    head = None
    if (OUT / "nutrient_head_test.csv").is_file():
        head = pd.read_csv(OUT / "nutrient_head_test.csv").set_index("dish_id").reindex(ids)
    sets["test"] = dict(t=s.true_weight.to_numpy(float), truth=truth, head=head, preds={
        SP: sp, SPK: np.clip(sp * np.exp(0.75 * delta), 10, 1500),
        DN: load(C / "dinov2_photo_only_n5k.csv", "mass", ids), DL: load(C / "dinov2_large_multidomain.csv", "mass", ids)})
    base = pd.read_csv(V1 / "dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id"); ids = base.index
    sp = qwen(S / "qwen_blind_val_fp8.csv", ids); delta = base.neighbor_log_residual.to_numpy(float)
    nut = pd.read_csv(V2 / "stage_b_imageonly_aux10_nutr/development_nutrients.csv").set_index("dish_id").reindex(ids)
    truth = nut[["true_calories", "true_fat", "true_carb", "true_protein"]].assign(true_mass=base.true_weight.to_numpy(float))
    sets["validation"] = dict(t=base.true_weight.to_numpy(float), truth=truth,
                              head=nut.rename(columns=lambda c: c), preds={
        SP: sp, SPK: np.clip(sp * np.exp(0.75 * delta), 10, 1500),
        DN: load(V2 / "stage_b_imageonly_aux10/development_predictions.csv", "mass", ids),
        DL: load(A / "runs/joint_large_consensus/n5k_development_predictions.csv", "mass", ids)})
    return sets


def external_sets():
    out = {}
    s = pd.read_csv(A / "asa24_aerial_sample.csv").set_index("dish_id"); ids = s.index
    med = float(pd.read_csv(V1 / "reference_train2253_qwen3vl_results.csv").true_weight.median())
    truth = pd.DataFrame({"true_mass": s.true_weight, "true_calories": s.kcal, "true_carb": s.carb}, index=ids)
    head = pd.read_csv(A / "finetuned_stage_b_imageonly_aux10_nutr.csv").set_index("dish_id").reindex(ids)
    out["ASA24"] = dict(t=s.true_weight.to_numpy(float), truth=truth, head=head, preds={
        SP: qwen(A / "qwen_blind_fp8_lenient.csv", ids, med), DN: load(A / "finetuned_stage_b_imageonly_aux10.csv", "mass", ids),
        DL: load(A / "runs/joint_large_consensus/asa24_test_predictions.csv", "mass", ids)})
    e = pd.read_csv(E / "external_ecustfd/ecustfd_all_predictions.csv"); smp = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").set_index("dish_id").loc[e.dish_id]
    out["ECUSTFD"] = dict(t=smp.true_weight.to_numpy(float), truth=None, head=None, preds={
        SP: e["Raw Qwen, blind"].to_numpy(float), DN: e["Nutrition5k-only photo model"].to_numpy(float), DL: e["Joint large + consensus"].to_numpy(float)})
    F = E / "external_fpb"; s = pd.read_csv(F / "fpb_val_eval.csv").set_index("dish_id")
    s = s[~s.portion.isin(set(pd.read_csv(F / "fpb_train.csv").portion))]; fb = float(pd.read_csv(F / "fpb_train_1024.csv").true_weight.median())
    q = pd.read_csv(F / "qwen_blind_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(s.index); qb = []
    for r, v, pw in zip(q.raw_response.fillna(""), q.format_valid.fillna(False), q.predicted_weight):
        if v and pw > 0:
            qb.append(pw); continue
        g = [float(x) for x in re.findall(r"^(?!\s*confidence).*?:\s*([\d.]+)\s*g\s*$", r, flags=re.M | re.I)]
        qb.append(sum(g) if g and sum(g) > 0 else fb)
    out["FPB"] = dict(t=s.true_weight.to_numpy(float), truth=None, head=None, preds={
        SP: np.clip(np.array(qb), 1, None), DN: load(F / "pred_n5k_imageonly.csv", "mass", s.index),
        "DINOv2-base, three-domain": load(E / "asa24_external/runs/joint3_base/fpb_val_predictions.csv", "mass", s.index)})
    return out


def bland_altman_figure(d, path):
    names = list(d["preds"]); t = d["t"]
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.8), sharey=True)
    for ax, k in zip(axes, names):
        e = d["preds"][k] - t; mean = (d["preds"][k] + t) / 2; b, sd = e.mean(), e.std(ddof=1)
        ax.scatter(mean, e, s=7, alpha=0.5, linewidths=0, color="#2a78d6" if "DINOv2" in k else "#eb6834")
        for y, ls in [(b, "-"), (b - 1.96 * sd, "--"), (b + 1.96 * sd, "--")]:
            ax.axhline(y, color="#1f1f1e", lw=0.8, ls=ls)
        ax.axhline(0, color="#b0afa8", lw=0.6)
        ax.set_xscale("log"); ax.set_xlim(5, 2000)
        ax.set_title(f"{k}\nbias {b:+.0f} g, LoA {b - 1.96 * sd:+.0f} to {b + 1.96 * sd:+.0f} g", fontsize=8.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted minus measured (g)")
    fig.supxlabel("Mean of predicted and measured mass (g)", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    results = {}
    for name, d in {**n5k_sets(), **external_sets()}.items():
        r = {k: mass_endpoints(p, d["t"]) for k, p in d["preds"].items()}
        if d["truth"] is not None:
            for k, p in d["preds"].items():
                r[k]["at_true_composition"] = composition_endpoints(p, d["truth"])
        if d["head"] is not None and d["truth"] is not None:
            r["DINOv2-base nutrient head (direct)"] = {"direct": direct_nutrients(d["head"].pred_calories.to_numpy(float), d["head"].pred_carb.to_numpy(float), d["truth"])}
        results[name] = r
        if name == "test":
            bland_altman_figure(d, OUT / "fig_bland_altman_test.pdf")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clinical_endpoints.json").write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        print(f"== {name}")
        for k, m in r.items():
            if "mae_g" in m:
                line = (f"  {k:32s} MAE {m['mae_g']:6.1f} WAPE {m['wape_pct']:5.1f}%  <=10% {m['within_10pct']:5.1f}  <=20% {m['within_20pct']:5.1f}  "
                        f"<=25g {m['within_25g']:5.1f}  <=50g {m['within_50g']:5.1f}  ratio LoA {m['ba_ratio_loa'][0]:.2f}-{m['ba_ratio_loa'][1]:.2f}")
                if "at_true_composition" in m:
                    c = m["at_true_composition"]; line += f"  | kcal {c['kcal_mae']:6.1f}  carb {c['carb_mae_g']:5.1f} g, <=10 g {c['carb_within_10g']:5.1f}%"
                print(line)
                print("      WAPE by band: " + ", ".join(f"{b} {v['wape']:.1f}% (n={v['n']})" for b, v in m["wape_by_band_pct"].items() if v["wape"] is not None))
                if "at_true_composition" in m:
                    print("      kcal MAE by energy density: " + ", ".join(f"{b} {v:.1f}" for b, v in m["at_true_composition"]["kcal_mae_by_energy_density_tertile"].items()))
            else:
                c = m["direct"]; print(f"  {k:32s} direct kcal MAE {c['kcal_mae']:6.1f} (WAPE {c['kcal_wape_pct']:.1f}%, within 20% {c['kcal_within_20pct']:.1f}%), carb MAE {c['carb_mae_g']:.1f} g, <=10 g {c['carb_within_10g']:.1f}%")


if __name__ == "__main__":
    main()
