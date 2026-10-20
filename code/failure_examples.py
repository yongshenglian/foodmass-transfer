"""Failure analysis (review request: show when the system is dangerously wrong). Run from code/.

Rates of large errors per method and dataset (absolute error at least 50% of the true mass; for Nutrition5k and ASA24
also an energy error of at least 100 kcal at the true composition), a table of the worst cases, and a figure of
selected failures. The figure needs the photographs, so it is drawn only where the image paths exist (workstation).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_endpoints import DL, DN, SP, external_sets, n5k_sets

E = Path("../experiments"); OUT = E / "failures"; OUT.mkdir(exist_ok=True)


def describe():
    s = pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv").set_index("dish_id")
    a = pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv").set_index("dish_id")
    e = pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv").set_index("dish_id")
    f = pd.read_csv(E / "external_fpb/fpb_val_eval.csv").set_index("dish_id")
    f = f[~f.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion))]
    short = lambda x, n=6: ", ".join(str(x).split(" | ")[:n]) + (" ..." if len(str(x).split(" | ")) > n else "")
    return {"Nutrition5k test": (s.index, s.image, s.ingredients.map(short)),
            "ASA24": (a.index, a.image, a.ImgTitle.fillna(a.fndds_description)),
            "ECUSTFD": (e.index, e.image, e.type), "FPB": (f.index, f.image, f.ingredients.map(lambda x: short(x, 5)))}


def main():
    sets = {"Nutrition5k test": n5k_sets()["test"], **{k: v for k, v in external_sets().items()}}
    best = {"Nutrition5k test": DL, "ASA24": DL, "ECUSTFD": DL, "FPB": "DINOv2-base, three-domain"}
    desc = describe(); rates = {}; worst = []
    for name, d in sets.items():
        t = d["t"]; truth = d["truth"]
        dens = (truth.true_calories.to_numpy(float) / t) if truth is not None else None
        rates[name] = {}
        for k, p in d["preds"].items():
            rel = np.abs(p - t) / t; r = {"n": int(len(t)), "error_ge_50pct": float(np.mean(rel >= 0.5) * 100),
                                          "error_ge_100pct": float(np.mean(rel >= 1.0) * 100)}
            if dens is not None:
                ek = np.abs(p - t) * dens; r["energy_error_ge_100kcal"] = float(np.mean(ek >= 100) * 100)
                r["error_ge_50pct_and_energy_ge_100kcal"] = float(np.mean((rel >= 0.5) & (ek >= 100)) * 100)
            rates[name][k] = r
        ids, img, txt = desc[name]
        for k in (best[name], SP):
            p = d["preds"][k]; ek = np.abs(p - t) * dens if dens is not None else np.full(len(t), np.nan)
            score = ek if dens is not None else np.abs(p - t)
            seen = set(); order = []
            for i in np.argsort(-score):  # five worst, one per food or tray description
                if txt.iloc[i] not in seen:
                    seen.add(txt.iloc[i]); order.append(i)
                if len(order) == 5:
                    break
            for i in order:
                worst.append({"dataset": name, "method": k, "dish_id": ids[i], "description": txt.iloc[i], "image": img.iloc[i],
                              "true_g": float(t[i]), "pred_g": float(p[i]), "error_pct": float((p[i] - t[i]) / t[i] * 100),
                              "energy_error_kcal": float(ek[i]) if dens is not None else None,
                              "other_method_pred_g": float(d["preds"][SP if k != SP else best[name]][i])})
    pd.DataFrame(worst).to_csv(OUT / "worst_cases.csv", index=False)
    (OUT / "failure_rates.json").write_text(json.dumps(rates, indent=2))
    for name, r in rates.items():
        print(f"== {name}")
        for k, v in r.items():
            extra = f"  energy>=100 kcal {v['energy_error_ge_100kcal']:5.1f}%  both {v['error_ge_50pct_and_energy_ge_100kcal']:5.1f}%" if "energy_error_ge_100kcal" in v else ""
            print(f"  {k:32s} >=50% {v['error_ge_50pct']:5.1f}%  >=100% {v['error_ge_100pct']:5.1f}%{extra}")
    w = pd.DataFrame(worst)
    print(w[["dataset", "method", "description", "true_g", "pred_g", "error_pct", "energy_error_kcal"]].to_string(max_colwidth=45))
    figure(w)


def figure(w):
    """Seven failures: the two worst of the best DINOv2 and the worst other dish of Qwen on Nutrition5k, two of DINOv2 on ASA24, one each on ECUSTFD and FPB."""
    q = w[(w.dataset == "Nutrition5k test") & (w.method == SP)]
    q = q[~q.dish_id.isin(w[(w.dataset == "Nutrition5k test") & (w.method == DL)].head(2).dish_id)]  # no dish twice
    picks = pd.concat([w[(w.dataset == "Nutrition5k test") & (w.method == DL)].head(2), q.head(1),
                       w[(w.dataset == "ASA24") & (w.method == DL)].head(2), w[(w.dataset == "ECUSTFD") & (w.method == DL)].head(1),
                       w[(w.dataset == "FPB") & (w.method != SP)].head(1)])
    picks.to_csv(OUT / "figure_failures.csv", index=False)
    if not all(Path(p).is_file() for p in picks.image):
        print(f"\n(figure skipped: photographs not found here; {OUT / 'figure_failures.csv'} lists the selection)")
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(2, 4, figsize=(10, 6.2))
    for ax in axes.ravel()[len(picks):]:
        ax.set_axis_off()
    for ax, (_, r) in zip(axes.ravel(), picks.iterrows()):
        im = Image.open(r.image)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im.convert("RGBA"))
        ax.imshow(im.convert("RGB")); ax.set_axis_off()
        d, q = (r.other_method_pred_g, r.pred_g) if r.method == SP else (r.pred_g, r.other_method_pred_g)
        pct = lambda p: f"{(p - r.true_g) / r.true_g * 100:+.0f}%"
        ax.set_title(f"{r.dataset}: {str(r.description)[:34]}\ntrue {r.true_g:,.0f} g; DINOv2 {d:,.0f} g ({pct(d)})\nQwen {q:,.0f} g ({pct(q)})",
                     fontsize=7.6, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_failure_examples.pdf"); fig.savefig(OUT / "fig_failure_examples.png", dpi=200)
    print(f"figure -> {OUT / 'fig_failure_examples.pdf'}")


if __name__ == "__main__":
    main()
