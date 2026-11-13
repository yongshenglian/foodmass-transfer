"""Reviewer comment 8: relative error and its energy consequences, from existing predictions.

Energy error assumes the composition is known: a photograph's mass error times its true energy density (kcal/g), from
Nutrition5k's ingredient metadata or ASA24's FNDDS values. It isolates the consequence of the mass error; a real system
would add errors in recognising the food and its composition. FPB and ECUSTFD have no composition and are left out.
Run from code/:  python comment8_energy.py
"""
import json
from pathlib import Path

import numpy as np

from clinical_endpoints import DL, DN, SP, external_sets, n5k_sets

OUT = Path("../experiments/comment8_energy"); OUT.mkdir(exist_ok=True)


def main():
    v = n5k_sets()["validation"]; a = external_sets()["ASA24"]; res = {}
    for name, d in (("Nutrition5k validation", v), ("ASA24 test", a)):
        t = d["t"]; kcal = d["truth"].true_calories.to_numpy(float); dens = kcal / t; ok = np.isfinite(dens) & (dens >= 0)
        q = np.quantile(dens[ok], [1 / 3, 2 / 3])
        terc = {f"low ({dens[ok].min():.2f}-{q[0]:.2f} kcal/g)": ok & (dens <= q[0]),
                f"middle ({q[0]:.2f}-{q[1]:.2f} kcal/g)": ok & (dens > q[0]) & (dens <= q[1]),
                f"high ({q[1]:.2f}-{dens[ok].max():.2f} kcal/g)": ok & (dens > q[1])}
        r = {"n": int(ok.sum()), "energy_density_kcal_per_g": {"median": float(np.median(dens[ok])), "p5": float(np.quantile(dens[ok], 0.05)),
                                                                "p95": float(np.quantile(dens[ok], 0.95))}, "models": {}}
        for k in (SP, DN, DL):
            p = d["preds"][k][ok]; tt = t[ok]; e_g = np.abs(p - tt); e_rel = e_g / tt; e_kcal = e_g * dens[ok]
            r["models"][k] = {
                "mae_g": float(e_g.mean()), "median_relative_error_pct": float(np.median(e_rel) * 100),
                "within_20pct": float(np.mean(e_rel <= 0.2) * 100), "energy_mae_kcal": float(e_kcal.mean()),
                "energy_error_100kcal_or_more_pct": float(np.mean(e_kcal >= 100) * 100),
                "by_energy_density": {tn: {"n": int(m[ok].sum()), "mae_g": float(e_g[m[ok]].mean()),
                                           "median_relative_error_pct": float(np.median(e_rel[m[ok]]) * 100),
                                           "energy_mae_kcal": float(e_kcal[m[ok]].mean())} for tn, m in terc.items()}}
        res[name] = r
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        e = r["energy_density_kcal_per_g"]
        print(f"== {name}: n {r['n']}, energy density median {e['median']:.2f} kcal/g (5th-95th {e['p5']:.2f}-{e['p95']:.2f})")
        for k, m in r["models"].items():
            print(f"  {k:30s} MAE {m['mae_g']:5.1f} g  median rel {m['median_relative_error_pct']:4.1f}%  within 20% {m['within_20pct']:4.1f}%  "
                  f"energy MAE {m['energy_mae_kcal']:5.1f} kcal  >=100 kcal {m['energy_error_100kcal_or_more_pct']:4.1f}%")
            for tn, x in m["by_energy_density"].items():
                print(f"      {tn:28s} n {x['n']:4d}  MAE {x['mae_g']:5.1f} g  rel {x['median_relative_error_pct']:4.1f}%  energy {x['energy_mae_kcal']:6.1f} kcal")


if __name__ == "__main__":
    main()
