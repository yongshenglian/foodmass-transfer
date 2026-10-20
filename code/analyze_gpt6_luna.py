"""gpt-6-luna versus Qwen3-VL-32B, single prompt, on every set Qwen was scored on (experiments/gpt6_luna/PLAN.md). Run from code/."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); G = E / "gpt6_luna"; B = 10000
N5K_MED = float(pd.read_csv(E / "qwen3vl_journal_v1/reference_train2253_qwen3vl_results.csv").true_weight.median())
FPB_MED = float(pd.read_csv(E / "external_fpb/fpb_train_1024.csv").true_weight.median())
rank = lambda x: pd.Series(x).rank().to_numpy()


def estimates(f, ids, fallback):
    q = pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id").reindex(ids); out, n_bad = [], 0
    for r, v, pw in zip(q.raw_response.fillna(""), q.format_valid.astype("boolean").fillna(False), q.predicted_weight):
        if v and pw > 0:
            out.append(pw); continue
        g = [float(x) for x in re.findall(r"^(?!\s*confidence).*?:\s*([\d.]+)\s*g\s*$", r, flags=re.M | re.I)]
        n_bad += 1; out.append(sum(g) if g and sum(g) > 0 else fallback)
    return np.array(out, float), n_bad


def metrics(p, t, series=None):
    r = {"mae": float(np.abs(p - t).mean()), "wape": float(np.abs(p - t).sum() / t.sum()), "scale_ratio": float(np.exp(np.mean(np.log(p / t)))),
         "spearman": float(np.corrcoef(rank(p), rank(t))[0, 1]), "within_20pct": float((np.abs(p - t) <= 0.2 * t).mean())}
    if series is not None:
        rho = [np.corrcoef(rank(p[series == s]), rank(t[series == s]))[0, 1] for s in np.unique(series) if (series == s).sum() >= 3 and np.ptp(t[series == s]) > 0]
        r["within_series_spearman"] = float(np.mean(np.nan_to_num(rho)))
    return r


def main():
    sets = {  # name: sample, Qwen file, cluster column, fallback, within-series column
        "Nutrition5k validation": (E / "qwen3vl_journal_v2_training/qwen_local/development_502_sample.csv", E / "n5k_single_prompt/qwen_blind_val_fp8.csv", None, N5K_MED, None, "n5k_val"),
        "ASA24 test": (E / "asa24_external/asa24_aerial_sample.csv", E / "asa24_external/qwen_blind_fp8.csv", "series", N5K_MED, "series", "asa24_test"),
        "ECUSTFD top": (E / "external_ecustfd/ecustfd_top_sample.csv", E / "external_ecustfd/qwen_blind_fp8.csv", "id", N5K_MED, None, "ecustfd_top"),
        "ECUSTFD side": (E / "ecustfd_side_view/ecustfd_side_sample.csv", E / "ecustfd_side_view/qwen_blind_fp8.csv", "id", N5K_MED, None, "ecustfd_side"),
        "FPB test (all)": (E / "external_fpb/fpb_val_eval.csv", E / "external_fpb/qwen_blind_fp8.csv", "portion", FPB_MED, None, "fpb_test"),
    }
    train_portions = set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion); res = {}; rng = np.random.default_rng(0)
    for name, (sample, qf, cl, fb, ser, tag) in sets.items():
        s = pd.read_csv(sample).drop_duplicates("dish_id").set_index("dish_id"); gf = G / f"gpt6luna_{tag}.csv"
        if not gf.is_file():
            print(f"{name}: no gpt-6-luna file yet"); continue
        q, qbad = estimates(qf, s.index, fb); g, gbad = estimates(gf, s.index, fb)
        subsets = {name: np.ones(len(s), bool)}
        if tag == "fpb_test":
            subsets = {"FPB test (2,034 signature-disjoint)": ~s.portion.isin(train_portions).to_numpy(), name: subsets[name]}
        for sub, m in subsets.items():
            t = s.true_weight.to_numpy(float)[m]; series = s[ser].to_numpy()[m] if ser else None
            clus = s[cl].to_numpy()[m] if cl else np.arange(m.sum())
            d = pd.DataFrame({"d": np.abs(g[m] - t) - np.abs(q[m] - t), "n": 1, "c": clus}).groupby("c").sum()
            i = rng.integers(0, len(d), (B, len(d))); bs = d.d.to_numpy()[i].sum(1) / d.n.to_numpy()[i].sum(1)
            res[sub] = {"n": int(m.sum()), "invalid_imputed": {"qwen": int(qbad), "gpt6_luna": int(gbad)},
                        "qwen": metrics(q[m], t, series), "gpt6_luna": metrics(g[m], t, series),
                        "mae_gpt_minus_qwen": [float(d.d.sum() / d.n.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            r = res[sub]; print(f"== {sub} (n={r['n']}; invalid Qwen {qbad}, GPT {gbad})")
            for k in ("qwen", "gpt6_luna"):
                x = r[k]; w = f"  within-series rho {x['within_series_spearman']:.2f}" if "within_series_spearman" in x else ""
                print(f"  {k:10s} MAE {x['mae']:6.1f}  WAPE {100 * x['wape']:5.1f}%  scale {x['scale_ratio']:.2f}  rho {x['spearman']:.2f}  within 20% {100 * x['within_20pct']:5.1f}%{w}")
            print("  GPT minus Qwen MAE {:+.1f} [{:+.1f}, {:+.1f}]".format(*r["mae_gpt_minus_qwen"]))
    tok = pd.concat([pd.read_csv(f) for f in G.glob("gpt6luna_*.csv")])
    res["usage"] = {"calls": int(len(tok)), "prompt_tokens": int(tok.prompt_tokens.sum()), "completion_tokens": int(tok.completion_tokens.sum()),
                    "cost_usd_standard": float(tok.prompt_tokens.sum() * 0.10e-6 + tok.completion_tokens.sum() * 0.50e-6)}
    print(res["usage"]); (G / "gpt6_luna.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
