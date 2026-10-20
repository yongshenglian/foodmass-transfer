"""Test-time flip averaging for a train_multidomain.py run, measured on validation only (ASA24 pool OOF + Nutrition5k OOF).

Rebuilds each fold's held-out rows exactly as in training, scores them with that fold's checkpoint under 4 flips
(none, h, v, hv), and reports validation MAE with and without averaging. Also writes TTA predictions for the ASA24
test sample (5-fold mean) to asa24_test_predictions_tta.csv without printing any test metric. Run from code/.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import finetune_dinov2_mass as ft
from finetune_dinov2_mass import HI, LO, Net
from train_multidomain import build, side


def predict_flips(net, frame, sd, bs=32):
    net.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, px, s in ft.batches(frame, sd, np.arange(len(frame)), bs, False, np.random.default_rng(0)):
            px, s = px.cuda(), s.cuda()
            views = [px, px.flip(-1), px.flip(-2), px.flip(-1).flip(-2)]
            out.append(np.stack([net(v, s)[0].float().cpu().numpy() for v in views], 1))
    return np.concatenate(out)  # (n, 4) log predictions


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    cfg = json.loads((a.run_dir / "config.json").read_text())
    args = argparse.Namespace(pool=cfg.get("pool", "asa24_train_pool.csv"), pad43=cfg["pad43"] == "True", qwen=cfg["qwen"] == "True", domain_flag=cfg["domain_flag"] == "True", text_features=cfg.get("text_features") == "True")
    ft.MODEL = cfg["backbone"]; ft.POOL = ThreadPoolExecutor(a.workers)
    n5k, asa, devf, testf = build(args)
    folds, seed = int(cfg["folds"]), int(cfg["seed"])
    n5k_fold = np.random.default_rng(seed).permutation(np.arange(len(n5k)) % folds)
    foods = sorted(asa.group.unique()); perm = np.random.default_rng(seed + 1).permutation(len(foods))
    asa_fold = asa.group.map({foods[i]: k % folds for k, i in enumerate(perm)}).to_numpy()
    oof_a = np.empty((len(asa), 4)); oof_n = np.empty((len(n5k), 4)); test = []
    for f in range(folds):
        ck = torch.load(a.run_dir / f"fold{f}.pt", map_location="cpu", weights_only=False)
        net = Net(ck["n_side"]); net.load_state_dict(ck["state_dict"]); net.cuda()
        for fr, fold, out in [(asa, asa_fold, oof_a), (n5k, n5k_fold, oof_n)]:
            sub = fr[fold == f].reset_index(drop=True)
            out[fold == f] = predict_flips(net, sub, side(sub, args))
        test.append(predict_flips(net, testf, side(testf, args)))
        del net; torch.cuda.empty_cache(); print(f"fold {f} done", flush=True)
    m = lambda r: np.clip(np.exp(r), LO, HI)
    res = {}
    for name, oof, t in [("asa24_pool", oof_a, asa.true.to_numpy()), ("n5k_ref", oof_n, n5k.true.to_numpy())]:
        res[name] = {"no_tta": float(np.mean(np.abs(m(oof[:, 0]) - t))), "flip4_tta": float(np.mean(np.abs(m(oof.mean(1)) - t)))}
    print(json.dumps(res, indent=1))
    (a.run_dir / "tta_validation.json").write_text(json.dumps(res, indent=2))
    tm = np.mean([x.mean(1) for x in test], 0)
    pd.DataFrame({"dish_id": testf.dish_id, "prediction": tm, "mass": m(tm)}).to_csv(a.run_dir / "asa24_test_predictions_tta.csv", index=False)


if __name__ == "__main__":
    main()
