"""Fine-tune DINOv2 on Nutrition5k reference dishes and/or the food-disjoint ASA24 training pool.

Validation only: 5-fold reference CV on Nutrition5k (same fold rule as finetune_dinov2_mass.py) and 5-fold CV on the
ASA24 pool grouped by food (a food never appears in both train and validation). Each fold model also scores the
Nutrition5k development cohort and the frozen ASA24 test sample; their predictions are written to files but no
test metric is printed (use evaluate_asa24_finals.py on pre-declared runs only).
Depth (volume proxy) exists for Nutrition5k only; its auxiliary loss is masked for ASA24 rows.
Run from code/ with DINOV2_PATH pointing at the backbone (base or large).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import finetune_dinov2_mass as ft
from build_nutrition5k_reference_bank import DATASET_ROOT, metadata
from finetune_dinov2_mass import EXP, HI, LO, Net, cohort, predict, side_features

V2 = Path("../experiments/qwen3vl_journal_v2_training")
A = Path("../experiments/asa24_external")


def qwen_log(frame_qwen):
    return np.log(np.clip(frame_qwen, 1, None)) - 5.0


def build(args):
    meta = metadata(DATASET_ROOT)
    ref_ids = [str(i) for i in np.load(EXP / "reference_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    dev_ids = [str(i) for i in np.load(EXP / "development_502_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    ref = cohort(EXP / "reference_train2253_qwen3vl_results.csv", ref_ids, meta)
    dev = cohort(EXP / "dinov2_gate_v1_full2253/development_predictions.csv", dev_ids, meta)
    vol = np.load(V2 / "reference_depth_volume.npy")
    img_col = "image_pad43" if args.pad43 else "image"
    pool = pd.read_csv(A / args.pool); test = pd.read_csv(A / "asa24_aerial_sample.csv")

    def frame(ids, images, true, qwen, domain, group, volume):
        f = pd.DataFrame({"dish_id": ids, "image": images, "true": np.asarray(true, float), "domain": domain, "group": group})
        f["qwen"] = qwen; f["volume"] = volume
        return f
    n5k = frame(ref.dish_id, ref.image, ref.true, ref.qwen.to_numpy(), "n5k", None, vol)
    asa = frame(pool.image_file, pool[img_col], pool.grams, np.nan, "asa24", pool.food_group, np.nan)
    devf = frame(dev.dish_id, dev.image, dev.true, dev.qwen.to_numpy(), "n5k", None, np.nan)
    testf = frame(test.dish_id, test[img_col], test.true_weight, np.nan, "asa24", None, np.nan)
    if args.qwen:  # Qwen FP8 estimates (blind) for ASA24; bf16 oracle-name estimates for Nutrition5k as in training
        for f, path in [(asa, A / "qwen_pool_blind_fp8_imputed.csv"), (testf, A / "qwen_blind_fp8_lenient_imputed.csv")]:
            q = pd.read_csv(path).drop_duplicates("dish_id").set_index("dish_id").reindex(f.dish_id)
            f["qwen"] = q.predicted_weight.to_numpy(float)
    fpb = fpbval = None
    if getattr(args, "fpb", False):  # Food Portion Benchmark: train split for training/CV (grouped by portion); val is a held-out test
        F = Path("../experiments/external_fpb")
        t = pd.read_csv(F / "fpb_train_1024.csv"); v = pd.read_csv(F / "fpb_val_eval.csv")
        fpb = frame(t.dish_id, t.image, t.true_weight, np.nan, "fpb", t.portion, np.nan)
        fpbval = frame(v.dish_id, v.image, v.true_weight, np.nan, "fpb", v.portion, np.nan)
    args._fpb = (fpb, fpbval)
    return n5k, asa, devf, testf


TEXT = {}


def side(f, args):
    if args.text_features and not TEXT:  # Qwen-named food text embeddings (bge-small), looked up by dish/image id
        z = np.load(A / "food_name_embeddings.npz", allow_pickle=True)
        TEXT.update(zip(z["dish_ids"].astype(str), z["features"]))
    width = 2 + (384 if args.text_features else 0)
    m = np.zeros((len(f), width), np.float32)
    if args.text_features:
        m[:, 2:] = np.stack([TEXT.get(str(d), np.zeros(384, np.float32)) for d in f.dish_id])
    if args.qwen:
        m[:, 0] = qwen_log(f.qwen.to_numpy(float))
    if args.domain_flag:
        m[:, 1] = (f.domain == "asa24").to_numpy(float)
    return m


def train_fold(train, args, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sd = side(train, args); y = np.log(train.true.to_numpy(float))
    vol = train.volume.to_numpy(float); has_v = np.isfinite(vol)
    mu, sdv = vol[has_v].mean(), vol[has_v].std()
    v = np.where(has_v, (vol - mu) / sdv, 0.0)
    w = np.where(train.domain == "asa24", args.asa_weight, 1.0)
    if args.mass_weight_power:  # tilt the log-space loss toward heavier dishes, closer to a grams MAE objective
        w = w * (train.true.to_numpy(float) / np.median(train.true)) ** args.mass_weight_power
    net = Net(sd.shape[1]).to("cuda")
    if args.grad_checkpoint:  # recompute activations in the backward pass: same results, less memory, slower
        net.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": args.backbone_lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.05)
    idx = np.arange(len(train)); steps = args.epochs * int(np.ceil(len(idx) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.head_lr], total_steps=steps, pct_start=0.1)
    for epoch in range(args.epochs):
        net.train(); total = 0
        for b, px, s in ft.batches(train, sd, idx, args.batch_size, True, rng):
            yt = torch.tensor(y[b], dtype=torch.float32, device="cuda")
            wt = torch.tensor(w[b], dtype=torch.float32, device="cuda")
            vt = torch.tensor(v[b], dtype=torch.float32, device="cuda"); mt = torch.tensor(has_v[b], dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr, pv = net(px.cuda(), s.cuda())
                loss = (wt * torch.nn.functional.smooth_l1_loss(pr.float(), yt, beta=0.1, reduction="none")).sum() / wt.sum()
                if args.aux_weight > 0 and mt.sum() > 0:
                    aux = torch.nn.functional.smooth_l1_loss(pv.float(), vt, reduction="none")
                    loss = loss + args.aux_weight * (aux * mt).sum() / mt.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); total += loss.item() * len(b)
        print(f"  epoch {epoch + 1}/{args.epochs} loss {total / len(idx):.4f}", flush=True)
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--domains", default="n5k,asa24", help="comma list from n5k, asa24, fpb")
    p.add_argument("--pool", default="asa24_train_pool.csv", help="ASA24 training pool CSV in experiments/asa24_external")
    p.add_argument("--pad43", action="store_true", help="use 4:3 white-padded ASA24 images instead of stretched")
    p.add_argument("--qwen", action="store_true", help="log Qwen estimate as a side input")
    p.add_argument("--domain-flag", action="store_true", help="input flag: 1 for ASA24 photos")
    p.add_argument("--fpb", action="store_true", help="add the FPB train split as a third domain (and score FPB val)")
    p.add_argument("--text-features", action="store_true", help="add a text embedding of Qwen's food names as side input")
    p.add_argument("--asa-weight", type=float, default=1.0)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--mass-weight-power", type=float, default=0.0, help="loss weight (mass/median)^p (0 = off)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--preprocess", choices=["stretch", "pad"], default="stretch", help="see finetune_dinov2_mass.PREPROC")
    p.add_argument("--grad-checkpoint", action="store_true", help="gradient checkpointing in the backbone (memory only)")
    args = p.parse_args()
    ft.PREPROC = args.preprocess
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "backbone": ft.MODEL}, indent=2))
    ft.POOL = ThreadPoolExecutor(args.workers)
    n5k, asa, devf, testf = build(args)
    use = set(args.domains.split(","))
    n5k_fold = np.random.default_rng(args.seed).permutation(np.arange(len(n5k)) % args.folds)
    foods = sorted(asa.group.unique()); perm = np.random.default_rng(args.seed + 1).permutation(len(foods))
    food_fold = {foods[i]: k % args.folds for k, i in enumerate(perm)}
    asa_fold = asa.group.map(food_fold).to_numpy()
    fpb, fpbval = args._fpb
    if fpb is not None:
        portions = sorted(fpb.group.unique()); pp = np.random.default_rng(args.seed + 2).permutation(len(portions))
        fpb_fold = fpb.group.map({portions[i]: k % args.folds for k, i in enumerate(pp)}).to_numpy()
    for f in range(args.folds):
        out = args.output_dir / f"fold{f}.npz"
        if out.is_file():
            continue
        parts = []
        if "n5k" in use: parts.append(n5k[n5k_fold != f])
        if "asa24" in use: parts.append(asa[asa_fold != f])
        if fpb is not None and "fpb" in use: parts.append(fpb[fpb_fold != f])
        train = pd.concat(parts, ignore_index=True)
        print(f"fold {f}: train {len(train)} ({', '.join(f'{d} {int((train.domain == d).sum())}' for d in ('n5k', 'asa24'))})", flush=True)
        net = train_fold(train, args, args.seed + f)
        res = {}
        evals = [("n5k_oof", n5k, n5k_fold == f), ("asa_oof", asa, asa_fold == f), ("dev", devf, None), ("test", testf, None)]
        if fpb is not None:
            evals += [("fpb_oof", fpb, fpb_fold == f), ("fpbval", fpbval, None)]
        for name, fr, mask in evals:
            sub = fr if mask is None else fr[mask].reset_index(drop=True)
            res[name] = predict(net, sub, side(sub, args), np.arange(len(sub)), "cuda")
            if mask is not None:
                res[name + "_index"] = np.flatnonzero(mask)
        e_n = np.mean(np.abs(np.clip(np.exp(res["n5k_oof"]), LO, HI) - n5k.true.to_numpy()[n5k_fold == f]))
        e_a = np.mean(np.abs(np.clip(np.exp(res["asa_oof"]), LO, HI) - asa.true.to_numpy()[asa_fold == f]))
        e_f = (f", FPB train OOF {np.mean(np.abs(np.clip(np.exp(res['fpb_oof']), LO, HI) - fpb.true.to_numpy()[fpb_fold == f])):.2f} g"
               if fpb is not None else "")
        print(f"fold {f} validation MAE: Nutrition5k OOF {e_n:.2f} g, ASA24 pool OOF {e_a:.2f} g{e_f}", flush=True)
        np.savez(out, **res)
        torch.save({"state_dict": net.state_dict(), "n_side": side(n5k.iloc[:1], args).shape[1], "args": vars(args)}, args.output_dir / f"fold{f}.pt")
        del net; torch.cuda.empty_cache()
    files = [args.output_dir / f"fold{f}.npz" for f in range(args.folds)]
    if all(x.is_file() for x in files):
        oof_n = np.empty(len(n5k)); oof_a = np.empty(len(asa)); dev, test = [], []
        for x in files:
            z = np.load(x)
            oof_n[z["n5k_oof_index"]] = z["n5k_oof"]; oof_a[z["asa_oof_index"]] = z["asa_oof"]
            dev.append(z["dev"]); test.append(z["test"])
        mass = lambda r: np.clip(np.exp(r), LO, HI)
        summary = {"validation_n5k_ref_cv_mae": float(np.mean(np.abs(mass(oof_n) - n5k.true))),
                   "validation_asa24_pool_cv_mae": float(np.mean(np.abs(mass(oof_a) - asa.true))),
                   "validation_asa24_pool_cv_median_ae": float(np.median(np.abs(mass(oof_a) - asa.true))),
                   "n5k_development_mae": float(np.mean(np.abs(mass(np.mean(dev, 0)) - devf.true))),
                   "asa24_test": "written to asa24_test_predictions.csv, not printed"}
        if fpb is not None:
            oof_f = np.empty(len(fpb)); fv = []
            for x in files:
                z = np.load(x); oof_f[z["fpb_oof_index"]] = z["fpb_oof"]; fv.append(z["fpbval"])
            summary["validation_fpb_train_cv_mae"] = float(np.mean(np.abs(mass(oof_f) - fpb.true)))
            summary["fpb_val"] = "written to fpb_val_predictions.csv, not printed"
            pd.DataFrame({"dish_id": fpbval.dish_id, "mass": mass(np.mean(fv, 0))}).to_csv(args.output_dir / "fpb_val_predictions.csv", index=False)
            pd.DataFrame({"dish_id": fpb.dish_id, "true": fpb.true, "oof": mass(oof_f), "portion": fpb.group}).to_csv(args.output_dir / "fpb_train_oof.csv", index=False)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame({"dish_id": testf.dish_id, "prediction": np.mean(test, 0), "mass": mass(np.mean(test, 0))}).to_csv(
            args.output_dir / "asa24_test_predictions.csv", index=False)
        pd.DataFrame({"dish_id": asa.dish_id, "true": asa.true, "oof": mass(oof_a), "food": asa.group}).to_csv(args.output_dir / "asa24_pool_oof.csv", index=False)
        pd.DataFrame({"dish_id": devf.dish_id, "true": devf.true, "mass": mass(np.mean(dev, 0))}).to_csv(args.output_dir / "n5k_development_predictions.csv", index=False)
        print(summary)


if __name__ == "__main__":
    main()
