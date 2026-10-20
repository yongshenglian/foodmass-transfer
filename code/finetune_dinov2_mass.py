"""Stage B: fine-tune DINOv2 to predict log mass or Qwen's log residual (RGB-only inference).

Training data are the 2,253 reference dishes only. Optional depth-privileged
auxiliary loss: an RGB-only head also regresses a depth-derived food-volume proxy,
so depth shapes the representation but is never an input. Five fold models are
trained with reference cross-validation (out-of-fold predictions for selection);
the development cohort is scored with the 5-model ensemble. The confirmation
cohort is never loaded. Run from code/ in an environment with torch + transformers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel

from build_nutrition5k_reference_bank import DATASET_ROOT, metadata

EXP = Path("../experiments/qwen3vl_journal_v1")
MODEL = os.environ.get("DINOV2_PATH", "/mnt/sdb/models/dinov2-base")
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
SIZE = (448, 336)  # width, height; multiples of the 14-pixel patch, keeps 4:3 aspect
LO, HI = 10.0, 1500.0


def depth_volume_proxy(path):
    """Food volume proxy from raw depth: height above the table plane summed over pixels."""
    d = np.asarray(Image.open(path), dtype=np.float64) / 1e4  # metres
    valid = d > 0
    border = np.concatenate([d[:20].ravel(), d[-20:].ravel(), d[:, :20].ravel(), d[:, -20:].ravel()])
    table = np.median(border[border > 0])
    height = np.clip(table - d, 0, 0.15) * valid
    height = np.where(height >= 0.02, height - 0.02, 0)  # drop the plate itself (~2 cm); best of 5-30 mm on a 200-dish sample
    return float(np.log1p(height.sum() * 1e3))


def load_rgb(path, train, rng, aug=None):
    im = Image.open(path).convert("RGB").resize(SIZE, Image.BICUBIC)
    x = np.asarray(im, dtype=np.float32) / 255
    if aug is not None:  # pre-drawn (flip_h, flip_v, gain, offset), used by the threaded loader
        if aug[0]: x = x[:, ::-1]
        if aug[1]: x = x[::-1]
        x = np.clip(x * aug[2] + aug[3], 0, 1)
    elif train:
        if rng.random() < 0.5: x = x[:, ::-1]
        if rng.random() < 0.5: x = x[::-1]
        x = np.clip(x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)  # mild photometric jitter, no crops (scale is signal)
    x = (x - MEAN) / STD
    return torch.tensor(x.transpose(2, 0, 1).copy(), dtype=torch.float32)


CNN_TYPES = ("resnet", "convnext", "regnet")


class Net(torch.nn.Module):
    def __init__(self, n_side, n_nutr=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL, local_files_only=True)
        # transformer backbones (DINOv2): [CLS] + mean patch token; CNN baselines (e.g. ResNet-50): mean- + max-pooled map
        self.cnn = getattr(self.backbone.config, "model_type", "") in CNN_TYPES
        width = (self.backbone.config.hidden_sizes[-1] if self.cnn else self.backbone.config.hidden_size) * 2
        self.side = torch.nn.Sequential(torch.nn.Linear(n_side, 128), torch.nn.GELU())
        self.trunk = torch.nn.Sequential(torch.nn.Linear(width + 128, 512), torch.nn.GELU(), torch.nn.Dropout(0.2))
        self.residual = torch.nn.Linear(512, 1)
        self.volume = torch.nn.Linear(width, 1)
        if n_nutr:  # optional direct nutrient head (log1p calories, fat, carbs, protein); created last so n_nutr=0 is unchanged
            self.nutrients = torch.nn.Linear(512, n_nutr)

    def forward(self, pixels, side):
        h = self.backbone(pixel_values=pixels).last_hidden_state
        if self.cnn:  # (B, C, H, W) feature map
            feat = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        else:
            feat = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        z = self.trunk(torch.cat([feat, self.side(side)], dim=1))
        if hasattr(self, "nutrients"):
            return self.residual(z).squeeze(1), self.volume(feat).squeeze(1), self.nutrients(z)
        return self.residual(z).squeeze(1), self.volume(feat).squeeze(1)


def cohort(qwen_csv, ids_from, meta, vocab=None):
    q = pd.read_csv(qwen_csv).drop_duplicates("dish_id")
    ids = ids_from
    q = q.set_index("dish_id").loc[ids]
    root = DATASET_ROOT / "imagery" / "realsense_overhead"
    frame = pd.DataFrame({
        "dish_id": ids, "true": q["true_weight"].to_numpy(float),
        "qwen": np.clip(q["predicted_weight"].to_numpy(float), 1, None),
        "confidence": q["confidence"].fillna(65).to_numpy(float),
        "n_ingr": q["ingredient_count"].to_numpy(float),
        "image": [str(root / i / "rgb.png") for i in ids],
        "depth": [str(root / i / "depth_raw.png") for i in ids],
        "ingr": [[x["id"] for x in meta[i]["ingredients"]] for i in ids],
    })
    return frame


def side_features(frame, vocab):
    index = {v: j for j, v in enumerate(vocab)}
    m = np.zeros((len(frame), len(vocab) + 3), np.float32)
    m[:, 0] = np.log(frame.qwen) - 5.0
    m[:, 1] = frame.confidence / 100 - 0.7
    m[:, 2] = np.log(frame.n_ingr) - 1.5
    for r, group in enumerate(frame.ingr):
        for g in group:
            if g in index:
                m[r, 3 + index[g]] = 1
    return m


POOL = None  # set by --workers; None keeps the original single-threaded loader (bit-identical RNG stream)


def batches(frame, side, idx, bs, train, rng):
    order = rng.permutation(idx) if train else np.asarray(idx)
    batch_list = [order[s:s + bs] for s in range(0, len(order), bs)]
    if POOL is None:
        for b in batch_list:
            pixels = torch.stack([load_rgb(frame.image.iloc[i], train, rng) for i in b])
            yield b, pixels, torch.tensor(side[b])
        return

    def load(b):
        augs = [(rng.random() < 0.5, rng.random() < 0.5, rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05)) if train else None
                for _ in b]  # drawn in the main thread, in order, so runs are reproducible
        return [POOL.submit(load_rgb, frame.image.iloc[i], train, None, a) for i, a in zip(b, augs)]

    pending = load(batch_list[0]) if batch_list else None
    for k, b in enumerate(batch_list):
        futures = pending
        pending = load(batch_list[k + 1]) if k + 1 < len(batch_list) else None  # prefetch the next batch
        yield b, torch.stack([f.result() for f in futures]), torch.tensor(side[b])


def predict(net, frame, side, idx, device, bs=32, with_nutrients=False):
    net.eval(); out = []; nut = []
    with torch.no_grad(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        for b, px, sd in batches(frame, side, idx, bs, False, np.random.default_rng(0)):
            res = net(px.to(device), sd.to(device))
            out.append(res[0].float().cpu().numpy())
            if with_nutrients:
                nut.append(res[2].float().cpu().numpy())
    return (np.concatenate(out), np.concatenate(nut)) if with_nutrients else np.concatenate(out)


def train_one(frame, side, target, volume, tr, device, args, seed, nutr=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(side.shape[1], 0 if nutr is None else nutr.shape[1]).to(device)
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone")]
    frozen = getattr(args, "freeze_backbone", False)
    if frozen:  # frozen-backbone baseline: only the head (and depth head) is trained
        net.backbone.requires_grad_(False)
        groups, max_lr = [{"params": head, "lr": args.head_lr}], [args.head_lr]
    else:
        groups = [{"params": net.backbone.parameters(), "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}]
        max_lr = [args.backbone_lr, args.head_lr]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=0.1)
    vol_mu, vol_sd = volume[tr].mean(), volume[tr].std()
    for epoch in range(args.epochs):
        net.train(); total = 0
        if frozen:
            net.backbone.eval()  # no dropout or batch-norm updates in a frozen backbone
        for b, px, sd in batches(frame, side, tr, args.batch_size, True, rng):
            y = torch.tensor(target[b], dtype=torch.float32, device=device)
            v = torch.tensor((volume[b] - vol_mu) / vol_sd, dtype=torch.float32, device=device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
                res = net(px.to(device), sd.to(device)); pr, pv = res[0], res[1]
                loss = torch.nn.functional.smooth_l1_loss(pr.float(), y, beta=0.1)
                if args.aux_weight > 0:
                    loss = loss + args.aux_weight * torch.nn.functional.smooth_l1_loss(pv.float(), v)
                if nutr is not None:
                    yn = torch.tensor(nutr[b], dtype=torch.float32, device=device)
                    loss = loss + args.nutrient_weight * torch.nn.functional.smooth_l1_loss(res[2].float(), yn, beta=0.1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); total += loss.item() * len(b)
        print(f"  epoch {epoch + 1}/{args.epochs} loss {total / len(tr):.4f}", flush=True)
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--aux-weight", type=float, default=0.0)
    p.add_argument("--target", choices=["residual", "log_mass"], default="log_mass",
                   help="log_mass won in stage A (frozen features): 34.4 g vs 48.9 g on development")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-folds", type=str, default="")
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="tiny CPU-sized run to check the pipeline end to end")
    p.add_argument("--no-ingredients", action="store_true",
                   help="ablation: zero the oracle ingredient multi-hot in the side features")
    p.add_argument("--zero-side", type=str, default="",
                   help="comma list of side inputs to zero: qwen,conf,count,ingr (all four = image-only model)")
    p.add_argument("--nutrient-weight", type=float, default=0.0,
                   help="add a direct head for log1p calories/fat/carbs/protein with this loss weight (0 = off)")
    p.add_argument("--workers", type=int, default=0, help="image-loading threads (0 = original single-threaded loader)")
    p.add_argument("--freeze-backbone", action="store_true", help="train only the head on frozen backbone features (baseline)")
    p.add_argument("--save-checkpoints", action="store_true",
                   help="also write foldN.pt (model state_dict + vocab) so the fold ensemble can score new cohorts")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "backbone": MODEL}, indent=2))

    meta = metadata(DATASET_ROOT)
    ref_ids = [str(i) for i in np.load(EXP / "reference_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    dev_ids = [str(i) for i in np.load(EXP / "development_502_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    ref = cohort(EXP / "reference_train2253_qwen3vl_results.csv", ref_ids, meta)
    dev = cohort(EXP / "dinov2_gate_v1_full2253/development_predictions.csv", dev_ids, meta)
    counts = pd.Series([g for grp in ref.ingr for g in set(grp)]).value_counts()
    vocab = sorted(counts[counts >= 5].index)
    ref_side, dev_side = side_features(ref, vocab), side_features(dev, vocab)
    if args.no_ingredients:
        ref_side[:, 3:] = 0; dev_side[:, 3:] = 0
    columns = {"qwen": [0], "conf": [1], "count": [2], "ingr": list(range(3, ref_side.shape[1]))}
    for name in filter(None, args.zero_side.split(",")):
        ref_side[:, columns[name]] = 0; dev_side[:, columns[name]] = 0
    if args.workers > 0:
        global POOL
        POOL = ThreadPoolExecutor(args.workers)
    target = (np.log(ref.true / ref.qwen) if args.target == "residual" else np.log(ref.true)).to_numpy()

    def to_mass(qwen, pred):
        return np.clip((qwen * np.exp(pred)) if args.target == "residual" else np.exp(pred), LO, HI)

    vol_cache = args.output_dir.parent / "reference_depth_volume.npy"
    if vol_cache.is_file():
        volume = np.load(vol_cache)
    elif args.smoke:
        volume = np.array([depth_volume_proxy(x) for x in ref.depth[:40]] + [0.0] * (len(ref) - 40))
    else:
        volume = np.array([depth_volume_proxy(x) for x in ref.depth]); np.save(vol_cache, volume)
    if not args.smoke:
        print("volume proxy vs log true mass corr:", np.corrcoef(volume, np.log(ref.true))[0, 1], flush=True)

    nutr_ref = nutr_dev = None
    if args.nutrient_weight > 0:  # dish totals: calories, fat, carbs, protein (dish_metadata_cafe*.csv columns 1, 3-5)
        totals = {}
        for cafe in (1, 2):
            for line in (DATASET_ROOT / "metadata" / f"dish_metadata_cafe{cafe}.csv").read_text().splitlines():
                r = line.split(","); totals[r[0]] = [float(r[1]), float(r[3]), float(r[4]), float(r[5])]
        nutr_ref = np.array([totals[d] for d in ref.dish_id]); nutr_dev = np.array([totals[d] for d in dev.dish_id])
    if args.smoke:
        ref, dev = ref.iloc[:40].reset_index(drop=True), dev.iloc[:8].reset_index(drop=True)
        ref_side, dev_side, target, volume = ref_side[:40], dev_side[:8], target[:40], volume[:40]
        if nutr_ref is not None:
            nutr_ref, nutr_dev = nutr_ref[:40], nutr_dev[:8]
    fold = np.random.default_rng(args.seed).permutation(np.arange(len(ref)) % args.folds)
    folds = [int(f) for f in args.only_folds.split(",")] if args.only_folds else range(args.folds)
    for f in folds:
        out = args.output_dir / f"fold{f}.npz"
        if out.is_file():
            continue
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        print(f"fold {f}: train {len(tr)} test {len(te)}", flush=True)
        net = train_one(ref, ref_side, target, volume, tr, args.device, args, args.seed + f,
                        None if nutr_ref is None else np.log1p(nutr_ref))
        extra = {}
        if nutr_ref is None:
            oof = predict(net, ref, ref_side, te, args.device)
            dev_pred = predict(net, dev, dev_side, np.arange(len(dev)), args.device)
        else:
            oof, extra["oof_nutrients"] = predict(net, ref, ref_side, te, args.device, with_nutrients=True)
            dev_pred, extra["dev_nutrients"] = predict(net, dev, dev_side, np.arange(len(dev)), args.device, with_nutrients=True)
        mass = to_mass(ref.qwen.to_numpy()[te], oof)
        print(f"fold {f} OOF MAE {np.mean(np.abs(mass - ref.true.to_numpy()[te])):.2f}", flush=True)
        if args.save_checkpoints:
            torch.save({"state_dict": net.state_dict(), "vocab": vocab, "target": args.target,
                        "n_nutr": 0 if nutr_ref is None else nutr_ref.shape[1]},
                       args.output_dir / f"fold{f}.pt")
        np.savez(out, test_index=te, oof_residual=oof, dev_residual=dev_pred, **extra)
        del net; torch.cuda.empty_cache()

    files = [args.output_dir / f"fold{f}.npz" for f in range(args.folds)]
    if all(x.is_file() for x in files):
        oof = np.empty(len(ref)); dev_res = []
        for x in files:
            z = np.load(x); oof[z["test_index"]] = z["oof_residual"]; dev_res.append(z["dev_residual"])
        ref_mass = to_mass(ref.qwen.to_numpy(), oof)
        dev_mass = to_mass(dev.qwen.to_numpy(), np.mean(dev_res, 0))
        summary = {"reference_cv_mae_g": float(np.mean(np.abs(ref_mass - ref.true))),
                   "development_mae_g": float(np.mean(np.abs(dev_mass - dev.true))),
                   "confirmation_cohort_used": False}
        if nutr_ref is not None:
            oof_n = np.empty_like(nutr_ref); dev_n = []
            for x in files:
                z = np.load(x); oof_n[z["test_index"]] = np.expm1(z["oof_nutrients"]); dev_n.append(z["dev_nutrients"])
            dev_n = np.expm1(np.mean(dev_n, 0))
            for j, k in enumerate(["calories", "fat", "carb", "protein"]):
                summary[f"reference_cv_{k}_mae"] = float(np.mean(np.abs(oof_n[:, j] - nutr_ref[:, j])))
                summary[f"development_{k}_mae"] = float(np.mean(np.abs(dev_n[:, j] - nutr_dev[:, j])))
            pd.DataFrame(np.c_[nutr_dev, dev_n], index=dev.dish_id,
                         columns=[f"{w}_{k}" for w in ("true", "pred") for k in ("calories", "fat", "carb", "protein")]).to_csv(
                args.output_dir / "development_nutrients.csv")
            pd.DataFrame(np.c_[nutr_ref, oof_n], index=ref.dish_id,
                         columns=[f"{w}_{k}" for w in ("true", "oof") for k in ("calories", "fat", "carb", "protein")]).to_csv(
                args.output_dir / "reference_oof_nutrients.csv")
        print(summary)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame({"dish_id": ref.dish_id, "true": ref.true, "qwen": ref.qwen, "oof_prediction": oof}).to_csv(
            args.output_dir / "reference_oof.csv", index=False)
        pd.DataFrame({"dish_id": dev.dish_id, "true": dev.true, "qwen": dev.qwen,
                      "prediction": np.mean(dev_res, 0), "mass": dev_mass}).to_csv(
            args.output_dir / "development_predictions.csv", index=False)


if __name__ == "__main__":
    main()
