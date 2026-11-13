"""Score external images (no Nutrition5k metadata) with a saved stage B fold ensemble.

Side features follow finetune_dinov2_mass.side_features: [log Qwen - 5, confidence/100 - 0.7, log count - 1.5, multi-hot].
Columns zeroed in the run's config (--zero-side / --no-ingredients) are zeroed here too; without --qwen-csv
the Qwen columns are zero, which is only valid for runs trained with them zeroed (image-only).
External foods are outside the Nutrition5k ingredient vocabulary, so the multi-hot is always zero.
Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import finetune_dinov2_mass as ft
from finetune_dinov2_mass import HI, LO, Net, predict

NUTR = ["calories", "fat", "carb", "protein"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--sample", type=Path, required=True, help="CSV with dish_id, image")
    p.add_argument("--qwen-csv", type=Path, help="Qwen outputs (dish_id, predicted_weight, confidence, ingredient_count)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    config = json.loads((args.run_dir / "config.json").read_text())
    zero = set(filter(None, config.get("zero_side", "").split(",")))
    ft.PREPROC = config.get("preprocess", "stretch")
    if args.qwen_csv is None and not {"qwen", "conf"} <= zero:
        raise SystemExit("this run uses Qwen inputs; pass --qwen-csv")
    from concurrent.futures import ThreadPoolExecutor
    ft.POOL = ThreadPoolExecutor(args.workers)

    s = pd.read_csv(args.sample)
    frame = pd.DataFrame({"dish_id": s.dish_id, "image": s.image})
    folds = sorted(args.run_dir.glob("fold*.pt"))
    ck0 = torch.load(folds[0], map_location="cpu", weights_only=False)
    side = np.zeros((len(frame), len(ck0["vocab"]) + 3), np.float32)
    if args.qwen_csv is not None:
        q = pd.read_csv(args.qwen_csv).drop_duplicates("dish_id").set_index("dish_id").reindex(frame.dish_id)
        side[:, 0] = np.log(np.clip(q.predicted_weight.to_numpy(float), 1, None)) - 5.0
        side[:, 1] = q.confidence.fillna(65).to_numpy(float) / 100 - 0.7
        side[:, 2] = np.log(np.clip(q.ingredient_count.fillna(1).to_numpy(float), 1, None)) - 1.5
    for name, cols in {"qwen": [0], "conf": [1], "count": [2]}.items():
        if name in zero:
            side[:, cols] = 0
    preds, nuts = [], []
    for path in folds:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = Net(side.shape[1], ck.get("n_nutr", 0)); net.load_state_dict(ck["state_dict"]); net.cuda()
        if ck.get("n_nutr", 0):
            pr, nu = predict(net, frame, side, np.arange(len(frame)), "cuda", with_nutrients=True); nuts.append(nu)
        else:
            pr = predict(net, frame, side, np.arange(len(frame)), "cuda")
        preds.append(pr); del net; torch.cuda.empty_cache()
    pred = np.mean(preds, 0)
    qwen = np.exp(side[:, 0] + 5.0)
    mass = np.clip(qwen * np.exp(pred) if ck0["target"] == "residual" else np.exp(pred), LO, HI)
    out = pd.DataFrame({"dish_id": frame.dish_id, "prediction": pred, "mass": mass})
    if nuts:
        out[[f"pred_{k}" for k in NUTR]] = np.expm1(np.mean(nuts, 0))
    out.to_csv(args.output, index=False)
    print(f"scored {len(out)} images with {len(folds)} fold models -> {args.output}")


if __name__ == "__main__":
    main()
