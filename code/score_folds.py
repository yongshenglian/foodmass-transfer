"""Score each fold model of a saved five-fold DINOv2 ensemble separately on a photograph set (photograph-only runs).
Writes one column per fold (grams, limited to 10-1,500 g) and the ensemble (geometric mean of the folds, limited),
exactly as the ensemble predictions in the paper were made. Works for finetune_dinov2_mass.py runs (stage_b_*) and
train_multidomain.py runs (joint_*). Run from code/ on the GPU workstation."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import finetune_dinov2_mass as ft
from finetune_dinov2_mass import HI, LO, Net, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--sample", type=Path, required=True, help="CSV with dish_id and image")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    cfg = json.loads((a.run_dir / "config.json").read_text())
    if "backbone" in cfg:  # train_multidomain.py runs record their backbone; stage_b runs use DINOV2_PATH
        ft.MODEL = cfg["backbone"]
    ft.POOL = ThreadPoolExecutor(a.workers)
    s = pd.read_csv(a.sample); frame = pd.DataFrame({"dish_id": s.dish_id, "image": s.image})
    out = pd.DataFrame({"dish_id": frame.dish_id}); logs = []
    folds = sorted(a.run_dir.glob("fold*.pt"))
    if len(folds) != 5:
        raise SystemExit(f"expected 5 fold checkpoints in {a.run_dir}, found {len(folds)}")
    for path in folds:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        n_side = ck["n_side"] if "n_side" in ck else len(ck["vocab"]) + 3
        net = Net(n_side, ck.get("n_nutr", 0)); net.load_state_dict(ck["state_dict"]); net.cuda()
        lp = predict(net, frame, np.zeros((len(frame), n_side), np.float32), np.arange(len(frame)), "cuda")
        if ck.get("target", "log_mass") != "log_mass":
            raise SystemExit("photograph-only log-mass runs only")
        logs.append(lp); out[path.stem] = np.clip(np.exp(lp), LO, HI)
        del net; torch.cuda.empty_cache()
    out["ensemble"] = np.clip(np.exp(np.mean(logs, 0)), LO, HI)
    a.output.parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.output, index=False)
    print(f"{len(out)} photographs, {len(folds)} folds -> {a.output}")


if __name__ == "__main__":
    main()
