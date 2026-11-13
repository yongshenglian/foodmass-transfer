"""Score an external image set with a train_multidomain.py fold ensemble (image-only runs; side inputs zero).
Run from code/."""
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
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    cfg = json.loads((a.run_dir / "config.json").read_text())
    if cfg.get("qwen") == "True" or cfg.get("text_features") == "True":
        raise SystemExit("run uses side inputs; this scorer is for image-only runs")
    ft.MODEL = cfg["backbone"]; ft.POOL = ThreadPoolExecutor(a.workers); ft.PREPROC = cfg.get("preprocess", "stretch")
    s = pd.read_csv(a.sample); frame = pd.DataFrame({"dish_id": s.dish_id, "image": s.image})
    preds = []
    for f in sorted(a.run_dir.glob("fold*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net = Net(ck["n_side"]); net.load_state_dict(ck["state_dict"]); net.cuda()
        preds.append(predict(net, frame, np.zeros((len(frame), ck["n_side"]), np.float32), np.arange(len(frame)), "cuda"))
        del net; torch.cuda.empty_cache()
    m = np.mean(preds, 0)
    pd.DataFrame({"dish_id": frame.dish_id, "prediction": m, "mass": np.clip(np.exp(m), LO, HI)}).to_csv(a.output, index=False)
    print(f"scored {len(frame)} images -> {a.output}")


if __name__ == "__main__":
    main()
