"""Smoke test for the aspect-ratio control (experiments/aspect_control/PLAN.md), on one GPU.

Loads a few photographs of every dataset under both preprocessing rules, then trains 6 steps of batch 16 at each
input size for DINOv2-base and -large, with and without gradient checkpointing, and prints peak memory and seconds
per step. Writes logs/smoke_examples.png. Run from code/ with DINOV2_BASE and DINOV2_LARGE set.
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import finetune_dinov2_mass as ft

E = Path("../experiments"); OUT = E / "aspect_control"
SETS = {"asa24": (E / "asa24_external/asa24_aerial_sample.csv", "image"), "fpb": (E / "external_fpb/fpb_val_eval.csv", "image"),
        "fpb_train": (E / "external_fpb/fpb_train_1024.csv", "image"), "ecustfd": (E / "external_ecustfd/ecustfd_top_sample.csv", "image"),
        "ecustfd_side": (E / "ecustfd_side_view/ecustfd_side_sample.csv", "image"),
        "asa24_pool": (E / "asa24_external/asa24_train_pool_consensus.csv", "image")}


def main():
    tiles = []
    n5k = Path(os.environ["N5K_ROOT"]) / "imagery/realsense_overhead"
    paths = [str(next(n5k.glob("dish_*/rgb.png")))]
    for name, (f, col) in SETS.items():
        d = pd.read_csv(f); paths += list(d[col].sample(2, random_state=0))
    for p in paths:
        for rule in ("stretch", "pad"):
            ft.PREPROC = rule; x = ft.load_rgb(p, False, None)
            assert torch.isfinite(x).all() and x.shape == ((3, 336, 448) if rule == "stretch" else (3, 448, 448)), (p, rule, x.shape)
        tiles.append(ft.to_input(Image.open(p).convert("RGB")).resize((224, 224)))
    sheet = Image.new("RGB", (224 * len(tiles), 224), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (224 * i, 0))
    (OUT / "logs").mkdir(parents=True, exist_ok=True); sheet.save(OUT / "logs/smoke_examples.png")
    print(f"loaded {len(paths)} photographs under both rules; examples in logs/smoke_examples.png", flush=True)

    ok = True
    for backbone in ("DINOV2_BASE", "DINOV2_LARGE"):
        ft.MODEL = os.environ[backbone]
        for shape, ckpt in (((336, 448), False), ((448, 448), True), ((448, 448), False)):
            r = trial(shape, ckpt); gc.collect(); torch.cuda.empty_cache()
            label = f"{backbone[7:].lower():5s} {shape[0]}x{shape[1]} checkpointing {str(ckpt):5s}"
            if r is None:
                print(f"{label} out of memory", flush=True)
                ok = ok and not (backbone == "DINOV2_LARGE" and ckpt) and not (backbone == "DINOV2_BASE" and shape == (448, 448) and not ckpt)
            else:
                print(f"{label} peak {r[0]:5.1f} GiB, {r[1]:.2f} s per step of 16", flush=True)
    sys.exit(0 if ok else 1)


def trial(shape, ckpt):
    """6 training steps of batch 16; peak GiB and seconds per step, or None when out of memory. Memory is freed on return."""
    torch.cuda.reset_peak_memory_stats()
    try:
        net = ft.Net(2).cuda(); net.train()  # from_pretrained returns eval mode, where checkpointing is inactive
        if ckpt:
            net.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt = torch.optim.AdamW(net.parameters(), lr=1e-5); t0 = None
        for step in range(6):
            if step == 2:
                torch.cuda.synchronize(); t0 = time.time()
            px = torch.randn(16, 3, *shape, device="cuda"); sd = torch.zeros(16, 2, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr, pv = net(px, sd); loss = pr.float().pow(2).mean() + pv.float().pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**30, (time.time() - t0) / 4
    except torch.OutOfMemoryError:
        return None


if __name__ == "__main__":
    main()
