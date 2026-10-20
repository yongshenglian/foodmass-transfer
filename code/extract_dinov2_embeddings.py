"""Checkpointed DINOv2 embedding extraction for Nutrition5k RGB images.

The input CSV must contain dish_id and image columns. Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def save(path, dish_ids, embeddings, model_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dish_ids=np.asarray(dish_ids),
        features=np.asarray(embeddings, dtype=np.float32),
        model_path=np.asarray(str(model_path)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/mnt/sdb/models/dinov2-base"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if not {"dish_id", "image"}.issubset(frame):
        raise RuntimeError("Input CSV must contain dish_id and image columns")
    if frame.dish_id.duplicated().any():
        raise RuntimeError("Input contains duplicate dish IDs")
    completed_ids, completed_embeddings = [], []
    if args.output.is_file():
        checkpoint = np.load(args.output)
        completed_ids = checkpoint["dish_ids"].astype(str).tolist()
        completed_embeddings = checkpoint["features"].astype(np.float32).tolist()
    done = set(completed_ids)
    pending = frame[~frame.dish_id.astype(str).isin(done)].copy()

    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16,
    ).to(args.device).eval()
    since_checkpoint = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending.iloc[start:start + args.batch_size]
        images = []
        for path in batch.image:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            result = model(**inputs)
            embedding = result.pooler_output if getattr(result, "pooler_output", None) is not None else result.last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding.float(), dim=1).cpu().numpy()
        completed_ids.extend(batch.dish_id.astype(str).tolist())
        completed_embeddings.extend(embedding.tolist())
        since_checkpoint += len(batch)
        if since_checkpoint >= args.checkpoint_every:
            save(args.output, completed_ids, completed_embeddings, args.model)
            since_checkpoint = 0
            print(f"{len(completed_ids)}/{len(frame)}", flush=True)
    save(args.output, completed_ids, completed_embeddings, args.model)
    print(json.dumps({
        "rows": len(completed_ids),
        "dimensions": len(completed_embeddings[0]) if completed_embeddings else 0,
        "model": str(args.model.resolve()), "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
