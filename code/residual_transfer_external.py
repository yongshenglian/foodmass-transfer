"""Training-free DINOv2 residual transfer (paper v1, fixed rule) for new images, with the frozen reference bank.

Embeds images exactly as extract_dinov2_embeddings.py does (AutoImageProcessor, fp16, pooled output, L2-normalized),
retrieves the 20 most similar reference dishes, and returns the similarity-weighted median of their Qwen log residuals
(transfer_features in evaluate_qwen3vl_dinov2_gate.py). `--check-development` reproduces the paper's saved
development deltas first. Run from code/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import evaluate_qwen3vl_retrieval_v1 as retrieval
from evaluate_qwen3vl_dinov2_gate import transfer_features

V1 = Path("../experiments/qwen3vl_journal_v1")


def bank():
    ref = np.load(V1 / "reference_dinov2_embeddings.npz", allow_pickle=True)
    ids = ref["dish_ids"].astype(str)
    q = pd.read_csv(V1 / "reference_train2253_qwen3vl_results.csv").drop_duplicates("dish_id").set_index("dish_id").loc[ids]
    residual = np.log(q.true_weight.to_numpy(float) / q.predicted_weight.to_numpy(float))
    return retrieval.unit_rows(ref["features"].astype(np.float32)), residual


def embed(paths, model_path, device="cuda", bs=64):
    proc = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
    out = []
    for s in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB").copy() for p in paths[s:s + bs]]
        inputs = {k: v.to(device) for k, v in proc(images=ims, return_tensors="pt").items()}
        with torch.inference_mode():
            r = model(**inputs)
            e = r.pooler_output if getattr(r, "pooler_output", None) is not None else r.last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=Path, help="CSV with dish_id, image")
    p.add_argument("--output", type=Path)
    p.add_argument("--model", default=str(Path.home() / "models/dinov2-base"))
    p.add_argument("--check-development", action="store_true")
    args = p.parse_args()
    feats, residual = bank()
    if args.check_development:
        dev = np.load(V1 / "development_502_dinov2_embeddings.npz", allow_pickle=True)
        delta, _ = transfer_features(retrieval.unit_rows(dev["features"].astype(np.float32)) @ feats.T, residual, 20)
        saved = pd.read_csv(V1 / "dinov2_gate_v1_full2253/development_predictions.csv").set_index("dish_id").loc[dev["dish_ids"].astype(str)]
        print("max |delta - saved| on development:", float(np.max(np.abs(delta - saved.neighbor_log_residual.to_numpy()))))
    if args.sample:
        s = pd.read_csv(args.sample)
        e = embed(list(s.image), args.model)
        delta, nb = transfer_features(e @ feats.T, residual, 20)
        pd.DataFrame({"dish_id": s.dish_id, "neighbor_log_residual": delta, "max_similarity": nb[:, 0],
                      "mean_similarity": nb[:, 1]}).to_csv(args.output, index=False)
        np.savez_compressed(args.output.with_suffix(".npz"), dish_ids=s.dish_id.astype(str).to_numpy(), features=e)
        print(f"{len(s)} images; max similarity to bank: median {np.median(nb[:, 0]):.3f}")


if __name__ == "__main__":
    main()
