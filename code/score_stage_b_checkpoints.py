"""Score a cohort with a saved stage B fold ensemble (foldN.pt from finetune_dinov2_mass.py --save-checkpoints).

Averages the fold models' log predictions, converts to grams and clips to [LO, HI],
exactly as finetune_dinov2_mass.py does for the development cohort. The Qwen CSV
needs the same columns as the development file (dish_id, true_weight,
predicted_weight, confidence, ingredient_count). Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_nutrition5k_reference_bank import DATASET_ROOT, metadata
from finetune_dinov2_mass import HI, LO, Net, cohort, predict, side_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="stage B run folder with fold*.pt and config.json")
    p.add_argument("--qwen-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ingredients-from", type=Path,
                   help="CSV with dish_id and ingredients_used ('a | b'): replaces the ground-truth ingredient list "
                        "(multi-hot) with these names, e.g. Qwen-recognized ingredients for the end-to-end test")
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    config = json.loads((args.run_dir / "config.json").read_text())
    folds = sorted(args.run_dir.glob("fold*.pt"))
    if len(folds) != int(config["folds"]):
        raise SystemExit(f"expected {config['folds']} checkpoints in {args.run_dir}, found {len(folds)}")

    ids = [str(i) for i in pd.read_csv(args.qwen_csv).drop_duplicates("dish_id").dish_id]
    frame = cohort(args.qwen_csv, ids, metadata(DATASET_ROOT))
    if args.ingredients_from:
        names = pd.read_csv(DATASET_ROOT / "metadata/ingredients_metadata.csv")
        to_id = {n.strip().lower(): f"ingr_{int(i):010d}" for n, i in zip(names.ingr, names.id)}
        used = pd.read_csv(args.ingredients_from).set_index("dish_id").ingredients_used.fillna("")
        frame["ingr"] = [[to_id[n.strip().lower()] for n in used.get(d, "").split(" | ") if n.strip()] for d in frame.dish_id]
    preds = []
    for path in folds:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        side = side_features(frame, ckpt["vocab"])
        if config.get("no_ingredients") == "True":
            side[:, 3:] = 0
        columns = {"qwen": [0], "conf": [1], "count": [2], "ingr": list(range(3, side.shape[1]))}
        for name in filter(None, config.get("zero_side", "").split(",")):
            side[:, columns[name]] = 0
        net = Net(side.shape[1], ckpt.get("n_nutr", 0)); net.load_state_dict(ckpt["state_dict"]); net.to(args.device)
        preds.append(predict(net, frame, side, np.arange(len(frame)), args.device))
        print(f"{path.name}: scored {len(frame)} dishes", flush=True)
        del net; torch.cuda.empty_cache()

    pred = np.mean(preds, 0)
    qwen = frame.qwen.to_numpy()
    mass = np.clip(qwen * np.exp(pred) if ckpt["target"] == "residual" else np.exp(pred), LO, HI)
    pd.DataFrame({"dish_id": frame.dish_id, "true": frame.true, "qwen": qwen, "prediction": pred,
                  "mass": mass}).to_csv(args.output, index=False)
    if frame.true.notna().all():
        print(f"MAE {np.mean(np.abs(mass - frame.true)):.2f} g over {len(frame)} dishes")


if __name__ == "__main__":
    main()
