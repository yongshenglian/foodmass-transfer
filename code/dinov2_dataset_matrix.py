"""Audit all trained DINOv2 models across datasets, using saved predictions.

Run from code/: python dinov2_dataset_matrix.py report
Fill missing cells on a GPU machine: python dinov2_dataset_matrix.py score-missing
No training, tuning, calibration or overwriting of existing predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments")
OUT = E / "dinov2_dataset_matrix"
DATASETS = {
    "n5k_internal": ("Nutrition5k internal", "confirmation_photo_only/confirmation_498_sample.csv"),
    "asa24": ("ASA24", "asa24_external/asa24_aerial_sample.csv"),
    "ecustfd_top": ("ECUSTFD overhead", "external_ecustfd/ecustfd_top_sample.csv"),
    "ecustfd_side": ("ECUSTFD side", "ecustfd_side_view/ecustfd_side_sample.csv"),
    "fpb": ("FPB", "external_fpb/fpb_val_eval.csv"),
    "nvr": ("NutritionVerse-Real", "nutritionverse_real/nvr_sample.csv"),
}
MODELS = {
    "base_n5k": {
        "backbone": "Base", "training": "Nutrition5k", "run": "qwen3vl_journal_v2_training/stage_b_imageonly_aux10",
        "predictions": ["confirmation_photo_only/dinov2_photo_only_n5k.csv", "asa24_external/finetuned_stage_b_imageonly_aux10.csv", "external_ecustfd/pred_n5k_imageonly.csv", "ecustfd_side_view/pred_n5k_imageonly.csv", "external_fpb/pred_n5k_imageonly.csv", "nutritionverse_real/pred_n5k_imageonly.csv"],
    },
    "base_n5k_asa24": {
        "backbone": "Base", "training": "Nutrition5k + strict ASA24", "run": "asa24_external/runs/joint_base_stretch",
        "predictions": [None, "asa24_external/runs/joint_base_stretch/asa24_test_predictions.csv", "external_ecustfd/pred_joint_base_stretch.csv", None, "external_fpb/pred_joint_base_stretch.csv", None],
    },
    "base_n5k_asa24_fpb": {
        "backbone": "Base", "training": "Nutrition5k + expanded ASA24 + FPB", "run": "asa24_external/runs/joint3_base",
        "predictions": [None, "asa24_external/runs/joint3_base/asa24_test_predictions.csv", None, None, "asa24_external/runs/joint3_base/fpb_val_predictions.csv", "nutritionverse_real/pred_joint3_base.csv"],
    },
    "large_n5k_asa24": {
        "backbone": "Large", "training": "Nutrition5k + expanded ASA24", "run": "asa24_external/runs/joint_large_consensus",
        "predictions": ["confirmation_photo_only/dinov2_large_multidomain.csv", "asa24_external/runs/joint_large_consensus/asa24_test_predictions.csv", "external_ecustfd/pred_joint_large_consensus.csv", "ecustfd_side_view/pred_joint_large_consensus.csv", "external_fpb/pred_joint_large_consensus.csv", "nutritionverse_real/pred_joint_large_consensus.csv"],
    },
}


def prediction_file(model, dataset):
    saved = MODELS[model]["predictions"][list(DATASETS).index(dataset)]
    return E / saved if saved else OUT / "predictions" / f"{model}_{dataset}.csv"


def sample(dataset):
    frame = pd.read_csv(E / DATASETS[dataset][1]).set_index("dish_id")
    assert frame.index.is_unique and (frame.true_weight > 0).all()
    if dataset == "fpb":
        train_trays = set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion)
        frame = frame[~frame.portion.isin(train_trays)]
        assert len(frame) == 2034
    return frame


def report():
    result = {"scope": "Retrospective completion of the model-by-dataset comparison; no model selection", "datasets": {}, "models": {}, "missing_predictions": []}
    for dataset, (label, filename) in DATASETS.items():
        frame = sample(dataset)
        path = E / filename
        result["datasets"][dataset] = {"label": label, "photographs": len(frame), "sample": str(path), "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for model, config in MODELS.items():
        cells = {}
        for dataset in DATASETS:
            frame = sample(dataset)
            path = prediction_file(model, dataset)
            if not path.is_file():
                cells[dataset] = {"status": "predictions unavailable", "expected_file": str(path)}
                result["missing_predictions"].append({"model": model, "dataset": dataset, "run_directory": str(E / config["run"]), "sample": str(E / DATASETS[dataset][1]), "output": str(path)})
                continue
            predictions = pd.read_csv(path).set_index("dish_id")
            assert predictions.index.is_unique, path
            mass = predictions.reindex(frame.index).mass.to_numpy(float)
            truth = frame.true_weight.to_numpy(float)
            assert np.isfinite(mass).all() and (mass > 0).all(), path
            ratio = float(np.exp(np.log(mass / truth).mean()))
            cells[dataset] = {
                "status": "evaluated", "n": len(frame), "mae_g": float(np.abs(mass - truth).mean()),
                "spearman": float(pd.Series(mass).rank().corr(pd.Series(truth).rank())),
                "scale_ratio": ratio, "oracle_mae_g": float(np.abs(mass / ratio - truth).mean()),
                "prediction_file": str(path), "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if dataset == "nvr":
                original_key = {"base_n5k": "n5k", "base_n5k_asa24_fpb": "three", "large_n5k_asa24": "large"}.get(model)
                if original_key:
                    original = json.loads((E / "nutritionverse_real/results.json").read_text())["mass"][original_key]["mae_g"]
                    assert np.isclose(cells[dataset]["mae_g"], original["estimate"])
                    cells[dataset]["mae_ci95"] = original["ci95"]
                    cells[dataset]["interval_source"] = "original NutritionVerse-Real results.json"
                else:
                    group = pd.factorize(frame.dish_group)[0]
                    totals = np.bincount(group, np.abs(mass - truth))
                    counts = np.bincount(group)
                    draws = np.random.default_rng(0).integers(0, len(totals), (10000, len(totals)))
                    bootstrap = totals[draws].sum(1) / counts[draws].sum(1)
                    cells[dataset]["mae_ci95"] = np.percentile(bootstrap, [2.5, 97.5]).tolist()
                    cells[dataset]["interval_source"] = "10,000 dish bootstrap resamples; seed 0; fixed predictions"
            metadata_file = path.with_suffix(".json")
            if path.parent == OUT / "predictions" and metadata_file.is_file():
                cells[dataset]["inference_metadata"] = json.loads(metadata_file.read_text())
        result["models"][model] = {"backbone": config["backbone"], "training": config["training"], "run_directory": str(E / config["run"]), "datasets": cells}
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "missing_predictions.json").write_text(json.dumps(result["missing_predictions"], indent=2) + "\n")
    for model, config in result["models"].items():
        print(model, {dataset: round(cell["mae_g"], 1) if cell["status"] == "evaluated" else None for dataset, cell in config["datasets"].items()})
    print(f"Missing combinations: {len(result['missing_predictions'])}")


def score_missing(args):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import finetune_dinov2_mass as ft

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("No usable CUDA GPU here. Run score-missing on the GPU machine; report works without a GPU.")
    for model, config in MODELS.items():
        for dataset in DATASETS:
            output = prediction_file(model, dataset)
            if output.exists():
                continue
            run = E / config["run"]
            saved_config = json.loads((run / "config.json").read_text())
            assert saved_config.get("qwen", "False") == "False" and saved_config.get("text_features", "False") == "False"
            backbone = args.backbone or Path(saved_config["backbone"])
            if not backbone.exists():
                raise SystemExit(f"Missing backbone {backbone}; pass --backbone with its local path.")
            ft.MODEL = str(backbone)
            ft.PREPROC = saved_config.get("preprocess", "stretch")
            frame = sample(dataset).reset_index()
            if args.data_root:
                frame["image"] = frame.image.map(lambda name: str(args.data_root / name.split("/data/", 1)[1]))
            missing = [name for name in frame.image if not Path(name).is_file()]
            if missing:
                raise SystemExit(f"Missing {len(missing)} images; first: {missing[0]}. Use --data-root to remap the original /home/lian/data/ root.")
            folds = sorted(run.glob("fold*.pt"))
            assert len(folds) == 5, f"Expected five checkpoints in {run}"
            ft.POOL = ThreadPoolExecutor(args.workers)
            log_predictions = []
            for checkpoint in folds:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                net = ft.Net(saved["n_side"]).to(args.device)
                net.load_state_dict(saved["state_dict"])
                side = np.zeros((len(frame), saved["n_side"]), np.float32)
                log_predictions.append(ft.predict(net, frame, side, np.arange(len(frame)), args.device, bs=args.batch_size))
                del net
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(model, dataset, checkpoint.name, "done", flush=True)
            ft.POOL.shutdown()
            mean_log = np.mean(log_predictions, axis=0)
            output.parent.mkdir(exist_ok=True, parents=True)
            pd.DataFrame({"dish_id": frame.dish_id, "prediction": mean_log, "mass": np.clip(np.exp(mean_log), ft.LO, ft.HI)}).to_csv(output, index=False)
            metadata = {"device": args.device, "torch": torch.__version__, "backbone": str(backbone), "checkpoints": [str(path) for path in folds], "preprocessing": ft.PREPROC, "batch_size": args.batch_size}
            output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("wrote", output, flush=True)
    report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "score-missing"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()
    report() if options.command == "report" else score_missing(options)
