"""Reviewer comment 9: versioned supplementary record of prompts, inference settings, checkpoints, splits, dependencies
and analysis plans. Prompts are rendered by the code that sent them, so the record cannot drift from what was run.

Run from code/:  python make_reproducibility_record.py  [--no-hash]
-> ../paper/reproducibility/prompts_and_settings.md, manifest.json, splits/*.csv, environments/*.txt
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(".."); E = R / "experiments"; OUT = R / "paper/reproducibility"
CHECKPOINTS = {  # the paper's models, by run directory
    "Nutrition5k-only, DINOv2-base (paper's final model)": E / "qwen3vl_journal_v2_training/stage_b_imageonly_aux10",
    "Two-domain, DINOv2-base (Nutrition5k + strict ASA24; data-addition control)": E / "asa24_external/runs/joint_base_stretch",
    "Two-domain, DINOv2-large (Nutrition5k + expanded ASA24)": E / "asa24_external/runs/joint_large_consensus",
    "Three-domain, DINOv2-base (Nutrition5k + ASA24 + FPB)": E / "asa24_external/runs/joint3_base",
    "Nutrition5k-only, DINOv2-large (backbone control)": E / "review_controls/runs/large_depth_s20261004"}
QWEN_OUTPUTS = {  # every Qwen result file in the paper; the file's last write is its inference date
    "Nutrition5k validation": E / "n5k_single_prompt/qwen_blind_val_fp8.csv", "Nutrition5k in-domain test": E / "n5k_single_prompt/qwen_blind_test_fp8.csv",
    "Nutrition5k training (grouped CV, second machine)": E / "n5k_grouped_cv/qwen_blind_train_fp8.csv",
    "ASA24 test": E / "asa24_external/qwen_blind_fp8_lenient.csv", "ECUSTFD top": E / "external_ecustfd/qwen_blind_fp8.csv",
    "ECUSTFD side": E / "ecustfd_side_view/qwen_blind_fp8.csv", "FPB test": E / "external_fpb/qwen_blind_fp8.csv",
    "NutritionVerse-Real": E / "nutritionverse_real/qwen_blind_fp8.csv"}
ENVS = ("n5k-train", "qwen3vl-serve")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=R, capture_output=True, text=True).stdout.strip()


def vllm_versions():
    """vLLM version per experiment, from its server logs (the grouped-CV log is the second machine's final run)."""
    out = {}
    for f in sorted(E.glob("*/logs/vllm*.log")):
        v = re.findall(r"version (\d+\.\d+\.\d+\S*)", f.read_text(errors="ignore"))
        out[str(f.relative_to(R))] = v[0] if v else None
    return out


def run_dates(folder):
    """Start and end of the Qwen steps recorded in an experiment's status log."""
    f = folder / "logs/status"
    return [l for l in f.read_text().splitlines() if re.search(r"qwen|vllm", l, re.I)] if f.is_file() else []


def prompts():
    sys.path.insert(0, ".")
    from run_nutrition5k_confidence_selective import SYSTEM_PROMPT, build_prompt
    from run_qwen_local import NEUTRAL_OPENING, blind_prompt, recognize_prompt
    block = lambda t: "```text\n" + t + "\n```\n"
    vllm = sorted({v for v in vllm_versions().values() if v})
    return "\n".join([
        "# Prompts and inference settings", "",
        f"Generated {datetime.now(timezone.utc).date()} from repository commit {git('rev-parse', 'HEAD')[:12]} by "
        "`code/make_reproducibility_record.py`, which renders each prompt with the function that sent it.", "",
        "## Qwen3-VL-32B-Instruct-FP8, single prompt (every photograph-only result)", "",
        "**System prompt**, sent with every request:", "", block(SYSTEM_PROMPT),
        "**User prompt** (`blind` mode), with the photograph attached:", "", block(blind_prompt()),
        "**Retry**: after an invalid response the same request is sent once more with this prompt:", "", block(blind_prompt(retry=True)),
        "**Opening for the prompt-wording check on FPB** (replaces \"The image displays one serving of food.\"):", "", block(NEUTRAL_OPENING.strip()),
        "## Variants", "",
        "**Names given** (`oracle` mode); `[NAMES]` is the JSON list of the dish's ground-truth ingredient names:", "", block(build_prompt(["[NAMES]"])),
        "**Two-step, step 1** (`recognize` mode); `[VOCABULARY]` is the JSON list of the Nutrition5k ingredient names:", "",
        block(recognize_prompt(["[VOCABULARY]"]).replace('["[VOCABULARY]"]', "[VOCABULARY]")),
        "Step 2 is the names-given prompt filled with the names returned by step 1.", "",
        "## Settings", "",
        "| Setting | Value |", "|---|---|",
        "| Model weights | Qwen/Qwen3-VL-32B-Instruct-FP8 (Hugging Face), local copy |",
        f"| Server | vLLM, OpenAI-compatible API, version {', '.join(vllm)} per run in manifest.json: 0.30.0 on the workstation; 0.11.0 for the grouped-CV training dishes, on a second machine whose driver did not support 0.30.0 |",
        "| Decoding | temperature 0.01, at most 650 new tokens, thinking disabled (`enable_thinking: false`) |",
        "| Attempts | 2 (the second with the retry prompt); invalid answers then receive the training-set median mass |",
        "| Image | the dataset file, unmodified, sent as a base64 data URL; one image per request |",
        "| Context | 16,384 tokens on the workstation; 8,192 on the second machine |",
        "", "## gpt-6-luna (comparator)", "",
        "OpenAI Chat Completions API, model `gpt-6-luna`, reasoning effort `none`, at most 650 completion tokens, "
        "temperature at the API default, image detail `auto`; the same system and user prompts and the same handling of "
        "invalid answers (experiments/gpt6_luna/PLAN.md). The API gives no direct control of internal reasoning; in all "
        "6,722 first-attempt calls the billed completion tokens equalled the visible answer (o200k_base encoding) plus "
        "3 tokens, so no reasoning tokens were billed.", "",
        "## Fine-tuned DINOv2: image processing", "",
        "| Step | Setting |", "|---|---|",
        "| Resize | to 448 x 336 pixels (width x height), bicubic, regardless of the photograph's shape (paper's rule) |",
        "| Aspect-ratio control | fit inside 448 x 448 keeping the shape, centred, mean-colour padding (experiments/aspect_control) |",
        "| ASA24 | 160 x 160 cut-outs placed on a white background before resizing |",
        "| FPB | reduced to at most 1,024 pixels on the longer side before resizing |",
        "| Normalisation | ImageNet mean (0.485, 0.456, 0.406) and SD (0.229, 0.224, 0.225) |",
        "| Training augmentation | horizontal and vertical flips (each p = 0.5), gain 0.9-1.1, offset -0.05 to 0.05; no crops |",
        "| Prediction | mean log mass of the 5 fold models, exponentiated, limited to 10-1,500 g |", ""])


def splits(out):
    s = {"nutrition5k_training_2253": pd.read_csv(E / "qwen3vl_journal_v1/reference_train2253_qwen3vl_results.csv").drop_duplicates("dish_id")[["dish_id"]],
         "nutrition5k_validation_502": pd.read_csv(E / "qwen3vl_journal_v1/dinov2_gate_v1_full2253/development_predictions.csv")[["dish_id"]],
         "nutrition5k_in_domain_test_498": pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv")[["dish_id"]],
         "nutrition5k_grouped_cv_folds": pd.read_csv(E / "n5k_grouped_cv/pool.csv").filter(regex=r"^(dish_id|set|group|sibling_group|fold_.*|inner_.*)$"),
         "asa24_training_pool_5099": pd.read_csv(E / "asa24_external/asa24_train_pool.csv")[["image_file", "food_group"]],
         "asa24_training_pool_expanded_6329": pd.read_csv(E / "asa24_external/asa24_train_pool_consensus.csv")[["image_file", "food_group"]],
         "asa24_test_1000": pd.read_csv(E / "asa24_external/asa24_aerial_sample.csv")[["dish_id", "series", "food_code"]],
         "fpb_training_9190": pd.read_csv(E / "external_fpb/fpb_train_1024.csv")[["dish_id", "portion"]],
         "fpb_test_2351": pd.read_csv(E / "external_fpb/fpb_val_eval.csv").assign(
             tray_not_in_training=lambda d: ~d.portion.isin(set(pd.read_csv(E / "external_fpb/fpb_train.csv").portion)))[["dish_id", "portion", "tray_not_in_training"]],
         "ecustfd_top_1429": pd.read_csv(E / "external_ecustfd/ecustfd_top_sample.csv")[["dish_id", "id", "type"]],
         "ecustfd_side_1441": pd.read_csv(E / "ecustfd_side_view/ecustfd_side_sample.csv")[["dish_id", "id", "type"]]}
    if (E / "nutritionverse_real/nvr_sample.csv").is_file():
        s["nutritionverse_real"] = pd.read_csv(E / "nutritionverse_real/nvr_sample.csv").filter(regex=r"^(dish_id|dish_group|image_file|true.*)$")
    if (E / "learning_curve/frame.csv").is_file():
        s["learning_curve_frame"] = pd.read_csv(E / "learning_curve/frame.csv").drop(columns=["image", "volume"])
    (out / "splits").mkdir(parents=True, exist_ok=True); info = {}
    for k, d in s.items():
        f = out / "splits" / f"{k}.csv"; d.to_csv(f, index=False); info[k] = {"rows": int(len(d)), "sha256": sha256(f)}
    return info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-hash", action="store_true", help="skip checkpoint hashes (slow)"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "prompts_and_settings.md").write_text(prompts())
    m = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repository_commit": git("rev-parse", "HEAD"),
         "uncommitted_changes": bool(git("status", "--porcelain")), "checkpoints": {}, "qwen_inference": {}, "analysis_plans": {}}
    for name, d in CHECKPOINTS.items():
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
        m["checkpoints"][name] = {"run_directory": str(d.relative_to(R)), "config": cfg,
                                  "fold_checkpoints": {p.name: (None if a.no_hash else sha256(p)) for p in sorted(d.glob("fold*.pt"))}}
    for name, f in QWEN_OUTPUTS.items():
        if f.is_file():
            m["qwen_inference"][name] = {"file": str(f.relative_to(R)), "rows": int(len(pd.read_csv(f))), "sha256": sha256(f),
                                         "status_log": run_dates(f.parent)}
    for p in sorted(E.glob("*/PLAN.md")):  # immutable version: the commit that first added the plan, and the current hash
        first = git("log", "--diff-filter=A", "--format=%H %cI", "--", str(p.relative_to(R))).splitlines()
        stated = re.findall(r"(?:[Ww]ritten|declared)[^.]{0,40}?(20\d\d-\d\d-\d\d)", p.read_text())
        m["analysis_plans"][str(p.relative_to(R))] = {"declared_in_text": stated[0] if stated else None,
                                                       "first_commit": first[-1] if first else "not committed", "sha256": sha256(p),
                                                       "last_commit": git("log", "-1", "--format=%H %cI", "--", str(p.relative_to(R))) or "not committed"}
    m["vllm_versions"] = vllm_versions()
    # Include exact saved follow-up endpoints without rerunning training or inference.
    result_files = ["n5k_grouped_cv/results.json", "asa24_base_training/results.json", "dinov2_dataset_matrix/results.json", "main_comparisons/results.json", "aspect_control/results.json", "learning_curve/results.json",
                    "intervals_consistent/results.json", "ecustfd_known_food/results.json",
                    "comment3_descriptives/results.json", "comment8_energy/results.json",
                    "ecustfd_backbone_control/results.json", "nutritionverse_real/results.json",
                    "comment6_statistics/group_weighting.json", "gpt6_luna/gpt6_luna.json"]
    (OUT / "results").mkdir(exist_ok=True)
    m["analysis_results"] = {}
    for name in result_files:
        source = E / name
        target = OUT / "results" / (name.split("/")[0] + ".json")
        shutil.copyfile(source, target)
        m["analysis_results"][name] = {"record_file": str(target.relative_to(OUT)), "sha256": sha256(target)}
    # Preserve the six additional inference outputs used to complete the model/dataset table.
    matrix_predictions = OUT / "predictions/dinov2_dataset_matrix"
    matrix_predictions.mkdir(parents=True, exist_ok=True)
    m["matrix_inference_files"] = {}
    for source in sorted((E / "dinov2_dataset_matrix/predictions").glob("*")):
        if source.suffix not in (".csv", ".json"):
            continue
        target = matrix_predictions / source.name
        shutil.copyfile(source, target)
        m["matrix_inference_files"][str(target.relative_to(OUT))] = sha256(target)
    m["manuscript_sources"] = {name: sha256(R / "paper" / name)
                               for name in ("main_jbhi.tex", "references.bib", "make_jbhi_submission.py")}
    photo_examples = OUT / "dataset_photo_examples.json"
    if photo_examples.is_file():
        m["photography_examples"] = {"record_file": photo_examples.name, "sha256": sha256(photo_examples)}
    m["training_choices_figure"] = {
        "script": "code/plot_training_choices.py",
        "script_sha256": sha256(R / "code/plot_training_choices.py"),
        "source_record": "results/learning_curve.json",
        "source_sha256": sha256(OUT / "results/learning_curve.json"),
        "figure": "figures/fig_training_choices.pdf",
        "figure_sha256": sha256(R / "paper/figures/fig_training_choices.pdf"),
        "selection": "Scalar calibration and both pooled training methods at all seven positive labeled-group counts; mean over three draws."
    }
    m["main_comparison_script"] = {
        "script": "code/analyze_main_comparisons.py",
        "sha256": sha256(R / "code/analyze_main_comparisons.py"),
        "result_record": "results/main_comparisons.json"
    }
    m["splits"] = splits(OUT); env_dir = OUT / "environments"; env_dir.mkdir(exist_ok=True)
    for env in ENVS:
        r = subprocess.run(["conda", "list", "-n", env, "--export"], capture_output=True, text=True)
        if r.returncode == 0:
            (env_dir / f"{env}.txt").write_text(r.stdout)
    m["environments"] = {p.stem: sha256(p) for p in sorted(env_dir.glob("*.txt"))}
    (OUT / "manifest.json").write_text(json.dumps(m, indent=2))
    print(f"wrote {OUT}: prompts_and_settings.md, manifest.json ({len(m['checkpoints'])} models, {len(m['analysis_plans'])} plans, "
          f"{len(m['splits'])} split files, {len(m['environments'])} environments)")


if __name__ == "__main__":
    main()
