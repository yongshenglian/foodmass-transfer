# Single-Image Food Mass Estimators Do Not Transfer Their Gram Scale Across Datasets

Code, set lists, analysis plans and model descriptions for the paper of the same title (Y. Lian, 2026; arXiv link to be added).

The study fine-tunes DINOv2 to estimate a dish's mass from one RGB photograph and compares it with zero-shot vision-language
models (Qwen3-VL-32B, gpt-6-luna) on Nutrition5k and three external datasets: ASA24, ECUSTFD and the Food Portion Benchmark (FPB).

## Contents

| Folder | What it holds |
|---|---|
| `sets/` | The photographs in every training and test set, by their identifiers in the original datasets |
| `code/` | Data preparation, training, scoring, analysis and figure scripts |
| `plans/` | The dated analysis plans, including the list of models declared before the ASA24 test set was scored |
| `models/` | Training configuration and SHA-256 checksums of the model weights; the weights themselves are on Zenodo |

## Data

No images are included. Download each dataset from its original source:

- Nutrition5k: https://github.com/google-research-datasets/Nutrition5k (CC BY 4.0)
- ASA24 Portion Size Image Database: https://epi.grants.cancer.gov/asa24/resources/portionsize.html
- FNDDS 2011–2012 portion weights, used for the ASA24 labels: https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds/
- ECUSTFD: https://github.com/Liang-yc/ECUSTFD-resized-
- Food Portion Benchmark: see Sanatbyek et al., IEEE Access 13 (2025), doi:10.1109/ACCESS.2025.3603287 (CC BY-NC)

### Set lists (`sets/`)

| File | Photographs | Columns |
|---|---|---|
| `nutrition5k_train.csv` | 2,253 | dish ID, measured mass |
| `nutrition5k_validation.csv` | 502 | dish ID, measured mass; 502 of the 507 dishes in the official test split |
| `nutrition5k_internal_test.csv` | 498 | dish ID, measured mass; drawn from the official training split |
| `asa24_train.csv` | 5,099 | image file, food series, FNDDS code, portion, label |
| `asa24_train_expanded.csv` | 6,329 | image file, food, FNDDS code, label |
| `asa24_test.csv` | 1,000 | image file, food series, FNDDS code, portion, label |
| `ecustfd_top_test.csv`, `ecustfd_side_test.csv` | 1,429 and 1,441 | image file, object ID, view, food type |
| `fpb_train.csv` | 9,190 | dish ID, image file, tray signature |
| `fpb_test.csv` | 2,034 | dish ID, image file, tray signature; the official validation split minus trays that also appear in training |

ASA24 labels are our own: the FNDDS weight of the stated portion times its multiplier, kept only when all FNDDS foods linked
to a photograph agree within 25% (`code/build_asa24_external_set.py`). For ECUSTFD and FPB only identifiers are given; take
the weights from the original datasets.

## Code

Scripts are run from `code/` and expect the datasets under `~/data/` (for example `~/data/asa24`, `~/data/fpb`,
`~/data/nutrition5k_dataset`) and the working outputs under `../experiments/`; adjust these paths to your setup. The main entry points:

| Step | Script |
|---|---|
| Build the external sets | `build_asa24_external_set.py`, `build_asa24_consensus_pool.py`, `build_fpb_external_set.py`, `build_ecustfd_side_sample.py` |
| Train the Nutrition5k-only model | `finetune_dinov2_mass.py` |
| Train the two- and three-domain models | `train_multidomain.py` |
| Score external datasets | `score_external_finetuned.py`, `score_external_multidomain.py`, `evaluate_asa24_finals.py` |
| Run the VLMs | `run_qwen_local.py` (Qwen3-VL served with vLLM), `run_openai_blind.py` (gpt-6-luna; needs `OPENAI_API_KEY`) |
| Ordering versus gram scale | `relative_vs_absolute.py`, `asa24_area_baselines.py`, `asa24_per_food_scale.py`, `ecustfd_scale.py` |
| Prediction intervals | `uncertainty_analysis.py` |
| Near-duplicate Nutrition5k plates | `n5k_sibling_leakage.py` |
| Figures | `plot_cross_dataset_figure.py`, `dataset_examples_figure.py`, `calibration_example_figure.py` |

The code was run with Python 3.13, PyTorch 2.6 and Transformers 5.18 on NVIDIA RTX A6000 GPUs. Some older helper scripts
(`run_nutrition5k_confidence_selective.py`) still import from a local copy of the DietAI24 code; their prompt and parsing
functions are the parts used here.

## Models

Three models are reported in the paper, each an ensemble of five fold models (`models/README.md`):

- Nutrition5k-only DINOv2-base (`stage_b_imageonly_aux10`)
- Two-domain DINOv2-large, trained on Nutrition5k and the expanded ASA24 training set (`joint_large_consensus`)
- Three-domain DINOv2-base, trained on Nutrition5k, ASA24 and FPB (`joint3_base`)

Weights: Zenodo DOI to be added.

## Citation

See `CITATION.cff`.

## License

Code: MIT (`LICENSE`). The set lists and plans are released under CC BY 4.0. The datasets keep their own licenses.
