# Lost in Transfer, Recovered with Local Data: Image-Based Food Mass Estimation Beyond Nutrition5k

Code, set lists, analysis plans, summary results and model descriptions for the paper of the same title (Y. Lian, 2026).

The study fine-tunes DINOv2 to estimate food mass from one RGB photograph and compares it with zero-shot vision-language
models (Qwen3-VL-32B, gpt-6-luna) on Nutrition5k and four other datasets: ASA24, ECUSTFD, the Food Portion Benchmark (FPB)
and NutritionVerse-Real. It then compares a single scale factor, ridge regression on pretrained features and fine-tuning
as ways to use labeled photographs from a new dataset.

## Contents

| Folder | What it holds |
|---|---|
| `sets/` | The photographs in every training and test set, by their identifiers in the original datasets, plus the plate-grouped CV folds and learning-curve groups |
| `code/` | Data preparation, training, scoring, analysis and figure scripts |
| `plans/` | The dated analysis plans, including the list of models declared before the ASA24 test set was scored |
| `results/` | Summary results behind the paper's tables, the exact prompts and settings, software environments and a checksum manifest |
| `models/` | Training configuration and SHA-256 checksums of the model weights; the weights themselves are attached to the v1.0 release |

## Data

No images are included. Download each dataset from its original source:

- Nutrition5k: https://github.com/google-research-datasets/Nutrition5k (CC BY 4.0)
- ASA24 Portion Size Image Database: https://epi.grants.cancer.gov/asa24/resources/portionsize.html
- FNDDS 2011–2012 portion weights, used for the ASA24 labels: https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds/
- ECUSTFD: https://github.com/Liang-yc/ECUSTFD-resized-
- Food Portion Benchmark: see Sanatbyek et al., IEEE Access 13 (2025), doi:10.1109/ACCESS.2025.3603287 (CC BY-NC)
- NutritionVerse-Real: see Tai et al. (2024), NutritionVerse

### Set lists (`sets/`)

| File | Rows | Columns |
|---|---|---|
| `nutrition5k_train.csv` | 2,253 | dish ID, measured mass |
| `nutrition5k_validation.csv` | 502 | dish ID, measured mass; the 502 valid dishes of the 507 in the official test split |
| `nutrition5k_in_domain_test.csv` | 498 | dish ID, measured mass; drawn from the official training split |
| `nutrition5k_plate_grouped_cv_folds.csv` | 3,253 | dish ID, set, inferred acquisition group, outer folds (random and plate-grouped, three seeds) and inner folds |
| `asa24_train.csv` | 5,099 | image file, food series, FNDDS code, portion, label |
| `asa24_train_expanded.csv` | 6,329 | image file, food, FNDDS code, label |
| `asa24_test.csv` | 1,000 | image file, food series, FNDDS code, portion, label |
| `ecustfd_top_test.csv`, `ecustfd_side_test.csv` | 1,429 and 1,441 | image file, object ID, view, food type |
| `fpb_train.csv` | 9,190 | dish ID, image file, tray signature |
| `fpb_test.csv` | 2,034 | dish ID, image file, tray signature; the official validation split minus trays that also appear in training |
| `nutritionverse_real.csv` | 889 | photograph ID, dish identifier |
| `learning_curve_groups.csv` | 16,542 | photograph ID, mass (blank for FPB), dataset, group, whether the group was held out, and the order in which groups were added in each draw |

ASA24 labels are our own: the FNDDS weight of the stated portion times its multiplier, kept only when all FNDDS foods linked
to a photograph agree within 25% (`code/build_asa24_external_set.py`). For ECUSTFD, FPB and NutritionVerse-Real only
identifiers are given; take the weights from the original datasets.

## Code

Scripts are run from `code/` and expect the datasets under `~/data/` (for example `~/data/asa24`, `~/data/fpb`,
`~/data/nutrition5k_dataset`) and the working outputs under `../experiments/`; adjust these paths to your setup. The main entry points:

| Step | Script |
|---|---|
| Build the evaluation sets | `build_asa24_external_set.py`, `build_asa24_consensus_pool.py`, `build_fpb_external_set.py`, `build_ecustfd_side_sample.py`, `build_nvr_sample.py` |
| Train the Nutrition5k-only model | `finetune_dinov2_mass.py` |
| Train models on several datasets | `train_multidomain.py` |
| Plate-grouped cross-validation on Nutrition5k | `n5k_grouped_cv.py` |
| Score other datasets | `score_external_finetuned.py`, `score_external_multidomain.py`, `evaluate_asa24_finals.py`, `dinov2_dataset_matrix.py`, `analyze_nvr.py` |
| Run the vision-language models | `run_qwen_local.py` (Qwen3-VL served with vLLM), `run_openai_blind.py` (gpt-6-luna; needs `OPENAI_API_KEY`), `analyze_gpt6_luna.py` |
| Scale factor, ridge regression and fine-tuning | `learning_curve.py` |
| Ordering versus gram scale | `relative_vs_absolute.py`, `asa24_area_baselines.py`, `asa24_per_food_scale.py`, `ecustfd_scale.py` |
| Paired comparisons and confidence intervals | `analyze_main_comparisons.py`, `intervals_consistent.py` |
| Reproducibility record | `make_reproducibility_record.py` |
| Figures | `plot_dataset_photography.py`, `plot_training_choices.py`, `plot_graphical_abstract.py` |

The code was run with Python 3.13, PyTorch 2.6 and Transformers 5.18 on NVIDIA RTX A6000 GPUs (`results/environments/`).
Some older helper scripts (`run_nutrition5k_confidence_selective.py`) still import from a local copy of the DietAI24 code;
their prompt and parsing functions are the parts used here.

## Models

The weights of three models are attached to the [v1.0 release](https://github.com/yongshenglian/foodmass-transfer/releases/tag/v1.0),
each an ensemble of five fold models (`models/README.md`):

- Nutrition5k-only DINOv2-base (`stage_b_imageonly_aux10`), used for all tests on other datasets
- DINOv2-large trained on Nutrition5k and the expanded ASA24 training set (`joint_large_consensus`)
- DINOv2-base trained on Nutrition5k, ASA24 and FPB (`joint3_base`)

Download the 15 `.pt` files from that release and check them against `models/README.md`.

## Citation

See `CITATION.cff`.

## License

Code: MIT (`LICENSE`). The set lists and plans are released under CC BY 4.0. The datasets keep their own licenses.
