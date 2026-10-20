# NutritionVerse-Real: untouched smartphone benchmark (pre-registered)

Written 2026-10-10, before the dataset was downloaded or any image or label was opened. Response to the second
reviewer's point 5 (`paper/RESPONSE_TO_COMMENTS.md`, item 20).

## Data
NutritionVerse-Real (Tai et al., arXiv:2401.08598; https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real):
889 smartphone photographs of 251 dishes (45 food types), taken from varying angles; ingredient weights measured, and
nutrients computed from them with package labels or the Canadian Nutrient File. It is a curated research dataset, not
prospective participant photographs. Downloaded by the author (Kaggle account); not redistributed.

Inclusion: every photograph whose dish has a positive total measured food mass. No photograph is excluded for model
output, low confidence, refusal, parse failure or error. True dish mass = sum of the measured ingredient weights (or
the dataset's total mass field if provided); nutrients = the dataset's dish totals (energy, carbohydrate, protein, fat).

Preprocessing (identical for every method, fixed before any image is seen): EXIF orientation applied, then the photo is
reduced to at most 1,024 pixels on the long side (as for FPB) and saved as PNG; each model then applies its own fixed
input pipeline (DINOv2: resize to 448x336, no crop; Qwen: the image as is).

## Frozen methods (no training, tuning, recalibration or model selection on this dataset)
Each scored once:
1. DINOv2-base, Nutrition5k only, photograph only (`stage_b_imageonly_aux10`), the paper's primary model: mass.
2. Nutrient-head DINOv2 (`stage_b_imageonly_aux10_nutr`): energy, fat, carbohydrate, protein.
3. DINOv2-large, Nutrition5k + ASA24 (`asa24_external/runs/joint_large_consensus`): mass.
4. DINOv2-base, Nutrition5k + ASA24 + FPB (`asa24_external/runs/joint3_base`): mass (the only model trained on
   several camera viewpoints).
5. Qwen3-VL-32B FP8, single prompt (blind mode, same prompt, vLLM settings and retry rule as before): mass.
6. kNN residual correction of 5 (fixed rule, Nutrition5k bank), secondary.
7. Constant: Nutrition5k training-set median, 186 g.
The same run gives each photograph's maximum cosine similarity to the Nutrition5k training bank (frozen DINOv2-base),
for the familiarity flag (threshold 0.652, the validation 5th percentile, fixed earlier).
Qwen nutrient routes are not evaluated: they depend on Nutrition5k's ingredient nutrient table.

## Endpoints and statistics (photograph = unit; all CIs: cluster bootstrap by dish, 10,000 resamples, seed 0)
- Mass, per method: MAE, median absolute error, WAPE, share within 20% and within 50 g, Spearman (ordering),
  geometric-mean scale ratio, MAE after an oracle global rescale (diagnostic only, uses labels).
- Primary comparison: method 1 minus method 5, paired difference in MAE. Secondary: 3, 4 and 6 versus 5.
- Dish-level consistency: mean within-dish SD of log predictions across a dish's photographs (angle sensitivity).
- Uncertainty: coverage and width of the fixed empirical 80/90/95% prediction intervals of methods 1 and 3, with the
  factors already calibrated in the paper (no recalibration); 90% is primary, as in the paper's table. Share of
  photographs flagged by the similarity threshold; MAE of flagged versus unflagged photographs.
- Nutrients (method 2): MAE and WAPE for energy, carbohydrate, protein, fat; share within 10 g carbohydrate; energy
  at the true composition (true energy density x predicted mass, as in the clinical endpoints) for methods 1, 3, 4, 5.
- Subgroups only if the metadata provides them without inspecting results: camera angle (if recorded), number of food
  items. No other subgroups.

## Reporting
All methods and endpoints above are reported, whatever the result, in a new Results subsection and the external-set
tables; the dataset is described as a curated smartphone dataset, not participant validation.

## Workflow
1. `code/nvr_inspect.py` (CPU): lists the folder layout, image sizes/EXIF orientation and the metadata headers, so the
   sample builder can be written against the real file format. It prints at most two metadata rows.
2. `code/build_nvr_sample.py` + `run_workstation.sh` (written after step 1): sample CSV, all methods, analysis.

## Amendment 1 (2026-10-10, after `inspect.txt`, before any model was run or any label analysed)
The file format was confirmed by `code/nvr_inspect.py` (folder layout, image sizes, metadata headers; two metadata
rows printed). Decisions fixed now:
- Photograph = one of the 889 JPEGs in `nutritionverse-manual/nutritionverse-manual/images`; dish = the number after
  `dish_` in the file name; labels = the dish totals in `nutritionverse_dish_metadata3.csv` (`total_food_weight`,
  `total_calories`, `total_carbohydrates`, `total_protein`, `total_fats`). All 889 photographs are used, regardless of
  the dataset's own Train/Validation/Test split column, because no method is trained on them. Photographs whose
  dish is missing from the metadata cannot be labelled and are counted and reported.
- The images carry no EXIF orientation (Roboflow export); about two thirds are portrait (3024x4032). The registered
  primary pipeline (resize to 448x336) therefore squeezes portrait photographs. As a pre-declared sensitivity analysis
  for the DINOv2 models only, portrait photographs are also turned 90 degrees (counter-clockwise) to landscape before
  the same resize; MAE is reported for both pipelines and for portrait and landscape photographs separately. The
  primary result remains the registered pipeline. Qwen receives the upright image in both cases (one run).
- Camera angle is not recorded, so there is no angle subgroup; the number-of-food-items subgroup uses the metadata
  (1, 2, 3 or more items). Qwen's stated confidence is summarised as in the paper (ECE, AUROC for within 50 g).
- Scripts: `code/build_nvr_sample.py`, `experiments/nutritionverse_real/run_workstation.sh`, `code/nvr_analysis.py`.
