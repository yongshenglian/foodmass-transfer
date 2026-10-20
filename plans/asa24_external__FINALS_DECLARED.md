# ASA24 external test: final models declared before test scoring

Written 2026-10-03, before any of the new models below was scored on the ASA24 test sample.

Test sample: `asa24_aerial_sample.csv`, 1,000 overhead photos in 162 portion series. It is frozen and was built before any training on ASA24. Every model below was trained only on Nutrition5k reference dishes plus the ASA24 training pool, which excludes every test food by all linked FNDDS codes, by description and by image. All choices (backbone, data, inputs, loss, calibration, flips, ensembling) were made on food-disjoint validation (5,099-photo pool, out-of-fold) and Nutrition5k CV.

Scored once on test (`code/evaluate_asa24_finals.py`):
1. Nutrition5k-only photo model (`stage_b_imageonly_aux10`), reference; its test result (67.9 g) was reported earlier.
2. `joint_base_stretch`: DINOv2-base, Nutrition5k + ASA24 pool.
3. `joint_large_stretch`: DINOv2-large, Nutrition5k + ASA24 pool.
4. `joint_large_consensus`: DINOv2-large, Nutrition5k + expanded consensus pool.
5. The geometric-mean ensemble of the candidate runs with the lowest validation MAE, chosen by the script before it reads test labels.

Not scored on test (validation showed no gain): Qwen-input, food-name text, mass-weighted loss, calibration, flip averaging. Their test prediction files exist but will not be evaluated, to avoid picking among them on test.

Validation results (ASA24, original pool, MAE): Nutrition5k-only zero-shot 56.3; joint base 40.2; + Qwen 40.3; + text 40.0; expanded pool 39.9; large 38.6; best ensemble so far 38.1; large + expanded pool: see compare_validation_runs.py output at scoring time.

## Addendum (2026-10-03): known-food scenario, declared before test scoring
Per-food k-shot calibration of `joint_large_consensus` test predictions: for each test food series with more than k photos, pick k labelled photos at random, divide predictions for the other photos of that food by exp(0.75 x median log error of the k photos), and score only those other photos; repeat 20 times and average. lambda = 0.75 was chosen on validation (k=1: 28.65, k=2: 25.43, k=3: 25.21 g; uncalibrated on the same images 37.67 g). Reported as a separate "food known in advance with k labelled photos" scenario, not as the unseen-food result.

## Post hoc reviewer sensitivity addendum (2026-10-10)

The original declaration above is preserved. To address reviewer point 10, we now report
`joint_base_consensus` test MAE alongside the existing strict-base and strict/relaxed-large
results as a post hoc training-recipe sensitivity diagnostic. This does not change the
primary model or select a model using test error. Configuration, seed and hyperparameters
match within backbone size; training-pool size, fold allocation and optimizer steps differ.
Results and paired food-series bootstrap intervals are in
`experiments/label_sensitivity/asa24_training_rule_sensitivity.json`.
