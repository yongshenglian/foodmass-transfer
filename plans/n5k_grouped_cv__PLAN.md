# Grouped and nested cross-validation on Nutrition5k (declared before any run)

Written 2026-10-29, before any model below was trained. Responds to the reviewer's request for a clean in-domain
estimate (paper/main_jbhi.tex, split description and sibling paragraph).

## Why

The reported in-domain MAE of 24.8 g is on the validation set, which was used to choose methods and settings. The internal
test set shares sibling plates with training. Neither is an untouched estimate of the selection procedure. The
comparison of 19.7 g (internal test) with 24.8 g (validation) mixes two sets with different dishes, so it does not
isolate the effect of siblings.

## Data

All 3,253 Nutrition5k dishes used in the paper: training (2,253), validation (502) and internal test (498). Overhead
RGB photographs; depth maps only as the auxiliary training target. One internal-test dish (dish_1564159636) has an empty
depth file and gets the median depth volume.

**Groups.** Dishes captured within 120 s of each other, chained, form one acquisition group: 754 groups, the largest
with 76 dishes. This contains every sibling pair of the paper's rule (120 s and one ingredient list contained in the
other, 1,059 groups), and also plates photographed in the same sequence without shared ingredients. Every grouped split
keeps whole groups together. Folds are balanced by dish count (651 or 650 dishes).

## Experiment 1: random against grouped splitting, matched

The final model (DINOv2-base, fine-tuned, depth auxiliary weight 1.0, 15 epochs, backbone learning rate 2e-5,
image only) is trained with 5-fold cross-validation over the 3,253 dishes, once with random folds and once with grouped
folds, for 3 seeds (20261101-3). Random and grouped runs use the same training seeds. Each dish gets an out-of-fold
prediction under both schemes, so the comparison is on the same dishes with the same model; only the split differs.

Reported: MAE per scheme and seed; the per-dish difference in absolute error (random minus grouped), averaged over
seeds, with a 95% interval from 10,000 bootstrap replicates that resample acquisition groups; the same difference for
dishes that have a sibling (paper's rule) in their training folds under the random split and for those without.

## Experiment 2: nested grouped cross-validation

- Outer: the grouped 5-fold split of seed 20261101.
- Inner: 4 grouped folds within each outer training set.
- Grid: the image-only configurations compared on the validation set during development:
  - fine-tuned base with depth auxiliary weight 1.0 (final), 0.3 and 0;
  - frozen base with weight 1.0;
  - fine-tuned large with weight 1.0.
- Selection: per outer fold, the configuration with the lowest pooled inner out-of-fold MAE.
- Outer score: the outer fold is scored with the mean log prediction of the selected configuration's 4 inner models, as
  the paper scored the validation set with its 5 fold models. No model is retrained on the full outer training set.

Reported:
- nested MAE over all 3,253 dishes with a group-bootstrap 95% interval, and by original set;
- the configuration chosen in each fold;
- the outer MAE of each configuration;
- DINOv2 against single-prompt Qwen3-VL-32B on the same outer photographs.

Qwen predictions:
- Existing for the validation and internal test dishes (n5k_single_prompt).
- To be added for the 2,253 training dishes with the same model, prompt and decoding (run.sh, step 3). That run uses
  two GPUs with tensor parallelism instead of one; this can change some answers slightly.
- Invalid answers get 186 g, as before.

## What this can and cannot establish

- It is a reanalysis of data that already informed the study, not a prospective holdout. The backbone, the training
  recipe and the grid itself were chosen while looking at the validation set. The nested procedure repeats only the
  choices inside the grid.
- The nested MAE estimates the accuracy of the selection procedure applied to new Nutrition5k plates from new
  acquisition sequences. It does not evaluate the single released model.
- Experiment 1 measures how much random splitting understates the error, on the same dishes, for this model.
- Other choices made on the validation set (image-only rather than Qwen side inputs, the ResNet-50 baseline) are
  outside the grid.

## Use of the held-out sets

The validation and internal test dishes become outer folds. This is the seventh use of the internal test set. It is
reported as a reanalysis. Nothing in the released models or in the paper's other results is changed or selected on
these results.

## Compute

130 training jobs: 30 for Experiment 1 and 100 for Experiment 2. On a 4090, about 5-7 min per base fine-tuning job,
2-3 min per frozen job and about 15 min per large job. That is roughly 8 h on two GPUs, plus 1-2 h for Qwen.
Code: code/n5k_grouped_cv.py; driver: run.sh; copying: copy.sh.
