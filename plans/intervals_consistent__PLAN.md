# One interval procedure for all models (reviewer comment 7), declared before computing

Written 2026-10-30, before any number below was computed. Existing predictions only; nothing is trained or tuned.

## Why

The paper's intervals used a different calibration source for each model and mixed fold-model with ensemble errors:
Nutrition5k-only, out-of-fold errors from random folds (which share sibling plates: only 169 of 2,253 dishes have no
acquisition-group member in their training folds); two-domain, the ensemble's validation errors pooled with fold-model
ASA24 errors; three-domain, fold-model FPB errors only. All were applied to the ensemble.

## Procedure, the same for every model

Empirical multiplicative interval [m/k, m*k] at nominal 90%, with log k the weighted 90% quantile of calibration errors
|log(predicted/true)|. No coverage guarantee: photographs of one food or tray are not exchangeable and test sets differ
from calibration data.

- **Prediction unit:** single fold models, for calibration and evaluation alike. Coverage and width are averaged over
  the 5 fold models. The deployed 5-model ensemble with the same k is reported as a secondary result.
- **Calibration errors, all from groups the fold model never saw:**
  - Nutrition5k: each fold model on the validation dishes, leaving out the dishes with a sibling in the training set
    (rule of the paper). The validation set was used for model selection; this is stated.
  - ASA24 (two- and three-domain models): out-of-fold errors on the training pool, folds grouped by food.
  - FPB (three-domain model): out-of-fold errors on the training split, folds grouped by tray.
  - Each training domain of a model receives equal total weight.
- **Models:** Nutrition5k-only (DINOv2-base), two-domain (DINOv2-large), three-domain (DINOv2-base), the paper's
  checkpoints and predictions.
- **Evaluation sets:** Nutrition5k internal test set, ASA24 test, ECUSTFD top views, FPB test (trays not in training),
  where fold-level predictions exist (the three-domain model has none for the Nutrition5k internal test set or ECUSTFD).

## Reported

Per model and set: coverage with a 95% interval from 10,000 bootstrap replicates of groups (dishes, food series,
objects, trays), median width in grams, and k. Subgroups: coverage and width by mass range (below 100, 100-299, 300-599,
600 g and more) and, on FPB, by viewpoint and item count.

Reading: coverage within the interval of 90 is "near nominal"; we do not call any interval trustworthy, and the text
states that same-domain calibration did not remove the shift (three-domain FPB coverage below nominal, if so).
