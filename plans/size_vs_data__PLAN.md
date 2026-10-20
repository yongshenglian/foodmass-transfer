# Model size versus ASA24 data (declared before training)

Written 2026-10-09, in response to review: on Nutrition5k the multi-domain DINOv2-large improved on the
Nutrition5k-only DINOv2-base, but backbone size and training data changed together.

Design (2 x 2, Nutrition5k validation set and Nutrition5k cross-validation only; the internal test set is not used):

| | Nutrition5k only | Nutrition5k + ASA24 |
|---|---|---|
| DINOv2-base | `stage_b_imageonly_aux10` (exists) | `joint_base_stretch`, `joint_base_consensus` (exist) |
| DINOv2-large | `stage_b_imageonly_aux10_large` (new) | `joint_large_stretch`, `joint_large_consensus` (exist) |

The new run uses exactly the settings of `stage_b_imageonly_aux10` (log mass, depth target weight 1.0, photograph only,
15 epochs, batch 16, learning rates 2e-5/1e-3, seed 20261004, 5 folds) with the DINOv2-large backbone. Outcome:
Nutrition5k validation MAE and cross-validation MAE of the four cells.
