# Per-food calibration with weighed references on ECUSTFD (declared before any result)

Written 2026-10-29, before any number below was computed. Responds to reviewer comment 2: the ASA24 gain from one
labeled photograph (42.6 to 28.3 g) measures agreement with assigned standard-portion weights, not with weighed mass.

## Data and models

ECUSTFD: every object was weighed. The 19 food types have 2 to 19 objects each, with different masses. There are top
and side photographs of each object, and a 25 mm coin in every photograph.

Existing predictions only; nothing is trained, rescored or tuned:
- DINOv2-base, Nutrition5k only (`pred_n5k_imageonly.csv`);
- DINOv2-large, two-domain (`pred_joint_large_consensus.csv`), the model used for the ASA24 calibration;
- top views from `external_ecustfd/`, side views from `ecustfd_side_view/`.

## Procedure, mirroring the ASA24 known-food analysis (code/asa24_area_baselines.py)

For each food type with at least 2 objects:
1. Draw one photograph of one object at random as the reference.
2. Score all photographs, in the target view, of the type's other objects. The reference object's own photographs are
   never scored.
3. Repeat for 20 draws (seed 0) and pool the absolute errors.

Rules, each given only the reference photograph and its weight:
- **Uncalibrated:** the model's prediction.
- **Calibrated:** prediction / exp(0.75 x log(reference prediction / reference weight)). The factor 0.75 is the one fixed
  on the ASA24 validation set.
- **Constant:** the reference weight.
- **Pixel-area scaling:** reference weight x (target box area / reference box area), in pixels, as on ASA24. Box: the
  annotated food bounding box.
- **Coin-scaled area:** reference weight x (target area / reference area)^1.5, with areas in mm² from the coin. The
  exponent 1.5 treats the objects as similar shapes.

Conditions:
- **Same view:** reference and targets from top views (primary). Also side to side.
- **Changed view:** reference from the top, targets from the side. Also side to top.
- **Changed camera distance:** within the same-view condition, targets whose field of view, measured with the coin,
  differs from the reference's by more than 25% in either direction, against those within 25%.

Reported: MAE per rule and condition, and calibrated minus each other rule, with a 95% interval from 10,000 bootstrap
replicates over food types (the independent units; there are only 19, so the intervals will be wide). Also the
mass spread within types, because the constant rule is strong where objects of a type weigh about the same.

## Limits, stated with the results

ECUSTFD shows whole items (different apples), not portions of one dish, and some types vary little in mass. It is
2 to 19 objects per type. Its photographs were taken by the dataset authors in one setting. This uses ECUSTFD labels in
a new analysis; the models and their predictions are unchanged.
