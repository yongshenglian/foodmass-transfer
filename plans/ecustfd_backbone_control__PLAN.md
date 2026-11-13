# Backbone or ASA24 training: which lowered the ECUSTFD error? (declared before scoring)

Written 2026-10-30, before the model below scored ECUSTFD. Responds to reviewer comment 4, which cites the two-domain
model's lower ECUSTFD error as a counterexample to "only same-setting data restores the scale". That comparison,
DINOv2-base trained on Nutrition5k (44.7 g on top views) against DINOv2-large trained on Nutrition5k and ASA24
(36.0 g), changes the backbone and the training data at once.

**Model:** DINOv2-large trained on Nutrition5k alone, image-only, depth weight 1.0, seed 20261004
(`review_controls/runs/large_depth_s20261004`). It was trained for the review controls and has not scored any test set.
Nothing is retrained or tuned.

**Scoring:** its 5-model ensemble scores ECUSTFD top views (1,429) and side views (1,441) once, with the paper's
preprocessing (`score_external_finetuned.py`). The user approved this use of the ECUSTFD test set on 2026-10-30.

**Analysis** (`code/ecustfd_backbone_control.py`): MAE and scale ratio per view, and two paired differences per
photograph, with 95% intervals from 10,000 bootstrap replicates of the 146 objects:
- **backbone:** large Nutrition5k-only minus base Nutrition5k-only;
- **ASA24 training:** two-domain (large) minus large Nutrition5k-only.

The two differences add up to the 8.7 g gain cited by the reviewer. Whichever has an interval that excludes 0 is
reported as contributing; an interval that includes 0 is reported as no detectable contribution.

**Limit:** one seed per model, so a difference of a few grams is within run-to-run variation; the paper's three extra
Nutrition5k-only seeds were not scored on ECUSTFD and give no reference here.
