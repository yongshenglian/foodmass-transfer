# Aspect-ratio control (reviewer comment 3), declared before any training or scoring

Written 2026-10-30, before any model below was trained. Responds to reviewer comment 3: square ASA24 and non-4:3 FPB
photographs are distorted by resizing every photograph to 448 x 336, which could explain part of the transfer failure.

## Preprocessing rules

- **Stretch (the paper's rule):** resize every photograph to 448 x 336.
- **Pad (the control):** keep the photograph's shape, resize it to fit a 448 x 448 square and fill the rest with the
  ImageNet mean colour, which is 0 after normalisation. The photograph is centred.

Each rule is used in training and in evaluation alike. Under the pad rule:

| Dataset | Photograph | Content in the square | Bars |
|---|---|---|---|
| Nutrition5k, ECUSTFD | 4:3 | 448 x 336, same pixels as stretch | above and below |
| ASA24 | square cut-out on white | 448 x 448 | none |
| FPB, lab camera | 16:9, 1024 x 576 | 448 x 252 | above and below |
| FPB, smartphone | 3:4, 768 x 1024 | 336 x 448 | left and right |

A square was chosen over a 4:3 frame so that the Nutrition5k-only model also sees padding in training. Its bars are
always above and below, so left and right bars (FPB smartphone photographs) remain unseen by that model; this is stated
with the results.

Nothing else changes: flips and brightness jitter, which also shifts the padding slightly in training, 15 epochs,
batch 16, learning rates, depth target, seeds, folds and fold ensembles are as in the paper. For DINOv2-large with the
pad rule, gradient checkpointing is used if the smoke test shows that it is needed to fit in 24 GB; it saves memory
and does not change the training.

## Models

Both rules, all six models trained on the same machine and software, so that the comparison is not affected by
hardware or library differences. Folds and seeds as in the paper (seed 20261004).

Amendment, 2026-10-30, before any model finished training: the runs moved from the 2 x RTX 4090 machine to the
workstation's RTX A6000 (48 GB), which became free. This is the machine and software environment that trained the
paper's models. All six models still share one machine. On 48 GB, DINOv2-large fits at 448 x 448 without gradient
checkpointing (smoke test peak 21.9 GiB), so none of the runs uses it.

| Model | Script | Training data | Backbone |
|---|---|---|---|
| Nutrition5k-only | finetune_dinov2_mass.py, image-only, depth weight 1.0 | 2,253 Nutrition5k dishes | DINOv2-base |
| Two-domain | train_multidomain.py | Nutrition5k + expanded ASA24 pool (consensus labels) | DINOv2-large |
| Three-domain | train_multidomain.py --fpb | Nutrition5k + ASA24 pool + FPB training split | DINOv2-base |

## Scoring

Each new model scores each set once, with the 5-model ensemble, no tuning and no selection:
- Nutrition5k validation set (502 dishes).
- ASA24 test set (1,000 photographs).
- FPB test set: the 2,034 photographs whose tray does not appear in FPB training (primary, as in the paper); all
  2,351 also reported.
- ECUSTFD top views (1,429) and side views (1,441).

This scores the test sets with new models. The user approved it for this control on 2026-10-30.

## Analysis (code/aspect_control_analyze.py)

Primary: for each model and set, MAE under pad minus MAE under stretch, paired by photograph, with a 95% interval from
10,000 bootstrap replicates of clusters: Nutrition5k acquisition groups, ASA24 food codes, FPB tray signatures and
ECUSTFD objects. Also reported: MAE, the scale ratio (geometric mean of predicted / true) and the share within 25%.

Reading the result:
- Preprocessing explains part of the transfer failure if the pad rule lowers the Nutrition5k-only model's MAE on ASA24
  or FPB with an interval that excludes 0.
- Its size is given as a share of the transfer gap: the Nutrition5k-only model's MAE minus the MAE of the model that was
  trained on that dataset (two-domain for ASA24, three-domain for FPB), both under the stretch rule.
- A pad-minus-stretch interval that includes 0 is reported as no detectable effect, with its interval, not as no effect.

Secondary:
- FPB by viewpoint, from file names: lab camera frames 01-02 (close overhead), lab camera frames 05-08 (distant
  oblique), smartphone (oblique), other. The two rules distort these groups differently.
- Reproducibility: the rerun stretch models against the paper's predictions for the same sets.

## Limits

- The pad rule changes the input size and the share of the input filled by food, not only the distortion. For 16:9
  FPB photographs the food occupies fewer pixels than under stretch. The comparison is between two complete
  preprocessing rules, not of distortion alone.
- One seed per model; seed-to-seed variation of the Nutrition5k-only model, from the paper's three extra seeds, is
  given as a reference for the size of run-to-run differences.
