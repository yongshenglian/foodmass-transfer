# Review controls: seeds and missing cells without touching any test set (declared 2026-10-12, before training)

Motivation (internal review of main_jbhi): single-seed contrasts of 1-2 g, and combined-recipe comparisons that do not
isolate backbone size or domain mixing. All test sets are already scored, so every control here is read on data that is
external to the model but is not a test set:

- Nutrition5k validation set (502 dishes; five-fold ensemble, as in the paper).
- ASA24 strict training pool (5,099 photographs, 728 series): never seen by a Nutrition5k-only model -> cross-dataset transfer.
- FPB training split (9,190 photographs): never seen by a Nutrition5k-only model -> cross-dataset transfer.
- FPB training split, 5-fold CV grouped by annotation signature: target-domain training.

No test set (Nutrition5k internal test, ASA24 test, ECUSTFD, FPB validation split) is scored. train_multidomain.py writes
test prediction files as a side effect; they are not read. Nothing here selects or changes a model in the paper; all
results are post hoc controls.

## A. Seed variability (Nutrition5k-only, DINOv2-base, settings of stage_b_imageonly_aux10)
Variants: fine-tuned + depth (paper), fine-tuned no depth, frozen + depth. Seeds: existing 20261004 plus new 20261011,
20261012, 20261013 (the seed also sets the fold split, so this is seed + split variability). Outcomes per run:
validation MAE; on the ASA24 pool and FPB training split: MAE, pooled Spearman, geometric-mean scale ratio, and (ASA24)
mean within-series Spearman. Report mean, SD and range over the four seeds; a contrast is called supported only if the
seed ranges do not overlap or the paired per-seed differences all share a sign.

## B. Backbone size without ASA24 data
stage_b_imageonly_aux10_large retrained with checkpoints (same settings and seed 20261004; the first run kept none),
scored on the ASA24 pool and the FPB training split, against the base model. Separates backbone size from added data in
the "Nutrition5k-only base -> multi-domain large" comparison.

## C. FPB-only control
DINOv2-base trained on the FPB training split alone (train_multidomain.py --fpb --domains fpb, seed 20261004, so the
grouped folds are identical to joint3_base) at (i) 15 epochs and (ii) 29 epochs, which matches the optimizer steps of the
three-domain run (17.8k vs 9.2k photographs per epoch). Outcome: grouped out-of-fold MAE on the FPB training split,
paired against joint3_base (58.6 g), bootstrap over annotation signatures. Tests whether mixing domains adds anything
beyond FPB data.
