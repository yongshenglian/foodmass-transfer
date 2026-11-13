# Matched learning curve (reviewer comment 5), declared before any training or analysis

Written 2026-10-30, before any model below was trained. Responds to reviewer comment 5: the same-dataset gains compare
models that differ in training-set size, backbone, optimizer exposure and labeling rule. The reviewer asks for a
matched learning curve: the same backbone and training budget, a varying number of target-domain food or tray groups,
target-only against pooled training, and simple scalar calibration and frozen-feature regression as baselines.

## Data, no test set

- **ASA24:** the standard training pool, 5,099 photographs of 719 foods, with the paper's standard label rule (not the
  expanded pool with relaxed labels, so all arms share one rule). Group: food.
- **FPB:** the official training split, 9,190 photographs of 960 trays. Group: tray (the paper's annotation signature).
- **Nutrition5k:** the 2,253 training dishes, with the depth target, as in the paper.

For each target dataset, a fixed 20% of groups is held out for evaluation (seed 20261201): 144 ASA24 foods and 192 FPB
trays. Every arm and every k is scored on the same held-out photographs. No test set is used.

From the other 80% of groups, k groups are used as target training data: k = 5, 10, 25, 50, 100, 250 and all (575
foods, 768 trays). Three random draws; within a draw the subsets are nested (the first k groups of one random order).

## Arms, at every k

All fine-tuned models: DINOv2-base, the paper's preprocessing (stretch to 448 x 336), flips and brightness jitter,
batch 16, AdamW, backbone learning rate 2e-5, head 1e-3, one-cycle schedule, log-mass target, and **exactly 4,000
optimizer steps** whatever the data size. Photographs are sampled uniformly from the training rows. The training seed
is 20261210 + draw.

1. **Target-only fine-tuning:** the k target groups alone.
2. **Pooled fine-tuning:** the 2,253 Nutrition5k dishes plus the k target groups, sampled uniformly from the union;
   the depth auxiliary loss (weight 1.0) applies to Nutrition5k rows only, as in the paper.
3. **Scalar calibration:** a Nutrition5k-only model trained with the same budget (pooled with k = 0; one per draw)
   predicts every target photograph; one factor, exp(mean log(true / predicted)) over the photographs of the k groups,
   multiplies its predictions on the held-out groups. At k = 0 it is the uncalibrated model.
4. **Frozen-feature ridge, target-only:** frozen DINOv2-base features (class token and mean patch token, 1,536
   numbers, same preprocessing without augmentation), standardized on the training rows; ridge regression of log mass.
   The penalty is chosen from 0.1 to 10,000 (factors of 10) by cross-validation over the k target groups (min(5, k)
   folds).
5. **Frozen-feature ridge, pooled:** as 4, with the Nutrition5k dishes added to every training fold.

Predictions are limited to 10-1,500 g, as in the paper.

**Machine:** the 2 x RTX 4090 machine (environment qwen-prompt-cu128), one worker per GPU. Amendment of 2026-10-30,
before any job ran: the plan first named the workstation's RTX A6000; the runs moved to the 4090 machine so they could
run alongside the aspect-ratio control. Every job, including the frozen features, runs on that one machine, so all arms
share hardware and software. On the A6000, rerunning the paper's Nutrition5k-only model gave 24.99 against 24.78 g on
the validation set, so differences of a few tenths of a gram between runs are run-to-run variation.

## Outcomes and reading

On the held-out groups, for each dataset, arm and k: MAE, scale ratio (geometric mean of predicted / true) and share
within 25%, averaged over the three draws. Intervals: 10,000 bootstrap replicates of held-out groups, applied to each
photograph's error averaged over draws. Paired differences at each k, with the same intervals:
- pooled fine-tuning minus scalar calibration (when does retraining beat calibration);
- target-only minus pooled fine-tuning (does Nutrition5k help);
- pooled fine-tuning minus pooled ridge, and target-only fine-tuning minus target-only ridge (is fine-tuning needed).

Reading rules, fixed now:
- Retraining beats calibration from the smallest k at which the better fine-tuned arm has a lower MAE than calibration
  with an interval that excludes 0, provided this also holds at every larger k. If no such k exists, we report that
  retraining did not detectably beat calibration in the range tested.
- Nutrition5k helps at a given k if target-only minus pooled is positive with an interval that excludes 0.
- An interval that includes 0 is reported as no detectable difference, not as equivalence.

## Limits

- One fixed budget, 4,000 steps. With few groups this means many passes over the same photographs, and a different
  budget or early stopping could change the small-k results; the budget is not tuned.
- Uniform sampling in the pooled arm means target photographs are a small share of the batches at small k. Other
  mixing ratios are not tested.
- Held-out groups come from the same split as the training groups (for FPB, mostly single items photographed by
  smartphone, unlike the mostly multi-item test trays), so the curves describe new foods or trays from the same
  setting, not the paper's test sets.
- Three draws give the variation due to the choice of groups and seed, not a full distribution.
