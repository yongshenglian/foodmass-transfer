# Matched supervised baselines (declared before training)

Written 2026-10-10, in response to review: show what fine-tuning and auxiliary depth supervision add under matched
conditions, and whether a conventional pretrained backbone gives the same findings.

All runs: photograph only (side inputs zeroed), Nutrition5k training set (2,253 dishes), 5-fold cross-validation, same
head (512-unit layer), log-mass target, 15 epochs, batch 16, AdamW, one-cycle schedule, seed 20261004, ensemble of the
five fold models, prediction limited to 10-1,500 g. Only the factor named changes.

| Run | Backbone | Backbone weights | Depth target |
|---|---|---|---|
| `stage_b_imageonly_aux10` (exists, the paper's model) | DINOv2-base | fine-tuned | yes |
| `stage_b_imageonly_nodepth` | DINOv2-base | fine-tuned | no |
| `stage_b_imageonly_frozen` | DINOv2-base | frozen | yes |
| `stage_b_imageonly_frozen_nodepth` | DINOv2-base | frozen | no |
| `stage_b_imageonly_resnet50` | ResNet-50 (ImageNet-1k, supervised) | fine-tuned, backbone lr 2e-5 | yes |
| `stage_b_imageonly_resnet50_lr1e4` | ResNet-50 | fine-tuned, backbone lr 1e-4 | yes |

ResNet-50 features: mean- and max-pooled final feature map (2 x 2,048), in place of DINOv2's [CLS] and mean patch
token. Because the DINOv2 learning rate may not suit a CNN, ResNet-50 is run at two backbone learning rates and the
one with the lower Nutrition5k validation MAE is reported (the same validation-based choice the DINOv2 settings had);
both are listed.

Evaluation: each run scored on the Nutrition5k validation set (used only for the ResNet-50 learning-rate choice) and
once on ASA24, ECUSTFD (top views) and FPB (`code/score_folds.py`). The internal Nutrition5k test set is deliberately
not scored by the follow-up job (`run_resnet_lr.sh`): it has already been accessed six times (the first job, launched before
this decision, also scored it), and the in-domain comparison uses cross-validation and
validation instead (see reviewer comment on repeated test access). Analysis: `code/matched_baselines.py` (MAE, paired bootstrap
against the paper's model, ordering and scale metrics).
