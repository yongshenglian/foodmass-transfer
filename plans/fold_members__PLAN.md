# Single fold models versus the five-model ensemble (declared before scoring)

Written 2026-10-09, in response to the question whether the fine-tuned DINOv2's advantage comes from averaging five
fold models. Every fine-tuned DINOv2 in the paper is the average of its five cross-validation fold models; the
published Nutrition5k comparators and Qwen are single models.

Scored once, with the saved fold checkpoints (no training): each fold model separately, and the ensemble recomputed
from them, for
- the Nutrition5k-only DINOv2-base (`stage_b_imageonly_aux10`) on the Nutrition5k validation and internal test sets,
  ASA24, ECUSTFD and FPB;
- the multi-domain DINOv2-large (`joint_large_consensus`) on the same five sets;
- the three-domain DINOv2-base (`joint3_base`) on FPB.
Analysis (`code/fold_vs_ensemble.py`): MAE of each fold model, their mean and range, the ensemble MAE, and the paired
difference ensemble minus average single model (bootstrap over dishes or clusters). The recomputed ensemble must
reproduce the paper's ensemble predictions. This is a further use of the internal test set for already-reported
models; nothing is selected on it.
