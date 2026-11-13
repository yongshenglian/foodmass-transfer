# Retrospective main-text confidence-interval reporting

Written 2026-11-11 after the manuscript results and preceding review comments were available. This documents a
reporting follow-up, not prospective registration of an experiment. It involves no training or inference.

The author requested paired confidence intervals in the main paper. Reuse the unchanged prediction files from
the completed model-by-dataset comparison and Qwen evaluations. Summarize Nutrition5k-only DINOv2-base versus
Qwen on ASA24, ECUSTFD overhead, FPB and NutritionVerse-Real; the fixed-base ASA24 addition; and three-dataset
versus two-dataset base on all five external dataset/view combinations. Report every requested contrast.

For each comparison, resample entire food series, objects, trays or dishes 10,000 times with replacement, seed 0
separately per comparison. Recalculate photograph-weighted paired MAE differences and percentile 95% intervals.
Predictions and fitted models stay fixed. Keep the original FPB training-tray exclusions and Qwen invalid-response
fallback. Record all input paths and hashes. These intervals do not include retraining variability.

Run `python analyze_main_comparisons.py` from `code/`. Results are written to this folder and copied into the
submission reproducibility record. For NutritionVerse-Real, the upper limit rounds to 140.3 g, versus 140.4 g in
the original experiment's RNG stream; retain the original result and identify the reporting rerun in the supplement.
