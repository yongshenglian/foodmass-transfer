# Open-vocabulary single-prompt Qwen on Nutrition5k (declared before scoring)

Written 2026-10-09, before any of the runs below.

Why: the two-step Nutrition5k Qwen baseline receives the closed list of 555 Nutrition5k ingredient names. That is
dataset-level vocabulary, not dish-specific information, but it is not photograph-only. This adds a truly
photograph-only Nutrition5k baseline: the same single-prompt ("blind") Qwen used on ASA24, ECUSTFD and FPB.

What is run (`run_qwen.sh`): Qwen3-VL-32B-Instruct FP8, `run_qwen_local.py --mode blind`, unchanged prompt and
decoding (temperature 0.01, 650 tokens, one retry), on
1. the Nutrition5k validation set (502 dishes, `qwen_local/development_502_sample.csv`), and
2. the Nutrition5k test set (498 dishes, `confirmation_photo_only/confirmation_498_sample.csv`).

What is scored (`code/score_n5k_single_prompt.py`), once:
- single-prompt Qwen; invalid answers get the Nutrition5k training-set median (186 g), as in every other run;
- the same frozen kNN residual correction applied to it (20 neighbours, factor 0.75, clip 10-1,500 g, the saved
  neighbour residuals already used for the paper);
- paired dish-level bootstrap (10,000 replicates, seed 0) against two-step Qwen and the fine-tuned DINOv2.

Nothing is tuned or selected on these results. They are a comparator only; the fine-tuned models are unchanged.
This is the fourth use of the test set, for a comparator, and is reported as such.
