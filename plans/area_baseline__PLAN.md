# Food-area and naive known-food baselines on ASA24 (post hoc, written 2026-10-12 before running)

Motivation (internal review of main_jbhi): (1) the within-food ordering result (Spearman 0.86) may be reachable by a
trivial visible-food-area cue; (2) per-food calibration (28.3 g at k=1) has no model-free reference.

No model is trained or selected. Existing prediction files are re-read; test labels are used only to score fixed rules.
Everything is reported as post hoc, on both the 1,000-photo test sample and the 5,099-photo strict training pool.

Fixed rules (not tuned; thresholds set here before any score is seen):
- Food area A: number of pixels of the original 160x160 RGBA cut-out with alpha > 127 and
  (max(R,G,B) - min(R,G,B) > 40 or luminance < 140), i.e. foreground that is not white plate. Divided by the image area.
  White foods on white plates will be missed; that is part of the baseline.
- Threshold-free variant D: sum over alpha > 127 pixels of (255 - luminance) / 255, divided by the image area.
- Ordering: mean within-series Spearman (series with >= 3 photos and non-constant weight; same definition as
  analyze_asa24_external.metrics) of A and D with true mass, beside the models and Qwen. Pooled Spearman also reported.
- Known-food baselines, same protocol as FINALS_DECLARED addendum (k = 1, 2, 3 labelled photos per food, 20 random
  draws, only the other photos scored, seed 0):
  N1 "labelled weight": predict the median weight of the k labelled photos.
  N2 "area-scaled": median over the k labelled photos of weight_l * A_x / A_l (falls back to N1 when A_l or A_x is 0).
  M  model calibration as declared (lambda 0.75), recomputed on the same draws, and the uncalibrated model.
- Differences N2 - M and N1 - M with a 10,000-replicate bootstrap over food series (averaged over draws).

## Note (2026-10-16): random draws aligned with Table XII
The declared known-food run (FINALS_DECLARED.md addendum) used one generator, default_rng(0), across k = 1, 2, 3 in
turn; this script first re-seeded it for each k, which reproduced k = 1 exactly but drew different photographs for
k = 2 and 3 (calibrated 26.9 / 25.9 g instead of the table's 27.3 / 26.7 g). The script now uses one generator across
k, reproducing the table to four decimals, so the baselines are scored on the same draws as the table. k = 3
area-scaled: 41.2 g (was 40.3 g on the other draws). Seeds 0-5 of the draw generator give calibrated k = 1 MAE of
28.3-29.3 g, k = 2 26.3-27.3 g, k = 3 25.3-26.7 g.
