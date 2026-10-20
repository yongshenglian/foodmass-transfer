# Nutrition5k incremental-scan siblings across our sets (post hoc check, written 2026-10-12 before running)

Nutrition5k plates were scanned incrementally (a capture after each added food), so consecutive dish IDs (Unix capture
times) can be the same plate at different stages. Our internal test set was drawn at random from the official training
split, so such siblings may sit in both the training set and the internal test set.

Fixed rule: two dishes are siblings if their capture times differ by at most 120 s and the ingredient-name set of one is
a subset of (or equal to) the other's. Sensitivity: 60 s and 300 s, and time-only (120 s, any ingredients).
Reported: share of internal test (498) and validation (502) dishes with a sibling in the training set (2,253); MAE of the
Nutrition5k-only DINOv2-base ensemble and single-prompt Qwen on dishes with and without a training sibling. No model is
retrained; this is a fixed-prediction subset analysis.
