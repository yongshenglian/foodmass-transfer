# Top-view versus side-view photographs on ECUSTFD (declared before scoring)

Written 2026-10-09, in response to the review request for subgroup errors by camera angle.

ECUSTFD photographed each object from the top and from the side. The paper used only the top views. Here every side
view of the same 146 objects (same weighed masses) is scored once, with the already-frozen models:
single-prompt Qwen3-VL-32B (FP8, unchanged prompt), the Nutrition5k-only DINOv2-base (`stage_b_imageonly_aux10`) and
the multi-domain DINOv2-large (`joint_large_consensus`). No model is trained, tuned or selected on these photographs.
Analysis (`code/subgroup_errors.py`): MAE and WAPE for top and side views, and the paired difference side minus top,
computed per object and bootstrapped over objects (10,000 replicates, seed 0).
