# ECUSTFD: is the gram-scale error a framing effect, and does matching the field of view remove it?
Post hoc and exploratory (written 2026-10-12 before running the analysis below). ECUSTFD was already scored once per
model; nothing here selects a model, and the paper's ECUSTFD results stay as they are.

Each ECUSTFD photograph contains a One Yuan coin (25.0 mm) with an annotated box. Millimetres per pixel = 25 / longer
side of the coin box; field of view (FOV) = mm per pixel x longer image side.

## A. Diagnostic on existing predictions (no inference)
Objects were photographed several times at different distances. Within-object regression (object fixed effects, so the
true mass is constant) of log(predicted/true) on log(FOV), top and side views separately and pooled, for the
Nutrition5k-only DINOv2-base, the multi-domain DINOv2-large and single-prompt Qwen. Slope with a 95% CI from a
10,000-replicate bootstrap over objects. A slope near 0 means the estimate does not depend on framing; a negative slope
means that the same object is judged lighter when photographed from further away.

## B. Field-of-view matching (one fixed rule, no tuning)
Nutrition5k overhead frames are 640 x 480 with the table at a median 406.9 mm (border of the depth maps); with the
RealSense D435 colour camera's nominal 69.4 degree horizontal FOV (fx about 462 px at 640 px) this is 0.88 mm per pixel,
a 563 mm wide view. Each ECUSTFD photograph is rescaled to 0.88 mm per pixel, pasted undistorted at the centre of a
640 x 480 canvas filled with the photograph's median border colour (cropped if larger), and scored with the unchanged
fold ensembles (code/score_folds.py). Outcomes: MAE, scale ratio and Spearman against the original predictions, paired
difference in MAE with a bootstrap over objects; and the within-object slope of part A on the matched images (expected
near 0 if matching works). Sensitivity: 0.77 mm per pixel (the 35.9 cm camera distance in the Nutrition5k release notes).
The padding is an artefact the model never saw in training; a null or negative result is therefore not evidence against
reference objects in general, and is reported as such.

## Unplanned follow-up (added 2026-10-12 after the planned outcomes above were seen)
Field-of-view matching lowered every prediction (scale ratio 0.28-0.56) but left ordering unchanged or better. We
therefore also report MAE after dividing by one global factor (geometric-mean ratio) estimated on other objects
(5 folds by object), for the original and the matched images. This uses ECUSTFD labels of other objects, so it is
target-domain calibration with a reference object, not cross-dataset transfer, and it was not planned in advance.
