# Retrospective model-by-dataset reporting completion

Prepared 2026-11-11 after the original results were known, at the author's request to compare the three trained
DINOv2-base models on every dataset and to account for omissions in the ordering/scale table. This is a retrospective
completion of reporting, not a prospective registration or a new model-selection experiment.

Models: Nutrition5k-only base; Nutrition5k plus strict ASA24 base; Nutrition5k plus expanded ASA24 plus FPB base;
and the Nutrition5k plus expanded ASA24 large model used in reference calibration. Checkpoints and all model settings
remain unchanged. The strict/expanded difference between the two base models trained with ASA24 is retained and
identified; their difference does not isolate adding FPB alone.

Datasets: the 498-photo Nutrition5k internal set (supporting only, related-plate overlap), the same 1,000-photo ASA24
test set, 1,429 ECUSTFD overhead and 1,441 side photographs, 2,034 FPB test photographs without a matching training
tray signature, and all 889 NutritionVerse-Real photographs. Do not substitute Nutrition5k validation results or
aspect-control reruns of different fitted models for missing cells.

Primary descriptive metric: photograph-weighted MAE in grams. Additional existing diagnostics: Spearman ordering,
geometric mean prediction/reference ratio, and evaluation-label-informed oracle-rescaled MAE. All outcomes are
reported for every available cell, regardless of performance. No calibration, fitting or model choice is allowed.

The report reuses saved predictions. The three-dataset base model's cached ASA24 predictions are now summarized
at the author's request (MAE 47.1 g); this was not part of the original main table. Missing predictions are listed
in missing_predictions.json and marked as unavailable in the manuscript.

Six missing evaluations: two-dataset base on Nutrition5k internal, ECUSTFD side and NutritionVerse-Real; three-dataset
base on Nutrition5k internal and both ECUSTFD views. Complete these by inference on the five saved fold checkpoints,
using the original stretched 448 x 336 preprocessing, mean log prediction, exponentiation and 10–1,500 g bounds.
The sandbox could not access CUDA. An approved check outside the sandbox found an idle NVIDIA RTX A6000, and inference for all six cells was completed there on 2026-11-11. No model was retrained or changed. Saved inference sidecars record the device, PyTorch version, checkpoint paths, preprocessing and batch size.

Run from code/:

    python dinov2_dataset_matrix.py report
    python dinov2_dataset_matrix.py score-missing --backbone /path/to/dinov2-base --data-root /path/to/data

The second command requires PyTorch/Transformers, a GPU, the original model checkpoints in the experiment run
folders, and the source images. --data-root remaps the original /home/lian/data/ prefix; its subfolder layout must
match the saved sample CSVs. Existing predictions are never overwritten. Re-run report after inference, then update
both manuscript tables from the resulting results.json. Preserve the inference environment sidecar JSON files.
