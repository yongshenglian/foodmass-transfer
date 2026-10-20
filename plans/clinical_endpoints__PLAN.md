# Clinically interpretable endpoints (declared before scoring)

Written 2026-10-09, in response to review comment 4 (MAE in grams is not enough for medical relevance).

Endpoints, computed for every photograph-only method already reported (no model is retrained or reselected):
- share of photographs within 10% and 20% of the true mass, and within 25 g and 50 g;
- weighted absolute percentage error (WAPE) by true-mass band (<100, 100-199, 200-299, 300-449, >=450 g);
- Bland-Altman agreement: bias and 95% limits of agreement, in grams and as ratios (log scale);
- energy error implied by the mass error at the dish's true composition, |predicted - true mass| x true energy
  density, overall and by energy-density tertile; the same for carbohydrate, with the share of dishes within 10 g of
  carbohydrate (a threshold used in insulin-dose studies);
- direct energy and macronutrient error of the nutrient-head DINOv2 (`stage_b_imageonly_aux10_nutr`).

Data: internal Nutrition5k test set (498) and validation set (502); ASA24 test set, whose labels include FNDDS energy
and macronutrients; ECUSTFD and FPB for the mass-based endpoints only (no nutrient labels).

New inference (workstation, `run_workstation.sh`): the nutrient-head DINOv2 is applied once to the 498 internal-test
photographs with its saved fold checkpoints, and the test dishes' true nutrients are exported from the Nutrition5k
metadata. This is the fifth use of the internal test set, for a model whose settings were fixed in October 2026 and
which was previously scored on the validation set only.
