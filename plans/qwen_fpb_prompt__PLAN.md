# Qwen prompt sensitivity on FPB (declared 2026-10-12, before any call; post hoc)

Question (internal review): Qwen's single prompt opens "The image displays one serving of food", while many FPB
photographs show a whole tray with several items. Does that wording cause Qwen's underestimation on FPB?

Data: FPB *training* split only (Qwen is zero-shot, so these photographs are external to it; the FPB validation split,
which is the paper's FPB test set, is not touched). Fixed sample: 600 single-item and 600 multi-item photographs drawn
with numpy default_rng(20261012) from fpb_train_1024.csv -> sample.csv.

Arms (same model, server settings, system prompt, parser, two attempts):
- blind: the paper's prompt, unchanged.
- blind_all: identical except the opening sentence is "The image displays food; it may be a single item or a whole
  meal with several items." Declared here; no other wording is tried.

Outcomes, overall and by single/multi-item: MAE, geometric-mean scale ratio, Spearman; paired difference in MAE with a
bootstrap over annotation signatures (10,000 replicates). Invalid responses are counted and excluded pairwise.
Reading: the wording explains the underestimation if the multi-item scale ratio moves materially towards 1.
Whatever the result, the paper's FPB test figure for Qwen is not re-scored.
