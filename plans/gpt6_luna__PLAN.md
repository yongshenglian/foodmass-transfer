# Second VLM: gpt-6-luna, single prompt (declared 2026-10-15, before any call)

Question: are the zero-shot results specific to Qwen3-VL-32B? Requested by the author; a cheap OpenAI model was chosen
for cost (about $2 for all sets). Post hoc comparator; nothing in the paper's models or selection changes.

Model: `gpt-6-luna` via the OpenAI Chat Completions API, reasoning effort "none", at most 650 completion tokens,
temperature left at the API default (reasoning models may reject other values). Prompt: the paper's single prompt,
unchanged (system prompt + `run_qwen_local.blind_prompt`, retry text on a format failure, two attempts), same parser.
Images are sent as the same files Qwen received, base64 PNG/JPEG data URLs, detail "auto".

Sets, each scored once with the full sample Qwen received:
- Nutrition5k validation (502; qwen_local/development_502_sample.csv). The internal test set is NOT scored.
- ASA24 test (1,000; asa24_aerial_sample.csv).
- ECUSTFD top (1,429) and side (1,441) views.
- FPB test (official validation split, 2,351; primary subset 2,034 signature-disjoint as in the paper).

Pilot: 20 photographs from the FPB *training* split (not a test set) to check format and token use; not reported.

Analysis (fixed now): invalid or non-positive answers imputed with the Nutrition5k training median, as for Qwen.
Per set: MAE, WAPE, scale ratio, Spearman, within-20%, and (ASA24) within-series Spearman; paired MAE difference
against Qwen's single prompt with the paper's cluster bootstrap (dish / series / object / annotation signature),
10,000 replicates. Reported whatever the result, as one added row per set and a sentence in limitation two.

## Clarification (2026-10-15, written before any gpt-6-luna result was read)
Invalid answers are handled exactly as the paper handled Qwen, for both models alike: a valid positive total is used;
otherwise the lines "name: N g" in the raw answer are summed (lenient parse); otherwise the fallback is the
Nutrition5k training median (186 g), or the FPB training median (204 g) on FPB. FPB is reported on the 2,034
signature-disjoint photographs (primary) and on all 2,351. Qwen's figures are recomputed from its raw files with
this rule and must reproduce the paper's values.
