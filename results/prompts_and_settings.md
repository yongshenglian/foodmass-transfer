# Prompts and inference settings

Generated 2026-11-12 from repository commit c6b4f7b42e88 by `code/make_reproducibility_record.py`, which renders each prompt with the function that sent it.

## Qwen3-VL-32B-Instruct-FP8, single prompt (every photograph-only result)

**System prompt**, sent with every request:

```text
You are an expert at analyzing images with computer vision. I will present you with a picture of food, which might be placed on a plate, inside a spoon, or contained within different vessels. Your job is to accurately identify the food depicted in the image.
```

**User prompt** (`blind` mode), with the photograph attached:

```text
The image displays one serving of food. Identify each food item in the image yourself. Make a best-effort integer gram estimate for EACH food item, including ingredients that are mixed in, hidden, or present in small amounts. Also report one dish-level confidence score from 0 to 100. Define confidence as your estimated percent probability that the SUM of your ingredient estimates is within 50 grams of the true total food weight. Judge confidence from visible boundaries, portion separability, available scale cues, and uncertainty from hidden ingredients. Do not inflate confidence. Use 90-100 only for unusually clear portions with reliable scale cues; 80-89 for mostly clear portions with moderate ambiguity; 60-79 for mixed foods, weak scale cues, or meaningful hidden ingredients; and 0-59 for severe ambiguity. Always provide the gram estimates even when confidence is low; filtering is performed externally. Respond with exactly these lines and no other text:
CONFIDENCE: integer%
food item name: integer g
(one line for every food item)
```

**Retry**: after an invalid response the same request is sent once more with this prompt:

```text
The image displays one serving of food. Identify each food item in the image yourself. Make a best-effort integer gram estimate for EACH food item, including ingredients that are mixed in, hidden, or present in small amounts. Also report one dish-level confidence score from 0 to 100. Define confidence as your estimated percent probability that the SUM of your ingredient estimates is within 50 grams of the true total food weight. Judge confidence from visible boundaries, portion separability, available scale cues, and uncertainty from hidden ingredients. Do not inflate confidence. Use 90-100 only for unusually clear portions with reliable scale cues; 80-89 for mostly clear portions with moderate ambiguity; 60-79 for mixed foods, weak scale cues, or meaningful hidden ingredients; and 0-59 for severe ambiguity. Always provide the gram estimates even when confidence is low; filtering is performed externally. Respond with exactly these lines and no other text:
CONFIDENCE: integer%
food item name: integer g
(one line for every food item) Your previous response was not in the required format. Include the confidence line and every listed ingredient exactly once.
```

**Opening for the prompt-wording check on FPB** (replaces "The image displays one serving of food."):

```text
The image displays food; it may be a single item or a whole meal with several items.
```

## Variants

**Names given** (`oracle` mode); `[NAMES]` is the JSON list of the dish's ground-truth ingredient names:

```text
The image displays one serving of a dish containing these known ground-truth ingredients: ["[NAMES]"]. Make a best-effort integer gram estimate for EACH listed ingredient, including ingredients that are mixed in, hidden, or present in small amounts. Also report one dish-level confidence score from 0 to 100. Define confidence as your estimated percent probability that the SUM of your ingredient estimates is within 50 grams of the true total food weight. Judge confidence from visible boundaries, portion separability, available scale cues, and uncertainty from hidden ingredients. Do not inflate confidence. Use 90-100 only for unusually clear portions with reliable scale cues; 80-89 for mostly clear portions with moderate ambiguity; 60-79 for mixed foods, weak scale cues, or meaningful hidden ingredients; and 0-59 for severe ambiguity. Always provide the gram estimates even when confidence is low; filtering is performed externally. Respond with exactly these lines and no other text:
CONFIDENCE: integer%
exact ingredient name: integer g
(one ingredient line for every listed item)
```

**Two-step, step 1** (`recognize` mode); `[VOCABULARY]` is the JSON list of the Nutrition5k ingredient names:

```text
List every ingredient in this dish, including ingredients that are mixed in, hidden, or present in small amounts. Choose names only from this list, copied exactly: [VOCABULARY]. Respond with one ingredient name per line and no other text.
```

Step 2 is the names-given prompt filled with the names returned by step 1.

## Settings

| Setting | Value |
|---|---|
| Model weights | Qwen/Qwen3-VL-32B-Instruct-FP8 (Hugging Face), local copy |
| Server | vLLM, OpenAI-compatible API, version 0.11.0, 0.30.0 per run in manifest.json: 0.30.0 on the workstation; 0.11.0 for the grouped-CV training dishes, on a second machine whose driver did not support 0.30.0 |
| Decoding | temperature 0.01, at most 650 new tokens, thinking disabled (`enable_thinking: false`) |
| Attempts | 2 (the second with the retry prompt); invalid answers then receive the training-set median mass |
| Image | the dataset file, unmodified, sent as a base64 data URL; one image per request |
| Context | 16,384 tokens on the workstation; 8,192 on the second machine |

## gpt-6-luna (comparator)

OpenAI Chat Completions API, model `gpt-6-luna`, reasoning effort `none`, at most 650 completion tokens, temperature at the API default, image detail `auto`; the same system and user prompts and the same handling of invalid answers (experiments/gpt6_luna/PLAN.md). The API gives no direct control of internal reasoning; in all 6,722 first-attempt calls the billed completion tokens equalled the visible answer (o200k_base encoding) plus 3 tokens, so no reasoning tokens were billed.

## Fine-tuned DINOv2: image processing

| Step | Setting |
|---|---|
| Resize | to 448 x 336 pixels (width x height), bicubic, regardless of the photograph's shape (paper's rule) |
| Aspect-ratio control | fit inside 448 x 448 keeping the shape, centred, mean-colour padding (experiments/aspect_control) |
| ASA24 | 160 x 160 cut-outs placed on a white background before resizing |
| FPB | reduced to at most 1,024 pixels on the longer side before resizing |
| Normalisation | ImageNet mean (0.485, 0.456, 0.406) and SD (0.229, 0.224, 0.225) |
| Training augmentation | horizontal and vertical flips (each p = 0.5), gain 0.9-1.1, offset -0.05 to 0.05; no crops |
| Prediction | mean log mass of the 5 fold models, exponentiated, limited to 10-1,500 g |
