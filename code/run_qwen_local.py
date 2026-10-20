"""Local Qwen3-VL runs on one GPU (vLLM OpenAI server), for the quantization check and the end-to-end test.

Modes (all use the original system prompt, temperature 0.01, 650 tokens, thinking off, 2 attempts):
  oracle     - the original mass prompt with the ground-truth Nutrition5k ingredient names (unchanged text).
  recognize  - step 1 of the end-to-end test: list the ingredients visible in the image, chosen from the
               Nutrition5k ingredient vocabulary (closed set, so names map to nutrient densities).
  predicted  - step 2: the original mass prompt, unchanged, but filled with the names from `recognize`.
  blind      - one call, no names: the original mass prompt with the ingredient clause replaced by
               "identify each food item yourself"; any item names are accepted (external datasets).
  blind_all  - as blind, but the opening sentence does not say "one serving" (prompt-sensitivity check).
Writes one row per dish after every dish (resumable). Run from code/.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from nutrient_estimate_mix import parse_food_items
from run_nutrition5k_confidence_selective import SYSTEM_PROMPT, build_prompt, image_url

N5K = Path(os.environ.get("N5K_ROOT", Path.home() / "data/nutrition5k_dataset"))


def vocabulary():
    return sorted(pd.read_csv(N5K / "metadata/ingredients_metadata.csv").ingr.str.strip().unique())


def recognize_prompt(vocab, retry=False):
    text = ("List every ingredient in this dish, including ingredients that are mixed in, hidden, or present in "
            "small amounts. Choose names only from this list, copied exactly: " + json.dumps(vocab) +
            ". Respond with one ingredient name per line and no other text.")
    if retry:
        text += " Your previous response contained no names from the list. Use only exact names from the list."
    return text


NEUTRAL_OPENING = "The image displays food; it may be a single item or a whole meal with several items. "


def blind_prompt(retry=False, opening="The image displays one serving of food. "):
    text = build_prompt(["X"], retry=retry)
    head = 'The image displays one serving of a dish containing these known ground-truth ingredients: ["X"]. '
    assert text.startswith(head)
    return (opening + "Identify each food item in the image yourself. "
            + text[len(head):].replace("EACH listed ingredient", "EACH food item")
                               .replace("exact ingredient name: integer g\n(one ingredient line for every listed item)",
                                        "food item name: integer g\n(one line for every food item)"))


def parse_names(response, vocab):
    lookup = {v.lower(): v for v in vocab}
    names = []
    for line in (response or "").splitlines():
        key = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"').lower()
        if key in lookup and lookup[key] not in names:
            names.append(lookup[key])
    return names


def chat(client, model, text, image):
    return client.chat.completions.create(
        model=model, temperature=0.01, max_tokens=650,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": [{"type": "text", "text": text},
                                               {"type": "image_url", "image_url": {"url": image_url(image)}}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}).choices[0].message.content


def mass_row(client, model, row, names, blind=False, opening=None):
    response, parsed, confidence, valid = "", {}, None, False
    for attempt in (1, 2):
        response = chat(client, model, blind_prompt(attempt > 1, **({"opening": opening} if opening else {})) if blind else build_prompt(names, retry=attempt > 1), row.image)
        lines = (response or "").strip().splitlines()
        confidence, food = None, []
        for line in lines:
            m = re.fullmatch(r"\s*confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?\s*", line, flags=re.IGNORECASE)
            if m and confidence is None:
                confidence = float(m.group(1))
            else:
                food.append(line)
        parsed = parse_food_items("\n".join(food).strip())
        names_ok = bool(parsed) if blind else (isinstance(parsed, dict) and set(parsed) == set(names))
        valid = (isinstance(parsed, dict) and names_ok and all(v == v for v in parsed.values())
                 and confidence is not None and 0 <= confidence <= 100)
        if blind and isinstance(parsed, dict):
            names = list(parsed)
        if valid:
            break
    return {"dish_id": row.dish_id, "true_weight": row.true_weight, "ingredient_count": len(names),
            "ingredients_used": " | ".join(names), "raw_response": response,
            "predicted_weight": sum(parsed.values()) if valid else np.nan, "confidence": confidence,
            "format_valid": valid, "attempt_count": attempt}


def run_one(row, args, vocab, recognized):
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    if args.mode == "recognize":
        response, names = "", []
        for attempt in (1, 2):
            response = chat(client, args.model, recognize_prompt(vocab, retry=attempt > 1), row.image)
            names = parse_names(response, vocab)
            if names:
                break
        return {"dish_id": row.dish_id, "true_ingredients": row.ingredients, "predicted_ingredients": " | ".join(names),
                "raw_response": response, "attempt_count": attempt}
    if args.mode in ("blind", "blind_all"):
        return mass_row(client, args.model, row, [], blind=True, opening=NEUTRAL_OPENING if args.mode == "blind_all" else None)
    names = ([n.strip() for n in str(row.ingredients).split(" | ") if n.strip()] if args.mode == "oracle"
             else [n for n in str(recognized.get(row.dish_id, "")).split(" | ") if n])
    if not names:
        return {"dish_id": row.dish_id, "true_weight": row.true_weight, "format_valid": False, "error": "no names"}
    return mass_row(client, args.model, row, names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, help="CSV with dish_id, image, true_weight, ingredients ('a | b')")
    p.add_argument("--mode", choices=["oracle", "recognize", "predicted", "blind", "blind_all"], required=True)
    p.add_argument("--recognized", help="output of --mode recognize (needed for --mode predicted)")
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    sample = pd.read_csv(args.sample)
    vocab = vocabulary()
    recognized = (pd.read_csv(args.recognized).set_index("dish_id").predicted_ingredients.fillna("").to_dict()
                  if args.recognized else {})
    rows = pd.read_csv(args.output).to_dict("records") if os.path.isfile(args.output) else []
    done = {r["dish_id"] for r in rows if "error" not in r or r.get("error") != r.get("error")}
    rows = [r for r in rows if r["dish_id"] in done]
    pending = [r for r in sample.itertuples(index=False) if r.dish_id not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_one, r, args, vocab, recognized): r for r in pending}
        for k, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # recorded, not raised, so one bad dish never stops the run
                rows.append({"dish_id": r.dish_id, "error": str(e)})
            pd.DataFrame(rows).to_csv(args.output, index=False)
            if k % 25 == 0 or k == len(pending):
                print(f"{k}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
