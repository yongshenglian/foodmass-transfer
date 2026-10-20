"""Evaluate Qwen weight estimates with post-hoc confidence-based selective prediction."""
import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

sys.path.insert(0, "/home/lian/DietAI24-original/code")
from nutrient_estimate_mix import parse_food_items


SYSTEM_PROMPT = (
    "You are an expert at analyzing images with computer vision. I will present you with a "
    "picture of food, which might be placed on a plate, inside a spoon, or contained within "
    "different vessels. Your job is to accurately identify the food depicted in the image."
)


def image_url(path):
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


def build_prompt(names, retry=False):
    retry_text = (
        " Your previous response was not in the required format. Include the confidence line "
        "and every listed ingredient exactly once."
        if retry else ""
    )
    return (
        "The image displays one serving of a dish containing these known ground-truth ingredients: "
        + json.dumps(names)
        + ". Make a best-effort integer gram estimate for EACH listed ingredient, including "
          "ingredients that are mixed in, hidden, or present in small amounts. Also report one "
          "dish-level confidence score from 0 to 100. Define confidence as your estimated percent "
          "probability that the SUM of your ingredient estimates is within 50 grams of the true "
          "total food weight. Judge confidence from visible boundaries, portion separability, "
          "available scale cues, and uncertainty from hidden ingredients. Do not inflate confidence. "
          "Use 90-100 only for unusually clear portions with reliable scale cues; 80-89 for mostly "
          "clear portions with moderate ambiguity; 60-79 for mixed foods, weak scale cues, or "
          "meaningful hidden ingredients; and 0-59 for severe ambiguity. Always provide the gram "
          "estimates even when confidence is low; filtering is performed externally. Respond with "
          "exactly these lines and no other text:\nCONFIDENCE: integer%\n"
          "exact ingredient name: integer g\n(one ingredient line for every listed item)"
        + retry_text
    )


def parse_response(response, names):
    lines = (response or "").strip().splitlines()
    confidence = None
    food_lines = []
    for line in lines:
        match = re.fullmatch(r"\s*confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?\s*", line,
                             flags=re.IGNORECASE)
        if match and confidence is None:
            confidence = float(match.group(1))
        else:
            food_lines.append(line)
    parsed = parse_food_items("\n".join(food_lines).strip())
    exact_names = isinstance(parsed, dict) and set(parsed) == set(names)
    finite = exact_names and all(value == value for value in parsed.values())
    valid_confidence = confidence is not None and 0 <= confidence <= 100
    return parsed, confidence, bool(finite and valid_confidence)


def estimate(row, model, base_url, threshold, max_attempts):
    names = [name.strip() for name in str(row.ingredients).split(" | ") if name.strip()]
    client = OpenAI(base_url=base_url, api_key="not-needed")
    response = ""
    parsed = {}
    confidence = None
    format_valid = False
    attempt_count = 0
    for attempt_count in range(1, max_attempts + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": build_prompt(names, retry=attempt_count > 1)},
                    {"type": "image_url", "image_url": {"url": image_url(row.image)}},
                ]},
            ],
            temperature=0.01,
            max_tokens=650,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).choices[0].message.content
        parsed, confidence, format_valid = parse_response(response, names)
        if format_valid:
            break
    predicted_weight = sum(parsed.values()) if format_valid else float("nan")
    accepted = bool(format_valid and confidence >= threshold)
    return {
        "dish_id": row.dish_id,
        "true_weight": row.true_weight,
        "ingredient_count": len(names),
        "raw_response": response,
        "predicted_weight": predicted_weight,
        "confidence": confidence,
        "format_valid": format_valid,
        "complete": accepted,
        "confidence_threshold": threshold,
        "attempt_count": attempt_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=80)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    rows = pd.read_csv(args.output).to_dict("records") if os.path.isfile(args.output) else []
    done = {row["dish_id"] for row in rows}
    pending = [row for row in sample.itertuples(index=False) if row.dish_id not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                estimate, row, args.model, args.base_url,
                args.confidence_threshold, args.max_attempts,
            ): row
            for row in pending
        }
        for number, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({
                    "dish_id": row.dish_id,
                    "true_weight": row.true_weight,
                    "format_valid": False,
                    "complete": False,
                    "error": str(error),
                })
            pd.DataFrame(rows).to_csv(args.output, index=False)
            if number % 10 == 0 or number == len(pending):
                print(f"{number}/{len(pending)} new; {len(rows)}/{len(sample)} total", flush=True)

    result = pd.DataFrame(rows)
    valid = result[result.format_valid.fillna(False)].copy()
    accepted = result[result.complete.fillna(False)].copy()
    error = accepted.predicted_weight - accepted.true_weight
    print(f"format_valid={len(valid)}/{len(result)} ({len(valid)/len(result):.1%})")
    print(f"accepted={len(accepted)}/{len(result)} ({len(accepted)/len(result):.1%})")
    if len(accepted):
        print(f"MAE={error.abs().mean():.2f} g; median={error.abs().median():.2f} g; "
              f"bias={error.mean():+.2f} g; WAPE={error.abs().sum()/accepted.true_weight.sum():.2%}")


if __name__ == "__main__":
    main()
