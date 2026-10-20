"""Single-prompt mass estimate with an OpenAI model (second VLM; experiments/gpt6_luna/PLAN.md).

Same system prompt, user prompt, retry rule and parser as run_qwen_local.py --mode blind. Writes one row per photograph
after every photograph (resumable); per-item errors are recorded, not raised; rate limits retry with backoff. Run from code/.
"""
import argparse
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

from nutrient_estimate_mix import parse_food_items
from run_nutrition5k_confidence_selective import SYSTEM_PROMPT, image_url
from run_qwen_local import blind_prompt

load_dotenv()


def chat(client, model, text, image):
    for k in range(8):
        try:
            r = client.chat.completions.create(
                model=model, max_completion_tokens=650, reasoning_effort="none",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": [{"type": "text", "text": text},
                                                       {"type": "image_url", "image_url": {"url": image_url(image)}}]}])
            return r.choices[0].message.content, r.usage
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
            if k == 7:
                raise
            time.sleep(min(60, 2 ** k) + random.random())


def mass_row(client, model, row):
    tokens_in = tokens_out = 0
    for attempt in (1, 2):
        response, usage = chat(client, model, blind_prompt(attempt > 1), row.image)
        tokens_in += usage.prompt_tokens; tokens_out += usage.completion_tokens
        confidence, food = None, []
        for line in (response or "").strip().splitlines():
            m = re.fullmatch(r"\s*confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?\s*", line, flags=re.IGNORECASE)
            if m and confidence is None:
                confidence = float(m.group(1))
            else:
                food.append(line)
        parsed = parse_food_items("\n".join(food).strip())
        valid = (isinstance(parsed, dict) and bool(parsed) and all(v == v for v in parsed.values())
                 and confidence is not None and 0 <= confidence <= 100)
        if valid:
            break
    return {"dish_id": row.dish_id, "true_weight": row.true_weight, "ingredients_used": " | ".join(parsed) if isinstance(parsed, dict) else "",
            "raw_response": response, "predicted_weight": sum(parsed.values()) if valid else np.nan, "confidence": confidence,
            "format_valid": valid, "attempt_count": attempt, "prompt_tokens": tokens_in, "completion_tokens": tokens_out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True); p.add_argument("--output", required=True)
    p.add_argument("--model", default="gpt-6-luna"); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    sample = pd.read_csv(args.sample)
    if args.limit:
        sample = sample.sample(args.limit, random_state=0)
    client = OpenAI()
    rows = pd.read_csv(args.output).to_dict("records") if os.path.isfile(args.output) else []
    rows = [r for r in rows if not isinstance(r.get("error"), str)]
    done = {r["dish_id"] for r in rows}
    pending = [r for r in sample.itertuples(index=False) if r.dish_id not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(mass_row, client, args.model, r): r for r in pending}
        for k, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # recorded, not raised, so one bad photograph never stops the run
                rows.append({"dish_id": r.dish_id, "true_weight": r.true_weight, "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(rows).to_csv(args.output, index=False)
            if k % 50 == 0 or k == len(pending):
                print(f"{k}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
