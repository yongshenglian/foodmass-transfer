"""Evaluate leakage-safe Nutrition5k retrieval and local Qwen residual transfer.

Retrieval keys use known ingredient names (matching the frozen Qwen baseline),
Qwen-predicted ingredient shares, and lightweight RGB features. Hyperparameters
are selected using training-only queries. The locked final holdout is never read.
Run from code/.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_nutrition5k_reference_bank import visual_features


SIGNAL_LABELS = {
    "visual": "R1_handcrafted_visual",
    "ingredient_text": "R2_ingredient_text",
    "composition": "R3_qwen_composition",
    "visual_text": "R4_visual_plus_text",
    "multimodal": "R5_visual_text_composition",
}


def valid_results(path):
    frame = pd.read_csv(path).copy()
    valid_column = "format_valid" if "format_valid" in frame else "complete"
    frame = frame[frame[valid_column].astype(str).str.lower().isin({"true", "1"})].copy()
    for column in ("true_weight", "predicted_weight", "confidence", "ingredient_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["dish_id", "true_weight", "predicted_weight"])


def normalize_name(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def names_from_sample(value):
    return [normalize_name(item) for item in str(value).split(" | ") if item.strip()]


def parse_qwen_masses(response, names):
    cleaned = str(response).replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    output = {}
    for name in names:
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}\s*:\s*(-?\d+(?:\.\d+)?)\s*g(?:rams?)?\s*$",
            cleaned,
        )
        if match:
            output[name] = max(0, float(match.group(1)))
    if len(output) != len(names) or sum(output.values()) <= 0:
        return {name: 1 / len(names) for name in names} if names else {}
    total = sum(output.values())
    return {name: value / total for name, value in output.items()}


def tokens(names):
    result = set()
    for name in names:
        words = re.findall(r"[a-z0-9]+", name)
        result.update(words)
        result.add("name=" + "_".join(words))
        result.update("bigram=" + words[index] + "_" + words[index + 1] for index in range(len(words) - 1))
    return result


def unit_rows(matrix):
    matrix = np.asarray(matrix, np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1
    return matrix / norm


def build_text_vectors(bank_names, query_name_groups):
    bank_tokens = [tokens(names) for names in bank_names]
    vocabulary = sorted(set().union(*bank_tokens))
    index = {token: number for number, token in enumerate(vocabulary)}
    document_frequency = np.zeros(len(vocabulary))
    for document in bank_tokens:
        for token in document:
            document_frequency[index[token]] += 1
    idf = np.log((1 + len(bank_tokens)) / (1 + document_frequency)) + 1

    def transform(documents):
        matrix = np.zeros((len(documents), len(vocabulary)), np.float32)
        for row, document in enumerate(documents):
            for token in tokens(document):
                if token in index:
                    matrix[row, index[token]] = idf[index[token]]
        return unit_rows(matrix)

    return transform(bank_names), [transform(groups) for groups in query_name_groups]


def build_composition_vectors(bank_ingredients, query_shares_groups):
    vocabulary = sorted({normalize_name(item["name"]) for items in bank_ingredients for item in items})
    index = {name: number for number, name in enumerate(vocabulary)}
    bank = np.zeros((len(bank_ingredients), len(vocabulary)), np.float32)
    for row, items in enumerate(bank_ingredients):
        total = sum(max(0, float(item["mass_g"])) for item in items)
        if total <= 0:
            continue
        for item in items:
            name = normalize_name(item["name"])
            bank[row, index[name]] += max(0, float(item["mass_g"])) / total

    def transform(groups):
        matrix = np.zeros((len(groups), len(vocabulary)), np.float32)
        for row, shares in enumerate(groups):
            for name, value in shares.items():
                if name in index:
                    matrix[row, index[name]] += max(0, float(value))
        return unit_rows(matrix)

    return unit_rows(bank), [transform(groups) for groups in query_shares_groups]


def scale_visual(reference, queries):
    center = np.median(reference, axis=0)
    scale = np.percentile(reference, 75, axis=0) - np.percentile(reference, 25, axis=0)
    scale[scale < 1e-6] = 1
    return unit_rows((reference - center) / scale), [unit_rows((query - center) / scale) for query in queries]


def row_z(matrix):
    center = matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1
    return (matrix - center) / scale


def similarities(bank_vectors, query_vectors):
    visual_bank, text_bank, composition_bank = bank_vectors
    visual_query, text_query, composition_query = query_vectors
    visual = visual_query @ visual_bank.T
    text = text_query @ text_bank.T
    composition = composition_query @ composition_bank.T
    return {
        "visual": visual,
        "ingredient_text": text,
        "composition": composition,
        "visual_text": row_z(visual) + row_z(text),
        "multimodal": row_z(visual) + row_z(text) + row_z(composition),
    }


def weighted_median(values, weights):
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), ordered_weights.sum() / 2)])


def retrieve(similarity, values, k, allowed=None):
    predictions = np.empty(similarity.shape[0])
    top_indices = []
    for row in range(similarity.shape[0]):
        scores = similarity[row].copy()
        if allowed is not None:
            scores[~allowed[row]] = -np.inf
        finite = np.isfinite(scores)
        if not finite.any() or np.max(scores[finite]) <= 0 and np.allclose(scores[finite], 0):
            candidates = np.flatnonzero(finite)
            predictions[row] = float(np.median(values[candidates]))
            top_indices.append(candidates[:k])
            continue
        count = min(k, int(finite.sum()))
        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        selected_scores = scores[indices]
        spread = max(float(np.std(selected_scores)), 0.05)
        weights = np.exp((selected_scores - selected_scores.max()) / spread)
        predictions[row] = weighted_median(values[indices], weights)
        top_indices.append(indices)
    return predictions, top_indices


def metric(truth, prediction):
    error = np.asarray(prediction) - np.asarray(truth)
    absolute = np.abs(error)
    return {
        "mae_g": float(absolute.mean()), "median_ae_g": float(np.median(absolute)),
        "rmse_g": float(np.sqrt(np.mean(error ** 2))), "bias_g": float(error.mean()),
        "within_50g": float((absolute <= 50).mean()),
    }


def paired_ci(truth, baseline, candidate, seed, iterations=10000):
    difference = np.abs(candidate - truth) - np.abs(baseline - truth)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(truth), len(truth))
        draws[index] = difference[sample].mean()
    return float(difference.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-features", type=Path, required=True)
    parser.add_argument("--train-sample", type=Path, required=True)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--development-sample", type=Path, required=True)
    parser.add_argument("--development-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    bank = pd.read_csv(args.bank)
    feature_archive = np.load(args.bank_features)
    feature_ids = feature_archive["dish_ids"].astype(str)
    if list(feature_ids) != list(bank.dish_id.astype(str)):
        raise RuntimeError("Reference CSV and visual-feature row orders do not match")
    bank_ids = bank.dish_id.astype(str).tolist()
    bank_index = {dish_id: index for index, dish_id in enumerate(bank_ids)}
    bank_ingredients = [json.loads(value) for value in bank.ingredients_json]
    bank_names = [[normalize_name(item["name"]) for item in items] for items in bank_ingredients]

    train_sample = pd.read_csv(args.train_sample)
    train_results = valid_results(args.train_results)
    development_sample = pd.read_csv(args.development_sample)
    development_results = valid_results(args.development_results)
    train = train_sample.merge(train_results.drop(columns=["true_weight", "ingredient_count"], errors="ignore"), on="dish_id", validate="one_to_one")
    development = development_sample.merge(development_results.drop(columns=["true_weight", "ingredient_count"], errors="ignore"), on="dish_id", validate="one_to_one")
    if set(train.dish_id) & set(development.dish_id):
        raise RuntimeError("Training and development dish IDs overlap")
    if not set(train.dish_id).issubset(set(bank_ids)):
        raise RuntimeError("Every residual-training dish must belong to the reference bank")

    train_names = [names_from_sample(value) for value in train.ingredients]
    dev_names = [names_from_sample(value) for value in development.ingredients]
    train_shares = [parse_qwen_masses(response, names) for response, names in zip(train.raw_response, train_names)]
    dev_shares = [parse_qwen_masses(response, names) for response, names in zip(development.raw_response, dev_names)]
    text_bank, (text_train, text_dev) = build_text_vectors(bank_names, [train_names, dev_names])
    composition_bank, (composition_train, composition_dev) = build_composition_vectors(bank_ingredients, [train_shares, dev_shares])

    raw_visual = feature_archive["features"].astype(np.float32)
    train_visual_raw = np.stack([raw_visual[bank_index[dish_id]] for dish_id in train.dish_id.astype(str)])
    dev_visual_raw = np.stack([visual_features(path)[0] for path in development.image])
    visual_bank, (visual_train, visual_dev) = scale_visual(raw_visual, [train_visual_raw, dev_visual_raw])
    bank_vectors = (visual_bank, text_bank, composition_bank)
    train_sims = similarities(bank_vectors, (visual_train, text_train, composition_train))
    dev_sims = similarities(bank_vectors, (visual_dev, text_dev, composition_dev))

    train_truth = train.true_weight.to_numpy(float)
    train_raw = train.predicted_weight.to_numpy(float)
    dev_truth = development.true_weight.to_numpy(float)
    dev_raw = development.predicted_weight.to_numpy(float)
    bank_mass = bank.total_mass_g.to_numpy(float)
    ks = (3, 5, 10, 20)
    rows = [{"method": "E0_raw_qwen", "family": "raw", "k": 0, "lambda": 0, **metric(dev_truth, dev_raw), "mae_difference_g": 0, "ci95_low_g": 0, "ci95_high_g": 0}]
    output = development[["dish_id", "true_weight"]].copy()
    output["E0_raw_qwen"] = dev_raw
    selected = {}

    # Reference-only k is chosen on training queries after excluding each query's own bank record.
    self_allowed = np.ones((len(train), len(bank)), dtype=bool)
    for row, dish_id in enumerate(train.dish_id.astype(str)):
        self_allowed[row, bank_index[dish_id]] = False
    for signal in SIGNAL_LABELS:
        choices = {}
        for k in ks:
            prediction, _ = retrieve(train_sims[signal], bank_mass, k, self_allowed)
            choices[k] = metric(train_truth, prediction)["mae_g"]
        best_k = min(choices, key=choices.get)
        dev_prediction, _ = retrieve(dev_sims[signal], bank_mass, best_k)
        name = SIGNAL_LABELS[signal] + "_reference_only"
        difference, low, high = paired_ci(dev_truth, dev_raw, dev_prediction, args.seed)
        rows.append({"method": name, "family": "reference_only", "k": best_k, "lambda": 1, **metric(dev_truth, dev_prediction), "mae_difference_g": difference, "ci95_low_g": low, "ci95_high_g": high})
        output[name] = dev_prediction
        selected[name] = {"training_mae_by_k": choices, "selected_k": best_k}

    # Residual transfer keys use inference-available Qwen shares for both train and development.
    residual_bank_vectors = (visual_train, text_train, composition_train)
    residual_train_sims = similarities(residual_bank_vectors, residual_bank_vectors)
    residual_dev_sims = similarities(residual_bank_vectors, (visual_dev, text_dev, composition_dev))
    fold = np.asarray([int.from_bytes(dish_id.encode(), "little") % 5 for dish_id in train.dish_id.astype(str)])
    log_residual = np.log(np.maximum(train_truth, 1)) - np.log(np.maximum(train_raw, 1))
    lambdas = (0.25, 0.5, 0.75, 1.0)
    for signal in SIGNAL_LABELS:
        candidates = {}
        for k in ks:
            allowed = fold[:, None] != fold[None, :]
            transferred, _ = retrieve(residual_train_sims[signal], log_residual, k, allowed)
            for strength in lambdas:
                prediction = np.clip(train_raw * np.exp(strength * transferred), 10, 1500)
                candidates[(k, strength)] = metric(train_truth, prediction)["mae_g"]
        best_k, best_strength = min(candidates, key=candidates.get)
        transferred, neighbor_indices = retrieve(residual_dev_sims[signal], log_residual, best_k)
        dev_prediction = np.clip(dev_raw * np.exp(best_strength * transferred), 10, 1500)
        name = SIGNAL_LABELS[signal] + "_local_residual"
        difference, low, high = paired_ci(dev_truth, dev_raw, dev_prediction, args.seed)
        rows.append({"method": name, "family": "local_residual", "k": best_k, "lambda": best_strength, **metric(dev_truth, dev_prediction), "mae_difference_g": difference, "ci95_low_g": low, "ci95_high_g": high})
        output[name] = dev_prediction
        selected[name] = {
            "selected_k": best_k, "selected_lambda": best_strength,
            "best_training_oof_mae_g": candidates[(best_k, best_strength)],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("mae_g")
    result.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    output.to_csv(args.output_dir / "development_predictions.csv", index=False)
    (args.output_dir / "retrieval_configuration.json").write_text(json.dumps({
        "bank_rows": len(bank), "residual_training_rows": len(train),
        "development_rows": len(development), "signals": SIGNAL_LABELS,
        "selection": selected, "seed": args.seed,
        "visual_features": "199-dimensional handcrafted RGB v1",
        "ingredient_setting": "known ground-truth ingredient names; Qwen-predicted target shares",
    }, indent=2))
    lines = [
        "# Qwen3-VL Retrieval v1: Development Results", "",
        f"Reference dishes: {len(bank)}  ", f"Residual-training dishes: {len(train)}  ",
        f"Development dishes: {len(development)}  ", "Final holdout evaluated: no", "",
        "| Method | Family | k | Strength | MAE (g) | Change vs raw (g) | Paired 95% CI (g) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['family']} | {row['k']} | {row['lambda']} | {row['mae_g']:.2f} | {row['mae_difference_g']:+.2f} | [{row['ci95_low_g']:+.2f}, {row['ci95_high_g']:+.2f}] |")
    lines.extend(["", "All retrieval hyperparameters were selected without development labels. The locked final holdout was not read.", ""])
    (args.output_dir / "RETRIEVAL_RESULTS.md").write_text("\n".join(lines))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
