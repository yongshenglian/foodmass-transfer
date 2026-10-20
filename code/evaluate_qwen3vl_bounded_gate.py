"""Train and evaluate a continuous trust gate for local Qwen residual transfer.

The gate is trained on out-of-fold neighbor corrections from the 600-dish
training cohort. Model and ridge selection never use development outcomes.
The locked final holdout is not read. Run from code/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_qwen3vl_retrieval_v1 as retrieval
from build_nutrition5k_reference_bank import visual_features


def ridge_fit(x, y, alpha):
    center = np.median(x, axis=0)
    scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    scale[scale < 1e-6] = 1
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.diag([0] + [alpha] * z.shape[1])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return center, scale, coefficients


def ridge_predict(model, x):
    center, scale, coefficients = model
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])
    return design @ coefficients


def transfer_features(similarity, residuals, k, allowed=None):
    rows, deltas = [], []
    for row in range(similarity.shape[0]):
        scores = similarity[row].copy()
        if allowed is not None:
            scores[~allowed[row]] = -np.inf
        finite = np.isfinite(scores)
        count = min(k, int(finite.sum()))
        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        selected_scores = scores[indices]
        spread = max(float(np.std(selected_scores)), 0.05)
        weights = np.exp((selected_scores - selected_scores.max()) / spread)
        selected_residuals = residuals[indices]
        delta = retrieval.weighted_median(selected_residuals, weights)
        deltas.append(delta)
        median = float(np.median(selected_residuals))
        rows.append([
            float(selected_scores.max()), float(selected_scores.mean()),
            float(selected_scores.std()), float(np.quantile(selected_scores, 0.25)),
            median, float(np.median(np.abs(selected_residuals - median))),
            float(selected_residuals.std()),
        ])
    return np.asarray(deltas), np.asarray(rows)


def gate_features(frame, delta, neighbor):
    prediction = np.maximum(frame.predicted_weight.to_numpy(float), 1)
    log_prediction = np.log(prediction)
    confidence = frame.confidence.fillna(frame.confidence.median()).to_numpy(float) / 100
    count = frame.ingredient_count.fillna(frame.ingredient_count.median()).to_numpy(float)
    hinges = np.column_stack([np.maximum(prediction - threshold, 0) / 100 for threshold in (100, 200, 300, 450)])
    return np.column_stack([
        log_prediction, confidence, np.log1p(count), delta, np.abs(delta),
        neighbor, hinges, hinges * delta[:, None],
        confidence * np.abs(delta), log_prediction * np.abs(delta),
    ])

def optimal_gate(truth, prediction, delta):
    wanted = np.log(np.maximum(truth, 1) / np.maximum(prediction, 1))
    gate = np.zeros(len(truth))
    useful = np.abs(delta) > 1e-8
    gate[useful] = wanted[useful] / delta[useful]
    return np.clip(gate, 0, 1)


def prepare(args):
    bank = pd.read_csv(args.bank)
    archive = np.load(args.bank_features)
    bank_ids = bank.dish_id.astype(str).tolist()
    if list(archive["dish_ids"].astype(str)) != bank_ids:
        raise RuntimeError("Reference feature rows do not match the reference CSV")
    bank_index = {dish_id: index for index, dish_id in enumerate(bank_ids)}
    bank_ingredients = [json.loads(value) for value in bank.ingredients_json]
    bank_names = [[retrieval.normalize_name(item["name"]) for item in items] for items in bank_ingredients]

    train_sample = pd.read_csv(args.train_sample)
    train_result = retrieval.valid_results(args.train_results)
    dev_sample = pd.read_csv(args.development_sample)
    dev_result = retrieval.valid_results(args.development_results)
    train = train_sample.merge(train_result.drop(columns=["true_weight", "ingredient_count"], errors="ignore"), on="dish_id", validate="one_to_one")
    dev = dev_sample.merge(dev_result.drop(columns=["true_weight", "ingredient_count"], errors="ignore"), on="dish_id", validate="one_to_one")
    if "ingredient_count" not in train:
        train["ingredient_count"] = train["ingredients"].map(lambda value: len(retrieval.names_from_sample(value)))
    if "ingredient_count" not in dev:
        dev["ingredient_count"] = dev["ingredients"].map(lambda value: len(retrieval.names_from_sample(value)))
    if set(train.dish_id) & set(dev.dish_id):
        raise RuntimeError("Training/development dish overlap")

    train_names = [retrieval.names_from_sample(value) for value in train.ingredients]
    dev_names = [retrieval.names_from_sample(value) for value in dev.ingredients]
    train_shares = [retrieval.parse_qwen_masses(response, names) for response, names in zip(train.raw_response, train_names)]
    dev_shares = [retrieval.parse_qwen_masses(response, names) for response, names in zip(dev.raw_response, dev_names)]
    text_bank, (text_train, text_dev) = retrieval.build_text_vectors(bank_names, [train_names, dev_names])
    composition_bank, (composition_train, composition_dev) = retrieval.build_composition_vectors(bank_ingredients, [train_shares, dev_shares])
    raw_visual = archive["features"].astype(np.float32)
    train_visual_raw = np.stack([raw_visual[bank_index[dish_id]] for dish_id in train.dish_id.astype(str)])
    dev_visual_raw = np.stack([visual_features(path)[0] for path in dev.image])
    _, (visual_train, visual_dev) = retrieval.scale_visual(raw_visual, [train_visual_raw, dev_visual_raw])
    train_vectors = (visual_train, text_train, composition_train)
    train_similarity = retrieval.similarities(train_vectors, train_vectors)["multimodal"]
    dev_similarity = retrieval.similarities(train_vectors, (visual_dev, text_dev, composition_dev))["multimodal"]
    return train, dev, train_similarity, dev_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-features", type=Path, required=True)
    parser.add_argument("--train-sample", type=Path, required=True)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--development-sample", type=Path, required=True)
    parser.add_argument("--development-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--fixed-strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    train, dev, train_similarity, dev_similarity = prepare(args)
    train_truth = train.true_weight.to_numpy(float)
    train_raw = train.predicted_weight.to_numpy(float)
    dev_truth = dev.true_weight.to_numpy(float)
    dev_raw = dev.predicted_weight.to_numpy(float)
    log_residual = np.log(np.maximum(train_truth, 1)) - np.log(np.maximum(train_raw, 1))
    fold = np.asarray([int.from_bytes(value.encode(), "little") % 5 for value in train.dish_id.astype(str)])
    allowed = fold[:, None] != fold[None, :]
    train_delta, train_neighbor = transfer_features(train_similarity, log_residual, args.neighbors, allowed)
    dev_delta, dev_neighbor = transfer_features(dev_similarity, log_residual, args.neighbors)
    train_x = gate_features(train, train_delta, train_neighbor)
    dev_x = gate_features(dev, dev_delta, dev_neighbor)
    target_gate = optimal_gate(train_truth, train_raw, train_delta)

    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    correction_bounds = (np.log(1.25), np.log(1.5), np.log(2.0), np.log(3.0), np.inf)
    alpha_scores, joint_scores, oof_gates = {}, {}, {}
    for alpha in alphas:
        gate = np.zeros(len(train))
        for held_out in range(5):
            fit = fold != held_out
            test = ~fit
            model = ridge_fit(train_x[fit], target_gate[fit], alpha)
            gate[test] = np.clip(ridge_predict(model, train_x[test]), 0, 1)
        oof_gates[alpha] = gate
        for bound in correction_bounds:
            correction = np.clip(gate * train_delta, -bound, bound)
            prediction = np.clip(train_raw * np.exp(correction), 10, 1500)
            score = retrieval.metric(train_truth, prediction)["mae_g"]
            joint_scores[(alpha, bound)] = score
            if np.isinf(bound):
                alpha_scores[alpha] = score
    selected_alpha, selected_bound = min(joint_scores, key=joint_scores.get)
    gate_model = ridge_fit(train_x, target_gate, selected_alpha)
    dev_gate = np.clip(ridge_predict(gate_model, dev_x), 0, 1)

    fixed = np.clip(dev_raw * np.exp(args.fixed_strength * dev_delta), 10, 1500)
    bounded_correction = np.clip(dev_gate * dev_delta, -selected_bound, selected_bound)
    gated = np.clip(dev_raw * np.exp(bounded_correction), 10, 1500)
    raw_metric = retrieval.metric(dev_truth, dev_raw)
    rows = []
    for name, prediction in (("raw_qwen", dev_raw), ("fixed_multimodal_residual", fixed), ("learned_bounded_gate", gated)):
        difference, low, high = retrieval.paired_ci(dev_truth, dev_raw, prediction, args.seed)
        rows.append({"method": name, **retrieval.metric(dev_truth, prediction), "mae_difference_g": difference, "ci95_low_g": low, "ci95_high_g": high})
    summary = pd.DataFrame(rows).sort_values("mae_g")
    output = dev[["dish_id", "true_weight", "predicted_weight", "confidence", "ingredient_count"]].copy()
    output["neighbor_log_residual"] = dev_delta
    output["learned_gate"] = dev_gate
    output["bounded_log_correction"] = bounded_correction
    output["fixed_multimodal_residual"] = fixed
    output["learned_bounded_gate"] = gated
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "gated_residual_summary.csv", index=False)
    output.to_csv(args.output_dir / "development_predictions.csv", index=False)
    configuration = {
        "neighbors": args.neighbors, "fixed_strength": args.fixed_strength,
        "gate_target": "per-training-row optimal bounded multiplier for OOF neighbor log residual",
        "selected_ridge_alpha": selected_alpha, "training_oof_mae_by_alpha": alpha_scores,
        "selected_max_mass_multiplier": None if np.isinf(selected_bound) else float(np.exp(selected_bound)),
        "training_oof_mae_by_alpha_and_max_multiplier": {
            f"alpha={alpha},max_multiplier={'unbounded' if np.isinf(bound) else float(np.exp(bound))}": score
            for (alpha, bound), score in joint_scores.items()
        },
        "gate_mean_development": float(dev_gate.mean()),
        "gate_zero_rate_development": float((dev_gate == 0).mean()),
        "gate_one_rate_development": float((dev_gate == 1).mean()),
        "final_holdout_evaluated": False, "seed": args.seed,
    }
    (args.output_dir / "gated_residual_configuration.json").write_text(json.dumps(configuration, indent=2))
    lines = ["# Qwen3-VL Learned Bounded Residual Gate: Development Results", "",
             "| Method | MAE (g) | Change vs raw (g) | Paired 95% CI (g) | Median AE (g) | Bias (g) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['mae_g']:.2f} | {row['mae_difference_g']:+.2f} | [{row['ci95_low_g']:+.2f}, {row['ci95_high_g']:+.2f}] | {row['median_ae_g']:.2f} | {row['bias_g']:+.2f} |")
    lines.extend(["", "All gate training and hyperparameter selection used training-only out-of-fold corrections. The locked final holdout was not evaluated.", ""])
    (args.output_dir / "GATED_RESIDUAL_RESULTS.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    main()
