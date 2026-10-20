"""Stage A: learned mass-correction heads on frozen features (no confirmation data).

Trains small models that predict Qwen's log-mass residual (or log mass directly)
from frozen DINOv2 embeddings, Qwen outputs, ingredient lists and the
out-of-fold DINOv2 neighbor residual. Model selection uses 5-fold cross-validation
on the 2,253-dish reference bank; the 502-dish development cohort is reported
as a held-out check. The 498-dish confirmation cohort is never loaded.
Run from code/ with an environment that has numpy, pandas and torch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_nutrition5k_reference_bank import DATASET_ROOT, metadata

EXP = Path("../experiments/qwen3vl_journal_v1")
K, SHRINK, LO, HI = 20, 0.75, 10.0, 1500.0


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


def neighbor_residual(sim, residuals, k=K):
    """Similarity-weighted median of neighbor residuals plus dispersion stats, per query row."""
    deltas, stats = np.empty(sim.shape[0]), np.empty((sim.shape[0], 4))
    for row in range(sim.shape[0]):
        scores = sim[row]
        finite = np.isfinite(scores)
        count = min(k, int(finite.sum()))
        idx = np.argpartition(np.where(finite, scores, -np.inf), -count)[-count:]
        s = scores[idx]
        weights = np.exp((s - s.max()) / max(float(np.std(s)), 0.05))
        r = residuals[idx]
        deltas[row] = weighted_median(r, weights)
        med = float(np.median(r))
        stats[row] = [s.max(), s.mean(), np.median(np.abs(r - med)), r.std()]
    return deltas, stats


def load_cohort(embeddings, qwen_csv, meta, mass_column="true_weight"):
    z = np.load(embeddings, allow_pickle=True)
    ids = [str(i) for i in z["dish_ids"]]
    feats = z["features"].astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    q = pd.read_csv(qwen_csv).drop_duplicates("dish_id").set_index("dish_id").loc[ids]
    frame = pd.DataFrame({
        "dish_id": ids,
        "true": q[mass_column].to_numpy(float),
        "qwen": np.clip(q["predicted_weight"].to_numpy(float), 1.0, None),
        "confidence": q["confidence"].fillna(65).to_numpy(float),
        "n_ingr": q["ingredient_count"].to_numpy(float),
        "ingr_ids": [[x["id"] for x in meta[i]["ingredients"]] for i in ids],
    })
    return frame, feats


def ingredient_matrix(groups, vocab):
    index = {v: j for j, v in enumerate(vocab)}
    m = np.zeros((len(groups), len(vocab)))
    for row, group in enumerate(groups):
        for g in group:
            if g in index:
                m[row, index[g]] = 1
    return m


def design(frame, feats, delta, stats, ingr, parts):
    blocks = []
    if "dino" in parts:
        blocks.append(feats)
    if "qwen" in parts:
        blocks.append(np.column_stack([np.log(frame.qwen), frame.confidence / 100, np.log(frame.n_ingr)]))
    if "ingr" in parts:
        blocks.append(ingr)
    if "knn" in parts:
        blocks.append(np.column_stack([delta, np.abs(delta), stats]))
    return np.column_stack(blocks)


def ridge_fit(x, y, alpha):
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-8] = 1
    zx = (x - mu) / sd
    a = np.column_stack([np.ones(len(zx)), zx])
    pen = alpha * np.eye(a.shape[1]); pen[0, 0] = 0
    return mu, sd, np.linalg.solve(a.T @ a + pen, a.T @ y)


def ridge_predict(model, x):
    mu, sd, coef = model
    return np.column_stack([np.ones(len(x)), (x - mu) / sd]) @ coef


def mlp_fit_predict(xtr, ytr, xte, seed, hidden=256, epochs=300, wd=1e-3, dropout=0.2):
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0); sd[sd < 1e-8] = 1
    xt = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), 128):
            b = perm[start:start + 128]
            # L1 on log scale tracks relative error and is robust to label outliers.
            loss = torch.nn.functional.smooth_l1_loss(net(xt[b]).squeeze(1), yt[b], beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor((xte - mu) / sd, dtype=torch.float32)).squeeze(1).numpy()


def to_mass(frame, pred, target):
    out = frame.qwen * np.exp(pred) if target == "residual" else np.exp(pred)
    return np.clip(out, LO, HI)


def mae(frame, mass):
    return float(np.mean(np.abs(mass - frame.true)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("../experiments/qwen3vl_journal_v2_training/stage_a"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = metadata(DATASET_ROOT)
    ref, ref_x = load_cohort(EXP / "reference_dinov2_embeddings.npz", EXP / "reference_train2253_qwen3vl_results.csv", meta)
    dev, dev_x = load_cohort(EXP / "development_502_dinov2_embeddings.npz",
                             EXP / "dinov2_gate_v1_full2253/development_predictions.csv", meta)
    ref_res = np.log(ref.true / ref.qwen)

    rng = np.random.default_rng(args.seed)
    fold = rng.permutation(np.arange(len(ref)) % args.folds)

    # Ingredient vocabulary from the reference bank only (ids seen >= 5 times).
    counts = pd.Series([g for group in ref.ingr_ids for g in set(group)]).value_counts()
    vocab = sorted(counts[counts >= 5].index)
    ref_ingr, dev_ingr = ingredient_matrix(ref.ingr_ids, vocab), ingredient_matrix(dev.ingr_ids, vocab)

    sim_rr = ref_x @ ref_x.T
    sim_dr = dev_x @ ref_x.T

    # Neighbor features: for CV rows, neighbors come only from other folds; for training
    # rows inside a fold, from the training folds excluding the row itself.
    def knn_feats(train_mask, query_sim, self_index=None):
        masked = np.where(train_mask[None, :], query_sim, -np.inf)
        if self_index is not None:
            masked[np.arange(len(self_index)), self_index] = -np.inf
        return neighbor_residual(masked, ref_res.to_numpy())

    candidates = []
    for target in ("residual", "log_mass"):
        for parts in (("dino",), ("dino", "qwen"), ("dino", "qwen", "ingr"),
                      ("dino", "qwen", "knn"), ("dino", "qwen", "ingr", "knn")):
            if target == "log_mass" and "qwen" not in parts and "ingr" not in parts:
                pass
            for model in ("ridge", "mlp"):
                alphas = (10, 100, 1000, 3000) if model == "ridge" else (None,)
                for alpha in alphas:
                    candidates.append((target, parts, model, alpha))

    oof_cache = {}
    results = []
    for target, parts, model, alpha in candidates:
        oof = np.empty(len(ref))
        for f in range(args.folds):
            tr, te = fold != f, fold == f
            tr_idx = np.flatnonzero(tr)
            key = f
            if key not in oof_cache:
                d_tr, s_tr = knn_feats(tr, sim_rr[tr], self_index=tr_idx)
                d_te, s_te = knn_feats(tr, sim_rr[te])
                oof_cache[key] = (d_tr, s_tr, d_te, s_te)
            d_tr, s_tr, d_te, s_te = oof_cache[key]
            sub_tr, sub_te = ref[tr].reset_index(drop=True), ref[te].reset_index(drop=True)
            x_tr = design(sub_tr, ref_x[tr], d_tr, s_tr, ref_ingr[tr], parts)
            x_te = design(sub_te, ref_x[te], d_te, s_te, ref_ingr[te], parts)
            y_tr = ref_res[tr].to_numpy() if target == "residual" else np.log(sub_tr.true.to_numpy())
            if model == "ridge":
                pred = ridge_predict(ridge_fit(x_tr, y_tr, alpha), x_te)
            else:
                pred = mlp_fit_predict(x_tr, y_tr, x_te, seed=args.seed + f)
            oof[te] = to_mass(sub_te, pred, target)
        cv = mae(ref, oof)
        results.append({"target": target, "features": "+".join(parts), "model": model,
                        "alpha": alpha, "cv_mae_g": cv})
        print(f"{target:9s} {'+'.join(parts):22s} {model:5s} alpha={alpha}  CV MAE {cv:6.2f}", flush=True)

    # Reference-CV baselines for the existing methods.
    oof_fixed = np.empty(len(ref))
    for f in range(args.folds):
        d_te = oof_cache[f][2]
        te = fold == f
        oof_fixed[te] = np.clip(ref.qwen[te] * np.exp(SHRINK * d_te), LO, HI)
    base = {"raw_qwen_cv_mae_g": mae(ref, ref.qwen.clip(LO, HI)), "fixed_residual_cv_mae_g": mae(ref, oof_fixed)}
    print(base)

    table = pd.DataFrame(results).sort_values("cv_mae_g")
    table.to_csv(args.output_dir / "reference_cv_results.csv", index=False)

    # Refit the top candidates on all reference dishes and score development once each.
    all_mask = np.ones(len(ref), bool)
    d_all, s_all = knn_feats(all_mask, sim_rr, self_index=np.arange(len(ref)))
    d_dev, s_dev = knn_feats(all_mask, sim_dr)
    dev_fixed = np.clip(dev.qwen * np.exp(SHRINK * d_dev), LO, HI)
    dev_rows = [{"method": "raw_qwen", "dev_mae_g": mae(dev, dev.qwen.clip(LO, HI))},
                {"method": "fixed_dinov2_residual", "dev_mae_g": mae(dev, dev_fixed)}]
    dev_pred = pd.DataFrame({"dish_id": dev.dish_id, "true": dev.true, "raw_qwen": dev.qwen, "fixed": dev_fixed})
    for _, row in table.head(6).iterrows():
        parts = tuple(row.features.split("+"))
        x_tr = design(ref, ref_x, d_all, s_all, ref_ingr, parts)
        x_dev = design(dev, dev_x, d_dev, s_dev, dev_ingr, parts)
        y_tr = ref_res.to_numpy() if row.target == "residual" else np.log(ref.true.to_numpy())
        if row.model == "ridge":
            pred = ridge_predict(ridge_fit(x_tr, y_tr, row.alpha), x_dev)
        else:
            pred = np.mean([mlp_fit_predict(x_tr, y_tr, x_dev, seed=args.seed + s) for s in range(3)], axis=0)
        name = f"{row.target}|{row.features}|{row.model}|{row.alpha}"
        dev_pred[name] = to_mass(dev, pred, row.target)
        dev_rows.append({"method": name, "cv_mae_g": row.cv_mae_g, "dev_mae_g": mae(dev, dev_pred[name])})
    dev_table = pd.DataFrame(dev_rows)
    print(dev_table.to_string(index=False))
    dev_table.to_csv(args.output_dir / "development_results.csv", index=False)
    dev_pred.to_csv(args.output_dir / "development_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "reference_cv_baselines": base, "folds": args.folds, "seed": args.seed,
        "ingredient_vocab_size": len(vocab), "confirmation_cohort_used": False}, indent=2))


if __name__ == "__main__":
    main()
