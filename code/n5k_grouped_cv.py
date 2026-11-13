"""Grouped and nested cross-validation on all 3,253 Nutrition5k dishes (experiments/n5k_grouped_cv/PLAN.md). Run from code/.

  python n5k_grouped_cv.py prepare             # this machine: pool, acquisition groups, folds, jobs, depth volumes, file lists
  python n5k_grouped_cv.py smoke               # GPU machine: one short base and one short large run, prints memory and speed
  python n5k_grouped_cv.py worker              # GPU machine, one per GPU (set CUDA_VISIBLE_DEVICES); resumable
  python n5k_grouped_cv.py status
  python n5k_grouped_cv.py qwen_sample         # GPU machine: training-dish sample for single-prompt Qwen, local image paths
  python n5k_grouped_cv.py analyze             # this machine, after the predictions are copied back

Experiment 1 trains the final configuration under random and grouped 5-fold splits of the same pool, with the same
seeds, so every dish gets an out-of-fold prediction under both. Experiment 2 is nested: 5 grouped outer folds, 4 grouped
inner folds, 5 configurations; the configuration with the lowest inner error is chosen per outer fold and the outer
fold is scored with the mean log prediction of its 4 inner models, as the paper's validation set was scored.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

E = Path("../experiments"); OUT = Path(os.environ.get("GCV_OUT", E / "n5k_grouped_cv"))
PRED, LOCK = OUT / "preds", OUT / "locks"
N5K = Path(os.environ.get("N5K_ROOT", Path.home() / "data/nutrition5k_dataset"))
CONF = Path(os.environ.get("N5K_CONF_ROOT", Path.home() / "data/n5k_confirmation_root"))  # internal test images
BACKBONES = {"base": os.environ.get("DINOV2_BASE", str(Path.home() / "models/dinov2-base")),
             "large": os.environ.get("DINOV2_LARGE", str(Path.home() / "models/dinov2-large"))}
SEEDS = (20261101, 20261102, 20261103)
GAP = 120  # s; dishes captured within GAP of each other, chained, form one acquisition group
# image-only configurations compared on the validation set during development (Table "Matched supervised baselines")
CONFIGS = {"ft_aux1": dict(backbone="base", aux_weight=1.0, freeze_backbone=False),   # the final model
           "ft_aux03": dict(backbone="base", aux_weight=0.3, freeze_backbone=False),
           "ft_aux0": dict(backbone="base", aux_weight=0.0, freeze_backbone=False),
           "frozen_aux1": dict(backbone="base", aux_weight=1.0, freeze_backbone=True),
           "large_ft_aux1": dict(backbone="large", aux_weight=1.0, freeze_backbone=False)}
FINAL = "ft_aux1"
LO, HI, QWEN_FALLBACK, B_BOOT = 10.0, 1500.0, 186.0, 10000


# ---------------------------------------------------------------- prepare (no GPU)
def group_folds(groups, k, seed):
    """Assign whole groups to k folds of near-equal size: shuffled groups, largest first, each to the smallest fold."""
    rng = np.random.default_rng(seed); u, inv, size = np.unique(groups, return_inverse=True, return_counts=True)
    order = rng.permutation(len(u)); order = order[np.argsort(-size[order], kind="stable")]
    fold_of, load = np.empty(len(u), int), np.zeros(k, int)
    for g in order:
        f = int(np.argmin(load)); fold_of[g] = f; load[f] += size[g]
    return fold_of[inv]


def chain_groups(ids, linked, gap):
    """Union of dishes captured within gap seconds of each other (and linked(a, b), if given)."""
    t = np.array([int(d.split("_")[1]) for d in ids]); o = np.argsort(t); par = list(range(len(ids)))

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a in range(len(o)):
        for b in range(a + 1, len(o)):
            i, j = o[a], o[b]
            if t[j] - t[i] > gap:
                break
            if linked is None or linked(ids[i], ids[j]):
                par[root(i)] = root(j)
    return pd.factorize(np.array([root(i) for i in range(len(ids))]))[0]


def jobs(pool):
    """Ordered so that partial results are useful: first seed of Experiment 1, the base configurations, then the rest."""
    split = lambda s: [{"name": f"split_{sc}_s{s}_f{f}", "kind": "split", "scheme": sc, "seed": s, "fold": f, "config": FINAL}
                       for f in range(5) for sc in ("grouped", "random")]
    nested = lambda cs: [{"name": f"nested_{c}_o{k}_i{j}", "kind": "nested", "config": c, "outer": k, "inner": j}
                         for c in cs for k in range(5) for j in range(4)]
    return split(SEEDS[0]) + nested([c for c in CONFIGS if c != "large_ft_aux1"]) + split(SEEDS[1]) + split(SEEDS[2]) + nested(["large_ft_aux1"])


def prepare():
    from finetune_dinov2_mass import depth_volume_proxy
    from n5k_sibling_leakage import ingredient_sets
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(E / "qwen3vl_journal_v1/reference_train2253_qwen3vl_results.csv").drop_duplicates("dish_id")
    ref = [str(i) for i in np.load(E / "qwen3vl_journal_v1/reference_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    va = pd.read_csv(E / "qwen3vl_journal_v2_training/qwen_local/development_502_sample.csv")
    te = pd.read_csv(E / "confirmation_photo_only/confirmation_498_sample.csv")
    p = pd.concat([pd.DataFrame({"dish_id": ref, "true": tr.set_index("dish_id").true_weight.reindex(ref).to_numpy(float), "set": "train", "root": "main"}),
                   pd.DataFrame({"dish_id": va.dish_id, "true": va.true_weight, "set": "validation", "root": "main"}),
                   pd.DataFrame({"dish_id": te.dish_id, "true": te.true_weight, "set": "internal_test", "root": "conf"})], ignore_index=True)
    assert p.dish_id.is_unique and len(p) == 3253 and p.true.notna().all()
    sets = ingredient_sets(); ids = list(p.dish_id)
    p["ingredients"] = [" | ".join(sorted(sets[d])) for d in ids]
    p["group"] = chain_groups(ids, None, GAP)  # acquisition groups, used for every grouped split
    p["sibling_group"] = chain_groups(ids, lambda a, b: sets[a] <= sets[b] or sets[b] <= sets[a], GAP)  # paper's sibling rule
    for s in SEEDS:
        p[f"fold_random_s{s}"] = np.random.default_rng(s).permutation(np.arange(len(p)) % 5)  # as in finetune_dinov2_mass.py
        p[f"fold_grouped_s{s}"] = group_folds(p.group.to_numpy(), 5, s)
    outer = p[f"fold_grouped_s{SEEDS[0]}"].to_numpy()
    for k in range(5):
        inner = np.full(len(p), -1); m = outer != k
        inner[m] = group_folds(p.group.to_numpy()[m], 4, SEEDS[0] + 100 + k); p[f"inner_o{k}"] = inner
    p.to_csv(OUT / "pool.csv", index=False)
    (OUT / "jobs.json").write_text(json.dumps(jobs(p), indent=1))
    vol = OUT / "depth_volume.npy"
    if not vol.is_file():  # depth is a training target only; an empty depth file gets the median volume
        v = np.array([depth_volume_proxy(str(f)) if (f := path(r, "depth_raw.png")).stat().st_size else np.nan for r in p.itertuples()])
        print(f"{np.isnan(v).sum()} empty depth files: {list(p.dish_id[np.isnan(v)])}"); v[np.isnan(v)] = np.nanmedian(v); np.save(vol, v)
    for root in ("main", "conf"):  # file lists for rsync --files-from, relative to imagery/realsense_overhead
        (OUT / f"files_{root}.txt").write_text("".join(f"{d}/{f}\n" for d in p.dish_id[p.root == root] for f in ("rgb.png", "depth_raw.png")))
    s = p.group.value_counts()
    print(f"{len(p)} dishes, {len(s)} acquisition groups (largest {s.max()}), {p.sibling_group.nunique()} sibling groups; {len(jobs(p))} jobs")
    for c in [c for c in p if c.startswith("fold_") or c.startswith("inner_")]:
        print(f"  {c:22s} fold sizes {np.bincount(p[c][p[c] >= 0]).tolist()}")


def path(r, name):
    return (N5K if r.root == "main" else CONF) / "imagery/realsense_overhead" / r.dish_id / name


def indices(p, job):
    """Training and evaluation rows of one job. Nested jobs evaluate their inner fold and their outer fold."""
    if job["kind"] == "split":
        f = p[f"fold_{job['scheme']}_s{job['seed']}"].to_numpy(); return np.flatnonzero(f != job["fold"]), np.flatnonzero(f == job["fold"])
    outer, inner = p[f"fold_grouped_s{SEEDS[0]}"].to_numpy(), p[f"inner_o{job['outer']}"].to_numpy()
    return np.flatnonzero((outer != job["outer"]) & (inner != job["inner"])), np.flatnonzero((outer == job["outer"]) | (inner == job["inner"]))


def train_seed(job):
    return job["seed"] + job["fold"] if job["kind"] == "split" else SEEDS[0] + 10 * job["outer"] + job["inner"]


# ---------------------------------------------------------------- GPU
def setup_gpu():
    from concurrent.futures import ThreadPoolExecutor
    import finetune_dinov2_mass as fd
    fd.POOL = ThreadPoolExecutor(int(os.environ.get("WORKERS", 16)))
    p = pd.read_csv(OUT / "pool.csv"); frame = pd.DataFrame({"image": [str(path(r, "rgb.png")) for r in p.itertuples()]})
    missing = [x for x in frame.image if not os.path.isfile(x)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[0]}"
    side = np.zeros((len(p), 3), np.float32)  # image-only model: side inputs are zero, as with --zero-side qwen,conf,count,ingr
    return fd, p, frame, side, np.log(p.true.to_numpy(float)), np.load(OUT / "depth_volume.npy")


def run_job(fd, frame, side, target, volume, cfg, tr, ev, seed, epochs=15):
    import torch
    fd.MODEL = BACKBONES[cfg["backbone"]]
    args = SimpleNamespace(epochs=epochs, batch_size=16, backbone_lr=2e-5, head_lr=1e-3, aux_weight=cfg["aux_weight"],
                           freeze_backbone=cfg["freeze_backbone"], nutrient_weight=0.0)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    net = fd.train_one(frame, side, target, volume, tr, "cuda", args, seed)
    pred = fd.predict(net, frame, side, ev, "cuda")
    del net; torch.cuda.empty_cache()
    return pred, time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 30


def worker():
    fd, p, frame, side, target, volume = setup_gpu(); PRED.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for job in json.loads((OUT / "jobs.json").read_text()):
        out = PRED / f"{job['name']}.npz"
        if out.is_file():
            continue
        try:
            (LOCK / job["name"]).mkdir()  # atomic claim; another worker has it otherwise
        except FileExistsError:
            continue
        tr, ev = indices(p, job); cfg = CONFIGS[job["config"]]
        print(f"{time.strftime('%F %T')} gpu {gpu} start {job['name']}: train {len(tr)} evaluate {len(ev)}", flush=True)
        try:
            pred, sec, mem = run_job(fd, frame, side, target, volume, cfg, tr, ev, train_seed(job))
        except Exception as e:  # recorded, not raised: the lock stays, so the job is retried only after run.sh clears locks
            print(f"{time.strftime('%F %T')} gpu {gpu} FAILED {job['name']}: {e!r}", flush=True); continue
        np.savez(out.with_suffix(".tmp.npz"), index=ev, log_pred=pred, seconds=sec, peak_gib=mem, job=json.dumps(job))
        out.with_suffix(".tmp.npz").rename(out)
        e = np.abs(np.clip(np.exp(pred), LO, HI) - p.true.to_numpy()[ev])
        print(f"{time.strftime('%F %T')} gpu {gpu} done {job['name']}: MAE {e.mean():.2f} g, {sec / 60:.1f} min, peak {mem:.1f} GiB", flush=True)


def smoke():
    """Two epochs on 160 dishes for the base and large backbones: checks the data, memory and speed before the real run."""
    fd, p, frame, side, target, volume = setup_gpu(); tr, ev = np.arange(0, 3253, 20)[:160], np.arange(1, 3253, 50)
    for c in ("ft_aux1", "large_ft_aux1"):
        pred, sec, mem = run_job(fd, frame, side, target, volume, CONFIGS[c], tr, ev, 1, epochs=2)
        per_epoch = sec / 2 * 2600 / len(tr) / 60  # minutes per epoch at the full training size, roughly
        print(f"{c}: peak {mem:.1f} GiB; about {per_epoch * 15:.0f} min per 15-epoch job on 2,600 dishes; predictions finite: {np.isfinite(pred).all()}", flush=True)


def status():
    js = json.loads((OUT / "jobs.json").read_text()); done = {x.stem for x in PRED.glob("*.npz") if not x.stem.endswith(".tmp")}
    locked = {x.name for x in LOCK.glob("*")} - done if LOCK.is_dir() else set()
    sec = [float(np.load(PRED / f"{j}.npz")["seconds"]) for j in done]
    print(f"{len(done)}/{len(js)} jobs done, {len(locked)} running or failed: {sorted(locked)}")
    if sec:
        print(f"mean {np.mean(sec) / 60:.1f} min per finished job")


def qwen_sample():
    p = pd.read_csv(OUT / "pool.csv"); t = p[p.set == "train"]
    sample = lambda t: pd.DataFrame({"dish_id": t.dish_id, "image": [str(path(r, "rgb.png")) for r in t.itertuples()],
                                     "true_weight": t.true, "ingredients": t.ingredients})
    sample(t).to_csv(OUT / "qwen_train_sample.csv", index=False)
    # agreement check: 200 validation dishes the paper's vLLM setup already answered, rescored with this machine's setup
    sample(p[p.set == "validation"].sample(200, random_state=SEEDS[0])).to_csv(OUT / "qwen_check_sample.csv", index=False)


# ---------------------------------------------------------------- analysis (no GPU)
def load_job(name):
    z = np.load(PRED / f"{name}.npz"); return z["index"], z["log_pred"]


def metrics(pred, true):
    e = np.abs(pred - true); r = pred / true
    return {"n": int(len(true)), "mae": float(e.mean()), "within_10pct": float(np.mean(np.abs(r - 1) <= 0.1) * 100),
            "within_20pct": float(np.mean(np.abs(r - 1) <= 0.2) * 100), "scale_ratio": float(np.exp(np.mean(np.log(r))))}


def cluster_boot(values, groups, rng):
    """Mean and 95% interval of values, resampling whole acquisition groups (B_BOOT replicates)."""
    u, inv = np.unique(groups, return_inverse=True); s = np.bincount(inv, values); n = np.bincount(inv)
    idx = rng.integers(0, len(u), (B_BOOT, len(u))); m = s[idx].sum(1) / n[idx].sum(1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975))}


def has_training_sibling(p, test_rows, train_rows):
    """True for test dishes whose sibling group (paper's rule) also has a dish among the training rows."""
    g = p.sibling_group.to_numpy(); return np.isin(g[test_rows], g[train_rows])


def near_match(p, rows, train_rows, gap):
    """True for rows with a training dish within gap seconds whose ingredient list contains or is contained in theirs
    (direct pairs, not chained). Used to drop possible rescans beyond the 120 s grouping."""
    t = np.array([int(d.split("_")[1]) for d in p.dish_id]); sets = [frozenset(s.split(" | ")) for s in p.ingredients.fillna("")]
    o = train_rows[np.argsort(t[train_rows])]; to = t[o]; out = np.zeros(len(rows), bool)
    for n, i in enumerate(rows):
        lo, hi = np.searchsorted(to, t[i] - gap), np.searchsorted(to, t[i] + gap, side="right")
        out[n] = any(sets[i] <= sets[j] or sets[j] <= sets[i] for j in o[lo:hi] if j != i)
    return out


def qwen_predictions(p):
    files = {"validation": E / "n5k_single_prompt/qwen_blind_val_fp8.csv", "internal_test": E / "n5k_single_prompt/qwen_blind_test_fp8.csv",
             "train": OUT / "qwen_blind_train_fp8.csv"}
    q = pd.Series(np.nan, index=p.dish_id)
    for s, f in files.items():
        if f.is_file():
            x = pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id").reindex(p.dish_id[p.set == s])
            v = x.predicted_weight.to_numpy(float).copy() if "predicted_weight" in x else np.full(len(x), np.nan)
            v[~(v > 0)] = QWEN_FALLBACK; q[x.index] = np.clip(v, 1, None)  # invalid answers get the training median, as in the paper
    return q.to_numpy()


def analyze():
    p = pd.read_csv(OUT / "pool.csv"); true = p.true.to_numpy(float); grp = p.group.to_numpy(); rng = np.random.default_rng(0)
    done = {x.stem for x in PRED.glob("*.npz")}; js = json.loads((OUT / "jobs.json").read_text()); res = {"jobs_done": f"{len(done)}/{len(js)}"}
    mass = lambda lp: np.clip(np.exp(lp), LO, HI)
    by_set = lambda pred: {s: metrics(pred[p.set == s], true[p.set == s]) for s in ("train", "validation", "internal_test")}

    # Experiment 1: random against grouped splitting, same pool, model and seeds
    e1, oof, clean = {}, {}, {}
    for s in SEEDS:
        for sc in ("random", "grouped"):
            names = [f"split_{sc}_s{s}_f{f}" for f in range(5)]
            if not all(n in done for n in names):
                continue
            lp = np.full(len(p), np.nan); sib = np.zeros(len(p), bool)
            for n in names:
                i, v = load_job(n); lp[i] = v
                f = p[f"fold_{sc}_s{s}"].to_numpy(); sib[i] = has_training_sibling(p, i, np.flatnonzero(f != int(n[-1])))
            oof[(sc, s)] = mass(lp)
            e1[f"{sc}_s{s}"] = {"all": metrics(oof[(sc, s)], true), "by_set": by_set(oof[(sc, s)]),
                                "share_with_training_sibling_pct": float(sib.mean() * 100), "with_sibling": sib.tolist() if sc == "random" else None}
            if sc == "grouped":  # sensitivity: dishes with no 300 s match in their training folds
                f = p[f"fold_grouped_s{s}"].to_numpy(); far = np.ones(len(p), bool)
                for k in range(5):
                    far[f == k] = ~near_match(p, np.flatnonzero(f == k), np.flatnonzero(f != k), 300)
                clean[s] = far
    paired = [s for s in SEEDS if ("random", s) in oof and ("grouped", s) in oof]
    if paired:
        d = np.mean([np.abs(oof[("random", s)] - true) - np.abs(oof[("grouped", s)] - true) for s in paired], 0)  # per dish, mean over seeds
        sib = np.mean([e1[f"random_s{s}"]["with_sibling"] for s in paired], 0) > 0.5
        e1["random_minus_grouped_mae"] = {"seeds": paired, "per_seed": [float(np.mean(np.abs(oof[("random", s)] - true) - np.abs(oof[("grouped", s)] - true))) for s in paired],
                                          "all": cluster_boot(d, grp, rng),
                                          "dishes_with_training_sibling_under_random": {"n": int(sib.sum()), **cluster_boot(d[sib], grp[sib], rng)},
                                          "dishes_without": {"n": int((~sib).sum()), **cluster_boot(d[~sib], grp[~sib], rng)}}
        c = np.all([clean[s] for s in paired], 0)  # no 300 s match in training under any seed's grouped split
        e1["no_300s_match"] = {"n": int(c.sum()), "grouped_mae": float(np.mean([np.abs(oof[("grouped", s)] - true)[c].mean() for s in paired])),
                               "random_mae": float(np.mean([np.abs(oof[("random", s)] - true)[c].mean() for s in paired])),
                               "random_minus_grouped": cluster_boot(d[c], grp[c], rng)}
    for k in e1:
        if isinstance(e1[k], dict):
            e1[k].pop("with_sibling", None)
    res["experiment1_split"] = e1

    # Experiment 2: nested grouped cross-validation
    outer = p[f"fold_grouped_s{SEEDS[0]}"].to_numpy(); e2 = {"outer_folds": {}}; sel_lp = np.full(len(p), np.nan); fixed = {}
    for k in range(5):
        inner = p[f"inner_o{k}"].to_numpy(); rows = np.flatnonzero(outer == k); info = {"inner_mae": {}}
        out_lp = {}
        for c in CONFIGS:
            names = [f"nested_{c}_o{k}_i{j}" for j in range(4)]
            if not all(n in done for n in names):
                continue
            ilp = np.full(len(p), np.nan); olp = []
            for j, n in enumerate(names):
                i, v = load_job(n); m = inner[i] == j; ilp[i[m]] = v[m]
                o = pd.Series(v[~m], index=i[~m]).reindex(rows); olp.append(o.to_numpy())
            ir = np.flatnonzero(inner >= 0); info["inner_mae"][c] = float(np.mean(np.abs(mass(ilp[ir]) - true[ir])))
            out_lp[c] = np.mean(olp, 0)  # mean log prediction of the 4 inner models
            fixed.setdefault(c, np.full(len(p), np.nan))[rows] = out_lp[c]
        if not out_lp:
            continue
        info["selected"] = min(info["inner_mae"], key=info["inner_mae"].get); info["complete_grid"] = len(out_lp) == len(CONFIGS)
        sel_lp[rows] = out_lp[info["selected"]]
        info["outer_mae"] = {c: float(np.mean(np.abs(mass(v) - true[rows]))) for c, v in out_lp.items()}
        e2["outer_folds"][k] = info
    ok = np.isfinite(sel_lp)
    if ok.all():
        sel = mass(sel_lp); q = qwen_predictions(p); hq = np.isfinite(q)
        e2["nested"] = {"all": metrics(sel, true), "mae_ci": cluster_boot(np.abs(sel - true), grp, rng), "by_set": by_set(sel)}
        e2["fixed_config"] = {c: {**metrics(mass(v), true), "mae_ci": cluster_boot(np.abs(mass(v) - true), grp, rng)}
                              for c, v in fixed.items() if np.isfinite(v).all()}
        if {"large_ft_aux1", "ft_aux1"} <= e2["fixed_config"].keys():  # selected backbone against the paper's final configuration
            e2["large_minus_final"] = cluster_boot(np.abs(mass(fixed["large_ft_aux1"]) - true) - np.abs(mass(fixed["ft_aux1"]) - true), grp, rng)
        far = np.ones(len(p), bool)
        for k in range(5):
            far[outer == k] = ~near_match(p, np.flatnonzero(outer == k), np.flatnonzero(outer != k), 300)
        e2["nested_no_300s_match"] = {**metrics(sel[far], true[far]), "mae_ci": cluster_boot(np.abs(sel - true)[far], grp[far], rng)}
        d = np.abs(sel - true) - np.abs(q - true)
        e2["dinov2_vs_qwen_same_photographs"] = {
            "n": int(hq.sum()), "sets": sorted(p.set[hq].unique()), "dinov2": metrics(sel[hq], true[hq]), "qwen": metrics(q[hq], true[hq]),
            "difference_mae": cluster_boot(d[hq], grp[hq], rng),
            "by_set": {s: {"n": int((m := hq & (p.set == s).to_numpy()).sum()), "dinov2_mae": float(np.abs(sel - true)[m].mean()),
                           "qwen_mae": float(np.abs(q - true)[m].mean()), "difference": cluster_boot(d[m], grp[m], rng)}
                       for s in ("train", "validation", "internal_test") if (hq & (p.set == s).to_numpy()).any()}}
        pd.DataFrame({"dish_id": p.dish_id, "set": p.set, "group": grp, "outer_fold": outer, "true": true, "nested_dinov2": sel, "qwen": q}).to_csv(
            OUT / "nested_predictions.csv", index=False)
    res["experiment2_nested"] = e2

    # agreement of the GPU machine's Qwen setup with the paper's, on the same 200 validation dishes
    if (f := OUT / "qwen_check_fp8.csv").is_file():
        a = pd.read_csv(f).drop_duplicates("dish_id").set_index("dish_id")
        b = pd.read_csv(E / "n5k_single_prompt/qwen_blind_val_fp8.csv").drop_duplicates("dish_id").set_index("dish_id").reindex(a.index)
        x, y, t = (np.clip(np.where(v > 0, v, QWEN_FALLBACK), 1, None) for v in (a.predicted_weight.to_numpy(float), b.predicted_weight.to_numpy(float), a.true_weight.to_numpy(float)))
        res["qwen_setup_check"] = {"n": int(len(a)), "identical_answer_pct": float(np.mean(x == y) * 100),
                                   "within_10pct_of_paper_pct": float(np.mean(np.abs(x / y - 1) <= 0.1) * 100),
                                   "mae_this_setup": float(np.abs(x - t).mean()), "mae_paper_setup": float(np.abs(y - t).mean())}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    report(res)


def report(res):
    print(f"jobs done {res['jobs_done']}")
    e1 = res["experiment1_split"]
    for k, v in e1.items():
        if "all" in v and "by_set" in v:
            print(f"  E1 {k:18s} MAE {v['all']['mae']:6.2f} g  within 10% {v['all']['within_10pct']:5.1f}%  dishes with a training sibling {v['share_with_training_sibling_pct']:5.1f}%")
    if "random_minus_grouped_mae" in e1:
        r = e1["random_minus_grouped_mae"]; f = lambda x: f"{x['mean']:+.2f} g (95% CI {x['lo']:+.2f} to {x['hi']:+.2f})"
        print(f"  E1 random minus grouped, same dishes: {f(r['all'])}; per seed {[round(x, 2) for x in r['per_seed']]}")
        print(f"     dishes with a training sibling under random ({r['dishes_with_training_sibling_under_random']['n']}): {f(r['dishes_with_training_sibling_under_random'])}")
        print(f"     dishes without ({r['dishes_without']['n']}): {f(r['dishes_without'])}")
        c = e1["no_300s_match"]
        print(f"  E1 dishes with no 300 s match in training ({c['n']}): grouped {c['grouped_mae']:.2f} g, random {c['random_mae']:.2f} g, "
              f"difference {f(c['random_minus_grouped'])}")
    e2 = res["experiment2_nested"]
    for k, v in e2["outer_folds"].items():
        print(f"  E2 outer {k}: selected {v['selected']}{'' if v['complete_grid'] else ' (grid incomplete)'}; inner " +
              ", ".join(f"{c} {m:.1f}" for c, m in v["inner_mae"].items()))
    if "nested" in e2:
        n = e2["nested"]; c = n["mae_ci"]
        print(f"  E2 nested MAE {c['mean']:.2f} g (95% CI {c['lo']:.2f} to {c['hi']:.2f}); by set " +
              ", ".join(f"{s} {m['mae']:.1f}" for s, m in n["by_set"].items()))
        c = e2["nested_no_300s_match"]["mae_ci"]
        print(f"  E2 nested MAE without dishes that have a 300 s match in training ({e2['nested_no_300s_match']['n']}): "
              f"{c['mean']:.2f} g (95% CI {c['lo']:.2f} to {c['hi']:.2f})")
        print("  E2 fixed configuration MAE: " + ", ".join(f"{c} {m['mae']:.2f}" for c, m in e2["fixed_config"].items()))
        q = e2["dinov2_vs_qwen_same_photographs"]; d = q["difference_mae"]
        print(f"  E2 same {q['n']} photographs ({', '.join(q['sets'])}): DINOv2 {q['dinov2']['mae']:.2f} g, Qwen {q['qwen']['mae']:.2f} g, "
              f"difference {d['mean']:+.2f} (95% CI {d['lo']:+.2f} to {d['hi']:+.2f})")
    if c := res.get("qwen_setup_check"):
        print(f"  Qwen setup check, {c['n']} validation dishes: identical answers {c['identical_answer_pct']:.0f}%, within 10% of the paper's "
              f"{c['within_10pct_of_paper_pct']:.0f}%, MAE {c['mae_this_setup']:.1f} g here against {c['mae_paper_setup']:.1f} g in the paper's run")


if __name__ == "__main__":
    {"prepare": prepare, "smoke": smoke, "worker": worker, "status": status, "qwen_sample": qwen_sample, "analyze": analyze}[sys.argv[1]]()
