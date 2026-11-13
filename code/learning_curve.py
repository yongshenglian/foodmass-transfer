"""Matched learning curve (experiments/learning_curve/PLAN.md): target-only and pooled fine-tuning with a fixed budget,
scalar calibration and frozen-feature ridge, against the number of target-domain food or tray groups.

Run from code/. Steps:
  python learning_curve.py prepare            # this workstation: frame.csv, splits and jobs.json (no GPU)
  python learning_curve.py smoke              # GPU machine: 40 steps, speed and finite predictions
  python learning_curve.py features           # GPU machine: frozen DINOv2-base features of every photograph
  python learning_curve.py worker             # GPU machine, one per GPU: claims and runs fine-tuning jobs
  python learning_curve.py status
  python learning_curve.py analyze            # this workstation, after copying preds/ and features.npy back
Environment: LC_OUT (default ../experiments/learning_curve), DINOV2_BASE, N5K_ROOT, WORKERS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("../experiments"); OUT = Path(os.environ.get("LC_OUT", E / "learning_curve")); PRED, LOCK = OUT / "preds", OUT / "locks"
DATASETS = ("asa24", "fpb"); KS = (5, 10, 25, 50, 100, 250, "all"); DRAWS = 3; STEPS, BS = 4000, 16
SPLIT_SEED, TRAIN_SEED = 20261201, 20261210; LO, HI = 10.0, 1500.0; ALPHAS = 10.0 ** np.arange(-1, 5); B = 10000


# ---------------------------------------------------------------- preparation (no GPU)
def prepare():
    sys.path.insert(0, "."); import finetune_dinov2_mass as ft
    from build_nutrition5k_reference_bank import metadata
    V1, V2 = E / "qwen3vl_journal_v1", E / "qwen3vl_journal_v2_training"
    ids = [str(i) for i in np.load(V1 / "reference_dinov2_embeddings.npz", allow_pickle=True)["dish_ids"]]
    q = pd.read_csv(V1 / "reference_train2253_qwen3vl_results.csv").drop_duplicates("dish_id").set_index("dish_id").loc[ids]
    root = Path.home() / "data/nutrition5k_dataset/imagery/realsense_overhead"
    n5k = pd.DataFrame({"dish_id": ids, "image": [str(root / i / "rgb.png") for i in ids], "true": q.true_weight.to_numpy(float),
                        "domain": "n5k", "group": "", "volume": np.load(V2 / "reference_depth_volume.npy")})
    a = pd.read_csv(E / "asa24_external/asa24_train_pool.csv")
    asa = pd.DataFrame({"dish_id": a.image_file, "image": a.image, "true": a.grams.astype(float), "domain": "asa24",
                        "group": a.food_group.astype(str), "volume": np.nan})
    f = pd.read_csv(E / "external_fpb/fpb_train_1024.csv")
    fpb = pd.DataFrame({"dish_id": f.dish_id, "image": f.image, "true": f.true_weight.astype(float), "domain": "fpb",
                        "group": f.portion.astype(str), "volume": np.nan})
    fr = pd.concat([n5k, asa, fpb], ignore_index=True)
    assert len(ids) == 2253 and len(asa) == 5099 and len(fpb) == 9190 and (fr.true > 0).all()
    fr["heldout"] = False
    for d in DATASETS:
        m = fr.domain == d; groups = np.array(sorted(fr.group[m].unique()))
        held = set(np.random.default_rng(SPLIT_SEED).permutation(groups)[:round(0.2 * len(groups))])
        fr.loc[m, "heldout"] = fr.group[m].isin(held)
        rest = np.array(sorted(set(groups) - held))
        for r in range(DRAWS):  # rank of each training group in draw r; the first k groups form the k-subset (nested)
            order = np.random.default_rng(SPLIT_SEED + 1 + r).permutation(rest)
            fr.loc[m, f"rank_d{r}"] = fr.group[m].map({g: i for i, g in enumerate(order)})
    fr.to_csv(OUT / "frame.csv", index=False)
    jobs = [{"name": f"n5k_only_d{r}", "dataset": None, "arm": "pooled", "k": 0, "draw": r} for r in range(DRAWS)]
    jobs += [{"name": f"{d}_{arm}_k{k}_d{r}", "dataset": d, "arm": arm, "k": k, "draw": r}
             for k in KS for d in DATASETS for arm in ("target", "pooled") for r in range(DRAWS)]
    (OUT / "jobs.json").write_text(json.dumps(jobs, indent=1))
    for d in DATASETS:
        m = fr.domain == d
        print(f"{d}: {fr.group[m].nunique()} groups, held out {fr.group[m & fr.heldout].nunique()} groups / {int((m & fr.heldout).sum())} photos, "
              f"training {fr.group[m & ~fr.heldout].nunique()} groups")
    print(f"{len(jobs)} fine-tuning jobs")


def rows(fr, job):
    """Training rows and evaluation rows of a job (indices into frame.csv)."""
    n5k = np.flatnonzero(fr.domain == "n5k")
    if job["k"] == 0:  # Nutrition5k-only model for calibration: predicts every ASA24 and FPB photograph
        return n5k, np.flatnonzero(fr.domain != "n5k")
    m = (fr.domain == job["dataset"]).to_numpy(); rank = fr[f"rank_d{job['draw']}"].to_numpy()
    tgt = np.flatnonzero(m & ~fr.heldout.to_numpy() & ((rank < job["k"]) if job["k"] != "all" else True))
    tr = tgt if job["arm"] == "target" else np.concatenate([n5k, tgt])
    return tr, np.flatnonzero(m & fr.heldout.to_numpy())


# ---------------------------------------------------------------- GPU
def setup():
    from concurrent.futures import ThreadPoolExecutor
    import finetune_dinov2_mass as ft
    ft.MODEL = os.environ.get("DINOV2_BASE", str(Path.home() / "models/dinov2-base")); ft.POOL = ThreadPoolExecutor(int(os.environ.get("WORKERS", 16)))
    fr = pd.read_csv(OUT / "frame.csv")
    missing = [x for x in fr.image if not os.path.isfile(x)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[0]}"
    return ft, fr


def train(ft, fr, tr, seed, steps=STEPS):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    side = np.zeros((len(fr), 2), np.float32); y = np.log(fr.true.to_numpy(float))
    vol = fr.volume.to_numpy(float); has = np.isfinite(vol); n5k = fr.domain.to_numpy() == "n5k"
    v = np.where(has, (vol - vol[n5k].mean()) / vol[n5k].std(), 0.0)
    net = ft.Net(2).to("cuda"); head = [p for n, p in net.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": 2e-5}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    stream = np.concatenate([rng.permutation(tr) for _ in range(int(np.ceil(steps * BS / len(tr))))])[:steps * BS]
    net.train(); done = 0
    for b, px, s in ft.batches(fr, side, stream, BS, True, rng):  # shuffled again inside: uniform sampling of training rows
        yt = torch.tensor(y[b], dtype=torch.float32, device="cuda"); vt = torch.tensor(v[b], dtype=torch.float32, device="cuda")
        mt = torch.tensor(has[b], dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr, pv = net(px.cuda(), s.cuda())
            loss = torch.nn.functional.smooth_l1_loss(pr.float(), yt, beta=0.1)
            if mt.sum() > 0:  # depth auxiliary loss on Nutrition5k rows only, weight 1.0
                loss = loss + (torch.nn.functional.smooth_l1_loss(pv.float(), vt, reduction="none") * mt).sum() / mt.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        done += 1
    assert done == steps, (done, steps)
    return net, side


def worker():
    import torch
    ft, fr = setup(); PRED.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for job in json.loads((OUT / "jobs.json").read_text()):
        out = PRED / f"{job['name']}.npz"
        if out.is_file():
            continue
        try:
            (LOCK / job["name"]).mkdir()  # atomic claim
        except FileExistsError:
            continue
        tr, ev = rows(fr, job); t0 = time.time()
        print(f"{time.strftime('%F %T')} gpu {gpu} start {job['name']}: train {len(tr)} evaluate {len(ev)}", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            net, side = train(ft, fr, tr, TRAIN_SEED + job["draw"])
            pred = ft.predict(net, fr, side, ev, "cuda"); del net; torch.cuda.empty_cache()
        except Exception as e:  # recorded, not raised: the lock stays until run.sh clears locks
            print(f"{time.strftime('%F %T')} gpu {gpu} FAILED {job['name']}: {e!r}", flush=True); continue
        sec = time.time() - t0
        np.savez(out.with_suffix(".tmp.npz"), index=ev, log_pred=pred, n_train=len(tr), seconds=sec, job=json.dumps(job))
        out.with_suffix(".tmp.npz").rename(out)
        e = np.abs(np.clip(np.exp(pred), LO, HI) - fr.true.to_numpy()[ev])
        print(f"{time.strftime('%F %T')} gpu {gpu} done {job['name']}: MAE {e.mean():.1f} g, {sec / 60:.1f} min", flush=True)


def smoke():
    import torch
    ft, fr = setup(); job = {"dataset": "fpb", "arm": "pooled", "k": 5, "draw": 0}; tr, ev = rows(fr, job)
    t0 = time.time(); net, side = train(ft, fr, tr, 1, steps=40); sec = (time.time() - t0) / 40
    pred = ft.predict(net, fr, side, ev[:64], "cuda")
    print(f"smoke: {sec:.2f} s per step, so about {sec * STEPS / 60:.0f} min per job; predictions finite: {np.isfinite(pred).all()}; "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    assert np.isfinite(pred).all()


def features():
    """Frozen DINOv2-base features: class token and mean patch token, as the fine-tuned network's head sees them."""
    import torch
    from transformers import AutoModel
    ft, fr = setup(); out = OUT / "features.npy"
    if out.is_file():
        return
    net = AutoModel.from_pretrained(ft.MODEL, local_files_only=True).cuda().eval(); feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, px, _ in ft.batches(fr, np.zeros((len(fr), 1), np.float32), np.arange(len(fr)), 64, False, np.random.default_rng(0)):
            h = net(pixel_values=px.cuda()).last_hidden_state; feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
    np.save(OUT / "features.tmp.npy", np.concatenate(feats).astype(np.float32)); (OUT / "features.tmp.npy").rename(out)
    print(f"features: {len(fr)} photographs -> {out}", flush=True)


def status():
    js = json.loads((OUT / "jobs.json").read_text()); done = {x.stem for x in PRED.glob("*.npz") if not x.stem.endswith(".tmp")}
    locked = {x.name for x in LOCK.glob("*")} - done if LOCK.is_dir() else set()
    sec = [float(np.load(PRED / f"{j}.npz")["seconds"]) for j in done]
    print(f"{len(done)}/{len(js)} jobs done, {len(locked)} running or failed: {sorted(locked)}; features {'done' if (OUT / 'features.npy').is_file() else 'not done'}")
    if sec:
        print(f"mean {np.mean(sec) / 60:.1f} min per finished job; about {np.mean(sec) * (len(js) - len(done)) / 3600 / 2:.1f} h left on 2 GPUs")


# ---------------------------------------------------------------- analysis (no GPU)
def ridge(X, y, tr, groups, pooled_rows):
    """Ridge on standardized features; penalty by grouped CV over the target rows tr (pooled rows always in training)."""
    def fit(rows, alphas):
        mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-6; Z = (X[rows] - mu) / sd; ym = y[rows].mean()
        U, S, Vt = np.linalg.svd(Z, full_matrices=False); uy = U.T @ (y[rows] - ym)
        return [(mu, sd, ym, Vt.T @ (S / (S ** 2 + a) * uy)) for a in alphas]
    pred = lambda m, rows: ((X[rows] - m[0]) / m[1]) @ m[3] + m[2]
    g = groups[tr]; ug = np.unique(g); nf = min(5, len(ug)); fold = {u: i % nf for i, u in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[x] for x in g]); err = np.zeros(len(ALPHAS))
    for i in range(nf):
        fits = fit(np.concatenate([pooled_rows, tr[f != i]]), ALPHAS)
        err += [np.abs(np.clip(np.exp(pred(m, tr[f == i])), LO, HI) - np.exp(y[tr[f == i]])).sum() for m in fits]
    best = ALPHAS[int(np.argmin(err))]
    return fit(np.concatenate([pooled_rows, tr]), [best])[0], pred, float(best)


def analyze():
    fr = pd.read_csv(OUT / "frame.csv"); y = np.log(fr.true.to_numpy(float)); t = fr.true.to_numpy(float); groups = fr.group.astype(str).to_numpy()
    jobs = json.loads((OUT / "jobs.json").read_text()); done = {x.stem for x in PRED.glob("*.npz") if not x.stem.endswith(".tmp")}
    X = np.load(OUT / "features.npy") if (OUT / "features.npy").is_file() else None
    n5k = np.flatnonzero(fr.domain == "n5k"); rng = np.random.default_rng(0); res = {"jobs_done": f"{len(done)}/{len(jobs)}", "datasets": {}}
    base = {}
    for r in range(DRAWS):
        if f"n5k_only_d{r}" in done:
            z = np.load(PRED / f"n5k_only_d{r}.npz"); p = np.full(len(fr), np.nan); p[z["index"]] = z["log_pred"]; base[r] = p
    for d in DATASETS:
        ev = np.flatnonzero((fr.domain == d) & fr.heldout); out = {}
        for k in (0,) + KS:
            arms = {}
            for r in range(DRAWS):
                tr, _ = rows(fr, {"dataset": d, "arm": "target", "k": k, "draw": r}) if k != 0 else (np.array([], int), None)
                if r in base:  # scalar calibration of the budget-matched Nutrition5k-only model
                    c = float(np.mean(y[tr] - base[r][tr])) if len(tr) else 0.0
                    arms.setdefault("scalar calibration", []).append(np.clip(np.exp(base[r][ev] + c), LO, HI))
                if X is not None and k != 0:
                    for arm, pooled in (("ridge, target-only", np.array([], int)), ("ridge, pooled", n5k)):
                        m, pr, a = ridge(X, y, tr, groups, pooled)
                        arms.setdefault(arm, []).append(np.clip(np.exp(pr(m, ev)), LO, HI))
                for arm, name in (("target", "fine-tuned, target-only"), ("pooled", "fine-tuned, pooled")):
                    j = f"{d}_{arm}_k{k}_d{r}"
                    if k != 0 and j in done:
                        z = np.load(PRED / f"{j}.npz"); assert np.array_equal(z["index"], ev)
                        arms.setdefault(name, []).append(np.clip(np.exp(z["log_pred"]), LO, HI))
            arms = {a: p for a, p in arms.items() if len(p) == DRAWS}
            err = {a: np.mean([np.abs(x - t[ev]) for x in p], 0) for a, p in arms.items()}  # per photograph, mean over draws
            row = {"n_target_photos": [int(len(rows(fr, {"dataset": d, "arm": "target", "k": k, "draw": r})[0])) if k else 0 for r in range(DRAWS)],
                   "mae": {a: float(e.mean()) for a, e in err.items()},
                   "mae_by_draw": {a: [float(np.abs(x - t[ev]).mean()) for x in p] for a, p in arms.items()},
                   "scale_ratio": {a: float(np.exp(np.mean([np.mean(np.log(x / t[ev])) for x in p]))) for a, p in arms.items()},
                   "within_25pct": {a: float(np.mean([np.mean(np.abs(x / t[ev] - 1) <= 0.25) for x in p]) * 100) for a, p in arms.items()}}
            row["differences"] = {f"{a} minus {b}": boot(err[a] - err[b], groups[ev], rng) for a, b in (
                ("fine-tuned, pooled", "scalar calibration"), ("fine-tuned, target-only", "scalar calibration"),
                ("fine-tuned, target-only", "fine-tuned, pooled"), ("fine-tuned, pooled", "ridge, pooled"),
                ("fine-tuned, target-only", "ridge, target-only"), ("ridge, pooled", "scalar calibration")) if a in err and b in err}
            out[str(k)] = row
        res["datasets"][d] = {"heldout_photos": int(len(ev)), "heldout_groups": int(len(np.unique(groups[ev]))), "by_k": out,
                              "retraining_beats_calibration_from_k": crossover(out)}
    (OUT / "results.json").write_text(json.dumps(res, indent=2)); figure(res); report(res)


def boot(d, g, rng):
    u, inv = np.unique(g, return_inverse=True); s = np.bincount(inv, d); n = np.bincount(inv)
    idx = rng.integers(0, len(u), (B, len(u))); m = s[idx].sum(1) / n[idx].sum(1)
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def crossover(by_k):
    """Smallest k from which the better fine-tuned arm beats calibration (interval below 0) at that and every larger k."""
    ok = {}
    for k, row in by_k.items():
        if k == "0":
            continue
        diffs = [row["differences"].get(f"fine-tuned, {a} minus scalar calibration") for a in ("pooled", "target-only")]
        diffs = [x for x in diffs if x is not None]
        if diffs:
            best = min(diffs, key=lambda x: x[0]); ok[k] = best[2] < 0
    ks = [k for k in map(str, KS) if k in ok]
    for i, k in enumerate(ks):
        if all(ok[x] for x in ks[i:]):
            return k
    return None


def figure(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    style = {"scalar calibration": ("#6b6a63", "o"), "ridge, target-only": ("#e0a458", "v"), "ridge, pooled": ("#b5651d", "^"),
             "fine-tuned, target-only": ("#7fb3e6", "s"), "fine-tuned, pooled": ("#1c4f99", "D")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, (d, x) in zip(axes, res["datasets"].items()):
        ks = [k for k in x["by_k"] if k not in ("0",)]; xs = [x["by_k"][k]["n_target_photos"][0] for k in ks]
        groups_all = {"asa24": 575, "fpb": 768}[d]; kx = [groups_all if k == "all" else int(k) for k in ks]
        for arm, (c, mk) in style.items():
            ys = [x["by_k"][k]["mae"].get(arm) for k in ks]
            if any(v is not None for v in ys):
                ax.plot([a for a, v in zip(kx, ys) if v is not None], [v for v in ys if v is not None], color=c, marker=mk, ms=4, lw=1.4, label=arm)
        if "scalar calibration" in x["by_k"]["0"]["mae"]:
            ax.axhline(x["by_k"]["0"]["mae"]["scalar calibration"], color="#c0392b", ls="--", lw=0.9, label="Nutrition5k only, uncalibrated")
        ax.set_xscale("log"); ax.set_xlabel(f"target {'foods' if d == 'asa24' else 'trays'} used for training or calibration")
        ax.set_ylabel("MAE on held-out groups (g)"); ax.set_title({"asa24": "ASA24", "fpb": "FPB"}[d], loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(True, color="#e4e3dc", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curve.pdf"); fig.savefig(OUT / "fig_learning_curve.png", dpi=160); plt.close(fig)


def report(res):
    print(f"jobs done {res['jobs_done']}")
    for d, x in res["datasets"].items():
        print(f"== {d}: held out {x['heldout_photos']} photos of {x['heldout_groups']} groups; retraining beats calibration from k = "
              f"{x['retraining_beats_calibration_from_k']}")
        for k, row in x["by_k"].items():
            print(f"  k {k:>4s} ({row['n_target_photos'][0]:5d} photos)  " + "  ".join(f"{a} {v:6.1f}" for a, v in row["mae"].items()))
            for n, v in row["differences"].items():
                print(f"{'':12s}{n:52s} {v[0]:+6.1f} [{v[1]:+.1f}, {v[2]:+.1f}]")


if __name__ == "__main__":
    {"prepare": prepare, "smoke": smoke, "features": features, "worker": worker, "status": status, "analyze": analyze}[sys.argv[1]]()
