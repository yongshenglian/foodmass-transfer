"""Visual check of the sibling rule (experiments/n5k_grouped_cv). Run from code/.

  python sibling_pairs_figure.py   -> ../experiments/n5k_grouped_cv/sibling_pairs_{120s,300s}.png

120s: 24 random internal-test dishes with a training sibling under the paper's rule (within 120 s, one ingredient list
contained in the other), each beside its nearest such training dish. 300s: 24 random dishes held out in the grouped
folds (seed 20261101) whose only match in training is 120-300 s away, beside that dish.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import n5k_grouped_cv as g
from n5k_sibling_leakage import ingredient_sets

N, COLS = 24, 4  # pairs, pairs per row


def matches(i, cand, t, ids, sets, lo, hi):
    """Rows in cand captured lo < |dt| <= hi from row i whose ingredient list contains or is contained in i's."""
    dt = np.abs(t[cand] - t[i]); a = sets[ids[i]]
    ok = [(lo < d <= hi) and (a <= sets[ids[c]] or sets[ids[c]] <= a) for c, d in zip(cand, dt)]
    c = cand[np.array(ok, bool)] if len(cand) else cand
    return c[np.argsort(np.abs(t[c] - t[i]))]


def plot(pairs, p, t, sets, title, out):
    rows = int(np.ceil(len(pairs) / COLS)); fig, ax = plt.subplots(rows, 2 * COLS, figsize=(2 * COLS * 1.9, rows * 1.75))
    for a in ax.ravel():
        a.axis("off")
    for k, (i, j) in enumerate(pairs):
        r, c = divmod(k, COLS)
        for s, row in enumerate((i, j)):
            x = p.iloc[row]; a = ax[r, 2 * c + s]
            a.imshow(Image.open(g.path(x, "rgb.png")).convert("RGB"))
            label = (f"#{k + 1} held out, {x.true:.0f} g" if s == 0 else f"train, {x.true:.0f} g, {abs(t[j] - t[i]):.0f} s")
            a.set_title(label, fontsize=6.5)
            a.text(0.02, 0.02, ", ".join(sorted(sets[x.dish_id]))[:42], transform=a.transAxes, fontsize=4.5, color="w",
                   bbox=dict(facecolor="k", alpha=0.6, pad=1, lw=0))
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)
    print(out)


def main():
    p = pd.read_csv(g.OUT / "pool.csv"); sets = ingredient_sets(); ids = p.dish_id.to_numpy()
    t = np.array([int(d.split("_")[1]) for d in ids]); rng = np.random.default_rng(0)
    test, train = np.flatnonzero(p.set == "internal_test"), np.flatnonzero(p.set == "train")
    pairs = []
    for i in rng.permutation(test):
        m = matches(i, train, t, ids, sets, 0, 120)
        if len(m):
            pairs.append((i, m[0]))
        if len(pairs) == N:
            break
    plot(pairs, p, t, sets, "Internal test dish (left) and its nearest training sibling (right), paper's rule: within 120 s, ingredient list contained",
         g.OUT / "sibling_pairs_120s.png")
    f = p[f"fold_grouped_s{g.SEEDS[0]}"].to_numpy(); pairs = []
    for i in rng.permutation(len(p)):
        tr = np.flatnonzero(f != f[i])
        if len(matches(i, tr, t, ids, sets, 0, 120)):
            continue  # cannot happen in grouped folds; kept as a check
        m = matches(i, tr, t, ids, sets, 120, 300)
        if len(m):
            pairs.append((i, m[0]))
        if len(pairs) == N:
            break
    plot(pairs, p, t, sets, "Grouped folds: held-out dish (left) and its nearest training match 120-300 s away (right)", g.OUT / "sibling_pairs_300s.png")


if __name__ == "__main__":
    main()
