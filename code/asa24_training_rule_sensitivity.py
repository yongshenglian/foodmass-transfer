"""Post hoc strict-versus-relaxed training-recipe sensitivity from saved runs.
Run from repository root. No model selection, new training or label correction.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'experiments/asa24_external'
OUT = ROOT / 'experiments/label_sensitivity'
sample = pd.read_csv(A / 'asa24_aerial_sample.csv').set_index('dish_id')
pools = {rule: pd.read_csv(A / name) for rule, name in
         [('strict', 'asa24_train_pool.csv'), ('relaxed', 'asa24_train_pool_consensus.csv')]}
assert set(pools['strict'].image_file) <= set(pools['relaxed'].image_file)
assert not set(sample.index) & set(pools['relaxed'].image_file)
common = set(pools['strict'].image_file)
truth = sample.true_weight.to_numpy(float)
groups = sample.series.to_numpy()
labels, cluster = np.unique(groups, return_inverse=True)
counts = np.bincount(cluster)
rows, contrasts = [], {}
def load_config(run):
    c = json.loads((A / 'runs' / run / 'config.json').read_text())
    for key, default in [('pool', 'asa24_train_pool.csv'), ('mass_weight_power', '0.0'),
                         ('text_features', 'False')]:
        c.setdefault(key, default)
    return c
for size in ['base', 'large']:
    runs = {'strict': f'joint_{size}_stretch', 'relaxed': f'joint_{size}_consensus'}
    config = {rule: load_config(run) for rule, run in runs.items()}
    ignore = {'output_dir', 'pool'}
    assert {k:v for k,v in config['strict'].items() if k not in ignore} == {
        k:v for k,v in config['relaxed'].items() if k not in ignore}
    errors = {}
    for rule, run in runs.items():
        path = A / 'runs' / run
        pred = pd.read_csv(path / 'asa24_test_predictions.csv').set_index('dish_id')
        assert set(pred.index) == set(sample.index)
        mass = pred.loc[sample.index, 'mass'].to_numpy(float)
        assert np.isfinite(mass).all()
        errors[rule] = abs(mass - truth)
        oof = pd.read_csv(path / 'asa24_pool_oof.csv').set_index('dish_id')
        subset = oof.loc[sorted(common)]
        folds, log_predictions = [], []
        for f in range(5):
            with np.load(path / f'fold{f}.npz') as z:
                assert len(z['test']) == len(sample)
                log_predictions.append(z['test'])
                # Existing trainer stores test log mass; this is the paper's model ensemble rule.
                folds.append(float(abs(np.clip(np.exp(z['test']), 10, 1500) - truth).mean()))
        assert np.allclose(np.clip(np.exp(np.mean(log_predictions, axis=0)), 10, 1500), mass, rtol=1e-6)
        rows.append({'backbone': size, 'rule': rule, 'run': run,
                     'training_photos': len(pools[rule]),
                     'training_foods': int(pools[rule].food_group.nunique()),
                     'asa24_test_mae_g': float(errors[rule].mean()),
                     'common_strict_pool_oof_mae_g': float(abs(subset.oof-subset.true).mean()),
                     'fold_test_mae_g': folds})
    d = errors['relaxed'] - errors['strict']
    sums = np.bincount(cluster, weights=d)
    draws = np.random.default_rng(20261010).integers(0, len(labels), (10000, len(labels)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    contrasts[size] = {'relaxed_minus_strict_mae_g': float(d.mean()),
                       'food_series_cluster_bootstrap_95ci_g': np.quantile(boot, [.025,.975]).tolist()}
result = {'n_test': len(sample), 'bootstrap_clusters': len(labels), 'runs': rows,
          'contrasts': contrasts,
          'scope': 'Post hoc saved-run recipe sensitivity. Matched backbone, seed and hyperparameters; pool size, ASA24 fold allocation and optimizer steps differ. Fold errors are descriptive, not independent training replicates. Confidence intervals describe evaluation-sample uncertainty only. No model selection.'}
(OUT/'asa24_training_rule_sensitivity.json').write_text(json.dumps(result, indent=2)+'\n')
pd.DataFrame([{k:v for k,v in row.items() if k != 'fold_test_mae_g'} for row in rows]).to_csv(OUT/'asa24_training_rule_sensitivity.csv', index=False)
print(json.dumps(result, indent=2))
