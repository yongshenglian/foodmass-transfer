"""Training-median carbohydrate reference and held-out label distribution. Run from repo root."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];E=R/'experiments';O=E/'clinical_endpoints'
t=pd.read_csv(E/'qwen3vl_journal_v2_training/stage_b_imageonly_aux10_nutr/reference_oof_nutrients.csv')
assert len(t)==2253 and t.dish_id.is_unique
constant=float(t.true_carb.median())
s=pd.read_csv(O/'test_true_nutrients.csv'); c=s.true_carb.to_numpy(float)
r={'constant_source':'median carbohydrate among 2,253 Nutrition5k training dishes; no test-label tuning','constant_g':constant,'n_test':len(c),'mean_g':float(c.mean()),'quantiles_g':dict(zip(['min','q25','median','q75','q90','max'],np.quantile(c,[0,.25,.5,.75,.9,1]).tolist())),'share_at_most_10g_pct':float(np.mean(c<=10)*100),'constant_mae_g':float(np.abs(c-constant).mean()),'constant_within_10g_pct':float(np.mean(np.abs(c-constant)<=10)*100)}
(O/'carbohydrate_constant_reference.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
