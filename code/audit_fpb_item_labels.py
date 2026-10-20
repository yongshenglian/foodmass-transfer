"""Fixed-model sensitivity excluding FPB images with any nonpositive item weight."""
import os
from pathlib import Path
import pandas as pd
import json
R=Path(__file__).resolve().parents[1];os.chdir(R/'code')
from clinical_endpoints import external_sets
E=R/'experiments';a=pd.read_csv(E/'label_sensitivity/fpb_source_label_audit.csv');v=pd.read_csv(E/'external_fpb/fpb_val_eval.csv');v=v[~v.portion.isin(set(pd.read_csv(E/'external_fpb/fpb_train.csv').portion))]
flag=a[a.split=='val'].set_index('dish_id').has_nonpositive_item.reindex(v.dish_id).to_numpy(bool)
d=external_sets()['FPB'];assert len(flag)==len(d['t'])
r={'n_signature_disjoint':len(flag),'excluded_nonpositive_items':int(flag.sum()),'n_complete_positive_item_labels':int((~flag).sum()),'mae_g':{k:float(abs(p[~flag]-d['t'][~flag]).mean()) for k,p in d['preds'].items()},'note':'Post hoc subset only; original models unchanged, including original training labels.'}
(E/'label_sensitivity/fpb_positive_item_sensitivity.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
