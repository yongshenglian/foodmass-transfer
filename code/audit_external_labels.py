"""Post hoc label sensitivity with fixed saved predictions; no model selection or retraining.
Run from repository root: python code/audit_external_labels.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_asa24_external_set import read_fndds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/label_sensitivity'
OUT.mkdir(exist_ok=True)
A = ROOT / 'experiments/asa24_external'
ASA = Path.home() / 'data/asa24'
x = pd.ExcelFile(ASA / 'ASA24_ImageMetadata.xlsx')
inv = x.parse('ImageFileInventory').set_index('FileName')
raw = x.parse('AssociatedCodes').dropna(subset=['FoodCode']).copy()
raw['FoodCode'] = raw.FoodCode.astype(int)
raw['PortionCode'] = raw.PortionCode.astype(int)
fw = read_fndds('FoodWeights.txt', ['food','sub','seq','portion','start','end','grams'])
fw = fw.assign(food=fw.food.astype(int), portion=fw.portion.astype(int), grams=fw.grams.astype(float))
fw = fw[fw.grams > 0].groupby(['food','portion']).grams.median()
raw['unit_g'] = [fw.get((f,p), np.nan) for f,p in zip(raw.FoodCode,raw.PortionCode)]
raw['grams'] = raw.unit_g * raw.Multiplier
matched = raw[raw.grams.notna() & (raw.grams > 0)]
g = matched.groupby('Image')
l = pd.DataFrame({'grams':g.grams.median(), 'minimum':g.grams.min(), 'maximum':g.grams.max(), 'matched_codes':g.FoodCode.nunique()})
l['original_codes'] = raw.groupby('Image').FoodCode.nunique()
l['all_rows_mapped'] = raw.groupby('Image').grams.apply(lambda v: bool(v.notna().all() and (v > 0).all()))
l['ratio'] = l.maximum / l.minimum
flow = {'inventory_images':len(inv), 'code_rows':len(raw), 'positive_mapped_code_rows':len(matched), 'images_with_positive_mapping':len(l)}
l = l[l.ratio <= 1.25]; flow['after_agreement']=len(l)
l = l.join(inv[['ImgDescription','ImgTitle','Display','Portion']], how='inner'); flow['after_inventory_join']=len(l)
l = l[[ (ASA/'Asa24PortionImages'/i).is_file() for i in l.index ]]; flow['after_existing_image']=len(l)
l = l[l.grams.between(5,1500)]; flow['after_mass_range']=len(l)
all_labels=pd.read_csv(A/'asa24_labeled_all.csv').set_index('image_file')
assert set(l.index)==set(all_labels.index)
assert np.allclose(l.grams.loc[all_labels.index],all_labels.grams)
t = pd.read_csv(A/'asa24_aerial_sample.csv').set_index('dish_id')
d = t.join(l[['ratio','original_codes','matched_codes','all_rows_mapped']])
for name, file, col in [('n5k','finetuned_stage_b_imageonly_aux10.csv','mass'), ('multi','runs/joint_large_consensus/asa24_test_predictions.csv','mass'), ('qwen','qwen_blind_fp8_lenient_imputed.csv','predicted_weight')]:
    z=pd.read_csv(A/file).set_index('dish_id')
    assert z.index.is_unique
    d[name]=z[col].reindex(d.index)
    assert d[name].notna().all()
rows=[]
for name,mask in [('all',np.ones(len(d),bool)), ('original_single_code_complete',(d.original_codes==1)&d.all_rows_mapped), ('all_codes_complete',d.all_rows_mapped)]+[(f'agreement_{v}',d.ratio<=v) for v in [1.0,1.1,1.2,1.25]]:
    q=d.loc[mask]
    rows.append({'subset':name,'n_images':len(q),'n_foods':q.ingredients.nunique(), **{f'{m}_mae_g':float((q[m]-q.true_weight).abs().mean()) for m in ['n5k','multi','qwen']}})
pd.DataFrame(rows).to_csv(OUT/'asa24_fixed_prediction_sensitivity.csv',index=False)
# Raw source rows for the ten largest multi-domain absolute errors, including unmapped codes.
ids=d.assign(error=(d.multi-d.true_weight).abs()).nlargest(10,'error').index
raw[raw.Image.isin(ids)].to_csv(OUT/'asa24_largest_error_source_rows.csv',index=False)
d.loc[ids,['ingredients','ImgTitle','Portion','true_weight','original_codes','matched_codes','all_rows_mapped','n5k','multi','qwen']].to_csv(OUT/'asa24_largest_errors.csv')
fig,axs=plt.subplots(2,5,figsize=(15,7))
for ax,i in zip(axs.flat,ids):
    im=Image.open(ASA/'Asa24PortionImages'/i).convert('RGBA')
    bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
    ax.imshow(bg.convert('RGB')); ax.axis('off')
    z=d.loc[i];ax.set_title(f'{i}\nstandard {z.true_weight:.1f} g; model {z.multi:.1f} g\n{str(z.ImgTitle)[:65]}',fontsize=8)
fig.tight_layout();fig.savefig(OUT/'asa24_largest_errors.pdf');plt.close(fig)
# Recompute FPB signatures from the original YOLO labels; don't infer physical identity.
F=ROOT/'experiments/external_fpb'; base=Path.home()/'data/fpb/FPB_Dataset/RGB'
frames={};audit=[]
for split,file in [('train','fpb_train.csv'),('val','fpb_val_eval.csv')]:
    q=pd.read_csv(F/file);q=q[q.true_weight>0].copy();frames[split]=q
    for z in q.itertuples():
        parts=[v.split() for v in (base/split/'labels'/f'{z.dish_id}.txt').read_text().splitlines() if v.strip()]
        signature=';'.join(f'{c}:{w}' for c,w in sorted((int(v[0]),float(v[5])) for v in parts))
        audit.append({'split':split,'dish_id':z.dish_id,'signature_matches':sorted((int(v[0]),float(v[5])) for v in parts)==sorted((int(c),float(w)) for c,w in (p.split(':') for p in z.portion.split(';'))),'has_nonpositive_item':any(float(v[5])<=0 for v in parts),'mass_matches':abs(sum(float(v[5]) for v in parts)-z.true_weight)<1e-6})
pd.DataFrame(audit).to_csv(OUT/'fpb_source_label_audit.csv',index=False)
assert all(a['signature_matches'] and a['mass_matches'] for a in audit)
tr,va=frames['train'],frames['val'];shared=set(tr.portion)&set(va.portion)
fpb={'train_positive_images':len(tr),'validation_positive_images':len(va),'train_signatures':tr.portion.nunique(),'validation_signatures':va.portion.nunique(),'shared_signatures':len(shared),'validation_images_with_shared_signature':int(va.portion.isin(shared).sum()),'signature_disjoint_validation_images':int((~va.portion.isin(shared)).sum()),'source_checks':len(audit),'positive_total_images_with_nonpositive_items':sum(a['has_nonpositive_item'] for a in audit),'physical_identity_verified':False}
groups=[]
for key,q in pd.concat(frames,names=['split']).reset_index(level=0).groupby('portion'):
    groups.append({'signature':key,'n_images':len(q),'train_images':int((q.split=='train').sum()),'val_images':int((q.split=='val').sum()),'ids':' | '.join(q.dish_id)})
pd.DataFrame(groups).to_csv(OUT/'fpb_signature_groups.csv',index=False)
(OUT/'summary.json').write_text(json.dumps({'asa24_flow':flow,'asa24_test_complete_mapping_images':int(d.all_rows_mapped.sum()),'fpb':fpb,'scope':'Post hoc fixed-prediction subset sensitivity. Threshold changes do not retrain models or add previously excluded test images. Source metadata audit is not verification of physical weights or tray identity.'},indent=2)+'\n')
print(json.dumps({'flow':flow,'sensitivity':rows,'fpb':fpb},indent=2))
