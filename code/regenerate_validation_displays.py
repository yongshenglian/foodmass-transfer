"""Regenerate primary Nutrition5k displays on the 502-dish validation set.
Run from repo root. Reads existing predictions; no model fitting or selection.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / 'code')
from clinical_endpoints import n5k_sets, mass_endpoints, composition_endpoints, direct_nutrients, bland_altman_figure, SP, SPK, DN, DL, load, qwen
from subgroup_errors import table, tertiles

E = ROOT / 'experiments'
OUT = E / 'validation_displays'
OUT.mkdir(exist_ok=True)
V1, V2 = E/'qwen3vl_journal_v1', E/'qwen3vl_journal_v2_training'
d = n5k_sets()['validation']
base = pd.read_csv(V1/'dinov2_gate_v1_full2253/development_predictions.csv').set_index('dish_id')
ids = base.index
t = d['t']
assert len(ids) == 502 and ids.is_unique
rows = dict(d['preds'])
ts = qwen(V2/'qwen_local/dev_predicted_fp8.csv', ids)
rows['Qwen, two-step'] = ts
rows['Qwen, two-step + kNN'] = np.clip(ts*np.exp(.75*base.neighbor_log_residual.to_numpy()),10,1500)
# Qwen reference uses matched FP8; the DINOv2 side-input model retains its saved bf16 inputs.
rows['Qwen, names given (FP8)'] = qwen(V2/'qwen_local/dev_oracle_fp8.csv',ids)
rows['Qwen, names given + kNN (FP8)'] = np.clip(rows['Qwen, names given (FP8)']*np.exp(.75*base.neighbor_log_residual.to_numpy()),10,1500)
rows['DINOv2 + Qwen estimate + names'] = load(V2/'stage_b_logmass_aux10_seed2/development_predictions.csv','mass',ids)
rows['Qwen, names given (FP8)'] = qwen(V2/'qwen_local/dev_oracle_fp8.csv',ids)
assert all(np.isfinite(p).all() and (p>0).all() for p in rows.values())
truth = d['truth']
assert truth.index.equals(ids) and np.allclose(truth.true_mass,t)
for file, col in [(V2/'stage_b_imageonly_aux10/development_predictions.csv','true'),
                   (E/'asa24_external/runs/joint_large_consensus/n5k_development_predictions.csv','true')]:
    if col in pd.read_csv(file).columns:
        assert np.allclose(load(file,col,ids),t)
idx = np.random.default_rng(0).integers(0,len(t),(10000,len(t)))
def paired(a,b):
    diff = abs(rows[a]-t)-abs(rows[b]-t)
    return [float(diff.mean()), *np.quantile(diff[idx].mean(1), [.025,.975]).tolist()]
metrics = {}
for k,p in rows.items():
    err=p-t
    metrics[k]={**mass_endpoints(p,t),'median_ae_g':float(np.median(abs(err))),
                'rmse_g':float(np.sqrt(np.mean(err**2))), 'minus_single_prompt':paired(k,SP)}
    if k in d['preds']:
        metrics[k]['oracle_composition']=composition_endpoints(p,truth)
        density=truth.true_calories.to_numpy()/t
        masks=tertiles(density,'kcal/g')
        energy_error=abs(p-t)*density
        metrics[k]['oracle_composition']['kcal_mae_by_energy_density_tertile'] = {name: float(energy_error[mask].mean()) for name,mask in masks.items()}
head=d['head']
direct=direct_nutrients(head.pred_calories.to_numpy(),head.pred_carb.to_numpy(),truth)
train=pd.read_csv(V2/'stage_b_imageonly_aux10_nutr/reference_oof_nutrients.csv')
assert train.dish_id.is_unique and len(train)==2253
constant=float(train.true_carb.median());carb=truth.true_carb.to_numpy()
carb_summary={'constant_g':constant,'mean_g':float(carb.mean()),'quantiles_g':np.quantile(carb,[0,.25,.5,.75,1]).tolist(),
              'at_most_10g_pct':float(np.mean(carb<=10)*100),'constant_mae_g':float(abs(carb-constant).mean()),
              'constant_within_10g_pct':float(np.mean(abs(carb-constant)<=10)*100)}
sample=pd.read_csv(V2/'qwen_local/development_502_sample.csv').set_index('dish_id').loc[ids]
assert np.allclose(sample.true_weight,t)
ni=sample.ingredients.str.split(r' \| ').str.len().to_numpy()
assert np.array_equal(ni,base.ingredient_count.to_numpy())
groups={'Mass <100 g':t<100,'Mass 100-299 g':(t>=100)&(t<300),'Mass >=300 g':t>=300,
        'Single ingredient':ni==1,'Mixed (>=2 ingredients)':ni>=2,'2-4 ingredients':(ni>=2)&(ni<=4),
        '5-9 ingredients':(ni>=5)&(ni<=9),'>=10 ingredients':ni>=10}
groups.update({f'Energy density {k}':v for k,v in tertiles(truth.true_calories.to_numpy()/t,'kcal/g').items()})
sub=table(t,d['preds'],groups)
sub.to_csv(OUT/'validation_subgroups.csv',index=False)
allpred=pd.DataFrame({'dish_id':ids,'true_mass_g':t,**rows})
allpred.to_csv(OUT/'validation_predictions.csv',index=False)
result={'cohort':'Nutrition5k validation (used for model selection; not an untouched test set)', 'n':len(t),
        'metrics':metrics,'direct_nutrient_head':direct,'carbohydrate':carb_summary,
        'mean_mass_g':float(t.mean()),'mean_kcal':float(truth.true_calories.mean()),
        'comparisons':{'large_minus_base':paired(DL,DN),'names_dino_minus_photo':paired('DINOv2 + Qwen estimate + names',DN),
                       'names_fp8_minus_single_prompt':paired('Qwen, names given (FP8)',SP),
                       'names_fp8_minus_two_step':paired('Qwen, names given (FP8)','Qwen, two-step')},
        'failure_rates':{k:{'relative_50pct':float(np.mean(abs(p-t)>=.5*t)*100),
                            'relative_50pct_and_kcal_100':float(np.mean((abs(p-t)>=.5*t)&(abs(p-t)*truth.true_calories.to_numpy()/t>=100))*100)} for k,p in d['preds'].items()}}
(OUT/'validation_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
bland_altman_figure(d,OUT/'fig_bland_altman_validation.pdf')
fig,axes=plt.subplots(1,4,figsize=(9.6,3),sharex=True,sharey=True)
colors=['#1baf7a','#eb6834','#2a78d6','#1c4f99']
for ax,(k,p),c in zip(axes,d['preds'].items(),colors):
    ax.plot([5,2000],[5,2000],ls='--',color='#6b6a63',lw=1)
    ax.scatter(t,p,s=9,color=c,alpha=.55,linewidths=0)
    ax.set(xscale='log',yscale='log',xlim=(5,1500),ylim=(5,1500),xlabel='Measured mass (g)')
    ax.set_title(f'{k}\nMAE {abs(p-t).mean():.2f} g',fontsize=8.5,loc='left')
    ax.grid(True,color='#e4e3dc',lw=.6);ax.set_axisbelow(True)
axes[0].set_ylabel('Predicted mass (g)')
fig.tight_layout();fig.savefig(OUT/'fig_scatter_validation_photo_only.pdf');plt.close(fig)
print(json.dumps(result,indent=2))

# Machine-generated table bodies keep rounding consistent with the saved JSON.
LINE = r' \\'
def save_body(label, lines):
    (OUT / f'{label}_body.tex').write_text('\n'.join(lines)+'\n')
def interval(c):
    return f'${c[0]:+.2f}$ (${c[1]:+.2f}$, ${c[2]:+.2f}$)'
names={SP:'Qwen3-VL-32B, single prompt',SPK:'Qwen3-VL-32B, single prompt + kNN residual correction',
       DN:'Fine-tuned DINOv2-base, Nutrition5k only',DL:'Fine-tuned DINOv2-large, multi-domain',
       'Qwen, two-step':'Qwen3-VL-32B, two-step','Qwen, two-step + kNN':'Qwen3-VL-32B, two-step + kNN residual correction',
       'Qwen, names given (FP8)':'Qwen3-VL-32B (FP8)',
       'Qwen, names given + kNN (FP8)':'Qwen3-VL-32B + kNN residual correction (FP8)',
       'DINOv2 + Qwen estimate + names':'Fine-tuned DINOv2-base + Qwen estimate + names (bf16 side inputs)'}
lines=[]
for title,ks in [('Photograph only',[SP,SPK,DN,DL]),('Photograph and closed list of Nutrition5k ingredient names',['Qwen, two-step','Qwen, two-step + kNN']),('Reference: ground-truth ingredient names given',['Qwen, names given (FP8)','Qwen, names given + kNN (FP8)','DINOv2 + Qwen estimate + names'])]:
    if lines:lines.append(r'\midrule')
    lines.append(r'\multicolumn{7}{l}{\emph{'+title+'}}'+LINE)
    for k in ks:
        m=metrics[k]
        lines.append(f"{names[k]} & {m['mae_g']:.2f} & {m['median_ae_g']:.2f} & {m['rmse_g']:.2f} & ${m['ba_bias_g']:+.2f}$ & {m['within_50g']:.1f}\\% & "+('--' if k==SP else interval(m['minus_single_prompt']))+LINE)
save_body('testphoto',lines)
lines=[]
for k in d['preds']:
    m=metrics[k];c=m['oracle_composition'];lo,hi=m['ba_ratio_loa']
    lines.append(f"{names[k]} & {m['wape_pct']:.1f}\\% & {m['within_10pct']:.1f}\\% & {m['within_20pct']:.1f}\\% & {m['within_50g']:.1f}\\% & {lo:.2f}--{hi:.2f} & {c['kcal_mae']:.1f} & {c['carb_within_10g']:.1f}\\%"+LINE)
lines += [r'\midrule',r' & & & & & & Direct energy MAE & Direct carb.\ within 10\,g'+LINE,
          f"DINOv2-base nutrient head & -- & -- & -- & -- & -- & {direct['kcal_mae']:.1f} & {direct['carb_within_10g']:.1f}\\%"+LINE,
          f"Constant carbohydrate (training median) & -- & -- & -- & -- & -- & -- & {carb_summary['constant_within_10g_pct']:.1f}\\%"+LINE]
save_body('clinical',lines)
bands=metrics[SP]['wape_by_band_pct']
lines=['Dishes & '+' & '.join(str(v['n']) for v in bands.values())+LINE,r'\midrule']
for k in d['preds']:
    lines.append(names[k]+' & '+' & '.join(f"{v['wape']:.1f}" for v in metrics[k]['wape_by_band_pct'].values())+LINE)
save_body('wapeband',lines)
labels=[r'\quad $<$100\,g',r'\quad 100--299\,g',r'\quad $\ge$300\,g',r'\quad Single ingredient',r'\quad Mixed ($\ge$2 ingredients)',r'\quad 2--4 ingredients',r'\quad 5--9 ingredients',r'\quad $\ge$10 ingredients']
lines=[]
for j,(_,row) in enumerate(sub.iterrows()):
    if j in [0,3,8]:lines.append(r'\multicolumn{6}{l}{\emph{'+{0:'Dish mass',3:'Composition',8:'Energy density'}[j]+'}}'+LINE)
    label=labels[j] if j<8 else r'\quad '+row.subgroup.replace('Energy density ','').replace('-','--')
    lines.append(label+f" & {row['n']} & "+' & '.join(f"{row[k+' MAE']:.1f} ({row[k+' WAPE']:.0f})" for k in d['preds'])+LINE)
save_body('subn5k',lines)
fold_stats={}
for k,file in [(DN,'base_n5k_n5k_val.csv'),(DL,'large_multidomain_n5k_val.csv')]:
    f=pd.read_csv(E/'fold_members'/file).set_index('dish_id').loc[ids]
    assert np.allclose(f.ensemble,d['preds'][k],rtol=1e-6)
    ae=np.abs(f[[f'fold{i}' for i in range(5)]].to_numpy()-t[:,None])
    means=ae.mean(0);gain=ae.mean(1)-abs(d['preds'][k]-t)
    fold_stats[k]={'mean_mae_g':float(means.mean()),'min_mae_g':float(means.min()),'max_mae_g':float(means.max()),
                   'single_mean_minus_ensemble_g':float(gain.mean()),'95ci_g':np.quantile(gain[idx].mean(1),[.025,.975]).tolist()}
result['fold_models']=fold_stats
result['comparisons']['base_minus_corrected_qwen']=paired(DN,SPK)
(OUT/'validation_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
