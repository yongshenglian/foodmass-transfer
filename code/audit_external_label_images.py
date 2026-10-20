from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'experiments/label_sensitivity';A=Path.home()/'data/asa24'
x=pd.read_excel(A/'ASA24_ImageMetadata.xlsx',sheet_name='ImageFileInventory').set_index('FileName')
ids=[i for i in x.index if 'Swordfish' in str(x.loc[i,'ImgTitle']) and x.loc[i,'Display']=='Aerial']
fig,ax=plt.subplots(2,8,figsize=(20,6))
for a,i in zip(ax.flat,ids):
 im=Image.open(A/'Asa24PortionImages'/i).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);a.imshow(bg);a.axis('off');a.set_title(f'{i}\n{x.loc[i,"ImgTitle"]}',fontsize=7)
for a in ax.flat[len(ids):]:a.axis('off')
fig.tight_layout();fig.savefig(O/'asa24_swordfish_series.pdf');plt.close(fig)
t=pd.read_csv(R/'experiments/external_fpb/fpb_train.csv');v=pd.read_csv(R/'experiments/external_fpb/fpb_val_eval.csv');t=t[t.true_weight>0];v=v[v.true_weight>0];shared=set(t.portion)&set(v.portion)
# deterministic selection of shared signatures; assess physical identity only as qualitative, not certified.
keys=sorted(shared,key=lambda k:(-(t.portion.eq(k).sum()+v.portion.eq(k).sum()),k))[:3]
fig,ax=plt.subplots(3,4,figsize=(14,10)); records=[]
for row,k in enumerate(keys):
 q=pd.concat([t[t.portion==k].head(2).assign(split='train'),v[v.portion==k].head(2).assign(split='val')])
 for a,z in zip(ax[row],q.itertuples()):
  path=Path.home()/'data/fpb/FPB_Dataset/RGB'/z.split/'images'/f'{z.dish_id}.jpg';a.imshow(Image.open(path));a.axis('off');a.set_title(f'{z.split}: {z.classes}\n{z.true_weight} g',fontsize=8);records.append({'signature':k,'split':z.split,'dish_id':z.dish_id})
 for a in ax[row,len(q):]:a.axis('off')
fig.tight_layout();fig.savefig(O/'fpb_shared_signature_examples.pdf');plt.close(fig);pd.DataFrame(records).to_csv(O/'fpb_visual_audit_selection.csv',index=False)
