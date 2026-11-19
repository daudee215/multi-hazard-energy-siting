"""Fixed-model wider spatial-gap robustness check; never used to select models."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from acquire import ROOT,CFG
from train import blocked_train,get_holdout,sample,metrics


def main():
 data=np.load(ROOT/'data/processed/stack.npz');X=data['X'].reshape(len(data['names']),-1).T;rows=[]
 for hazard in ['flood','landslide','wildfire']:
  info=json.loads((ROOT/f'results/{hazard}_training_manifest.json').read_text())
  name=info['primary_model'];saved=joblib.load(ROOT/f'models/{hazard}_uniform_{name}.joblib')
  label=np.load(ROOT/f'data/processed/{hazard}_labels.npz');groups,bg=label['groups'],label['background']
  hold=get_holdout(data,hazard);held=pd.read_csv(ROOT/f'results/{hazard}_uniform_{name}_holdout.csv')
  hi=held.pixel_id.to_numpy();hy=held.y_presence_background.to_numpy()
  for buffer in [5000,20000]:
   train=blocked_train(data['mask']&~hold,hold,groups,buffer_m=buffer)
   try:
    ti,ty=sample(train,groups,bg,X,'uniform',CFG['seed'],CFG['max_positive_train'])
    estimator=clone(saved);estimator.fit(X[ti],ty)
    score=estimator.predict_proba(X[hi])[:,1]
    rows.append(dict(hazard=hazard,model=name,buffer_m=buffer,training_cells=int(train.sum()),training_presence_n=int(ty.sum()),
     training_presence_groups=int(np.unique(groups.ravel()[ti][ty==1]).size),**metrics(hy,score),status='done'))
   except ValueError as exc:rows.append(dict(hazard=hazard,model=name,buffer_m=buffer,status='infeasible',reason=str(exc)))
 pd.DataFrame(rows).to_csv(ROOT/'results/spatial_buffer_sensitivity.csv',index=False)
 print(pd.DataFrame(rows)[['hazard','buffer_m','training_presence_n','roc_auc','average_precision','status']].to_string(index=False),flush=True)


if __name__=='__main__':main()
