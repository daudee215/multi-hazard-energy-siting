"""Spatially buffered nested evaluation of presence/background classifiers.

Output scores are relative inventory/zonation suitability, NOT event probabilities.
Every sample selection and model fit is executed inside its training region.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,confusion_matrix,roc_curve,precision_recall_curve
from xgboost import XGBClassifier
from acquire import ROOT,CFG
from prepare import cog


def get_holdout(data,hazard):
 if hazard=='wildfire' and CFG.get('wildfire_holdout')=='northern_20_percent_bbox_height':
  rows=np.indices(data['mask'].shape)[0]
  return (rows<=int(data['mask'].shape[0]*.2))&data['mask']
 return data['holdout']


def get_folds(data):
 rows,cols=np.indices(data['mask'].shape)
 return ((cols*CFG['resolution_m']//CFG['block_m'])+2*(rows*CFG['resolution_m']//CFG['block_m']))%CFG['cv_folds']


def owned_validation(base,folds,groups,fold):
 """One validation-fold owner per positive group, determined without predictors/metrics."""
 counts=np.zeros((int(groups.max())+1,int(folds.max())+1),dtype=np.int64)
 present=base&(groups>0)
 np.add.at(counts,(groups[present],folds[present]),1)
 owner=counts.argmax(axis=1)
 return base&(folds==fold)&((groups==0)|(owner[groups]==fold))


def blocked_train(base,test,groups,buffer_m=None):
 distance=ndimage.distance_transform_edt(~test)*CFG['resolution_m']
 near=distance<=(buffer_m if buffer_m is not None else CFG['exclusion_buffer_m'])
 forbidden=np.unique(groups[near]);forbidden=forbidden[forbidden>0]
 train=base&~near&~np.isin(groups,forbidden)
 assert not ((set(np.unique(groups[train]))&set(np.unique(groups[test])))-{0})
 return train


def sample(mask,groups,bg,X,strategy,seed,max_positive):
 rng=np.random.default_rng(seed)
 positive=np.flatnonzero(mask&(groups>0))
 negative=np.flatnonzero(mask&bg)
 n=min(len(positive),len(negative),max_positive)
 if n<12:raise ValueError(f'Infeasible label support: {len(positive)} presence, {len(negative)} background')
 p=rng.choice(positive,n,replace=False)
 if strategy=='uniform':b=rng.choice(negative,n,replace=False)
 else:
  # Match training presence elevation quartiles; no held-out values inform this sampling.
  edges=np.unique(np.quantile(X[p,0],[0,.25,.5,.75,1]))
  if len(edges)<3:b=rng.choice(negative,n,replace=False)
  else:
   bins=np.digitize(X[negative,0],edges[1:-1]);posbins=np.digitize(X[p,0],edges[1:-1])
   chosen=[]
   for bin_id in np.unique(posbins):
    pool=negative[bins==bin_id];count=int((posbins==bin_id).sum())
    chosen.extend(rng.choice(pool,min(len(pool),count),replace=False).tolist())
   if len(chosen)<n:
    pool=np.setdiff1d(negative,np.asarray(chosen,dtype=int))
    chosen.extend(rng.choice(pool,n-len(chosen),replace=False).tolist())
   b=np.asarray(chosen)
 ids=np.r_[p,b];y=np.r_[np.ones(n,dtype=int),np.zeros(n,dtype=int)]
 return ids,y


def model(name,params,seed):
 if name=='logistic':
  learner=LogisticRegression(C=1.,max_iter=1000,random_state=seed)
  return Pipeline([('missing',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',learner)])
 if name=='random_forest':learner=RandomForestClassifier(n_estimators=64,max_depth=params['max_depth'],min_samples_leaf=5,max_features=.8,n_jobs=2,random_state=seed)
 else:learner=XGBClassifier(n_estimators=80,max_depth=params['max_depth'],learning_rate=.07,subsample=.85,colsample_bytree=.8,
  reg_lambda=2,objective='binary:logistic',eval_metric='logloss',n_jobs=2,random_state=seed,tree_method='hist')
 return Pipeline([('missing',SimpleImputer(strategy='median')),('model',learner)])


def candidates(name):
 return [{}] if name=='logistic' else [{'max_depth':6},{'max_depth':12}] if name=='random_forest' else [{'max_depth':3},{'max_depth':5}]


def tune(name,base,groups,bg,X,blocks,strategy,seed):
 configs=candidates(name)
 if len(configs)==1:return configs[0],[]
 records=[]
 for params in configs:
  scores=[]
  for fold in range(2):
   spatial_validation=base&(blocks%2==fold)
   validate=owned_validation(base,blocks%2,groups,fold)
   train=blocked_train(base,spatial_validation,groups)
   try:
    ti,ty=sample(train,groups,bg,X,strategy,seed+fold,CFG['max_positive_train']//2)
    vi,vy=sample(validate,groups,bg,X,'uniform',seed+100+fold,1000)
    estimator=model(name,params,seed);estimator.fit(X[ti],ty)
    scores.append(float(roc_auc_score(vy,estimator.predict_proba(X[vi])[:,1])))
   except ValueError as exc:
    records.append(dict(params=params,inner_fold=fold,status='infeasible',reason=str(exc)))
  records.append(dict(params=params,inner_mean_auc=float(np.mean(scores)) if scores else None,feasible_inner_folds=len(scores)))
 valid=[r for r in records if r.get('inner_mean_auc') is not None]
 # If tuning lacks support, use the first conservative predeclared configuration and disclose it.
 return (max(valid,key=lambda r:r['inner_mean_auc'])['params'] if valid else configs[0]),records


def metrics(y,score):
 prediction=(score>=.5).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
 return dict(roc_auc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
  balanced_accuracy=float(balanced_accuracy_score(y,prediction)),sensitivity=float(tp/(tp+fn)),background_specificity=float(tn/(tn+fp)),
  tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n=len(y),n_presence=int(y.sum()),n_background=int((y==0).sum()),threshold=.5)


def block_ci(y,score,blocks,seed):
 rng=np.random.default_rng(seed);unique=np.unique(blocks);values=[]
 for _ in range(CFG['bootstrap_replicates']):
  chosen=rng.choice(unique,len(unique),replace=True)
  indices=np.concatenate([np.where(blocks==b)[0] for b in chosen])
  if len(np.unique(y[indices]))<2:continue
  values.append([roc_auc_score(y[indices],score[indices]),average_precision_score(y[indices],score[indices])])
 if not values:return dict(bootstrap_valid=0,roc_auc_low=None,roc_auc_high=None,ap_low=None,ap_high=None)
 interval=np.quantile(values,[.025,.975],axis=0)
 return dict(bootstrap_valid=len(values),roc_auc_low=float(interval[0,0]),roc_auc_high=float(interval[1,0]),ap_low=float(interval[0,1]),ap_high=float(interval[1,1]))


def predict_grid(estimator,X,valid,shape):
 result=np.full(len(X),np.nan,dtype='float32');ids=np.flatnonzero(valid)
 for begin in range(0,len(ids),30000):
  chosen=ids[begin:begin+30000];result[chosen]=estimator.predict_proba(X[chosen])[:,1]
 return result.reshape(shape)


def explain(hazard,estimator,Xtrain,Xtest,ytest,testblocks,names):
 rng=np.random.default_rng(CFG['seed'])
 base=roc_auc_score(ytest,estimator.predict_proba(Xtest)[:,1]);rows=[]
 featuregroups={'relief':['elevation_m','slope_deg','curvature_per_m','tpi_1km_m','tpi_5km_m'],
  'orientation':['aspect_sin','aspect_cos'],'land_cover_water':[n for n in names if n.startswith('clc') or n.startswith('distance')]}
 for group,features in featuregroups.items():
  columns=[names.index(n) for n in features];effects=[]
  for _ in range(5):
   perm=Xtest.copy()
   for b in np.unique(testblocks):
    ids=np.where(testblocks==b)[0];shuffled=rng.permutation(ids)
    perm[np.ix_(ids,columns)]=Xtest[np.ix_(shuffled,columns)]
   effects.append(base-roc_auc_score(ytest,estimator.predict_proba(perm)[:,1]))
  rows.append(dict(group=group,auc_drop_mean=float(np.mean(effects)),auc_drop_std=float(np.std(effects)),
   scope='Joint feature-group permutation restricted within held-out spatial blocks; fine-scale autocorrelation not preserved'))
 pd.DataFrame(rows).to_csv(ROOT/f'results/{hazard}_grouped_permutation.csv',index=False)


def shap_values(hazard,estimator,Xtest,names):
 try:
  import shap
  transformed=estimator[:-1].transform(Xtest[:100])
  values=np.asarray(shap.TreeExplainer(estimator.named_steps['model']).shap_values(transformed))
  if values.ndim==3:values=values[:,:,1]
  pd.DataFrame(dict(feature=names,mean_abs_shap=np.abs(values).mean(axis=0))).to_csv(ROOT/f'results/{hazard}_xgboost_shap.csv',index=False)
  np.savez_compressed(ROOT/f'results/{hazard}_xgboost_shap_values.npz',values=values,X=transformed,names=names)
  return dict(status='done',model='uniform-background XGBoost',samples=len(transformed),warning='Correlated feature attributions are not causal effects')
 except Exception as exc:return dict(status='blocked',reason=str(exc))


def train_hazard(hazard,data,profile,fingerprint):
 marker=ROOT/f'results/{hazard}_training_manifest.json'
 if marker.exists() and json.loads(marker.read_text()).get('fingerprint')==fingerprint:
  print(hazard,'reusing completed, input-matched run',flush=True);return
 label=np.load(ROOT/f'data/processed/{hazard}_labels.npz')
 groups,bg=label['groups'],label['background']
 valid=data['mask'];holdout=get_holdout(data,hazard);blocks=data['blocks'];folds=get_folds(data)
 cog(ROOT/f'results/{hazard}_spatial_folds.tif',np.where(valid,np.where(holdout,3,folds),np.nan),profile,
  tags=dict(fold_3='Northern 20% of bbox height, selected from label-support audit before wildfire fitting' if hazard=='wildfire' else 'Eastern independent holdout',block_m=str(CFG['block_m']),buffer_m=str(CFG['exclusion_buffer_m'])))
 feature_names=data['names'].tolist();X=data['X'].reshape(len(feature_names),-1).T
 development=blocked_train(valid&~holdout,holdout,groups)
 hi,hy=sample(holdout,groups,bg,X,'uniform',CFG['seed']+500,CFG['max_positive_test'])
 records,tuning,split_audit=[],[],[]
 fold_models={}
 begin=time.perf_counter()
 for strategy in CFG['background_strategies']:
  for fold in range(CFG['cv_folds']):
   spatial_validation=development&(folds==fold)
   validate=owned_validation(development,folds,groups,fold)
   training=blocked_train(development&~spatial_validation,spatial_validation,groups)
   try:
    ti,ty=sample(training,groups,bg,X,strategy,CFG['seed']+fold,CFG['max_positive_train'])
    vi,vy=sample(validate,groups,bg,X,'uniform',CFG['seed']+100+fold,CFG['max_positive_test'])
   except ValueError as exc:
    records.append(dict(hazard=hazard,strategy=strategy,fold=fold,split='spatial_cv',status='infeasible',reason=str(exc)));continue
   tg=set(groups.ravel()[ti]);vg=set(groups.ravel()[vi]);shared=(tg&vg)-{0}
   assert not shared
   split_audit.append(dict(strategy=strategy,fold=fold,training_pixels=len(ti),validation_pixels=len(vi),
    training_presence_groups=len(tg-{0}),validation_presence_groups=len(vg-{0}),shared_groups=len(shared),
    training_area_km2=float(training.sum()*CFG['resolution_m']**2/1e6),validation_area_km2=float(validate.sum()*CFG['resolution_m']**2/1e6)))
   pd.DataFrame(dict(pixel_id=vi,event_group=groups.ravel()[vi],y_presence_background=vy)).to_csv(ROOT/f'results/{hazard}_{strategy}_fold{fold}_validation_support.csv',index=False)
   for name in CFG['models']:
    params,inner=tune(name,training,groups,bg,X,blocks,strategy,CFG['seed']+fold)
    tuning.append(dict(strategy=strategy,outer_fold=fold,model=name,chosen=params,inner=inner))
    estimator=model(name,params,CFG['seed']+fold);estimator.fit(X[ti],ty)
    score=estimator.predict_proba(X[vi])[:,1]
    records.append(dict(hazard=hazard,strategy=strategy,model=name,fold=fold,split='spatial_cv',status='done',**metrics(vy,score)))
    fold_models.setdefault((strategy,name),[]).append(estimator)
    print(hazard,strategy,'fold',fold,name,'AUC',round(records[-1]['roc_auc'],4),'n',len(vy),flush=True)
  ti,ty=sample(development,groups,bg,X,strategy,CFG['seed'],CFG['max_positive_train'])
  for name in CFG['models']:
   params,inner=tune(name,development,groups,bg,X,blocks,strategy,CFG['seed'])
   tuning.append(dict(strategy=strategy,outer_fold='final_development',model=name,chosen=params,inner=inner))
   estimator=model(name,params,CFG['seed']);estimator.fit(X[ti],ty)
   joblib.dump(estimator,ROOT/f'models/{hazard}_{strategy}_{name}.joblib')
   score=estimator.predict_proba(X[hi])[:,1]
   records.append(dict(hazard=hazard,strategy=strategy,model=name,fold='eastern_holdout',split='independent_spatial_holdout',status='done',
    **metrics(hy,score),**block_ci(hy,score,blocks.ravel()[hi],CFG['seed'])))
   pd.DataFrame(dict(pixel_id=hi,y_presence_background=hy,relative_score=score,block=blocks.ravel()[hi],
    event_group=groups.ravel()[hi])).to_csv(ROOT/f'results/{hazard}_{strategy}_{name}_holdout.csv',index=False)
   if strategy=='uniform' and name=='xgboost':
    shap_status=shap_values(hazard,estimator,X[hi],feature_names)
   print(hazard,strategy,name,'HOLDOUT AUC',round(records[-1]['roc_auc'],4),'n',len(hy),flush=True)
 table=pd.DataFrame(records);table.to_csv(ROOT/f'results/{hazard}_evaluation.csv',index=False)
 cv=table[(table.split=='spatial_cv')&(table.status=='done')&(table.strategy=='uniform')]
 if cv.empty:raise ValueError('No feasible outer spatial validation; no validated surface can be claimed')
 primary=cv.groupby('model').roc_auc.mean().idxmax()
 estimator=joblib.load(ROOT/f'models/{hazard}_uniform_{primary}.joblib')
 surface=predict_grid(estimator,X,valid.ravel(),valid.shape)
 cog(ROOT/f'results/{hazard}_relative_suitability.tif',surface,profile,tags=dict(target='P2 regulatory zonation emulation, not independent flood prediction' if hazard=='flood' else 'inventory presence-background relative suitability',model=primary,probability='uncalibrated case-control score; not event probability'))
 reference=np.sort(surface[development]);percentile=np.searchsorted(reference,surface,side='right')/len(reference)
 percentile[~valid]=np.nan
 cog(ROOT/f'results/{hazard}_screening_percentile.tif',percentile,profile,tags=dict(scale='empirical score percentile against development-region grid cells',risk='not annual probability or expected loss'))
 predictions=[predict_grid(m,X,valid.ravel(),valid.shape) for m in fold_models[('uniform',primary)]]
 uncertainty=np.std(np.stack(predictions),axis=0)
 cog(ROOT/f'results/{hazard}_model_dispersion.tif',uncertainty,profile,tags=dict(definition='Standard deviation of outer-CV fitted models at each cell; not calibrated uncertainty interval'))
 errors=np.full(valid.size,np.nan,dtype='float32');testscore=estimator.predict_proba(X[hi])[:,1];errors[hi]=((testscore>=.5).astype(int)!=hy).astype(float)
 cog(ROOT/f'results/{hazard}_heldout_errors.tif',errors.reshape(valid.shape),profile,tags=dict(definition='sampled presence/background classification errors; background is unknown, not true absence'))
 explain(hazard,estimator,X[ti],X[hi],hy,blocks.ravel()[hi],feature_names)
 # Report held-out metrics by land-cover and broad north/south support, not just the average.
 group_reports=[]
 for kind,labels in [('land_cover',data['landcover'].ravel()[hi]),('north_south',((hi//valid.shape[1])>valid.shape[0]//2).astype(int))]:
  for value in np.unique(labels):
   keep=labels==value
   group_reports.append(dict(grouping=kind,group=str(value),n=int(keep.sum()),n_presence=int(hy[keep].sum()),
    roc_auc=float(roc_auc_score(hy[keep],testscore[keep])) if len(np.unique(hy[keep]))==2 else None))
 pd.DataFrame(group_reports).to_csv(ROOT/f'results/{hazard}_subgroup_evaluation.csv',index=False)
 pd.DataFrame(split_audit).to_csv(ROOT/f'results/{hazard}_split_audit.csv',index=False)
 (ROOT/f'results/{hazard}_tuning.json').write_text(json.dumps(tuning,indent=2))
 marker.write_text(json.dumps(dict(fingerprint=fingerprint,hazard=hazard,primary_model=primary,seed=CFG['seed'],
  development_cells=int(development.sum()),holdout_sample_n=len(hi),holdout_blocks=int(np.unique(blocks.ravel()[hi]).size),
  holdout_positive_groups=int(np.unique(groups.ravel()[hi][hy==1]).size),
  holdout_definition='northern 20% of bbox height; revised before wildfire fitting because eastern holdout had only one fire group' if hazard=='wildfire' else 'eastern approximately 20% of bbox width',
  shared_development_holdout_groups=len((set(groups[development])&set(groups[holdout]))-{0}),
  event_date_scope='2016-2024' if hazard=='wildfire' else 'undated inventory / zonation; no prospective claim',
  background='unknown, not confirmed absence',shap=shap_status,runtime_seconds=time.perf_counter()-begin),indent=2))


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--hazards',nargs='+',default=['landslide','flood','wildfire']);args=parser.parse_args()
 data=np.load(ROOT/'data/processed/stack.npz')
 with rasterio.open(ROOT/'results/predictors.tif') as src:profile=dict(crs=src.crs,transform=src.transform,width=src.width,height=src.height)
 fingerprint=hashlib.sha256((ROOT/'configs/pilot.yaml').read_bytes()+(ROOT/'data/processed/stack.npz').read_bytes()+Path(__file__).read_bytes()).hexdigest()
 for hazard in args.hazards:
  hazard_fingerprint=hashlib.sha256(fingerprint.encode()+(ROOT/f'data/processed/{hazard}_labels.npz').read_bytes()).hexdigest()
  try:train_hazard(hazard,data,profile,hazard_fingerprint)
  except Exception as exc:
   (ROOT/f'results/{hazard}_failure.json').write_text(json.dumps(dict(status='blocked',reason=str(exc)),indent=2));raise


if __name__=='__main__':main()
