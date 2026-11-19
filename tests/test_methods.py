import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rio_cogeo.cogeo import cog_validate
from sklearn.impute import SimpleImputer
from train import blocked_train,sample,model,get_holdout,get_folds,owned_validation
from acquire import ROOT,CFG


def test_entire_group_is_excluded_across_split():
 groups=np.zeros((1,200),dtype=int);groups[0,10]=7;groups[0,180]=7;groups[0,2]=11
 held=np.zeros_like(groups,dtype=bool);held[0,175:]=True
 train=blocked_train(np.ones_like(held),held,groups)
 assert not train[0,10] and train[0,2]
 assert not ((set(groups[train])&set(groups[held]))-{0})
 assert not np.any(train&held)


def test_background_never_samples_known_presence():
 groups=np.zeros((1,100),dtype=int);groups[0,:25]=1
 bg=groups==0;features=np.arange(200).reshape(100,2).astype(float)
 for strategy in ['uniform','elevation_matched']:
  ids,y=sample(np.ones_like(bg),groups,bg,features,strategy,42,20)
  assert len(ids)==40 and len(np.unique(ids))==40
  assert (groups.ravel()[ids[y==1]]>0).all()
  assert (groups.ravel()[ids[y==0]]==0).all()


def test_validation_group_has_one_fold_owner():
 groups=np.array([[1,1,1,2,2,2],[1,1,0,2,0,0]])
 folds=np.array([[0,0,1,0,1,1],[2,2,0,2,0,1]])
 base=np.ones_like(groups,dtype=bool);seen=set()
 for fold in range(3):
  selected=owned_validation(base,folds,groups,fold)
  ids=set(groups[selected])-{0}
  assert not seen&ids
  seen|=ids
 assert seen=={1,2}


def test_missing_predictors_use_training_only_imputation():
 estimator=model('logistic',{},42)
 X=np.array([[1,0],[2,1],[np.nan,1],[4,0],[5,1],[6,0]],float)
 estimator.fit(X,[0,0,0,1,1,1])
 assert estimator['missing'].statistics_[0]==4
 assert np.isfinite(estimator.predict_proba([[np.nan,1]])).all()
 assert estimator['missing'].statistics_[0]==4


def test_composite_monotonicity_and_bounds():
 low=np.array([.1,.3,.4]);high=low+np.array([.2,0,0])
 for weight in [[1,1,1],[2,1,1],[1,2,1],[1,1,2]]:
  assert np.average(high,weights=weight)>=np.average(low,weights=weight)
 assert high.max()>=low.max()
 assert 0<=np.mean(low)<=1


def test_actual_raster_alignment_and_cogs():
 paths=list((ROOT/'results').glob('*.tif'))
 if not paths:pytest.skip('Run prepare.py first')
 with rasterio.open(ROOT/'results/predictors.tif') as src:
  reference=(src.crs,src.transform,src.width,src.height)
 for path in paths:
  with rasterio.open(path) as src:
   assert (src.crs,src.transform,src.width,src.height)==reference
   assert src.nodata==-9999
   assert all(shape==(256,256) for shape in src.block_shapes) and src.overviews(1)
  valid,errors,warnings=cog_validate(path,strict=True,quiet=True)
  assert valid,(str(path),errors,warnings)


def test_actual_no_shared_event_groups_and_predictor_coverage():
 path=ROOT/'data/processed/stack.npz'
 if not path.exists():pytest.skip('Run prepare.py first')
 data=np.load(path);assert np.isfinite(data['X'][:,data['mask']]).all()
 for hazard in ['landslide','flood','wildfire']:
  labels=np.load(ROOT/f'data/processed/{hazard}_labels.npz')
  groups=labels['groups'];holdout=get_holdout(data,hazard);base=data['mask']&~holdout
  train=blocked_train(base,holdout,groups)
  assert not ((set(groups[train])&set(groups[holdout]))-{0})
  seen=set()
  for fold in range(CFG['cv_folds']):
   spatial_test=train&(get_folds(data)==fold)
   test=owned_validation(train,get_folds(data),groups,fold)
   ids=set(groups[test])-{0}
   assert not seen&ids
   seen|=ids
   fitting=blocked_train(train&~spatial_test,spatial_test,groups)
   assert not ((set(groups[fitting])&set(groups[test]))-{0})


def test_figures_tables_do_not_receive_computation_dates():
 for path in (ROOT/'results').glob('*.csv'):
  assert 'computed on' not in path.read_text(encoding='utf-8')
