"""Aligned projected raster stack and audited presence/background label support."""
from __future__ import annotations
import hashlib,json,math,zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize,geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject,Resampling
from rasterio.shutil import copy as rio_copy
from scipy import ndimage,sparse
from scipy.sparse.csgraph import connected_components
from shapely import make_valid
from acquire import ROOT,CFG


def cog(path,array,profile,descriptions=None,tags=None):
 """Write with GDAL COG driver, then validate externally with rio-cogeo."""
 if array.ndim==2:array=array[None,:,:]
 temp=path.with_suffix('.working.tif')
 meta=dict(profile,driver='GTiff',count=array.shape[0],dtype='float32',nodata=-9999.,tiled=True,blockxsize=256,blockysize=256,compress='deflate')
 with rasterio.open(temp,'w',**meta) as dst:
  dst.write(np.where(np.isfinite(array),array,-9999).astype('float32'))
  if descriptions:
   for i,name in enumerate(descriptions,1):dst.set_band_description(i,name)
  dst.update_tags(**(tags or {}))
 rio_copy(temp,path,driver='COG',BLOCKSIZE=256,COMPRESS='DEFLATE',OVERVIEWS='AUTO',RESAMPLING='AVERAGE')
 temp.unlink()


def grid(study):
 res=CFG['resolution_m'];w,s,e,n=study.total_bounds
 w,s,e,n=[math.floor(w/res)*res,math.floor(s/res)*res,math.ceil(e/res)*res,math.ceil(n/res)*res]
 h,width=round((n-s)/res),round((e-w)/res)
 profile=dict(crs=CFG['crs'],transform=from_origin(w,n,res,res),height=h,width=width)
 mask=geometry_mask(study.geometry,invert=True,out_shape=(h,width),transform=profile['transform'],all_touched=False)
 return profile,mask


def terrain(profile):
 arr=np.full((profile['height'],profile['width']),np.nan,dtype='float32')
 for path in sorted((ROOT/'data/raw').glob('Copernicus_DSM_COG_30_*.tif')):
  with rasterio.open(path) as src:
   tmp=np.full_like(arr,np.nan)
   reproject(rasterio.band(src,1),tmp,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
    dst_transform=profile['transform'],dst_crs=profile['crs'],dst_nodata=np.nan,resampling=Resampling.average)
   arr[np.isfinite(tmp)]=tmp[np.isfinite(tmp)]
 if not np.isfinite(arr).all():
  indices=ndimage.distance_transform_edt(~np.isfinite(arr),return_distances=False,return_indices=True)
  arr=arr[tuple(indices)]
 res=CFG['resolution_m'];gy,gx=np.gradient(arr.astype('float64'),res)
 slope=np.degrees(np.arctan(np.hypot(gx,gy)))
 aspect=np.arctan2(-gx,gy)
 features=dict(elevation_m=arr,slope_deg=slope,aspect_sin=np.sin(aspect),aspect_cos=np.cos(aspect),
  curvature_per_m=ndimage.laplace(arr.astype('float64'))/res**2,
  tpi_1km_m=arr-ndimage.uniform_filter(arr,size=5),tpi_5km_m=arr-ndimage.uniform_filter(arr,size=21))
 return features


def labels(hazard,profile,mask):
 frame=gpd.read_file(ROOT/f'data/processed/{hazard}_events.gpkg',layer='events')
 frame=frame.reset_index(drop=True)
 frame['training_eligible']=True
 if hazard=='wildfire':
  frame['training_eligible']=pd.to_datetime(frame.event_date,utc=True).dt.year>=CFG['fire_model_min_year']
 # Merge strictly overlapping polygons into groups; mere shared boundaries are not overlaps.
 pairs=frame.sindex.query(frame.geometry,predicate='overlaps')
 graph=sparse.coo_matrix((np.ones(pairs.shape[1]),(pairs[0],pairs[1])),shape=(len(frame),len(frame))).tocsr()
 n,groups=connected_components(graph,directed=False)
 frame['group_id']=groups+1
 eligible=frame[frame.training_eligible]
 shapes=((geom,int(group)) for geom,group in zip(eligible.geometry,eligible.group_id))
 group_grid=rasterize(shapes,out_shape=mask.shape,transform=profile['transform'],fill=0,dtype='int32',all_touched=False)
 group_grid[~mask]=0
 # Preserve complete original event footprints in background exclusion, even if no cell centre lies inside.
 footprint=rasterize(((g,1) for g in frame.geometry),out_shape=mask.shape,transform=profile['transform'],fill=0,dtype='uint8',all_touched=True).astype(bool)
 distance=ndimage.distance_transform_edt(~footprint)*CFG['resolution_m']
 bg=(distance>CFG['inventory_uncertainty_buffer_m'])&mask
 frame['supported_by_centres']=frame.group_id.isin(np.unique(group_grid[group_grid>0]))
 frame.drop(columns='geometry').to_csv(ROOT/f'results/{hazard}_label_records.csv',index=False)
 np.savez_compressed(ROOT/f'data/processed/{hazard}_labels.npz',groups=group_grid,background=bg)
 # Actual mask geometry, not arbitrary mock points, is exported as a COG.
 labelmap=np.where(mask,np.where(group_grid>0,1,np.where(bg,0,np.nan)),np.nan)
 cog(ROOT/f'results/{hazard}_label_support.tif',labelmap,profile,tags=dict(definition='1=presence; 0=eligible background, NOT confirmed absence; nodata=unknown/excluded',provenance='independent reconstruction'))
 return dict(hazard=hazard,polygon_records=len(frame),overlap_groups=n,groups_with_grid_support=int(np.unique(group_grid[group_grid>0]).size),
  records_without_centre_support=int((~frame.supported_by_centres).sum()),positive_cells=int((group_grid>0).sum()),eligible_background_cells=int(bg.sum()),unknown_in_region_cells=int((mask&~bg&(group_grid==0)).sum()))


def population(profile,mask):
 paths=list((ROOT/'data/raw').glob('GHS_POP*.zip'))
 if not paths:return False
 path=paths[0]
 with zipfile.ZipFile(path) as z:name=next(n for n in z.namelist() if n.endswith('.tif'))
 with rasterio.open(f'/vsizip/{path.as_posix()}/{name}') as src:
  arr=np.full(mask.shape,np.nan,dtype='float32')
  reproject(rasterio.band(src,1),arr,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
   dst_transform=profile['transform'],dst_crs=profile['crs'],dst_nodata=np.nan,resampling=Resampling.average)
 # Native 100m equal-area cell counts / 0.01km² -> residential density.
 arr*=100;arr[~mask]=np.nan
 cog(ROOT/'results/population_context_density.tif',arr,profile,tags=dict(units='approximate residents/km2; 2020',
  source='GHS-POP R2023A 100m equal-area counts, area-average reprojected to analysis grid',
  warning='Context only; not social vulnerability, grid customers, or a training predictor'))
 return True


def main():
 study=gpd.read_file(ROOT/'data/processed/study_area.gpkg').to_crs(CFG['crs'])
 profile,mask=grid(study)
 features=terrain(profile)
 cover=gpd.read_file(ROOT/'data/processed/clc2012.gpkg')
 cover=cover[cover.geometry.notna()]
 classes=pd.to_numeric(cover.clc12,errors='coerce').fillna(0).astype(int)//100
 land=rasterize(((g,int(c)) for g,c in zip(cover.geometry,classes)),out_shape=mask.shape,transform=profile['transform'],fill=0,dtype='int16')
 mask=mask&np.isfinite(features['elevation_m'])&(land>0)
 for c,name in [(1,'built'),(2,'agriculture'),(3,'forest_seminatural'),(4,'wetland'),(5,'water')]:features['clc2012_'+name]=(land==c).astype(float)
 features['distance_clc2012_water_m']=ndimage.distance_transform_edt(land!=5)*CFG['resolution_m']
 names=list(features)
 stack=np.stack([features[n] for n in names]).astype('float32');stack[:,~mask]=np.nan
 cog(ROOT/'results/predictors.tif',stack,profile,names,tags=dict(provenance='Independent reconstruction; common 250m grid does not imply every source has 250m precision',source='Copernicus DEM GLO90; CLC2012'))
 rows,cols=np.indices(mask.shape)
 block_cols=(cols*CFG['resolution_m'])//CFG['block_m']
 block_rows=(rows*CFG['resolution_m'])//CFG['block_m']
 blocks=block_rows*(int(block_cols.max())+1)+block_cols
 # Fixed eastern 20% by width is held out before looking at labels or metrics.
 edge=int(np.quantile(np.unique(block_cols[mask]),.8))
 holdout=(block_cols>=edge)&mask
 folds=(block_cols+2*block_rows)%CFG['cv_folds']
 folds=np.where(holdout,3,folds)
 cog(ROOT/'results/spatial_folds.tif',np.where(mask,folds,np.nan),profile,tags=dict(fold_0_2='development spatial block folds',fold_3='independent eastern spatial holdout',block_m=str(CFG['block_m']),buffer_m=str(CFG['exclusion_buffer_m'])))
 np.savez_compressed(ROOT/'data/processed/stack.npz',X=stack,mask=mask,landcover=land,blocks=blocks,folds=folds,holdout=holdout,
  names=np.asarray(names),transform=np.asarray(tuple(profile['transform'])),crs=np.asarray(CFG['crs']))
 audit=[labels(h,profile,mask) for h in ['landslide','flood','wildfire']]
 pd.DataFrame(audit).to_csv(ROOT/'results/label_support_audit.csv',index=False)
 population(profile,mask)
 (ROOT/'results/grid.json').write_text(json.dumps(dict(width=profile['width'],height=profile['height'],resolution_m=CFG['resolution_m'],
  crs=CFG['crs'],transform=list(profile['transform']),eligible_cells=int(mask.sum()),heldout_cells=int(holdout.sum()),features=names,block_m=CFG['block_m']),indent=2))
 print(pd.DataFrame(audit).to_string(index=False),flush=True)


if __name__=='__main__':main()
