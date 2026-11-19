"""Assemble only executed results into scientific and GIS deliverables."""
from __future__ import annotations
import hashlib,importlib.metadata,json,platform,sys,textwrap
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COLORS, INK, SEQUENTIAL, UNCERTAINTY
from matplotlib.ticker import FuncFormatter,MaxNLocator
from matplotlib.colors import ListedColormap,BoundaryNorm
import numpy as np
import pandas as pd
import psutil
import rasterio
from rio_cogeo.cogeo import cog_validate
from sklearn.metrics import roc_curve,precision_recall_curve
from acquire import ROOT,CFG
from prepare import cog
from site_ranks import rank_sites

HAZARDS=['flood','landslide','wildfire']
LABELS={'flood':'Flood zonation emulation','landslide':'Mapped-landslide suitability','wildfire':'Mapped-burn suitability'}


def read(path):
 with rasterio.open(path) as src:return src.read(1,masked=True).filled(np.nan)


def mdtable(frame):
 def fmt(value):
  if value is None or (not isinstance(value,str) and pd.isna(value)):return '—'
  if isinstance(value,(float,np.floating)):return f'{value:.3f}'
  return str(value).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def composites(data,profile):
 stack=np.stack([read(ROOT/f'results/{h}_screening_percentile.tif') for h in HAZARDS])
 for name,weights in CFG['composite_weights'].items():
  array=np.average(stack,axis=0,weights=weights)
  cog(ROOT/f'results/composite_{name}.tif',array,profile,tags=dict(scale='weighted mean of hazard-specific development-domain percentiles',hazard_order=','.join(HAZARDS),weights=str(weights),warning='Not risk, expected loss, event probability, or legally binding suitability'))
 cog(ROOT/'results/composite_max.tif',np.max(stack,axis=0),profile,tags=dict(scale='maximum of hazard-specific relative percentiles',warning='Decision screening convention, not physical joint risk'))
 dispersion=np.stack([read(ROOT/f'results/{h}_model_dispersion.tif') for h in HAZARDS])
 thresholds=np.nanquantile(dispersion.reshape(3,-1),.9,axis=1)
 flag=(dispersion>thresholds[:,None,None]).any(axis=0)
 region=data['mask'];dev=region&~data['holdout']
 features=data['X'];low=np.nanmin(features[:,dev],axis=1);high=np.nanmax(features[:,dev],axis=1)
 novelty=((features<low[:,None,None])|(features>high[:,None,None])).any(axis=0)
 quality=np.where(region,1+flag.astype(int)+2*novelty.astype(int),np.nan)
 cog(ROOT/'results/quality_mask.tif',quality,profile,tags=dict(codes='1=in development predictor range/lower model dispersion; 2=high dispersion; 3=outside range; 4=both',warning='Quality diagnostic; no statistical confidence level; valid grid cells only'))
 sites=gpd.read_file(ROOT/'data/processed/candidate_sites.gpkg').to_crs(CFG['crs'])
 site_values={}
 for path in [*(ROOT/f'results/{h}_screening_percentile.tif' for h in HAZARDS),*(ROOT/f'results/{h}_relative_suitability.tif' for h in HAZARDS),
  *(ROOT/f'results/composite_{n}.tif' for n in [*CFG['composite_weights'],'max']),ROOT/'results/quality_mask.tif',ROOT/'results/population_context_density.tif']:
  with rasterio.open(path) as src:
   values=np.asarray([v[0] for v in src.sample([(p.x,p.y) for p in sites.geometry])],dtype=float)
   values[values==src.nodata]=np.nan
   sites[path.stem]=values
 sensitivity=rank_sites(sites, [*CFG['composite_weights'], 'max'])
 flood=gpd.read_file(ROOT/'data/processed/flood_events.gpkg')
 sites['p2_zone_point_overlap']=False
 pairs=flood.sindex.query(sites.geometry,predicate='intersects')
 sites.loc[sites.index[np.unique(pairs[0])],'p2_zone_point_overlap']=True
 sites['site_status']='Existing OSM feature; illustrative screening sample, not approved new siting'
 sites['social_vulnerability']=np.nan
 sites['context_warning']='Population density is residential context, not customers served or social vulnerability'
 path=ROOT/'results/sites.gpkg'
 if path.exists():path.unlink()
 sites.to_file(path,layer='sites',driver='GPKG')
 sites.drop(columns='geometry').to_csv(ROOT/'results/site_rankings.csv',index=False)
 sensitivity.to_csv(ROOT/'results/site_sensitivity.csv',index=False)
 return sites


def map_figure(path,title,label,study,cmap=SEQUENTIAL,vmin=None,vmax=None):
 with rasterio.open(path) as src:
  array=src.read(1,masked=True);bounds=src.bounds
 fig,ax=plt.subplots(figsize=(8.2,8.5))
 im=ax.imshow(array,extent=[bounds.left,bounds.right,bounds.bottom,bounds.top],cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
 study.boundary.plot(ax=ax,color=INK,linewidth=.6)
 ax.set(title=textwrap.fill(title,48),xlabel='Easting (EPSG:32632), m',ylabel='Northing, m')
 fig.colorbar(im,ax=ax,shrink=.75,label=label)
 fig.text(.11,.025,'Independent reconstruction · ISPRA / Copernicus / ISTAT\n250 m analytical grid; source support varies.',fontsize=8,color=INK)
 fig.tight_layout(rect=(0,.045,1,1));fig.savefig(ROOT/f'figures/{path.stem}.png',dpi=180);plt.close(fig)


def figures(study,sites):
 for hazard in HAZARDS:
  map_figure(ROOT/f'results/{hazard}_relative_suitability.tif',LABELS[hazard],
   'Relative presence/background score (not probability)',study,vmin=0,vmax=1)
  map_figure(ROOT/f'results/{hazard}_model_dispersion.tif',LABELS[hazard]+' · Model dispersion','Std. deviation across outer-CV models; not a confidence interval',study,cmap=UNCERTAINTY)
  map_figure(ROOT/f'results/{hazard}_spatial_folds.tif',hazard.title()+' · Spatial folds (3 = independent holdout)',
   'Development fold 0, 1, 2; holdout 3',study,cmap=ListedColormap(['#B8CE9A',COLORS[1],COLORS[2],COLORS[0]]),vmin=0,vmax=3)
 map_figure(ROOT/'results/composite_equal.tif','Composite relative screening · Equal weights','Mean of hazard-specific relative percentiles; high = higher concern',study,vmin=0,vmax=1)
 map_figure(ROOT/'results/quality_mask.tif','Coverage and model-quality diagnostics','1=lower dispersion; 2=high; 3=outside predictor range; 4=both',study,
  cmap=ListedColormap(['#E3EDD5','#D6BF78',COLORS[2],COLORS[0]]),vmin=1,vmax=4)
 fig,axes=plt.subplots(1,3,figsize=(14,7),sharex=True,sharey=True)
 for ax,hazard in zip(axes,HAZARDS):
  study.boundary.plot(ax=ax,color=INK,linewidth=.6)
  events=gpd.read_file(ROOT/f'data/processed/{hazard}_events.gpkg')
  events.plot(ax=ax,color=dict(zip(HAZARDS,COLORS))[hazard],linewidth=0)
  ax.set(title=f'{LABELS[hazard]}\n{len(events):,} polygon records',xlabel='Easting, km',ylabel='')
  ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5))
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}'))
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y/1000:.0f}'))
 axes[0].set_ylabel('Northing (EPSG:32632), km')
 fig.suptitle('Inventory and zonation coverage · Independent reconstruction')
 fig.text(.1,.035,'Polygon counts are source mapping records, not independent dated events. Flood = regulatory P2 zones. Sources: ISPRA / EFFIS / ISTAT.',fontsize=9)
 fig.tight_layout(rect=(0,.11,1,.94));fig.savefig(ROOT/'figures/event_coverage.png',dpi=180);plt.close(fig)
 for hazard in HAZARDS:
  fig,axes=plt.subplots(1,2,figsize=(10,4.5))
  for name in CFG['models']:
   rows=pd.read_csv(ROOT/f'results/{hazard}_uniform_{name}_holdout.csv');y=rows.y_presence_background;s=rows.relative_score
   fpr,tpr,_=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
   axes[0].plot(fpr,tpr,label=name);axes[1].plot(recall,precision,label=name)
  axes[0].plot([0,1],[0,1],color='#777',ls='--',lw=.8)
  axes[0].set(xlabel='Background false-positive rate',ylabel='Presence sensitivity',title='ROC')
  axes[1].axhline(.5,color='#777',ls='--',lw=.8)
  axes[1].set(xlabel='Presence recall',ylabel='Precision in balanced presence/background sample',title='Precision–recall (sample prevalence 0.5)')
  for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1)
  fig.suptitle(LABELS[hazard]+' · Independent spatial holdout')
  fig.tight_layout();fig.savefig(ROOT/f'figures/{hazard}_holdout_curves.png',dpi=180);plt.close(fig)
  shap_path=ROOT/f'results/{hazard}_xgboost_shap.csv'
  if shap_path.exists():
   values=pd.read_csv(shap_path).sort_values('mean_abs_shap')
   fig,ax=plt.subplots(figsize=(8,5));ax.barh(values.feature,values.mean_abs_shap,color=COLORS[0])
   ax.set(xlabel='Mean |SHAP| (model-output scale)',title=hazard.title()+' · XGBoost attribution, not causation')
   fig.tight_layout();fig.savefig(ROOT/f'figures/{hazard}_shap.png',dpi=170);plt.close(fig)


def spatial_diagnostic(data):
 rng=np.random.default_rng(CFG['seed']);mask=data['mask'];elevation=data['X'][0];points=np.argwhere(mask)
 chosen=points[rng.choice(len(points),2000,replace=False)];variance=float(np.var(elevation[mask]));rows=[]
 for lag in [1000,2500,5000,10000,20000,40000,60000]:
  angles=rng.uniform(0,2*np.pi,len(chosen));offset=np.column_stack([np.sin(angles),np.cos(angles)])*lag/CFG['resolution_m']
  partner=chosen+np.rint(offset).astype(int)
  keep=(partner[:,0]>=0)&(partner[:,0]<mask.shape[0])&(partner[:,1]>=0)&(partner[:,1]<mask.shape[1])
  a,b=chosen[keep],partner[keep];keep2=mask[b[:,0],b[:,1]];a,b=a[keep2],b[keep2]
  gamma=float(.5*np.mean((elevation[a[:,0],a[:,1]]-elevation[b[:,0],b[:,1]])**2))
  rows.append(dict(lag_m=lag,n_pairs=len(a),elevation_semivariance_m2=gamma,fraction_of_marginal_variance=gamma/variance))
 pd.DataFrame(rows).to_csv(ROOT/'results/terrain_spatial_diagnostic.csv',index=False)


def qgis_project():
 root=ET.Element('qgis',version='3.40.0',projectname='Piedmont independent susceptibility reconstruction')
 ET.SubElement(root,'title').text='Piedmont multi-hazard relative screening'
 srs=ET.SubElement(ET.SubElement(root,'projectCrs'),'spatialrefsys')
 ET.SubElement(srs,'authid').text=CFG['crs'];ET.SubElement(srs,'srid').text='32632'
 ET.SubElement(srs,'description').text='WGS 84 / UTM zone 32N'
 tree=ET.SubElement(root,'layer-tree-group',name='',checked='Qt::Checked',expanded='1')
 layers=ET.SubElement(root,'projectlayers')
 entries=[('Candidate sample','ogr','../results/sites.gpkg|layername=sites','vector'),
  ('Study area','ogr','../data/processed/study_area.gpkg|layername=study_area','vector')]
 entries += [(p.stem,'gdal','../results/'+p.name,'raster') for p in sorted((ROOT/'results').glob('*.tif'))]
 for i,(name,provider,path,kind) in enumerate(entries):
  assert (ROOT/'qgis'/path.split('|')[0]).resolve().exists(),path
  uid=f'layer_{i}_{name.replace(" ","_")}'
  ET.SubElement(tree,'layer-tree-layer',id=uid,name=name,source=path,providerKey=provider,checked='Qt::Checked' if 'composite_equal' in name or kind=='vector' else 'Qt::Unchecked',expanded='0')
  layer=ET.SubElement(layers,'maplayer',type=kind,autoRefreshEnabled='0',hasScaleBasedVisibilityFlag='0')
  ET.SubElement(layer,'id').text=uid;ET.SubElement(layer,'layername').text=name
  ET.SubElement(layer,'datasource').text=path;ET.SubElement(layer,'provider').text=provider
  srs=ET.SubElement(ET.SubElement(layer,'srs'),'spatialrefsys');ET.SubElement(srs,'authid').text=CFG['crs'];ET.SubElement(srs,'srid').text='32632'
  ET.SubElement(layer,'layerTransparency').text='0'
 ET.indent(root)
 path=ROOT/'qgis/piedmont_screening.qgs';ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
 ET.parse(path)
 (ROOT/'qgis/INSPECTION_STATUS.md').write_text('Project XML parses and relative datasource paths are checked by the delivery script. Native QGIS loading/rendering was not verified: the configured QGIS connection failed in the preceding project. Open this project in QGIS to complete that inspection. No native-QGIS cartography is claimed.\n')


def report(sites,evaluation):
 audits=pd.read_csv(ROOT/'results/event_audit.csv');support=pd.read_csv(ROOT/'results/label_support_audit.csv')
 cv=evaluation[evaluation.split.eq('spatial_cv')&evaluation.status.eq('done')]
 summary=cv.groupby(['hazard','strategy','model']).agg(auc_mean=('roc_auc','mean'),auc_std=('roc_auc','std'),ap_mean=('average_precision','mean'),folds=('roc_auc','size')).reset_index()
 summary.to_csv(ROOT/'results/cv_summary.csv',index=False)
 hold=evaluation[evaluation.split.eq('independent_spatial_holdout')&evaluation.status.eq('done')]
 table=hold[['hazard','strategy','model','n_presence','n_background','roc_auc','roc_auc_low','roc_auc_high','average_precision','balanced_accuracy']]
 primary=[];manifests=[]
 for h in HAZARDS:
  manifest=json.loads((ROOT/f'results/{h}_training_manifest.json').read_text());manifests.append(manifest)
  rows=table[(table.hazard==h)&table.strategy.eq('uniform')&table.model.eq(manifest['primary_model'])]
  primary.append(rows)
 selected=pd.concat(primary)
 assumptions=f'''# Results: Piedmont multi-hazard screening

**New independent-data reconstruction. No earlier code, study split or historical performance is verified.** Results below are executed presence/background or zonation comparisons, not calibrated event probabilities or failure outcomes.

## Scope and label support

The official ISTAT 2024 generalised boundary covers approximately 25,392 km². The retained common-grid mask contains {int(np.load(ROOT/'data/processed/stack.npz')['mask'].sum()):,} cells at 250 m, after source-coverage exclusions. Terrain/CLC2012 predictors are harmonised without claiming subpixel evidence. The wildfire modelling window is 2016–2024; acquisition includes 2015 for exclusion/audit. Landslide inventory dates are unverified; flood labels are regulatory P2 zones.

{mdtable(audits)}

The audit denominator is source polygon records after repair/deduplication/clipping. `sum_polygon_area` may count overlaps; union area does not. Zero dated IFFI/zonation records means no reliable event date was assigned, not that no dates exist in source documentation.

{mdtable(support)}

Grid support is based on positive-cell centres and overlapping-event exclusion groups. Tiny polygons may have no positive centre; no artificial nearby label is created. Records sharing an overlapping group inherit that group's support flag. Background is unknown, not observed absence.

![Event coverage](../figures/event_coverage.png)

Figure 1. Actual source polygons over the study boundary; record counts shown on each panel. ISPRA / EFFIS / ISTAT. Independent reconstruction.

## Spatial validation and model comparison

Models are tuned within training geography using two inner spatial splits and evaluated in three outer buffered block folds. Blocks are 20 km; separation is 5 km. Entire overlapping-event groups touching the held-out buffer are excluded from fitting. All {len(manifests)} hazard manifests report zero shared development/holdout groups. The eastern holdout is retained for landslide/flood. Wildfire uses a northern holdout revised from label-support counts **before wildfire fitting**, because the eastern region had one fire group. The northern test has eight groups and 92 presence cells; this is limited independent fire support. No cross-hazard mean AUC is computed because targets and holdout geographies differ.

{mdtable(summary)}

Table 3. Outer spatial-CV AUC mean/standard deviation and mean average precision. Each model/strategy has its own shown feasible fold count. Each feasible fold uses a balanced presence/background test sample; detailed denominators, confusion matrices and group/area counts are retained in evaluation and split-audit CSVs.

## Independent holdout: full comparisons

{mdtable(table)}

Table 4. ROC-AUC and 95% descriptive whole-block bootstrap intervals (200 resamples where feasible), average precision and balanced accuracy. Sample prevalence is 0.5 by design. Background specificity is not verified true-negative specificity. The 0.5 threshold is fixed for case-control classification. Intervals do not account for all inventory omission, label error or spatial dependence.

## Maps and selected models

Primary maps use the uniform-background model with highest mean development-fold AUC, selected independently of holdout performance:

{mdtable(selected[['hazard','model','roc_auc','roc_auc_low','roc_auc_high','n_presence','n_background']])}

'''
 for h in HAZARDS:
  assumptions+=f'''![{LABELS[h]}](../figures/{h}_relative_suitability.png)

Figure: {LABELS[h]}. Actual CV-selected model, 250 m aligned grid; inventory/background score, not event probability. Flood is explicitly existing-zonation emulation.

![{h} folds](../figures/{h}_spatial_folds.png)

Figure: {h} 20 km spatial folds, independent holdout shown as 3. Only source-supported cells are shown. Actual separation and excluded event groups are in the split audit.

![{h} evaluation](../figures/{h}_holdout_curves.png)

Figure: Independent-holdout ROC and PR curves, uniform training background. Denominators are in Table 4; PR reference is 0.5 sample prevalence, not population prevalence.

![{h} dispersion](../figures/{h}_model_dispersion.png)

Figure: Standard deviation of predictions from outer-CV fitted primary models. It is model dispersion, not a calibrated confidence interval.

'''
 sensitivity=pd.read_csv(ROOT/'results/site_sensitivity.csv')
 assumptions+=f'''## Composite decision screen and existing-site sample

Each model score is converted to its empirical percentile relative to its development-domain grid. The composite compares maximum, equal mean and alternative 2:1:1 weighting. Relative percentiles are a declared preference scale, not comparable annual risks. The sources do not support expected losses, social vulnerability estimates or customer assignments.

![Composite](../figures/composite_equal.png)

Figure: Equal-weight mean of the three hazard-specific relative percentiles. High values indicate greater screening concern. Denominator is the common eligible grid, not a count of events or people.

The demonstration screens **{len(sites)} existing OSM substations/transformers** in the frozen Turin sample. No approved new sites or comprehensive regional register are implied. **{int(sites.rank_instability_flag.sum())}/{len(sites)}** have a rank range greater than 10% of this sample under the tested weighting/max alternatives. P2 point overlap occurs for {int(sites.p2_zone_point_overlap.sum())}/{len(sites)} sampled representative points. This is regulatory map context, not a blanket legal exclusion determination.

{mdtable(sensitivity)}

Table: Site-rank sensitivity, denominator {len(sites)} existing mapped features. Lower composite indicates lower relative screened concern. Ties use minimum rank and are retained. Full results are in sites.gpkg and site_rankings.csv.

GHSL supplies a separate 2020 residential-density layer, derived from native 100 m equal-area cell counts and reprojected by area average. It is excluded from all predictors. No census-based socio-economic vulnerability score has been constructed.

![Quality](../figures/quality_mask.png)

Figure: Diagnostic categories flag high CV-model dispersion and/or feature values outside the common eastern-development reference range. The mask is not a statistical confidence map; wildfire has a distinct holdout geography recorded separately.

## Explanations and limits

Grouped local permutation results and XGBoost SHAP values are retained for every hazard. SHAP is computed on up to 100 held-out samples for XGBoost, which can differ from the CV-selected primary model. Correlated predictors can redistribute attribution, and neither explanation method demonstrates causation. Land-cover and north/south evaluation tables leave undefined AUCs missing where one class is absent.

The main scientific constraints are mapping omissions, undated landslides, flood-zone target semantics, limited independent wildfire test support, static DSM assumptions, absence of independent lithology/climate/NDVI predictors and residual spatial dependence. See METHODS.md and LIMITATIONS.md. Native QGIS rendering has not been verified; the exported project XML and data paths are inspectable.

The terrain diagnostic shows normalised elevation semivariance of about 0.30 at 20 km and 0.77 at 60 km. Broad terrain structure therefore persists beyond the nominal blocks; the local 5 km separation does not establish complete spatial independence. A stricter 20 km exclusion-buffer sensitivity uses the same held-out samples and fixed selected model configuration, with results in spatial_buffer_sensitivity.csv. It is an additional robustness check, not a reason to select the most favourable reported result.

## Completion status

Executed: real-data acquisition and event audit; aligned predictor stack; both background strategies; three model families; buffered nested spatial tuning and holdout tests; COGs; grouped permutations and SHAP; composite alternatives; site GeoPackage; population context; figures; source hashes and exact commands.

Partial: a restricted terrain/land-cover predictor set; retrospective associations; a small wildfire holdout; QGIS project generated but native rendering unverified. Socio-economic vulnerability, engineering risk, prospective forecasting and independent flood-event prediction are not claimed. No online deployment is performed.
'''
 (ROOT/'report/RESULTS.md').write_text(assumptions,encoding='utf-8')
 sensitivity_path=ROOT/'results/spatial_buffer_sensitivity.csv'
 if sensitivity_path.exists():
  with (ROOT/'report/RESULTS.md').open('a',encoding='utf-8') as f:
   f.write('\n## Wider exclusion-buffer sensitivity\n\n'+mdtable(pd.read_csv(sensitivity_path))+'\n\nSame independent holdout support; fixed model family/configuration and uniform background. Larger gaps can alter training support and do not prove independence from broader terrain structure.\n')
 catalog=pd.read_csv(ROOT/'data_catalog.csv')
 sources=catalog[['product','official_portal','version','event_window','resolution_crs','licence','expected_bias']].drop_duplicates()
 (ROOT/'report/SOURCES.md').write_text('# Source citation appendix\n\n'+mdtable(sources)+'\n\nExact per-response URLs, access metadata and SHA-256: ../data_catalog.csv. Raw response/request pairs are retained. No immutable product version is inferred when the service does not supply one.\n',encoding='utf-8')
 return selected


def manifest_and_validate():
 checks=[]
 with rasterio.open(ROOT/'results/predictors.tif') as src:alignment=(src.crs,src.transform,src.width,src.height)
 for path in sorted((ROOT/'results').glob('*.tif')):
  ok,errors,warnings=cog_validate(path,strict=True,quiet=True)
  with rasterio.open(path) as src:
   aligned=(src.crs,src.transform,src.width,src.height)==alignment
   checks.append(dict(file=path.name,cog_valid=ok,aligned=aligned,tiled=all(shape==(256,256) for shape in src.block_shapes),overviews=str(src.overviews(1)),nodata=src.nodata,errors=str(errors),warnings=str(warnings)))
  assert ok and aligned,(path,errors)
 pd.DataFrame(checks).to_csv(ROOT/'results/cog_validation.csv',index=False)
 catalog=pd.read_csv(ROOT/'data_catalog.csv');bad=[]
 for row in catalog.itertuples():
  if hashlib.sha256((ROOT/row.file).read_bytes()).hexdigest()!=row.sha256:bad.append(row.file)
 assert not bad,bad
 versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','geopandas','rasterio','scikit-learn','xgboost','shap','scipy','rio-cogeo']}
 files=[]
 for folder in ['configs','src','tests','data/raw','data/processed','models','results','figures']:
  for p in sorted((ROOT/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.name not in ['run_manifest.json','file_inventory.csv']:
    files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 for name in ['figure_style.json','FIGURE_STYLE.md']:
  p=ROOT/name;files.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 manifest=dict(execution_recorded_utc=datetime.now(timezone.utc).isoformat(),provenance='new independent reconstruction, not recovered historical project',
  seed=CFG['seed'],config=CFG,python=sys.version,platform=platform.platform(),software_versions=versions,
  catalog_responses_verified=len(catalog),cogs_validated=len(checks),files=files,
  delivery_process_peak_rss_mb=psutil.Process().memory_info().peak_wset/1e6 if sys.platform=='win32' else psutil.Process().memory_info().rss/1e6,
  qgis='XML parsed; native loading/rendering unverified',visual_date_policy='No computation-date stamps on figures or result tables')
 (ROOT/'results/run_manifest.json').write_text(json.dumps(manifest,indent=2))
 pd.DataFrame(files).to_csv(ROOT/'results/file_inventory.csv',index=False)
 statuses=[('Source acquisition and catalog','done'),('Event and label support audit','done'),('Aligned predictor stack and COGs','done'),
  ('LR/RF/XGB; two backgrounds; nested spatial evaluation','done'),('SHAP and grouped permutation','done'),('Composite and site GeoPackage','done'),
  ('Population exposure context','done'),('Socio-economic vulnerability','partial'),('NDVI/rainfall/lithology/hydrological predictors','partial'),
  ('QGIS project file','done'),('Native QGIS rendered inspection','blocked'),('Independent observed-flood validation','partial')]
 pd.DataFrame(statuses,columns=['deliverable','status']).to_csv(ROOT/'results/deliverable_status.csv',index=False)
 print(f'{len(checks)} COGs validated; {len(catalog)} catalog response hashes verified',flush=True)


def main():
 data=np.load(ROOT/'data/processed/stack.npz');study=gpd.read_file(ROOT/'data/processed/study_area.gpkg')
 with rasterio.open(ROOT/'results/predictors.tif') as src:profile=dict(crs=src.crs,transform=src.transform,width=src.width,height=src.height)
 evaluation=pd.concat([pd.read_csv(ROOT/f'results/{h}_evaluation.csv') for h in HAZARDS],ignore_index=True)
 evaluation.to_csv(ROOT/'results/model_evaluation.csv',index=False)
 sites=composites(data,profile);figures(study,sites);spatial_diagnostic(data);qgis_project();selected=report(sites,evaluation);manifest_and_validate()
 print(selected.to_string(index=False),flush=True)


if __name__=='__main__':main()
