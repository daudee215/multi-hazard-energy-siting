"""Public source discovery and cached acquisition; never bypasses authentication."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math,shutil,time,zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import make_valid,normalize
from shapely.geometry import shape
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'configs/pilot.yaml').read_text())
BASE='https://sinacloud.isprambiente.it/arcgisina/rest/services/'
CAT=[]


def dirs():
 for name in ['data/raw','data/processed','models','results','figures','qgis','report','tests','notebooks']:
  (ROOT/name).mkdir(parents=True,exist_ok=True)


def fetch(name,url,params=None):
 path=ROOT/'data/raw'/name; meta=path.with_suffix(path.suffix+'.request.json')
 if not path.exists():
  last=None
  for attempt in range(3):
   try:
    r=requests.get(url,params=params,timeout=150)
    r.raise_for_status()
    if 'json' in r.headers.get('Content-Type',''):
     value=r.json()
     if isinstance(value,dict) and value.get('error'): raise ValueError(str(value)[:400])
    temp=path.with_suffix(path.suffix+'.part'); temp.write_bytes(r.content); temp.replace(path)
    meta.write_text(json.dumps(dict(url=r.url,params=params,access_utc=datetime.now(timezone.utc).isoformat(),headers={k:r.headers.get(k) for k in ['Last-Modified','ETag','Content-Type']}),indent=2))
    break
   except Exception as exc:
    last=exc; time.sleep(2*(attempt+1))
  else: raise RuntimeError(f'{name}: {last}')
 else:
  saved=json.loads(meta.read_text())
  if saved['params']!=params: raise ValueError(f'Request changed for immutable cache {name}')
 return path


def catalog(path,product,portal,version,period,resolution,licence,bias,role):
 meta=json.loads(path.with_suffix(path.suffix+'.request.json').read_text())
 CAT.append(dict(product=product,official_portal=portal,version=version,source_date=meta.get('headers',{}).get('Last-Modified'),
  event_window=period,resolution_crs=resolution,expected_bias=bias,licence=licence,download_endpoint=meta['url'],
  access_utc=meta['access_utc'],file=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),role=role))


def boundary():
 url='https://www.istat.it/storage/cartografia/confini_amministrativi/generalizzati/2024/Limiti01012024_g.zip'
 path=fetch('istat_boundaries_2024.zip',url)
 catalog(path,'ISTAT administrative boundaries','https://www.istat.it/notizia/confini-delle-unita-amministrative-a-fini-statistici-al-1-gennaio-2018-2/','2024 generalised','2024-01-01','WGS84 UTM32N; generalised polygons','ISTAT reuse attribution; package terms retained','Boundary generalisation; not an operational service area','study boundary')
 with zipfile.ZipFile(path) as z:
  names=z.namelist(); main=next(n for n in names if n.endswith('.shp') and '/Reg' in n)
 region=gpd.read_file(f'zip://{path}!{main}')
 name=next(c for c in region.columns if c.upper() in ['DEN_REG','NOME_REG'])
 region=region[region[name].str.casefold().eq('piemonte')].to_crs(CFG['crs'])
 assert len(region)==1
 region=region[['geometry']].copy();region['name']='Piemonte';region['area_km2']=region.area/1e6
 region.to_file(ROOT/'data/processed/study_area.gpkg',layer='study_area',driver='GPKG')
 print('Boundary',float(region.area.sum()/1e6),'km2',flush=True)
 return region


def arcgis(name,service,layer,study,where='1=1'):
 url=BASE+service+f'/MapServer/{layer}'
 meta=fetch(name+'_metadata.json',url,{'f':'pjson'})
 metadata=json.loads(meta.read_text()); assert 'Query' in metadata['capabilities']
 bounds=study.to_crs(4326).total_bounds
 common=dict(where=where,geometry=','.join(map(str,bounds)),geometryType='esriGeometryEnvelope',inSR=4326,spatialRel='esriSpatialRelIntersects')
 count_path=fetch(name+'_count.json',url+'/query',dict(common,f='json',returnCountOnly='true'))
 count=json.loads(count_path.read_text())['count']
 pages=[]; size=min(metadata.get('maxRecordCount',1000),1000)
 oid=next(f['name'] for f in metadata['fields'] if f['type']=='esriFieldTypeOID')
 offsets=[]
 for start in range(0,count,size):
  # The service fails on its final large flood page; smaller, explicitly recorded pages work.
  if name=='flood' and start>=5000:offsets.extend((i,100) for i in range(start,min(start+size,count),100))
  else:offsets.append((start,size))
 for offset,request_size in offsets:
  params=dict(common,f='geojson',outFields='*',outSR=4326,returnGeometry='true',resultOffset=offset,resultRecordCount=request_size,orderByFields=oid)
  suffix=f'_{request_size}' if request_size!=size else ''
  if name=='flood' and offset>=5300:
   # GeoJSON conversion fails for this service tail. ESRI JSON plus ~10 m geometry
   # generalisation is a documented fallback, well below the 250 m analysis cell.
   params.update(f='json',maxAllowableOffset=.0001)
   path=fetch(f'{name}_{offset:06}{suffix}_esri.json',url+'/query',params)
   fallback=gpd.read_file(path)
   if fallback.crs is None:fallback=fallback.set_crs(4326)
   payload=json.loads(fallback.to_crs(4326).to_json())
  else:
   path=fetch(f'{name}_{offset:06}{suffix}.geojson',url+'/query',params)
   payload=json.loads(path.read_text())
  catalog(path,name,'https://idrogeo.isprambiente.it/' if name!='clc2012' else 'https://www.isprambiente.gov.it/it/servizi/sistema-carta-della-natura/banca-dati/corine-land-cover',
   'Live ArcGIS layer; snapshot metadata retained', 'Inventory dates not assumed to be event dates' if name=='landslide' else '2012' if name=='clc2012' else 'Regulatory zonation; no event dates',
   'Vector EPSG:4326; mapping support in source metadata','CC BY-SA 4.0 (ISPRA inventory/zonation); Copernicus free/open use for CLC',
   'Inventory completeness and positional uncertainty; public service snapshot','labels' if name!='clc2012' else 'pre-2015 static predictor')
  pages.extend(payload['features'])
  print(name,'page',offset,'/',count,flush=True)
 if len(pages)!=count:raise ValueError(f'Truncated {name}: {len(pages)} vs {count}')
 if not pages: raise RuntimeError(f'No features for {name}')
 frame=gpd.GeoDataFrame.from_features(pages,crs=4326).to_crs(study.crs)
 frame['geometry']=frame.geometry.map(make_valid)
 frame=frame[frame.intersects(study.geometry.iloc[0])].copy()
 frame['geometry']=frame.geometry.intersection(study.geometry.iloc[0])
 frame=frame[~frame.is_empty].copy()
 return frame,metadata


def fires(study):
 frames=[]
 # Exact province filters are supported. __in was tested and ignored: never use it.
 for province in ['Torino','Cuneo','Biella','Novara','Asti','Alessandria','Verbano-Cusio-Ossola','Vercelli']:
  offset=0;count=None;features=[]
  while count is None or offset<count:
   params=dict(country='IT',province=province,firedate__gte=CFG['fire_start'],firedate__lt=CFG['fire_end_exclusive'],limit=100,offset=offset,ordering='id')
   path=fetch(f'effis_{province}_{offset}.json','https://api.effis.emergency.copernicus.eu/rest/2/burntareas/current/',params)
   data=json.loads(path.read_text());count=data['count']
   catalog(path,'EFFIS burned perimeters','https://forest-fire.emergency.copernicus.eu/applications/data-and-services','Public REST v2 snapshot; no immutable release',
    '2015-2024','EPSG:4326; sensor-dependent minimum mapped size','European Commission reuse Decision 2011/833/EU, acknowledge EFFIS',
    'Omission of smaller fires; sensor/mapping changes in 2018; non-burned areas unknown','labels')
   for r in data['results']:
    assert r['country']=='IT' and r['province']==province
    assert 2015<=int(r['firedate'][:4])<=2024
    features.append(dict(event_id='effis/'+str(r['id']),event_date=r['firedate'],province=province,reported_area_ha=r['area_ha'],geometry=make_valid(shape(r['shape']))))
   if not data['results'] and offset<count:raise ValueError('Truncated EFFIS page')
   offset+=100
  print('EFFIS',province,count,flush=True)
  frames+=features
 frame=gpd.GeoDataFrame(frames,crs=4326).to_crs(study.crs)
 frame=frame[frame.intersects(study.geometry.iloc[0])].copy()
 frame['geometry']=frame.geometry.intersection(study.geometry.iloc[0])
 return frame


def dem(study):
 west,south,east,north=study.to_crs(4326).total_bounds
 tiles=[]
 for lat in range(math.floor(south),math.ceil(north)):
  for lon in range(math.floor(west),math.ceil(east)):
   name=f'Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM'
   tiles.append((name,f'https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif'))
 def one(item):
  name,url=item
  path=fetch(name+'.tif',url)
  return path
 with ThreadPoolExecutor(max_workers=3) as pool: paths=list(pool.map(one,tiles))
 for path in paths:
  catalog(path,'Copernicus DEM GLO-90','https://registry.opendata.aws/copernicus-dem/','2021 AWS release; TanDEM-X mainly 2011-2015',
   'Main acquisition 2011-2015; older gap filling possible','3 arcseconds; EPSG:4326; DSM, not bare-earth DTM',
   'Copernicus DEM GLO-90 free/open licence; source attribution required','Canopy/buildings and resampling; static association for undated inventory','terrain predictors')
 print('DEM',len(paths),'tiles',flush=True)


def events(frame,hazard,idcol=None):
 initial=len(frame)
 frame=frame[frame.geometry.geom_type.isin(['Polygon','MultiPolygon'])].copy()
 frame['event_id']=frame[idcol].astype(str) if idcol else [f'{hazard}/{i}' for i in frame.index]
 frame['geometry_hash']=frame.geometry.map(lambda g:hashlib.sha256(normalize(g).wkb).hexdigest())
 frame=frame.drop_duplicates('geometry_hash').copy()
 if hazard=='landslide':
  # Components of a phenomenon share its ID; do not treat them as independent events.
  frame=frame.dissolve(by='event_id',as_index=False,aggfunc='first')
 if 'event_date' not in frame:frame['event_date']=None
 frame['hazard']=hazard
 frame['mapping_confidence']='not independently quantified'
 frame['label_definition']='mapped inventory polygon' if hazard=='landslide' else 'mapped burned perimeter' if hazard=='wildfire' else 'P2 regulatory flood zone; not an observed event'
 frame.to_file(ROOT/f'data/processed/{hazard}_events.gpkg',layer='events',driver='GPKG')
 return dict(hazard=hazard,raw_clipped_features=initial,unique_polygon_records=len(frame),
  dated_records=int(frame.event_date.notna().sum()),sum_polygon_area_km2=float(frame.area.sum()/1e6),
  union_area_km2=float(frame.geometry.union_all().area/1e6),definition=frame.label_definition.iloc[0])


def main():
 dirs();study=boundary();audit=[]
 landslide,_=arcgis('landslide','iffi/Progetto_IFFI_WMS_public',3,study,where="regione = 'Piemonte'")
 audit.append(events(landslide,'landslide','idfrana'))
 flood,_=arcgis('flood','Alluvioni/Aree_pericolosita_idraulica_',1,study)
 audit.append(events(flood,'flood'))
 fire=fires(study);audit.append(events(fire,'wildfire','event_id'))
 clc,_=arcgis('clc2012','corine_land_cover/CorineLandCover',7,study)
 clc.to_file(ROOT/'data/processed/clc2012.gpkg',layer='land_cover',driver='GPKG')
 dem(study)
 # Reuse only real public OSM site geometry, not Project 1's climate or priority scores.
 prior=ROOT.parent/'climate-risk-energy-grid/results/energy_assets.gpkg'
 snapshot=ROOT/'data/raw/candidate_site_sample.gpkg'
 if prior.exists():
  candidates=gpd.read_file(prior,layer='assets').to_crs(study.crs)
  candidates=candidates[candidates.power.isin(['substation','transformer'])].copy()
  candidates['geometry']=candidates.geometry.representative_point()
  candidates.to_file(ROOT/'data/processed/candidate_sites.gpkg',layer='sites',driver='GPKG')
  source=ROOT.parent/'climate-risk-energy-grid/data/raw/osm_power.json'
  request=source.with_suffix('.request.json')
  target=ROOT/'data/raw/osm_power_project1.json';shutil.copy2(source,target)
  metadata=json.loads(request.read_text());metadata['params']=None
  metadata['access_utc']=metadata.pop('retrieved_utc')
  target.with_suffix('.json.request.json').write_text(json.dumps(metadata,indent=2))
  catalog(target,'OSM substation/transformer candidate screening sample','https://www.openstreetmap.org/copyright','Frozen Project 1 query; OSM timestamp in raw response',
   'Public mapping snapshot; no construction date','OSM geometry, EPSG:4326','ODbL 1.0; OpenStreetMap contributors',
   'Convenience sample limited to Turin research window, not all regional energy sites; existing assets are not proposed developments','candidate/exposure context only; not predictors')
 elif snapshot.exists():
  shutil.copy2(snapshot,ROOT/'data/processed/candidate_sites.gpkg')
  target=ROOT/'data/raw/osm_power_project1.json'
  catalog(target,'OSM substation/transformer candidate screening sample','https://www.openstreetmap.org/copyright','Frozen raw response; source timestamp retained','Public mapping snapshot','OSM EPSG:4326','ODbL 1.0; OpenStreetMap contributors','Frozen Turin convenience sample only','context, not predictors')
 elif (ROOT/'data/raw/osm_power_project1.json').exists():
  from shapely.geometry import Point,Polygon
  target=ROOT/'data/raw/osm_power_project1.json';payload=json.loads(target.read_text(encoding='utf-8'));rows=[]
  for el in payload['elements']:
   tags=el.get('tags',{})
   if tags.get('power') not in ['substation','transformer']:continue
   if el['type']=='node':geom=Point(el['lon'],el['lat'])
   elif el['type']=='way':
    coords=[(p['lon'],p['lat']) for p in el.get('geometry',[])]
    if len(coords)<4 or coords[0]!=coords[-1]:continue
    geom=make_valid(Polygon(coords)).representative_point()
   else:continue
   rows.append(dict(asset_id=f'osm/{el["type"]}/{el["id"]}',power=tags['power'],name=tags.get('name'),geometry=geom))
  candidates=gpd.GeoDataFrame(rows,crs=4326).to_crs(study.crs)
  candidates=candidates[candidates.intersects(study.geometry.iloc[0])]
  candidates.to_file(ROOT/'data/processed/candidate_sites.gpkg',layer='sites',driver='GPKG')
  catalog(target,'OSM substation/transformer candidate screening sample','https://www.openstreetmap.org/copyright','Frozen raw response; OSM timestamp retained','Mapping snapshot','OSM EPSG:4326','ODbL 1.0; OpenStreetMap contributors','Convenience sample only','context, not predictors')
 if (ROOT/'data/processed/candidate_sites.gpkg').exists():
  if not snapshot.exists():
   shutil.copy2(ROOT/'data/processed/candidate_sites.gpkg',snapshot)
   snapshot.with_suffix('.gpkg.request.json').write_text(json.dumps(dict(url=None,params=None,access_utc=datetime.now(timezone.utc).isoformat(),
    origin='Locally derived Project 1 OSM convenience sample; exact frozen geometry input to Project 2, not an external download'),indent=2))
  catalog(snapshot,'Frozen derived OSM candidate-site input','https://www.openstreetmap.org/copyright','Locally derived frozen Project 1 sample','Mapping snapshot, no event date',
   'EPSG:32632 representative points','ODbL 1.0; OpenStreetMap contributors','Derived convenience sample; no external download endpoint applies','Imported geometry input; raw OSM response catalogued separately')
 pd.DataFrame(audit).to_csv(ROOT/'results/event_audit.csv',index=False)
 pd.DataFrame(CAT).to_csv(ROOT/'data_catalog.csv',index=False)
 print(pd.DataFrame(audit).to_string(index=False),flush=True)


if __name__=='__main__':main()
