from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import requests
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'data/raw/discovery').mkdir(parents=True,exist_ok=True)
URLS={
 'iffi_wfs':'https://idrogeo.isprambiente.it/geoserver/idrogeo/frane/ows?service=WFS&version=2.0.0&request=GetCapabilities',
 'iffi_polygons':'https://sinacloud.isprambiente.it/arcgisina/rest/services/iffi/Progetto_IFFI_WMS_public/MapServer/3?f=pjson',
 'ispra_services':'https://sinacloud.isprambiente.it/arcgisina/rest/services?f=pjson',
 'iffi_api':'https://idrogeo.isprambiente.it/api/',
 'fire':'https://api.effis.emergency.copernicus.eu/rest/2/burntareas/current/?country=IT&limit=1&ordering=-firedate',
 'istat':'https://www.istat.it/notizia/confini-delle-unita-amministrative-a-fini-statistici-al-1-gennaio-2018-2/',
 'dem_tiles':'https://copernicus-dem-90m.s3.amazonaws.com/tileList.txt',
}
def get(item):
 name,url=item
 try:
  r=requests.get(url,timeout=35); (ROOT/f'data/raw/discovery/{name}.txt').write_bytes(r.content)
  if name=='iffi_wfs':
   root=ET.fromstring(r.content); details=[el.text for el in root.iter() if el.tag.endswith('Name')][:40]
  elif name=='istat':
   import re
   details=re.findall(r'href=[\"\']([^\"\']+(?:2026|2024)[^\"\']*\.zip)',r.text)
  elif name=='dem_tiles':
   details=[v for v in r.text.splitlines() if any(f'N{lat}_00_E00{lon}_' in v for lat in [44,45,46] for lon in [6,7,8,9])]
  else:
   data=r.json()
   if name=='iffi_api': details={'paths':list(data.get('paths',{})),'info':data.get('info')}
   else: details=data
  return name,r.status_code,details
 except Exception as exc: return name,str(exc)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=6) as pool:
  for result in pool.map(get,URLS.items()): print(json.dumps(result,ensure_ascii=True)[:11000],flush=True)
