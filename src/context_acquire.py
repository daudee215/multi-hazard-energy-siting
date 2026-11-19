"""GHSL population is contextual exposure, never a susceptibility predictor."""
import json
import pandas as pd
from acquire import ROOT,fetch,catalog,CAT

def main():
 index=fetch('ghsl_tile_schema.zip','https://human-settlement.emergency.copernicus.eu/download/GHSL_data_54009_shapefile.zip')
 catalog(index,'GHSL tile schema','https://human-settlement.emergency.copernicus.eu/download.php?ds=pop','R2023A schema','not applicable','ESRI:54009 vector','European Commission reuse with acknowledgement','Tiling index only','select covering tile')
 name='GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R4_C19'
 url='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/'+name+'.zip'
 tile=fetch(name+'.zip',url)
 catalog(tile,'GHS-POP epoch 2020','https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php','R2023A V1.0','2020 estimated residential population',
  '100 m equal-area Mollweide ESRI:54009, people per cell','European Commission reuse with proper acknowledgement',
  'Residential population allocation, not daytime occupancy, social vulnerability or actual grid customers','context only, excluded from all model inputs')
 existing=pd.read_csv(ROOT/'data_catalog.csv')
 existing=existing[~existing['product'].isin([v['product'] for v in CAT])]
 pd.concat([existing,pd.DataFrame(CAT)],ignore_index=True).to_csv(ROOT/'data_catalog.csv',index=False)
 print('GHSL selected tile R4_C19 acquired and hashed',flush=True)

if __name__=='__main__':main()
