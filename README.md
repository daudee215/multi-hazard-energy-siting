# Piedmont multi-hazard energy-site screening

A 250 m screening exercise comparing landslide inventory, regulatory P2 flood zones and burned perimeters. For 145 of 176 mapped Turin features, ranks spanned at least 18 places across five weighting and aggregation scenarios. The regenerated maximum-overlay comparison uses its own tie-aware ranks and shows a maximum shift of 155 places. The scores are relative indicators for the sample.

[Portfolio case study](https://www.daudee.com/work/piedmont-energy-screening)

## Inspect the published results

- [Site rank sensitivity](results/site_sensitivity.csv) compares weighting and aggregation scenarios.
- [Spatial buffer sensitivity](results/spatial_buffer_sensitivity.csv) compares the exclusion gaps used for spatial evaluation.
- [Raster validation](results/cog_validation.csv) records checks on the generated Cloud Optimized GeoTIFFs.

## Run

Use Python 3.12 in a clean virtual environment. The [reproduction guide](REPRODUCE.md) lists the provider downloads, generated files and frozen inputs. Reproducing the historical site ranks requires the original OSM geometry snapshot.

```sh
python -m pip install -r requirements-lock.txt
python src/acquire.py
python src/context_acquire.py
python src/prepare.py
python src/train.py
python src/spatial_sensitivity.py
python src/deliver.py
python -m pytest -q
```

## Data and attribution

ISPRA and Regione Piemonte (IFFI), ISPRA flood mapping, EFFIS, ISTAT, Copernicus and © OpenStreetMap contributors. The release contains source code and aggregate validation. The exact historical OSM convenience sample relies on the original frozen geometry. An updated source snapshot may yield a different sample.
