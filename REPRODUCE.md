# Ordered reproduction

1. Use Python 3.12 and install the exact environment from `requirements-lock.txt`. `uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt` is an equivalent installation command after creating the environment. `environment.yml` is supplied for conda users.
2. Run `python src/acquire.py`. It acquires the ISTAT boundary; paginates IFFI polygons, P2 flood zones and CLC2012; downloads EFFIS separately for each supported province filter; downloads all GLO-90 tiles covering the boundary; and imports the frozen public OSM site sample from `../climate-risk-energy-grid/` when available.
3. Run `python src/context_acquire.py` to retrieve the GHSL tile selected using the official Mollweide tiling schema. It appends its hashes to the catalog.
4. Run `python src/prepare.py`. This creates the aligned predictor COG, audited presence/background masks, event-group mapping, spatial folds and context layer. It records events too small to contain a grid-cell centre instead of snapping them into invented labels.
5. Run `python src/train.py --hazards landslide` for the first full pilot. It fits logistic regression, Random Forest and XGBoost with two background strategies, nested buffered spatial tuning, three outer folds and an independent eastern holdout. If a fold lacks both classes, it is reported as infeasible; a surface is not called validated without feasible spatial evaluation.
6. Run `python src/train.py` to extend to floods and wildfire. Existing completed hazards are reused only for a matching configuration, code, predictor and label fingerprint. Then run `python src/spatial_sensitivity.py` for the fixed-model 5 km versus 20 km exclusion-gap comparison. Frozen raw bytes make this path reproducible despite changes to live services.
7. Run `python src/deliver.py` to assemble composites, site summaries, COG validation, maps, reports, QGIS XML and the run manifest.
8. Run `python -m pytest -q` and retain output in `results/validation.txt`. Tests cover the split barrier, whole-group exclusion, training-only imputation, background construction, actual COGs/alignment and monotonic decision aggregation.
9. Open `qgis/piedmont_screening.qgs` in QGIS. An XML parse and datasource check are automated; native QGIS rendering requires a functioning local QGIS connection and must not be claimed when unavailable.

Run commands from this repository's root using `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux. No credentials or online publication are involved.

## Cache and refresh

Each raw response has request metadata and a checksum. Cache parameters are checked. A changed request requires an explicitly new cache filename or archiving the previous dataset; the code does not silently mix snapshots. Do not delete source files to obtain better metrics. The ArcGIS service's problematic final flood page is retrieved as ESRI JSON with approximately 10 m generalisation; all other source pages retain original served geometry. The full expected feature count is checked.

`acquire.py` rebuilds the core catalog. Rerun `context_acquire.py` after it to reappend GHSL lineage. `prepare.py` rebuilds harmonised inputs; training fingerprints include configuration, predictors and hazard labels. `deliver.py` regenerates derived documentation and maps from computed results. Model-selection decisions use development CV, not holdout performance.

If the sibling Project 1 directory is unavailable on a clean machine, the frozen imported geometry input `data/raw/candidate_site_sample.gpkg` reconstructs the exact sample. It is a locally derived snapshot, not an external download, and is catalogued separately from the original `osm_power_project1.json` response and its query metadata. The acquisition code supports this retained-copy route. The convenience sample is not a full regional electricity register.
