# Methods

## Interpretation and temporal design

This is a new reconstruction with independent public data. Landslide labels represent mapped IFFI inventory polygons; inventory existence is not necessarily a dated event. Flood labels represent P2 regulatory zones; modelling them estimates spatial similarity to existing zonation, not observed inundation occurrence or independent flood prediction. Wildfire labels represent EFFIS mapped burned perimeters. Background is unknown for all three.

The official 2024 ISTAT generalised Piedmont polygon defines the boundary. The aligned grid is 250 m, EPSG:32632, origin snapped to integer multiples of 250 m, common nodata -9999. This is an analytical support choice, not a promise that each source has 250 m accuracy. GLO-90 terrain is resampled by area averaging before differentiation. CLC2012 is rasterized at cell centres. GHSL population is excluded from all model features and supplies a separate context layer.

Wildfire acquisition covers 2015–2024, but training eligibility starts in **2016**, after the principal 2011–2015 DEM acquisition window and CLC2012. Earlier fires remain excluded from background. Terrain gap filling can be older, and exact per-pixel DEM acquisition dates are not established, so no prospective forecasting performance is claimed. IFFI dates are not assumed; landslide results are static inventory associations and cannot demonstrate temporal causation.

## Label audit and geometry

Only polygons are modelled. IFFI points, linear features, diffuse-area classes and deep-seated deformation layers are not silently merged into a single event definition. Invalid polygons are repaired. Exact normalised geometry duplicates are removed; pieces with the same IFFI identity are dissolved. Strictly overlapping polygons are connected into exclusion groups. Shared boundaries alone do not merge adjacent polygons. For flood polygons these groups represent mapping support rather than independent hydrological events.

Positive cells have their centres within eligible polygons. No-centre-support records are reported and not shifted to nearby pixels. Original all-touched footprints, including ineligible early fires, define conservative background exclusion. Background must be more than 500 m from these footprints. Event-count and overlap tables precede modelling in `event_audit.csv` and `label_support_audit.csv`. No unmapped fire is declared absent.

## Predictor dictionary and mechanisms

| Feature | Formula / units | Mechanism hypothesis and caution |
|---|---|---|
| elevation_m | Area-averaged GLO-90 DSM, m | Broad terrain/climate proxy; buildings/canopy remain |
| slope_deg | atan(sqrt((dz/dx)^2+(dz/dy)^2)) in degrees, spacing 250 m | Slope instability / runoff / fire spread; smooths subpixel slopes |
| aspect_sin, aspect_cos | Sine/cosine of downslope aspect from projected gradients | Solar-exposure/orientation proxy; flat-terrain aspect is weakly meaningful |
| curvature_per_m | Discrete Laplacian(z) / 250² | Local convergence/convexity proxy; not hydrologically conditioned flow accumulation |
| tpi_1km_m | z minus 5×5 moving-window mean | Relief over a 1.25 km window; name is approximate scale |
| tpi_5km_m | z minus 21×21 moving-window mean | Relief over a 5.25 km window |
| clc2012 indicators | Five one-hot level-one CLC classes | Vegetation/fuel and surface-cover context; 25 ha mapping unit is coarser than a pixel |
| distance_clc2012_water_m | Euclidean grid distance to rasterized class-5 cells | Broad water proximity, not river hydraulics or distance to a complete drainage network |

Terrain, cover and proximity form a deliberately small transparent pilot feature set. NDVI, rainfall climatology, independent lithology and verified pre-event roads/river networks are not represented by fabricated substitutes. Event-record lithology is **not** used as a spatial predictor: it is collected at labelled inventory sites and would leak target-related sampling. Flow accumulation/TWI are omitted because a conditioned drainage model has not been executed. CLC water proximity is not labelled as a hydrological wetness model.

All predictors are finite on the retained analysis mask; missing CLC/DEM coverage is excluded. Training pipelines include median imputation fitted on training observations only; logistic regression adds training-only scaling. There is no target-informed global feature selection or use of zonation as a predictor. Correlated terrain descriptors are retained for comparison, so feature attributions are explicitly non-causal.

## Spatial design, sampling and tuning

20 km blocks and a 5 km train/test separation buffer are conservative landscape-scale pilot choices. The buffer exceeds the 500 m inventory positional exclusion and the approximately 2.6 km radius of the largest TPI window. These distances are not represented as an estimated universal independence range. An empirical terrain semivariance diagnostic is supplied, and residual long-range spatial dependence remains a limitation. The three development folds follow `(block_column + 2×block_row) mod 3`. The eastern approximately 20% of region width is reserved as a geographically independent holdout before viewing outcomes.

For every split, remove training pixels within 5 km of validation/holdout, and remove the **entire exclusion group** of any positive group touching that buffer. This is applied again within nested tuning splits. No positive group can appear in train and test. Held-out background pools are fixed uniform samples shared across training strategies and models.

Training compares uniform background with elevation-matched background. The latter allocates background according to training-presence elevation quartiles, with bounded fallback when a bin lacks candidates; no held-out elevations determine these bins. Per split, training retains at most 3,000 presence cells and the same number of background cells; outer/holdout evaluation at most 2,000 each. The case-control prevalence is artificial and is reported; PR baselines depend on it.

Baseline logistic regression uses C=1. Random Forest uses 64 trees, minimum leaf size 5 and max_features=0.8; depths 6 and 12 are compared. XGBoost uses 80 trees, learning rate 0.07, subsample 0.85, colsample_bytree 0.8 and L2 penalty 2; depths 3 and 5 are compared. Two inner spatial splits tune depth within each outer training region. Infeasible inner folds are recorded; when none is feasible the conservative first configuration is used and disclosed. All model families and both sampling strategies are evaluated; the primary uniform-background map model is chosen by mean development-fold AUC, never holdout AUC.

### Wildfire holdout revision before fitting

The label-support audit found only one fire group (33 positive cells) in the originally planned eastern holdout. Before fitting or inspecting wildfire performance, its independent holdout was revised to the northern 20% of the bounding-box height. That region contains eight mapped fire groups and 92 positive cells on this grid. The revision is based on label feasibility, not a performance target. It covers about 6.5% of the irregular region's area, not 20% of its area. Landslide and flood retain the eastern holdout. Hazard-specific fold maps and manifests record the distinction. Cross-hazard performance is therefore not a matched-geography comparison and no mean AUC across hazards is reported.

### Unique validation ownership

Within each CV level, each positive exclusion group is assigned to its modal geographic fold, with deterministic tie-breaking. Validation uses only its cell centres in that owner fold; group pixels in other folds do not become additional validation replicates. The entire geographic validation block still defines the spatial buffer, and full touched groups remain excluded from training. This prevents one event group contributing to multiple validation folds while preserving the geographic separation. Actual validation support IDs are exported and disjointness is tested.

## Evaluation and explanations

Report each hazard, model, background strategy and split separately: ROC-AUC, average precision, balanced accuracy, presence sensitivity, background specificity and confusion matrix. The 0.5 threshold is a prespecified case-control classification threshold, not a calibrated event threshold. No calibration curve or physical probability interpretation is claimed. Holdout uncertainty uses 200 resamples of whole spatial blocks, retaining replicates with both classes; intervals are descriptive given limited and dependent spatial support.

Joint predictor-group permutations are restricted within held-out spatial blocks. This preserves block distributions but not fine-scale autocorrelation; it is a local discrimination diagnostic, not a causal importance test. SHAP is computed for the uniform-background XGBoost model on up to 100 held-out samples, separately from any other CV-selected primary model. Correlated predictors can redistribute attribution. Subgroup metrics are shown only when both classes have support; undefined results stay missing.

The dispersion map is the standard deviation of predictions from the three outer-CV fitted primary models. It is **not** a calibrated confidence interval. Error maps show mistakes at sampled presence/background support, where background is not confirmed absence.

## Composite and sites

Transform each hazard-specific score to its empirical percentile relative to development-region grid cells. Percentiles put relative standings on a declared decision scale but do not make the underlying phenomena commensurate in physical risk. Compare maximum, equal mean and 2:1:1 focus weights. No composite is called expected annual loss or calibrated risk.

Screen a frozen public OSM sample of existing Turin substations and transformers at representative points. These are illustrative candidate features, not approved development sites or a complete regional register. Retain original geometry metadata and OSM IDs; export per-hazard score/percentile, composite alternatives, rank range, population context and data-quality caveats. No population is assigned as customers served. P2 intersection is authoritative mapping context, not a blanket legally binding exclusion rule. Legal development eligibility requires jurisdiction-specific rules not applied here.
