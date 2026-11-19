# Limitations and appropriate claims

This is retrospective reconstruction, not recovery or validation of earlier CV work. The previously mentioned historical AUC is not a target or result and is not used for tuning.

| Limitation | Effect on interpretation | Handling |
|---|---|---|
| Undated IFFI inventory | Event chronology and pre-event predictor status are unverified | Static inventory association only; no forecast claim |
| Flood targets are regulatory zones | Reproducing mapping conventions is not independently predicting inundation | Explicit zonation-emulator target, separate metrics and map labels |
| EFFIS detection thresholds/sensor change | Small-fire omissions and temporal heterogeneity | Presence/background design; early-event eligibility; no true-negative claims |
| EFFIS historical completeness unverified | The public current-perimeter service accepts historical date filters, but returned records do not prove a complete historical archive | Call the 85 returned perimeters the acquired sample; older raw-perimeter completeness may require an official EFFIS data request |
| 250 m centre-based support | Small events can lack positive cells | Quantify lost support; never snap labels into new pixels |
| Incomplete predictor set | Missing geology, climate and hydrology can bias transfer | Terrain/CLC pilot scope; no invented NDVI, lithology or TWI |
| Static DSM | Buildings/vegetation and mixed acquisition vintages affect derivatives | Native-source disclosure, smoothing and no temporal causation |
| Spatial dependence beyond chosen blocks | Scores may remain optimistic despite buffers | Separate holdout, whole-group exclusion, terrain diagnostic and explicit unestimated independence range |
| Case-control sampling | Scores and AP do not represent deployment prevalence | No calibration or failure/event probabilities; show denominators and background sensitivity |
| Limited event/holdout groups | Pixel counts can overstate independent support | Group/block counts, whole-block bootstrap, report infeasible folds |
| Correlated terrain descriptors | Permutation and SHAP attributions can be unstable | Grouped local permutations, non-causal caveats |
| Composite decision scale | Percentile weights are preferences, not physical joint risk | Max and alternative weights; rank-instability flags |
| Existing OSM convenience sample | Cannot claim comprehensive Italian siting or infrastructure coverage | Explicit Turin sample and existing-asset status |
| GHSL residential estimates | No service areas, age/income vulnerability or daytime population | Separate context layer; never customers served |
| No local legal rules interpreted | P2 overlap does not alone determine development eligibility | No automatic legal no-go classification |
| Live unversioned services | Future responses may differ | Freeze raw responses, exact requests and hashes |
| Final flood service page requires fallback | Approximately 10 m generalisation affects a small tail of source records | Exact request retained; below 250 m cell scale; report as source processing |
| QGIS connection unverified | XML/datasources can be checked without proving native render | Retain project and native inspection status separately |

No susceptibility layer is an energy-grid risk estimate. Exposure, social vulnerability, engineering vulnerability and operational resilience require independent evidence that is not supplied by a fitted presence/background classifier.
