# Calibrated configurations

The configurations the calibration suite covers, one entry per
model constructor of `crates/thiessen/tests/calibration.rs`: each
runs simulation-based calibration and Geweke tests at two sizes
(`docs/testing.md`). Rendered by the suite itself and checked
against this file, so the list cannot drift; regenerate with
`THIESSEN_UPDATE_DOCS=1 cargo test --features experimental --test
calibration calibrated_configuration_list`. Every other
combination of the documented options is valid to run and is not
separately verified (`docs/models.md`, Validation).

## gaussian

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## probit

```json
{"outcome":{"probit":{"offset":-0.2}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## heteroscedastic

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":2,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## tobit (experimental)

```json
{"outcome":{"tobit":{"lower":-0.25,"upper":0.3,"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## aft (experimental)

```json
{"outcome":{"aft":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## interval censored (experimental)

```json
{"outcome":{"interval_censored":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## student_t (experimental)

```json
{"outcome":{"student_t":{"df":4.0,"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## laplace (experimental)

```json
{"outcome":{"laplace":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## ordinal (experimental)

```json
{"outcome":{"ordinal":{"categories":4,"offset":-0.1,"cutpoint_sd":1.0}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## spherical metric

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"spherical":{"sphere":0}},{"spherical":{"sphere":0}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"spherical":{"sphere":0}},{"spherical":{"sphere":0}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## categorical metric

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":["euclidean","categorical"],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":["euclidean","categorical"],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## minkowski metric (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"minkowski":{"p":1.0}},{"minkowski":{"p":1.0}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"minkowski":{"p":1.0}},{"minkowski":{"p":1.0}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## cosine metric (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"cosine":{}},{"cosine":{}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"cosine":{}},{"cosine":{}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## gower metric (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"gower":{"kind":"numeric"}},{"gower":{"kind":"categorical"}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"gower":{"kind":"numeric"}},{"gower":{"kind":"categorical"}}],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## mahalanobis metric (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":["mahalanobis","mahalanobis"],"sigma_c":0.8,"precision":[2.0,0.6,0.6,1.0]},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":["mahalanobis","mahalanobis"],"sigma_c":0.8,"precision":[2.0,0.6,0.6,1.0]},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## composite metric (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"manhattan":{}},"categorical"],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[{"manhattan":{}},"categorical"],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## weighted inclusion (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"inclusion":{"weighted":{"weights":[0.75,0.25]}},"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"inclusion":{"weighted":{"weights":[0.75,0.25]}},"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## dart inclusion (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"inclusion":{"dart":{"a":0.5,"b":1.0,"rho":2.0}},"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"inclusion":{"dart":{"a":0.5,"b":1.0,"rho":2.0}},"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## linear cell basis (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{"basis":"linear"}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```

## soft membership (experimental)

```json
{"outcome":{"gaussian":{"nu":6.0,"q":0.85}},"mean_params":{"tessellations":3,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8,"membership":{"soft":{"rate":10.0}}},"structure":{"omega":0.8},"cell":{}},"variance_params":{"tessellations":null,"k":3.0,"lambda_c":2.0,"geometry":{"metric":[],"sigma_c":0.8,"membership":{"soft":{"rate":10.0}}},"structure":{"omega":0.8},"cell":{}},"general_params":{"burn_in":200,"draws":1000,"thinning":1,"prior_only":false}}
```
