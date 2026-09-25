# Changelog

Notable changes to the Python package, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versions
follow semantic versioning. The sampled values for a fixed seed are those of
the core crate version the package builds against, which
`thiessen.CORE_VERSION` reports.

## [Unreleased]

### Fixed

- The `posterior_predictive` replicates of `to_inference_data` under the
  Student-t and Laplace families are drawn at sigma, the scale of the t
  or Laplace error, where they were drawn at the error standard
  deviation, sigma sqrt(df / (df - 2)) or sigma sqrt(2), and so were too
  wide; a Student-t grid admitting df <= 2 no longer fails for want of
  a variance.

### Changed

- `to_inference_data` names the AFT replicate and log-likelihood
  variable `time`, the name of the observed variable it pairs with, so
  `arviz.plot_ppc` and `loo_pit` find the pair without renaming; the
  interval-censored replicate keeps `y`, its observation being a pair of
  bounds with no single partner.
- Equality of parameter objects compares field by field, an array
  element-wise, so a value holding one (`GeometryParams(precision=)`,
  `weighted_inclusion()`) compares rather than raising.
- A saved model naming an item the build gates raises
  `RequiresFeatureError` through the core's typed load path.
- `Model.fit` and the estimators default to `n_chains=4`, the number
  Vehtari and others (2021) recommend for the convergence checks, on one
  thread (`n_threads=1`, `n_jobs=None`), the scikit-learn convention. A
  call that sets nothing pays four chains on one core; `n_jobs=-1` or
  `n_threads=os.cpu_count()` runs them on every core for the same
  draws, about 2.7 times faster on four cores at n = 200. On Friedman #1
  with n = 200 and p = 10 the default schedule reaches a smallest
  effective sample size of about 290 and a largest R-hat of about 1.014,
  so a default fit warns; more draws per chain is the remedy.
  `FittedModel.load` and `Sampler` are unchanged.

### Added

- The response selects the outcome family where `Model` and `Sampler`
  name none: a numeric array the Gaussian family, a boolean array or a
  two-category pandas `Categorical` the probit family, an ordered
  `Categorical` the ordinal family, a structured survival array in the
  scikit-survival layout the AFT family and an `(n, 2)` array of bounds
  the interval-censored family. A named family is checked against the
  shape and a mismatch raises `ValueError` naming both. `log_likelihood`,
  `to_inference_data` and `Sampler.set_response` take the same shapes.
  `ordinal(categories=None)`, the new default, takes the count from the
  categories.
- Every row of `docs/experimental.md` is reachable in an extension built
  with the core's `experimental` feature: the Minkowski, Manhattan,
  cosine, Gower and Mahalanobis metrics and the per-column composite as
  entries of `GeometryParams(metric=)`, `GeometryParams(membership=
  soft_membership())` and `precision=`, `StructureParams(inclusion=
  weighted_inclusion() | dart_inclusion())`, and
  `TermParams(cell=CellParams(basis="linear"))`. A default build accepts
  the published defaults and raises `RequiresFeatureError` for the rest.
- `FittedModel.predict_proba` (the ordinal category probabilities),
  `dfs`, `cutpoints`, `bandwidths`, `inclusion_weights` and
  `concentrations`, empty where the model samples none; `to_inference_data`
  carries them as `df`, `cutpoint`, `bandwidth`, `inclusion_weight` and
  `concentration`, its predictive replicates follow each family's own
  observation model, and `observed_data` carries `time` and `event` or
  `lower` and `upper` for a censored response.

- `Model.fit(n_threads=)` and the estimators' `n_jobs` (joblib convention,
  `None` one thread, -1 every core): the chains of a fit run on up to that
  many threads, each chain on one thread with its own generator, and the
  fitted model splits the rows of a prediction, and of its quantiles and
  intervals, over the same number (`FittedModel.n_threads`, settable;
  `load(n_threads=)`; the estimators read `n_jobs` again at each
  prediction). The draws and the predictions do not depend on the
  count. The GIL is released for the fit.

- `thiessen.sampler.Sampler` (experimental): the core's Gibbs loop driven
  one call at a time, after the updatable sampler of dbarts and the
  low-level interface of stochtree. `step(n)`, `keep`, `set_response`,
  `fitted_values`, `noise_variances`, and `finish()` returning the
  ordinary `FittedModel`. Burn-in and thinning are the caller's loop, the
  response may be replaced between sweeps, and driving the configured
  schedule by hand reproduces `fit` bit for bit at the same seed.

- `Model`, the configuration, in the shape the core stores it: an outcome
  family from `gaussian(nu, q)` or `probit(offset)`, one `TermParams` group
  per ensemble (`mean_params`, `variance_params`), the flat run-length
  settings, and `fit(X, y, random_state=None, n_chains=1)`. A positive
  tessellation count on `variance_params` selects the heteroscedastic
  model. The family and group objects implement `get_params`/`set_params`
  with `<group>__<parameter>` routing, compare by value and reproduce
  their constructor call in `repr`.
- `n_chains` runs that many chains, each with a seed the core derives from
  the resolved seed, and pools their draws; the chain dimension of
  `to_inference_data` holds them. Two or more chains warn where R-hat
  exceeds 1.01 or a bulk or tail effective sample size falls below 400
  (Vehtari and others, 2021), computed by arviz on sigma and on the mean
  function at up to twenty training rows; a fit says so where arviz is not
  installed. `AddiVortesRegressor` and `AddiVortesClassifier` take
  `n_chains` as a constructor parameter.
- `FittedModel`, the kept draws: `predict`, `predict_draws`,
  `predict_latent`, `predict_variance`, `predict_quantiles`,
  `credible_interval`, `prediction_interval`, `log_likelihood`, `sigma`,
  `cell_counts`, `dimension_counts`,
  `variable_inclusion_proportions`, and the resolved configuration, the
  fit-time warnings and the in-sample root mean squared error. Pickling
  through the core's serde representation.
- `ThiessenError`, a `ValueError`, carrying the core's message.
- `random_state` taking an `int`, a `numpy.random.Generator`, a
  `numpy.random.RandomState` or `None`, resolved by one rule; an integer
  passes through unchanged, and the resolved seed is on the fitted object.
- `CORE_VERSION`, the core crate version the extension was built from.
- `thiessen.estimators` with `AddiVortesRegressor` and
  `AddiVortesClassifier`, meeting the scikit-learn estimator contract:
  explicit `__init__` parameters, `n_features_in_`, `feature_names_in_`,
  `classes_`, `predict(X, return_std=False)`, `predict_proba`,
  `__sklearn_tags__`, `clone` and pickling. The configuration groups are
  the same objects as on `Model`, so grid search routes into them:
  `GridSearchCV(model, {"mean_params__tessellations": [50, 200]})`.
  `scikit-learn` >= 1.6 is the `sklearn` extra.
- `categorical_features` on the estimators, of the
  `HistGradientBoostingRegressor` shape: `None`, `'from_dtype'`, an index
  array or a boolean mask. A named column becomes d - 1 treatment-contrast
  indicators, the first level as reference, unless its entry in the
  geometry's `metric` is `'categorical'`, in which case it passes as
  integer level codes.
- Coverage reporting for the Python suite, uploaded under the `python` flag.
- Packaging metadata completed against the pyOpenSci guide: the issue
  tracker, the platform and language classifiers, and a statement of need,
  contributing and citation sections in the README.
- A documentation site built with mkdocs-material and mkdocstrings: quick
  start, the model menu with parameter tables, priors and scaling with the
  BART correspondence table, the input-data contract, the scikit-learn and
  arviz pages, reproducibility, and the shared testing strategy. Every code
  example is executed at build and `mkdocs build --strict` runs in CI. The
  site is a CI artefact in this release; it is not deployed.
- `FittedModel.to_inference_data(X, y)`, returning the arviz `DataTree` of
  the fit: `posterior` with `mu` per draw, `sigma` under the Gaussian model
  only, and the per-draw cell and dimension counts; `posterior_predictive`
  and `log_likelihood` with `y`; and `observed_data`. The chain dimension is
  one, as the sampler runs one chain, and the predictive replicates are drawn
  in numpy from the fit's resolved seed rather than by the core. `arviz>=1.0`
  is the `arviz` extra; the groups are built through `arviz.from_dict` with
  no version dispatch.
- `FittedModel.save` and `FittedModel.load`, taking `str` or `os.PathLike`
  and writing the core's serde representation, which reloads bit-exact.
  Failures to read or write are `OSError`; contents that are not a fitted
  model are `ThiessenError`.
- The core's fit-time warnings are raised as `UserWarning` at fit, by `Model`
  and by the estimators, as well as staying on `FittedModel.warnings`.
- Overloads on `AddiVortesRegressor.predict`, so `return_std=True` types as a
  pair of arrays rather than a union.
- `numpy.integer` accepted for `random_state`, which already worked at
  runtime.
- `_native.EXPERIMENTAL`, whether the extension was built with the core's
  `experimental` feature. It is off in every wheel, so a released wheel
  reaches the published models only. The gated outcome families have
  constructors whatever the build carries (`tobit`, `aft`,
  `interval_censored`, `ordinal`, `student_t` and `laplace`), and an
  extension accepting them is built from source with
  `pip install ./python --config-settings build-args="--features
  experimental"`. Such a build is outside semantic versioning, and a model
  saved from one does not load in a build without the feature.
- `RequiresFeatureError`, a subclass of `ThiessenError` raised where a
  configuration or a saved model names an item the extension gates, so it
  is caught apart from an invalid configuration.
- `thiessen.families.Outcome`, the base of the family objects, which
  `Model` and `Sampler` take in place of the two published classes.
