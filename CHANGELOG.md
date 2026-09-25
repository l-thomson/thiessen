# Changelog

Notable changes to the core crate, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Versions follow semantic versioning. Patch releases preserve sampled
values for a fixed seed; minor releases may change them and the entry
says "Sampled values changed" with the reason.

## [Unreleased]

### Fixed

- The Student-t df grid draws the degrees of freedom from their
  conditional given the residuals and sigma^2, the weights integrated
  out, then the weights under the new df. The conditional given the
  weights alone could not leave the grid value the weights were drawn
  under, so a fit sat at the top of its grid whatever the data. Sampled
  values changed under a df grid; a fixed df is untouched.

### Added

- `Fitted::load`: a saved model read through any serde deserialiser with
  every error typed, `RequiresFeature` for a configuration the build
  gates among them, where the `Deserialize` impl reports each through
  the format's error as text. Loading also checks that the DART
  inclusion-weight draws are present exactly under `Inclusion::Dart`,
  one per covariate, and a soft-membership bandwidth exactly under
  `Membership::Soft`. `Tessellation::bandwidth` is present in every
  build, `None` without the feature.
- (experimental) `fit_aft_chains_with_threads` and
  `fit_interval_censored_chains_with_threads`: the censored models as
  `fit_chains_with_threads` fits the plain ones, through the one chain
  schedule; `fit_aft` and `fit_interval_censored` are their one-chain
  form.
- `fit_chains_with_threads` and `Sampler::advance_all`: the chains of a
  fit spread over at most `n_threads` scoped threads, each chain on one
  thread with its own generator, so the draws are those of the chains
  run in turn on any count. `Fitted::threads` and `set_threads`: the
  rows of a prediction, and of the per-row quantile and interval pass
  after it, split into that many contiguous chunks, each evaluated on a
  thread of its own by the same operations in the same order, so the
  values do not depend on the count; an execution setting, not persisted
  with the model. Either count is bounded by the parallelism available
  to the process.

- `Sampler::training_response`: the response the in-sample fit is
  measured against, caller scale, the observed values under the models
  whose working response is replaced by latents. `Fitted::cutpoint_draws`,
  `inclusion_weight_draws`, `concentration_draws` and `bandwidth_draws`
  are present in every build, empty without the feature.
- (experimental) `Fitted::inclusion_weight_draws` and
  `Fitted::concentration_draws`: the DART inclusion weights s and their
  concentration theta of each kept draw, which the sampler drew and no
  kept draw held. s is the sampled prior weight Linero (2018) reports and
  `BART::wbart` returns as `varprob`, not
  `Fitted::variable_inclusion_proportions`, which counts the usage the
  tessellations realised. Kept only under `Inclusion::Dart`, so a payload
  written by another configuration is byte-identical and one written
  before this change still loads. Outside the semver promise
  (docs/experimental.md).
- (experimental) `Fitted::bandwidth_draws`: the soft-membership
  bandwidth of each mean tessellation, one row per kept draw, on the
  scaled covariate space its prior is on. The values were already kept
  on each tessellation; this reads them off in one call. Outside the
  semver promise (docs/experimental.md).
- (experimental) `Metric::Minkowski` with order p >= 1 and
  `Metric::Manhattan`, its p = 1 alias: Minkowski distance on the scaled
  active coordinates, the active columns of one order combined as
  (sum |x_d - c_d|^p)^(2 / p), so p = 2 is Euclidean bit-for-bit. Centre
  coordinates and structural moves are those of Euclidean; only the
  assignment of rows to cells changes. Outside the semver promise
  (docs/experimental.md).
- (experimental) `Metric::Cosine`: 1 minus the cosine similarity of the
  active Cosine coordinates, squared into the key; 1 against a single
  zero vector, 0 when both are zero. The triangle inequality does not
  hold, the [-0.5, 0.5] scaling makes the origin data-dependent, and the
  option is intended for covariates that are directions already. Outside
  the semver promise (docs/experimental.md).
- (experimental) `Metric::Gower` with a per-column kind: the mean, over
  the active Gower columns, of the range-normalised absolute difference
  (numeric, the [-0.5, 0.5] scaling) or the plain mismatch of integer
  level codes (categorical, levels learnt at fit, uniform centre
  coordinates), squared into the key. Missing-value weighting is not
  implemented. Outside the semver promise (docs/experimental.md).
- (experimental) `Metric::Mahalanobis` with `GeometryParams::precision`,
  a user-supplied row-major p x p matrix over the encoded design, checked
  at fit (symmetric, positive definite). The active-subspace distance
  uses the principal submatrix on the active columns, which is not the
  conditional precision; the identity matrix reproduces Euclidean
  bit-for-bit. Outside the semver promise (docs/experimental.md).
- (experimental) per-column composite: a `group` label (default 0) on
  the Minkowski, Manhattan, Cosine and Gower entries. Columns sharing a
  metric and a label form one group; group distances enter the key as a
  sum of squares, the composition the sphere and categorical columns
  already use, so all-Euclidean-like groups reproduce Euclidean
  bit-for-bit. Every column belongs to exactly one group by
  construction. The axioms of a composite are those of its weakest
  member metric. Manhattan and Cosine entries are now objects
  (`{"manhattan": {}}`, `{"cosine": {}}`), a change within the same
  unreleased series. Outside the semver promise (docs/experimental.md).
- (experimental) `StructureParams::inclusion` with `Inclusion::Weighted`
  (bartMachine `cov_prior_vec`): a fixed non-negative weight per column.
  The subset prior given the dimension count is proportional to the
  product of member weights, the add-dimension, remove-dimension and
  swap proposals carry the weights in both directions, a zero weight
  excludes the column, and equal weights are the uniform prior on its
  code path, reproducing the default draws exactly. Nothing is sampled.
  Outside the semver promise (docs/experimental.md).
- (experimental) `Inclusion::Dart` (Linero 2018; BART `sparse = TRUE`
  with `a`, `b`, `rho`): the inclusion weights are a sampled vector
  s ~ Dirichlet(theta / p), updated by a Metropolis step whose
  Dirichlet(theta / p + counts) proposal leaves exactly the subset-prior
  normalisers in the ratio; the concentration theta is sampled on the
  BART grid of 1000 points, which is the prior itself, not an
  approximation. In the API a component of the term group; in validation
  model-grade, with SBC and Geweke at both sizes and a broken-sampler
  fixture for the weight update. Outside the semver promise
  (docs/experimental.md).
- (experimental) `CellParams::basis` with `Basis::Linear` on the mean
  slot: each cell contributes mu + beta' (x_A - c) with the slopes under
  the cell-value prior, the (d + 1)-dimensional conjugate update drawn
  jointly and the structural moves integrating the whole coefficient
  vector out (Prado, Moral and Parnell 2021 for the BART-family
  precedent). Needs min-max scaled columns; the variance ensemble's
  inverse-gamma cells keep the constant basis. Outside the semver
  promise (docs/experimental.md).
- (experimental) `GeometryParams::membership` with `Membership::Soft` on
  the mean slot: kernel-weighted membership over the tessellation's
  centres, exp(-d^2 / (2 tau^2)) normalised, with a per-tessellation
  bandwidth tau ~ Exponential(rate) updated by a Metropolis step on
  ln tau (Linero and Yang 2018 and the SoftBart package for the
  precedent). The cell values are drawn jointly from the b-dimensional
  conjugate normal and the structural moves integrate them out; the
  empty-cell rule still counts nearest-centre members. Constant cell
  basis and constant spread only; the probit model composes. Outside
  the semver promise (docs/experimental.md).
- (experimental) `Outcome::Tobit` with `TobitParams`: the type-I tobit
  model for a response censored at known limits (Tobin 1958), fitted by
  Chib (1992) data augmentation, each censored row's latent refreshed
  from a truncated normal before the sweep and the completed response
  running the Gaussian sweep unchanged. A response value equal to a
  limit is read as censored; a value beyond one is
  `Error::ResponseBeyondLimit`. sigma^2 is sampled, so a variance
  ensemble composes; uncensored data reproduces the Gaussian chain
  draw for draw at the same seed. Validated by a censored-likelihood
  quadrature known-answer test and SBC and Geweke at both sizes.
  Outside the semver promise (docs/experimental.md).
- (experimental) `Outcome::Aft` with `AftParams`, `fit_aft`,
  `Sampler::aft` and `Fitted::log_likelihood_survival`: the lognormal
  accelerated failure time model for a right-censored time-to-event
  response (Wei 1992; the BART package's `abart`), fitted by
  censored-data augmentation on the log scale with the censored refresh
  shared with the tobit model. The times and the event indicator are
  data, entering through the survival entry points; `fit` rejects the
  outcome. All-event data reproduces the Gaussian chain on log times
  draw for draw at the same seed. Validated by a censored-likelihood
  quadrature known-answer test, SBC and Geweke at both sizes, and an
  informational comparison against `abart`
  (benchmarks/upstream/aft_abart.R). Outside the semver promise
  (docs/experimental.md).
- (experimental) `Outcome::IntervalCensored` with
  `IntervalCensoredParams`, `fit_interval_censored`,
  `Sampler::interval_censored` and
  `Fitted::log_likelihood_interval_censored`: the model for a response
  known only to lie between two row-specific bounds (Sun 2006), fitted
  by censored-data augmentation with the tobit model's refresh extended
  to a two-sided truncated draw (Robert 1995, section 2). A pair of
  bounds per row is data, entering through the bound entry points; an
  equal pair is an exact value, an infinite endpoint one-sided
  censoring, and `fit` rejects the outcome. Exact data reproduces the
  Gaussian chain draw for draw at the same seed. Validated by an
  interval-likelihood quadrature known-answer test and SBC and Geweke
  at both sizes. Outside the semver promise (docs/experimental.md).
- (experimental) `Outcome::Ordinal` with `OrdinalParams`,
  `Fitted::predict_category_probabilities`, `Fitted::cutpoint_draws`
  and `Sampler::cutpoints`: the ordinal probit model over K ordered
  categories (Albert and Chib 1993, s. 5), latent variance 1 and first
  cutpoint 0 for identification, the offset resolved from the marginal
  shares. The K - 2 interior cutpoints are sampled by a blocked MH move
  with the latents integrated out (Cowles 1996) on the log-gap
  transformation (Albert and Chib 2001), walk scale
  2.38 / sqrt(n (K - 2)) (Roberts, Gelman and Gilks 1997), against
  independent N(0, cutpoint_sd^2) log-gap priors; the cutpoint draws
  are stored on the fitted model. `predict` is the expected category;
  `log_likelihood` the ordinal likelihood; a variance ensemble is
  rejected for identification. Two-category data reproduces the probit
  chain draw for draw at the same seed. Validated by a cutpoint and
  cell-mean quadrature known-answer test, SBC and Geweke at both sizes
  covering the cutpoints, a broken-sampler fixture dropping the prior
  ratio from the cutpoint acceptance, and a full-size cutpoint
  effective-sample-size check. Outside the semver promise
  (docs/experimental.md).
- (experimental) `Outcome::StudentT` with `StudentTParams`,
  `DegreesOfFreedom`, `Posterior::dfs` and `Sampler::student_df`: the
  independent Student-t model (Geweke 1993) as a scale mixture of
  normals, per-observation Gamma weights redrawn each sweep from their
  conditional and entering the precisions, sigma^2 drawn from the
  weighted inverse-gamma conditional. The error degrees of freedom are
  fixed (default 4) or drawn over a declared grid by their exact
  discrete conditional; no continuous-df sampler. `predict_variance` is
  sigma^2 df / (df - 2), `prediction_interval` the t mixture,
  `log_likelihood` the t log density; a variance ensemble is rejected
  pending an identification argument. Validated by a marginal
  t-likelihood quadrature known-answer test at fixed and grid df, and
  SBC and Geweke at both sizes. Outside the semver promise
  (docs/experimental.md).
- (experimental) `Outcome::Laplace` with `LaplaceParams`: Laplace
  errors as a scale mixture of normals with exponential mixing (Andrews
  and Mallows 1974), the per-observation weights redrawn each sweep
  from their inverse-Gaussian conditional (Park and Casella 2008)
  through the scale-mixture refresh shared with the Student-t model; no
  parameters of its own. `predict_variance` is 2 sigma^2,
  `prediction_interval` the Laplace mixture, `log_likelihood` the
  Laplace log density; a variance ensemble is rejected pending an
  identification argument. Validated by a marginal Laplace-likelihood
  quadrature known-answer test and SBC and Geweke at both sizes.
  Outside the semver promise (docs/experimental.md).
- `Outcome::catalogue()`: every outcome family the build carries, at its
  defaults. A binding renders its family constructors from it rather than
  keeping a list of names.
- `Fitted::pool`, the kept draws of chains of the same model and data as
  one fitted model in chain order, with `Error::MismatchedChains` for
  chains that disagree. Chain seeds come from `chain_seed`; the pooled
  in-sample RMSE is that of the pooled posterior mean.
- `fit_with_progress`, `fit` with a callback taking the number of
  completed sweeps and `burn_in + draws * thinning`. The draws for a
  fixed seed are those of `fit`.
- `VERSION`, the crate version as the bindings report it.
- `Metric` and `Config::metric`: the metric of each covariate column,
  Euclidean (default) or spherical, the great-circle distance of CRAN
  AddiVortes `metric = "S"` with `members` as the sphere label. Spherical
  columns are radians, unscaled, with the upstream centre-coordinate
  law; an upstream comparison fixture on a sphere joins the suite. The
  Euclidean chain for a fixed seed is unchanged.
- `Metric::Categorical`: integer level codes with the Eskin mismatch
  weight 2 / n^2 of CRAN AddiVortes `metric = "C"`, uniform centre
  coordinates over the training levels, `Error::InvalidCategoryCode` for
  a non-integer value; the levels are stored on `Fitted`.
- Cargo feature `experimental` for components and models beyond the
  published method, outside the semver promise; `Error::RequiresFeature`
  and `FromStr` for `Model`; the stability contract in the crate-root
  documentation and `docs/experimental.md`.

- `Model::{Gaussian, Probit, Heteroscedastic}` on `Config` and `Fitted`,
  with the per-model fields `Config::offset` (probit) and `Config::m_var`
  (heteroscedastic). The sampler composes a mean ensemble with one of
  three noise models: the global sigma^2 draw, the Albert-Chib latent
  refresh with unit variance, or a multiplicative ensemble of
  inverse-gamma variance tessellations. The Gaussian chain for a fixed
  seed is unchanged.
- `Fitted::model`, `Fitted::predict_latent`, `Fitted::predict_variance`;
  `Sampler::variance_tessellations`, `Sampler::noise_variances`;
  `Posterior::variance_tessellations`; `Error::InvalidLabel` and
  `Error::NotApplicable`. Saved Gaussian models from earlier builds load
  unchanged.
- Probit model (`Model::Probit`): Albert-Chib augmentation, sigma_mu =
  3 / (k sqrt m) on the latent scale, offset Phi^-1(ybar) by default;
  known-answer test by quadrature, SBC and Geweke tests at two sizes, a
  simulation-recovery test and a fixed-seed snapshot; `docs/models.md`
  with the model statements and parameter correspondence tables;
  `benchmarks/upstream/binary_variant.R` for the informational comparison
  against the authors' script.
- Heteroscedastic model (`Model::Heteroscedastic`): a multiplicative
  ensemble of `m_var` inverse-gamma variance tessellations with the
  (nu', lambda') prior matching of HBART; SBC and Geweke tests at two
  sizes, a broken-sampler fixture dropping the inverse-gamma cell
  normaliser, a prior-matching test, a simulation-recovery test and a
  fixed-seed snapshot; `benchmarks/upstream/heteroscedastic_variant.R`
  for the informational comparison against the authors' script. The
  Geweke tests thin at 45 for every model.

- Gaussian AddiVortes model: `fit`, the `Sampler` step API, and `Fitted`
  with prediction, interval, likelihood and ensemble-summary methods.
- `Config::prior_only`: sampling with the likelihood switched off, so
  `predict` gives prior predictive draws.
- `Sampler::pinned_prior`: a sampler whose prior is fixed by the caller,
  for calibration tests; `Sampler::lambda` reports the sigma^2 prior
  scale in force.
- Simulation-based calibration and Geweke joint-distribution tests for
  the Gaussian model at two sizes: small in `cargo test`, full in the
  nightly suite with an R evaluation (SBC package ECDF difference
  bands).
- Broken-sampler fixtures under `cfg(test)`, each rejected by the small
  SBC configuration.
- `Config::paper()`: the paper's settings, lambda_c = 25. The default
  lambda_c is 5 for every model, following CRAN AddiVortes >= 0.6.8.
- Upstream comparison: posterior summaries against upstream AddiVortes 0.7.1
  on fixed datasets, within 4 combined Monte Carlo standard errors.

### Changed

- A configuration naming an item the build gates behind `experimental` is
  reported by `Config::validate` as `Error::RequiresFeature` naming the
  item, in place of a serde unknown-variant or unknown-field error, so a
  caller matches the error rather than its text. `geometry.membership`,
  `geometry.precision`, `structure.inclusion` and `cell.basis` are fields
  of every build with only their experimental variants gated: a build
  without the feature accepts the published defaults `"hard"`, `"uniform"`
  and `"constant"`, which are what the model does when the field is
  absent. Serialised output and sampled values are unchanged.
- The entry points of the gated models (`fit_aft`,
  `fit_interval_censored`, `Sampler::aft`, `Sampler::interval_censored`,
  `Sampler::set_aft_response`, `Sampler::set_interval_censored_response`,
  `Fitted::log_likelihood_survival`,
  `Fitted::log_likelihood_interval_censored` and
  `Fitted::predict_category_probabilities`) are present in every build
  and return `Error::RequiresFeature` naming the outcome in a build
  without the feature; `Fitted::cutpoint_draws`, `Sampler::cutpoints`,
  `Sampler::student_df` and `Sampler::inclusion_state` are present and
  empty. A binding compiles against one surface under either build.
- Breaking: the `Model` enum is gone. The outcome layer is the whole
  selection surface: `Fitted::model_name()` returns "gaussian",
  "probit" or "heteroscedastic", `Fitted::outcome()` and
  `Fitted::has_variance_ensemble()` carry the identity, and
  `Error::NotApplicable` names the model as a string. A saved flat
  configuration fails to load with an error naming the replacement
  groups. Sampled values are unchanged for a fixed seed.
- Breaking: the configuration is four groups. `outcome` names the
  observation model and carries its own parameters and the sigma^2
  prior (`Outcome::Gaussian(GaussianParams { nu, q })`,
  `Outcome::Probit(ProbitParams { offset })`); `mean_params` and
  `variance_params` are one term-group struct instantiated per slot
  (`tessellations`, `k`, `lambda_c`, `geometry` with `metric` and
  `sigma_c`, `structure` with `omega`, `cell`); `general_params` is the
  sweep schedule. H-AddiVortes is `variance_params.tessellations`
  above 0 in place of a `Model` variant, defaulting to the paper's 40
  through `with_model`. Validity is derived from the outcome's scale
  mode: a variance ensemble under probit is rejected at config
  assembly, naming identification. The `with_*` setters remain and
  write into the groups; geometry and structure setters write both
  slots, which must agree while per-ensemble geometry awaits its
  identification argument. Sampled values are unchanged for a fixed
  seed.
- `Config` is non-exhaustive; construct it with `Config::new` and the
  `with_*` setters.
- A response lying exactly on a least-squares fit no longer degenerates
  the sigma^2 prior; sigma_hat falls back to the response standard
  deviation.
- Reproducibility contract in the crate-root documentation, with
  determinism and fixed-seed snapshot tests.
- CI pipeline, nightly statistical suite and branch protection.
- Input-data contract in the crate-root documentation; a design matrix
  with no columns is rejected with `Error::NoFeatures`.
- The crate readme is the repository README, so `cargo package` finds it.
- The end of a fit makes no prediction pass. The sampler sums the mean
  function at the training rows at each kept draw, from the cell
  assignments it already holds and in the order `Fitted::predict`
  evaluates a draw, so `Fitted::pool_samplers` and `fit_chains` take the
  fitted values and the in-sample RMSE from those sums: bit-identical to
  the pass for one chain, and within the last bit of it for several,
  whose per-chain sums are added before the division. `Sampler::finish`
  takes its per-chain RMSE from the same sums. A chain that kept a soft
  draw (experimental) still predicts once. On a default Friedman fit at
  n = 500 the pass was 2.5 to 3.7 s of a 6 to 10 s fit.
- The end of a fit predicts the training set once. `Sampler::finish`
  computed a per-chain in-sample RMSE that `Fitted::pool` discarded and
  recomputed from its own prediction pass, and the R binding predicted a
  third time for the fitted values. `Fitted::pool_samplers` and
  `fit_chains` pool the chains of a fit and return the fitted values
  from the one pass that gives the pooled in-sample RMSE; both bindings
  use them. Sampled values, the pooled in-sample RMSE and the fitted
  values are unchanged for a fixed seed.
- The backfitting step writes its proposal, the proposal's assignment,
  the partials, both sets of cell statistics and the cell-value draw into
  buffers the ensemble keeps across steps, and swaps an accepted proposal
  into the ensemble, in place of allocating each of them per proposal;
  the arithmetic and the draw order are unchanged, so sampled values are
  the same for a fixed seed. Under hard membership a sweep allocates
  nothing; a default fit went from about 22 heap allocations per proposal
  to none outside the kept draws.
- The backfitting step carries one number per training row, the
  family's partial (the partial residual under Gaussian cells, the
  product of the other tessellations under inverse-gamma cells), in
  place of a three-field record, and reads the input and the weight
  from the sampler's own vectors; the current tessellation's cell
  statistics are accumulated in the same pass that forms the partials,
  in place of a pass of their own; and each cell's statistics sit in one
  record. The per-row operations and their order are unchanged, so
  sampled values are the same for a fixed seed. A sweep at n = 500
  executes 21 per cent fewer instructions.
- The full reassignment of a dimension move writes the first active
  column's term in place of adding it to a zeroed buffer, takes the
  nearest centre of every row in one branch-free pass per centre over
  the column-major keys in place of a loop per row, and the rows that
  lose their centre under a change or removal take the all-Euclidean
  nearest-centre search without the per-column metric dispatch. Every
  key is the same sum in the same order and the winner the same strict
  comparison, so sampled values are the same for a fixed seed. The
  sweep benches run 1.16 to 1.23 times faster on one machine.
- Prediction evaluates each kept tessellation over every row at once,
  through the same streaming assignment the sampler uses under an
  all-Euclidean geometry, in place of a nearest-centre search per row per
  tessellation; every row's sum over the tessellations is formed in the
  same order, so the per-draw values are unchanged. The mixture quantile
  behind `prediction_interval` stops its bisection once the bracket has
  collapsed to adjacent doubles, where the remaining fixed iterations
  changed nothing. `Fitted::predict_with_interval` and `IntervalKind`
  return the posterior mean and a central interval from one traversal;
  the R `predict(interval = )` uses it.
- Cell assignment under an all-Euclidean geometry runs over a
  column-major mirror of the scaled design, one active column at a time
  for every row, in place of a per-row gather over the active columns:
  the full reassignment a dimension move makes, and the one-centre keys
  a centre move makes, stream over contiguous columns. Each row's sum
  over its active columns is formed in the same order as before, so the
  keys, the assignments and the sampled values are unchanged for a fixed
  seed. The mirror is built once per fit and doubles the memory of the
  scaled design.
- The all-Euclidean distance path is chosen once per geometry rather
  than by a scan of every column at every distance evaluation, which
  made the default fit grow with the number of covariates: a Friedman
  fit at n = 200, m = 200 and 1200 sweeps takes 1.4 s at p = 5, 10 and
  40 in place of 1.7, 2.2 and 4.6 s. Sampled values are unchanged for a
  fixed seed.
