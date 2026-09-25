# Changelog

## thiessen 0.0.0.9000

- [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  fits the Gaussian, binary probit and heteroscedastic models to a
  numeric matrix, with
  [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)
  carrying the configuration in the shape the core stores it: an outcome
  family from
  [`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md)
  or
  [`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md),
  one
  [`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)
  group per ensemble (with
  [`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md)
  and
  [`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md)
  nested inside), and the sweep schedule from
  [`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md).
  A positive tessellation count on `variance_params` selects the
  heteroscedastic model, and `thiessen_control(tessellations = )` is the
  one shortcut, setting the mean ensemble’s size. Methods:
  [`predict()`](https://rdrr.io/r/stats/predict.html) for the posterior
  mean, the per-draw mean function and variance, and central credible
  and posterior predictive intervals;
  [`sigma()`](https://rdrr.io/r/stats/sigma.html),
  [`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
  [`residuals()`](https://rdrr.io/r/stats/residuals.html),
  [`nobs()`](https://rdrr.io/r/stats/nobs.html),
  [`print()`](https://rdrr.io/r/base/print.html) and
  [`summary()`](https://rdrr.io/r/base/summary.html).
- `thiessen(formula, data)` and `thiessen(data.frame, y)` through
  hardhat, so `predict(newdata = )` matches columns by name and type and
  reports a missing one. A factor covariate becomes d - 1
  treatment-contrast indicators, the first level as reference; where
  `control` declares a `metric`, factors pass as integer level codes and
  each factor column must declare `"categorical"`. A two-level factor
  response becomes 0 and 1 with the first level as the zero.
- Methods on the established generics:
  [`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html),
  [`as_draws_array()`](https://mc-stan.org/posterior/reference/draws_array.html),
  [`ndraws()`](https://mc-stan.org/posterior/reference/draws-index.html)
  and
  [`nchains()`](https://mc-stan.org/posterior/reference/draws-index.html),
  registered on demand so posterior is needed only by a session that
  calls them, with R-hat and effective sample sizes coming from
  [`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html);
  and
  [`posterior_predict()`](https://mc-stan.org/rstantools/reference/posterior_predict.html),
  [`posterior_epred()`](https://mc-stan.org/rstantools/reference/posterior_epred.html),
  [`log_lik()`](https://mc-stan.org/rstantools/reference/log_lik.html)
  and
  [`predictive_interval()`](https://mc-stan.org/rstantools/reference/predictive_interval.html)
  from rstantools, re-exported so no
  [`library(rstantools)`](https://mc-stan.org/rstantools/) is needed.
  [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html) runs on
  [`log_lik()`](https://mc-stan.org/rstantools/reference/log_lik.html).
  The draws carry `mu[i]`, `sigma` under the Gaussian model only, and
  the per-draw `cell_count` and `dimension_count`.
- `chains` runs that many chains, each with a seed the core derives from
  `seed`, and pools their draws. Two or more chains give rank-normalised
  split R-hat and the bulk and tail effective sample sizes of sigma and
  of the mean function at up to twenty training rows, through
  [`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html).
  A fit warns, and [`print()`](https://rdrr.io/r/base/print.html) and
  [`summary()`](https://rdrr.io/r/base/summary.html) repeat the warning,
  where R-hat exceeds 1.01 or an effective sample size falls below 400
  (Vehtari and others, 2021); a fit of one chain says so instead.
- The defaults are `chains = 4` and
  `threads = getOption("mc.cores", 1L)`, read when the fit is called, as
  Stan’s interfaces do: a session that sets nothing runs the four chains
  on one thread and pays four chains, and `options(mc.cores = 4)` runs
  them on four cores for the same draws, in less than half the
  one-thread time (about 45 per cent at n = 200 on four cores of a 2025
  laptop). On Friedman
  [\#1](https://github.com/l-thomson/thiessen/issues/1) with n = 200 and
  p = 10 the default schedule reaches a smallest effective sample size
  of about 100 and a largest R-hat of about 1.05, so a default fit
  warns; more draws per chain is the remedy.
- Thirteen guides, executed at build and grouped on the site: getting
  started; one page per published model on one template (likelihood,
  priors, posterior, example against a known truth) with a model
  description page holding the notation; the draws through posterior,
  bayesplot, loo and tidybayes; chains, convergence and compute; what
  each prior does; covariates and the covariate space; the control
  surface; the sampler API with a worked censored-response imputation;
  troubleshooting; and related software. Nine are installed as
  vignettes; the Gaussian, H-AddiVortes, convergence and covariates
  guides run longer than a package check allows and are website
  articles. A pkgdown site configuration groups the reference by
  surface.
- [`thiessen_sampler()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_sampler.md)
  (experimental) drives the core’s Gibbs loop one call at a time:
  `$step(n)`, `$keep()`, `$set_response()`, `$fitted_values()`,
  `$noise_variances()`, `$finish()` returning the fit
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  returns. Burn-in and thinning are the caller’s loop, the response may
  be replaced between sweeps, and driving the configured schedule by
  hand reproduces a one-chain fit bit for bit.
- [`plot()`](https://rdrr.io/r/graphics/plot.default.html) on a fit
  traces the per-draw diagnostics, one panel per quantity and one line
  per chain; distributional displays go through
  [`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html)
  and bayesplot.
- Conformance with the rOpenSci statistical software standards (general
  and Bayesian) is tagged through srr in `R/srr-stats-standards.R`,
  every standard met or waived with its reason.
- [`thiessen_diagnostics()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_diagnostics.md)
  returns the per-draw sigma, mean cells and mean active covariates,
  with the chain each draw comes from, and
  [`variable_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/variable_inclusion.md)
  the share of the active dimensions falling on each covariate.
- Progress over the whole fit is signalled with progressr, so a session
  reports it after
  [`progressr::handlers()`](https://progressr.futureverse.org/reference/handlers.html)
  and nothing is printed by default. The sweeps, pooling the draws,
  saving the state and the convergence summary each report, and each
  names itself in a sticky message a terminal handler pushes above the
  bar, so the bar stands until the fit is complete and the phase running
  is named. The phases after the sweeps take a share of the bar set from
  their cost against the sweeps’, which scale with the kept draws and
  the training rows respectively. The draws do not depend on whether a
  handler is set.
- A fit is a plain R object, so
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) writes one and a
  later session reads it without a refit. The fitted state lives on the
  Rust side behind a handle, with a byte encoding alongside for
  persistence, so no method serialises or parses the whole model.
- The call is stored on a fit, so
  [`stats::update()`](https://rdrr.io/r/stats/update.html) refits with
  an argument replaced.
- `seed = NULL` draws the chain’s seed from R’s stream, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) governs; a whole
  number passes to the core unchanged.
- The core is built without its `experimental` feature by default, so a
  released build reaches the published models only. The gated outcome
  families have constructors here whatever the build carries
  ([`tobit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/tobit_outcome.md),
  [`aft_outcome()`](https://l-thomson.github.io/thiessen/r/reference/aft_outcome.md),
  [`interval_censored_outcome()`](https://l-thomson.github.io/thiessen/r/reference/interval_censored_outcome.md),
  [`ordinal_outcome()`](https://l-thomson.github.io/thiessen/r/reference/ordinal_outcome.md),
  [`student_t_outcome()`](https://l-thomson.github.io/thiessen/r/reference/student_t_outcome.md)
  and
  [`laplace_outcome()`](https://l-thomson.github.io/thiessen/r/reference/laplace_outcome.md)),
  and a build accepting them is installed from source with
  `THIESSEN_EXPERIMENTAL=1` in the environment;
  [`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
  reports the build’s setting. Such a build is outside semantic
  versioning, and a fit saved from one does not load in a build without
  the feature.
- Errors carry the condition class `thiessen_error` and warnings
  `thiessen_warning`; an error naming the core’s `experimental` feature
  carries `thiessen_requires_feature` before it, so a configuration
  needing an opt-in build is handled apart from an invalid one. Scalar
  arguments are checked by rlang’s input checks, re-signalled under that
  class, so a rejected value is reported with the argument’s name and
  what was passed.
- [`print()`](https://rdrr.io/r/base/print.html) and
  [`format()`](https://rdrr.io/r/base/format.html) on an outcome family
  or a parameter group give the constructor call that builds it: the
  string parses back to the object.
- The response selects the outcome family where
  [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)
  names none (its `outcome` now defaults to `NULL`): a numeric vector
  the Gaussian family, a two-level factor the probit family, an ordered
  factor the ordinal family (previously encoded as an unordered one), a
  [`survival::Surv()`](https://rdrr.io/pkg/survival/man/Surv.html) of
  type `"right"` the AFT family and one of type `"interval2"` the
  interval-censored family. A named family is checked against the
  response and a mismatch is an error naming both. The same shapes are
  taken by the formula method, by
  [`thiessen_sampler()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_sampler.md)
  and its `$set_response()`, and by `log_lik(y = )`.
  `predict(type = "probs")` gives the ordinal category probabilities;
  [`sigma()`](https://rdrr.io/r/stats/sigma.html) is 1 under the ordinal
  family.
- Experimental component options, gated as the outcome families are: the
  Minkowski, Manhattan, cosine, Gower and Mahalanobis entries of
  `geometry_params(metric = )` with `precision` for the last and a
  `group` label for composites;
  `geometry_params(membership = soft_membership())`;
  `structure_params(inclusion = weighted_inclusion() or dart_inclusion())`;
  and `term_params(cell = cell_params(basis = "linear"))`. The draws
  carry `df`, `cutpoint[k]`, `bandwidth[j]`, `inclusion_weight[j]` and
  `concentration` where the model samples them.
- Builds and links the core crate offline from vendored sources, the
  core as source under `src/rust/core` and the third-party crates in
  `src/rust/vendor.tar.xz`;
  [`core_version()`](https://l-thomson.github.io/thiessen/r/reference/core_version.md)
  reports the core version.
- `predict(interval = )` takes the mean and the interval from one pass
  over the draws rather than two; the values are unchanged.
- The sampler keeps the working buffers of a backfitting step across
  steps rather than allocating them per proposal; sampled values are
  unchanged.
- The backfitting step carries one number per training row and builds
  the current tessellation’s cell statistics in the pass that forms
  them, with each cell’s statistics in one record; sampled values are
  unchanged and a sweep executes about a fifth fewer instructions.
- The reassignment of the training rows after a dimension move takes the
  nearest centre of every row in one branch-free pass per centre, and
  the rows that lose their centre under a change or removal take a plain
  Euclidean search; sampled values are unchanged and a sweep runs about
  a fifth faster on one machine.
- `thiessen(threads = )` runs the chains of a fit on up to that many
  threads, each chain on one thread with its own generator, and
  [`predict()`](https://rdrr.io/r/stats/predict.html) on the fit,
  intervals included, splits its rows over the same number or over its
  own `threads` argument; the draws and the predictions do not depend on
  the count. The chains of a fit now advance together rather than in
  turn, so a progress handler reports “sampling 2 chains” in place of
  one message per chain.
- The library is built with the workspace release profile
  (`lto = "fat"`, `codegen-units = 1`), which the detached manifest
  under `src/rust` did not carry: fits about 7 per cent and predictions
  10 to 15 per cent faster on one machine, the shared library half the
  size, sampled values unchanged.
- [`posterior_predict()`](https://mc-stan.org/rstantools/reference/posterior_predict.html)
  draws from each family’s own observation model: category codes under
  the ordinal family, times under the AFT family, values clipped to the
  limits under the tobit family, the working value under the
  interval-censored family, and Student-t or Laplace errors at the drawn
  scale under those families; the Gaussian and probit replicates are
  unchanged. [`sigma()`](https://rdrr.io/r/stats/sigma.html) and the
  family resolution at fit dispatch on the outcome’s class. A fit no
  longer stores `response_levels`; the levels are read from `response`.
  A saved fit naming an item the build gates signals
  `thiessen_requires_feature` through the core’s typed load path, and an
  `NA` event flag is refused at the boundary.
- Bug fix: a fit no longer fails where the calling environment defines a
  function named `rhat`, `ess_bulk` or `ess_tail`. The convergence
  measures are passed to
  [`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html)
  as functions rather than by name, so a user’s function of the same
  name is not resolved in their place.
