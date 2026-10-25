# Fit an AddiVortes model

AddiVortes is Bayesian regression on a sum of Voronoi tessellations
(Stone and Gosling, 2025): the mean function is a sum of m
tessellations, each with a mean per cell, drawn by the Gibbs sampler of
the paper. It stands to BART (Chipman, George and McCulloch, 2010) as a
tessellation stands to a tree: a cell is a region of the covariate space
rather than a box, so a boundary oblique to the axes costs one cell
rather than many splits.

## Usage

``` r
thiessen(x, ...)

# Default S3 method
thiessen(
  x,
  y,
  control = thiessen_control(),
  seed = NULL,
  chains = 4,
  threads = getOption("mc.cores", 1L),
  init = NULL,
  init_draw = NULL,
  start = c("cold", "fast"),
  fast = NULL,
  ...
)

# S3 method for class 'data.frame'
thiessen(
  x,
  y,
  control = thiessen_control(),
  seed = NULL,
  chains = 4,
  threads = getOption("mc.cores", 1L),
  init = NULL,
  init_draw = NULL,
  start = c("cold", "fast"),
  fast = NULL,
  ...
)

# S3 method for class 'formula'
thiessen(
  formula,
  data,
  control = thiessen_control(),
  seed = NULL,
  chains = 4,
  threads = getOption("mc.cores", 1L),
  init = NULL,
  init_draw = NULL,
  start = c("cold", "fast"),
  fast = NULL,
  ...
)
```

## Arguments

- x:

  A numeric matrix of covariates, one row per observation, or a data
  frame. A numeric vector is taken as one column.

- ...:

  Passed to the method.

- y:

  The response, with one value per row of `x`: a numeric vector, a
  two-level factor, an ordered factor, or a
  [`survival::Surv()`](https://rdrr.io/pkg/survival/man/Surv.html) of
  type `"right"` or `"interval2"`. Under the probit model a numeric
  vector must hold 0 and 1.

- control:

  An object of class `"thiessen_control"`, from
  [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md).

- seed:

  The seed of the chain. `NULL`, the default, draws one from R's stream,
  so [`set.seed()`](https://rdrr.io/r/base/Random.html) governs; a whole
  number in `[0, 2^53]` passes to the core unchanged, so the same value
  reproduces the same draws for a given package version and platform.

- chains:

  The number of chains to run, a whole number. Each chain has its own
  seed, derived from `seed` in the core, and the draws of the chains are
  pooled. Two or more chains give the convergence diagnostics; one chain
  does not. Default 4, the number Vehtari and others (2021) recommend
  and the Stan default.

- threads:

  The number of threads, a whole number. The chains are spread over at
  most this many threads, each chain on one thread with its own
  generator, so the draws do not depend on it;
  [`predict()`](https://rdrr.io/r/stats/predict.html) on the fit splits
  the rows over the same number. The default reads
  `getOption("mc.cores", 1L)` when the fit is called, as Stan's
  interfaces do, so `options(mc.cores = 4)` is the one-line opt-in and a
  session that sets nothing runs on one thread, where a default fit is
  four chains long.

- init:

  `NULL`, the default, for the cold start; or a previous fit of the same
  model to the same data, from `thiessen()`. Each chain then starts from
  one of its kept draws instead of the cold start; only the starting
  state changes, not the posterior. Without `init_draw` and with as many
  chains as `init`, chain k continues from the last draw of chain k of
  `init`. Otherwise chain k (from 1) starts from pooled draw
  `init_draw - k + 1`, `init_draw` defaulting to the last; with fewer
  draws than chains up to it, every chain starts from it and the fit
  warns. This is the warm-start rule of stochtree. Pass a `seed` other
  than the one `init` was fitted with: the same seed replays the random
  numbers of its chains, and the fit warns. A clean R-hat across chains
  started from one previous fit is a check within that run, not across
  dispersed starts (Vehtari and others, 2021). Experimental: needs a
  build with the `experimental` feature.

- init_draw:

  The pooled draw of `init` the first chain starts from, a whole number
  from 1.

- start:

  Where each chain starts: `"cold"`, the sampler's own start, or
  `"fast"`, its own XAddiVortes fit
  ([`xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes.md)),
  after which the same burn-in and draws run. Only the starting state
  differs, so the chains target the same posterior either way. `"fast"`
  needs the Gaussian family, cannot be combined with `init`, and needs a
  build with the `experimental` feature.

- fast:

  The XAddiVortes settings of a fast start, from
  [`xaddivortes_params()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_params.md);
  `NULL` takes the defaults. Given with `start = "cold"` it is an error.

- formula:

  A two-sided formula. The left side names the response and the right
  side the covariates, `.` for every remaining column.

- data:

  A data frame holding the columns the formula names.

## Value

An object of class `"thiessen"`: a list with the fitted state, the
resolved configuration, the number of chains, the thread count, the
number of kept draws, the convergence diagnostics where two or more
chains ran, the seed used, the design, the response the in-sample fit is
measured against (the numeric response, the factor codes, or under a
censored family the observed values on the model's scale, the log times
under the AFT family), the fitted values, the residuals against that
response, the hardhat blueprint where one applies, and the call.

## Details

A factor covariate becomes d - 1 treatment-contrast indicators, the
first level as reference, as `model.matrix` and upstream AddiVortes
encode it. Where `control` declares a `metric`, one entry per column,
factors are passed as integer level codes instead and each factor column
must declare `"categorical"`.

The response selects the outcome family where `control` names none: a
numeric vector the Gaussian family, a two-level factor the probit family
(0 and 1 with the first level as the zero, as `glm` treats one), an
ordered factor the ordinal family (the
[`MASS::polr`](https://rdrr.io/pkg/MASS/man/polr.html) convention), a
[`survival::Surv()`](https://rdrr.io/pkg/survival/man/Surv.html) of type
`"right"` the AFT family and one of type `"interval2"` the
interval-censored family (an `NA` bound is one-sided censoring and an
equal pair an exact value). A family named in `control` is checked
against the response and never coerced: the probit family also takes the
numbers 0 and 1, and every other family takes only the shape that
selects it. This is `glm`'s rule, the declared family authoritative and
the response validated against it, with "not declared" representable.
The families beyond the Gaussian and probit ones need a build with the
core's `experimental` feature; see
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md).

Missing (`NA`) and non-finite values in the covariates or the response
are rejected with an error; no row is dropped silently.

[`stats::update()`](https://rdrr.io/r/stats/update.html) works on a fit:
the call is stored, so `update(fit, seed = 2)` refits with that argument
replaced.

With `chains` of two or more, the chains are run together with the seeds
the core derives from `seed`, each on one of `threads` threads, their
draws are pooled, and the fit carries rank-normalised split R-hat and
the bulk and tail effective sample sizes of sigma and of the mean
function at up to twenty training rows
([`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html)).
A fit warns, and [`print()`](https://rdrr.io/r/base/print.html) and
[`summary()`](https://rdrr.io/r/base/summary.html) repeat the warning,
where R-hat exceeds 1.01 or an effective sample size falls below 400
(Vehtari and others, 2021). A fit of one chain says so instead.

A default fit runs four chains of 200 burn-in and 1000 draws each, on
`getOption("mc.cores", 1L)` threads, so on one thread it costs four
chains; `options(mc.cores = 4)` runs the chains on four cores for the
same draws, and the fit then takes less than half the one-thread time
(about 45 per cent at n = 200 on four cores of a 2025 laptop; pooling
the draws and the diagnostics run on one thread), so it costs under two
chains rather than four. On Friedman \#1 with n = 200 and p = 10 the
smallest effective sample size over the monitored points is about 100
(threshold 400) and the largest R-hat about 1.05 (threshold 1.01), so
the default fit warns. More draws per chain, `general_params(draws = )`,
is the answer; `thinning` is not.

## Progress

Progress over the whole fit is signalled with progressr, so a session
reports it after
[`progressr::handlers()`](https://progressr.futureverse.org/reference/handlers.html)
and nothing is printed by default; `progressr::handlers(global = TRUE)`
sets one for a whole session. The schedule raises one progression per
sweep, to a maximum of a hundred over the sweeps of every chain, then
pooling the draws, saving the state and the convergence summary, so the
report closes when the fit is complete rather than at the last sweep.
The phases after the sweeps scale with the kept draws and run on one
thread, while the sweeps scale with the training rows and run on
`threads`, so their share of the bar is set from that ratio: the bar is
most of the way along when the sweeps of a one-chain fit end, and about
a third of the way for four chains on four threads at n = 200. Each
phase names itself in a sticky message, which a terminal handler pushes
above the bar rather than overwriting, so the phase that is running is
named whatever handler is set. The draws do not depend on whether a
handler is set.

## Persistence

A fit is a plain R object holding the sampler state, so
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) writes one and a
later session reads it and predicts the same values, with no refit. A
fit written by a build with the core's `experimental` feature and read
by a build without it errors with the condition class
`thiessen_requires_feature`, naming the item and the feature, at the
first call that needs the state.

## Conditions

Errors raised by this package and by the core carry the condition class
`thiessen_error`, and its warnings carry `thiessen_warning`, so either
can be handled or silenced by class rather than by message. An error
naming the core's `experimental` feature carries
`thiessen_requires_feature` before it, so a configuration needing an
opt-in build is told apart from an invalid one. The convergence warning
fires on any short schedule, so silencing it deliberately is a routine
need:

    withCallingHandlers(
      fit <- thiessen(x, y, control),
      thiessen_warning = function(condition) {
        invokeRestart("muffleWarning")
      }
    )

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. *The Annals of Applied Statistics*
4(1), 266-298.
[doi:10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285)

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
[doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Buerkner, P.-C.
(2021). Rank-normalization, folding, and localization: an improved R-hat
for assessing convergence of MCMC. *Bayesian Analysis* 16(2), 667-718.
[doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
fit <- thiessen(x, y, control, seed = 1)
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 160 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.03107, seed 1
#> 4 chains, largest R-hat 1.777, smallest effective sample size 7
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.

# The default four chains carry the convergence diagnostics, and warn on
# a schedule this short; one chain reports none.
thiessen(x, y, control, seed = 1, chains = 1)
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1, chains = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 40 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.04022, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains

frame <- data.frame(y = y, a = x[, 1], b = factor(x[, 2] > 0))
thiessen(y ~ a + b, frame, control, seed = 1)
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
#> AddiVortes fit
#> Call: thiessen(formula = y ~ a + b, data = frame, control = control,      seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 160 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.03107, seed 1
#> 4 chains, largest R-hat 1.777, smallest effective sample size 7
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
```
