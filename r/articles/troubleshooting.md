# Troubleshooting

Each entry is a symptom, then the remedy. The chunks that run below use
a small fit so the messages on the page are the ones you will see.

``` r

library(thiessen)

set.seed(1)
n <- 60
x <- cbind(a = runif(n), b = runif(n))
y <- x[, "a"] + rnorm(n, sd = 0.1)
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
```

## The install fails with “cargo not found”

The package compiles Rust, so the build needs rustc 1.74 or later with
Cargo, from [rustup](https://rustup.rs). Install the toolchain before
starting R, or restart the session afterwards: rustup puts `cargo` on
the path of new sessions only, and a session older than the install
fails the build as though no toolchain were there. On Windows, Rtools
and the target `configure.win` names are needed as well; the
[README](https://github.com/l-thomson/thiessen/tree/dev/r#installation)
has the commands. The compilation uses no network: every crate ships
with the package.

## The fit prints nothing while it runs

That is the default, not a hang. Progress is signalled with progressr
and nothing is printed until the session chooses a handler:

``` r

progressr::handlers(global = TRUE)
progressr::handlers("txtprogressbar")
```

The bar stands until the fit is complete, since pooling the draws,
saving the state and the convergence summary run after the last sweep
and report too.

## The fit warns that the chains may not have converged

[`summary()`](https://rdrr.io/r/base/summary.html) reports R-hat and the
effective sample sizes. More draws per chain,
`general_params(draws = )`, is the first remedy, then more chains;
`thinning` reduces the stored draws without adding information, so it is
not the remedy for a low effective sample size. The [chains and
convergence](https://l-thomson.github.io/thiessen/r/articles/convergence.html)
page shows a fit that warns beside one that does not.

The warning fires on any short schedule, this page’s included, and
carries the condition class `thiessen_warning`, so a script that has
chosen a short schedule silences it by class rather than by message:

``` r

withCallingHandlers(
  quiet <- thiessen(x, y, control, seed = 1),
  thiessen_warning = function(condition) invokeRestart("muffleWarning")
)
quiet$convergence$rhat
#> [1] 1.646019
```

## `predict()` says a column is missing

A fit from a formula or a data frame matches new data by column name and
type through hardhat, and a column the new data lack is an error naming
it. Supply every covariate the fit used, with the same names and types;
a factor needs the same levels.

``` r

d <- data.frame(a = x[, "a"], b = x[, "b"], y = y)
fit <- thiessen(y ~ a + b, d, control, seed = 1)
#> Warning in thiessen(y ~ a + b, d, control, seed = 1): The chains may not have
#> converged: largest R-hat 1.646 (threshold 1.01), smallest effective sample size
#> 8 (threshold 400). Run more draws or more chains.
predict(fit, d["a"])
#> Error in `predict()`:
#> ! The required column "b" is missing.
```

## A factor column is rejected under a declared metric

Where `geometry_params(metric = )` is given, it has one entry per column
of the data and a factor column must be declared `"categorical"`; the
factor then passes as level codes rather than as indicators. Without a
declared metric every column is Euclidean and a factor becomes d - 1
indicators. The
[covariates](https://l-thomson.github.io/thiessen/r/articles/covariates.html)
page shows both.

## A configuration is rejected

Every value is validated by the core at construction, with the reason
kept. The message names the argument and what was passed:

``` r

thiessen_control(outcome = gaussian_outcome(q = 1.5))
#> Error in `thiessen_control()`:
#> ! invalid hyperparameter `q`: must be in the open interval (0, 1), got 1.5
```

Two combinations are refused by the model rather than by a bound: a
variance ensemble under the probit family, whose latent scale is fixed,
and a variance ensemble with `nu` at or below 2, whose prior mean would
not exist.

``` r

thiessen_control(
  outcome = probit_outcome(),
  variance_params = term_params(tessellations = 40)
)
#> Error in `thiessen_control()`:
#> ! invalid hyperparameter `variance_params.tessellations`: a variance ensemble needs a sampled sigma^2 to carry, and the probit and ordinal latent scales are fixed at 1 for identification
```

## `predict(interval = "prediction")` errors under the probit model

A two-point distribution has no continuous predictive interval. Use
`interval = "credible"` for an interval on the probability scale, or
[`posterior_predict()`](https://mc-stan.org/rstantools/reference/posterior_predict.html)
for replicate labels.

## `sigma()` errors under the heteroscedastic model

There is no single residual scale; the spread is
`predict(type = "variance")`, s^2(x) per draw.

## loo reports many Pareto k values above 0.7

The importance sampling behind PSIS-LOO fails for those observations. A
short schedule on a small data set produces several; more draws is the
answer, not a different estimator. The
[posterior](https://l-thomson.github.io/thiessen/r/articles/posterior.md)
page shows the count.

## An error names the core’s `experimental` feature

The configuration, or a saved fit, uses an outcome family or a component
option compiled only into a build made with `THIESSEN_EXPERIMENTAL=1`.
The condition carries the class `thiessen_requires_feature`, so it is
told apart from an invalid configuration. The
[experimental](https://l-thomson.github.io/thiessen/r/articles/experimental.md)
page has the install command and the catalogue;
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
reports the setting of the build in use.

``` r

c(core = core_version(), experimental = core_experimental())
#>         core experimental 
#>      "0.3.0"      "FALSE"
```

## A saved fit will not load

A fit written by a build carrying the experimental feature and read by a
build without it errors with that same class, naming the item and the
feature, at the first call that needs the state. Reinstall with the
feature, or refit under the default build.

## Where to ask

Bug reports and questions go to the [issue
tracker](https://github.com/l-thomson/thiessen/issues). A bug report
should carry
[`core_version()`](https://l-thomson.github.io/thiessen/r/reference/core_version.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md),
[`sessionInfo()`](https://rdrr.io/r/utils/sessionInfo.html) and the call
that failed, with the seed.
