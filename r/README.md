
<!-- README.md is generated from README.Rmd. Please edit that file -->

# thiessen

Bayesian regression on a sum of Voronoi tessellations (AddiVortes; Stone
and Gosling, 2025, <doi:10.1080/10618600.2024.2414104>), a variant of
BART (Chipman, George and McCulloch, 2010) in which a cell is a region
of the covariate space rather than a box. The package provides the
Gaussian model of the paper together with its published variants, Binary
AddiVortes (probit classification) and H-AddiVortes (heteroscedastic
variance), with a formula interface, four-chain convergence diagnostics,
and methods for the posterior, rstantools, loo, bayesplot and tidybayes
generics.

The sampler is the `thiessen` Rust crate, built from sources vendored in
the package. The method and all credit for it belong to its authors;
this package is an independent implementation, and its test suite
compares posterior summaries against the authors' R package,
[AddiVortes](https://github.com/johnpaulgosling/AddiVortes).

## Installation

The package is not yet on CRAN, and it compiles Rust, so the build needs
a toolchain besides R's own:

- rustc 1.74 or later with Cargo, from [rustup](https://rustup.rs).
- On Windows, [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
  as well, and the Rust target the R build links against. Under the
  usual Rtools that is `x86_64-pc-windows-gnu`, so
  `rustup target add x86_64-pc-windows-gnu`; `configure.win` prints the
  target it selected, which differs on the clang and ARM builds.

Install the toolchain before starting R, or restart the session
afterwards: rustup puts `cargo` on the path of new sessions only, and a
session older than the install fails the build as though no toolchain
were there. Every crate the build needs ships with the package, so the
compilation itself uses no network.

``` r
install.packages("remotes")
remotes::install_github("l-thomson/thiessen", subdir = "r")
```

The vignettes are on the website named below, so the install leaves them
out. `build_vignettes = TRUE` installs them locally instead, and needs
`knitr` and `rmarkdown` present to build them.

## Example

``` r
library(thiessen)

set.seed(1)
n <- 200
x <- cbind(a = runif(n), b = runif(n))
y <- 2 * (x[, "a"] - 0.5)^2 + 0.5 * x[, "b"] + rnorm(n, sd = 0.1)

fit <- thiessen(x, y, seed = 1)
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, seed = 1)
#> gaussian model, 200 observations, 2 covariates
#> 200 tessellations, 4000 draws kept after 200 burn-in, thinning 1
#> In-sample RMSE 0.07966, seed 1
#> 4 chains, largest R-hat 1.008, smallest effective sample size 899
```

The last line is the convergence check over the four chains a fit runs
by default. `predict()` gives the posterior mean and, with `interval =`,
a credible interval for the mean function or a predictive interval for a
new observation:

``` r
grid <- cbind(a = seq(0, 1, length.out = 100), b = 0.5)
band <- predict(fit, grid, interval = "credible", level = 0.9)

plot(grid[, "a"], band[, "fit"], type = "l", col = "steelblue",
     ylim = range(band), xlab = "a", ylab = "f(a, b = 0.5)")
polygon(c(grid[, "a"], rev(grid[, "a"])), c(band[, "lower"], rev(band[, "upper"])),
        col = adjustcolor("steelblue", 0.3), border = NA)
lines(grid[, "a"], 2 * (grid[, "a"] - 0.5)^2 + 0.25, lty = 2)
```

<img src="man/figures/README-figure-1.png" alt="" width="100%" />

The dashed line is the truth. `sigma()`, `summary()`, `plot()`,
`posterior::as_draws_df()` and `loo::loo(log_lik(fit))` all take the
fit.

## Documentation

The [package website](https://l-thomson.github.io/thiessen/r/) renders
every page below with its output, beside the reference pages.

- Get started: `vignette("thiessen")`, one data set end to end.
- Models: `vignette("gaussian")`, `vignette("binary-addivortes")` and
  `vignette("h-addivortes")`, one page per published model on one
  template, with `vignette("model-description")` holding the notation
  and the symbol-to-argument table.
- Using a fit: `vignette("posterior")` for the draws through posterior,
  bayesplot, loo and tidybayes; `vignette("convergence")` for the
  chains, R-hat, effective sample size and compute; `vignette("priors")`
  for what each prior does; `vignette("covariates")` for factors,
  scaling and the covariate space; `vignette("control-surface")` for
  every configuration group.
- Building your own model: `vignette("sampler-api")`, the Gibbs loop
  driven from R, so an outcome family the package does not ship is built
  without Rust.
- Help: `vignette("troubleshooting")` and
  `vignette("related-software")`.

## Scope

The published surface is the Gaussian, probit and heteroscedastic
models. The core crate also carries further outcome families (tobit,
accelerated failure time, interval-censored, ordinal, Student-t,
Laplace) and component options behind its `experimental` Cargo feature,
which a released build does not enable. Their constructors exist in
every build, so a script naming one is portable, and a build that
accepts them is installed from source with `THIESSEN_EXPERIMENTAL=1` in
the environment; `core_experimental()` reports the setting of the build
in use. Each item graduates on its own once calibrated and derived,
under the policy in
[`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).

## Where to ask

Bug reports and questions go to the [issue
tracker](https://github.com/l-thomson/thiessen/issues). A bug report
should carry `core_version()`, `core_experimental()`, `sessionInfo()`
and the call that failed, with the seed.

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. The Annals of Applied Statistics
4(1), 266-298. <doi:10.1214/09-AOAS285>

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. Journal of Computational and Graphical Statistics
34(3), 859-871. <doi:10.1080/10618600.2024.2414104>
