# Getting started

AddiVortes is Bayesian regression on a sum of Voronoi tessellations
(Stone and Gosling, 2025). It stands to BART (Chipman, George and
McCulloch, 2010) as a tessellation stands to a tree: a cell is a region
of the covariate space rather than a box, so a boundary oblique to the
axes costs one cell rather than many splits. The priors, the Gibbs
sampler and the posterior summaries are those of BART, so a reader who
has fitted BART, dbarts or bartMachine will recognise every argument.

This page fits one data set end to end with the defaults: the fit, its
predictions and intervals, the convergence diagnostics, and saving the
result. It assumes you are comfortable with regression and have met
MCMC; the two papers above are the only background it draws on.

## A first fit

[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
takes a formula and a data frame, a data frame and a response, or a
numeric matrix and a response. The data below have two numeric
covariates and a factor, and a response that is smooth in `a`, linear in
`b`, shifted for one level of `g`, with noise of standard deviation 0.1.

``` r

library(thiessen)
library(ggplot2)

set.seed(1)
n <- 200
d <- data.frame(
  a = runif(n),
  b = runif(n),
  g = factor(sample(c("low", "mid", "high"), n, replace = TRUE))
)
d$y <- 3 * (d$a - 0.4)^2 + 0.5 * d$b + 0.3 * (d$g == "high") +
  rnorm(n, sd = 0.1)

fit <- thiessen(y ~ ., d, seed = 1)
fit
#> AddiVortes fit
#> Call: thiessen(formula = y ~ ., data = d, seed = 1)
#> gaussian model, 200 observations, 4 covariates
#> 200 tessellations, 4000 draws kept after 200 burn-in, thinning 1
#> In-sample RMSE 0.06568, seed 1
#> 4 chains, largest R-hat 1.009, smallest effective sample size 618
```

The defaults are the published configuration: 200 tessellations, 200
burn-in sweeps and 1000 kept draws, run as four chains whose draws are
pooled. The last line of the print is the convergence check over the
four chains, R-hat and the smallest effective sample size; the [chains
and
convergence](https://l-thomson.github.io/thiessen/r/articles/convergence.html)
page says what it monitors and what to do when it warns.

A factor covariate becomes d - 1 treatment-contrast indicators, the
first level as reference, as
[`model.matrix()`](https://rdrr.io/r/stats/model.matrix.html) encodes
one; the
[covariates](https://l-thomson.github.io/thiessen/r/articles/covariates.html)
page covers the encoding and the alternatives.

## Predictions and intervals

[`predict()`](https://rdrr.io/r/stats/predict.html) returns the
posterior mean of the response and matches new data by column name and
type. `interval = "credible"` adds the interval of the mean function;
`interval = "prediction"` covers a new observation, so it is wider by
the noise.

``` r

new <- data.frame(a = c(0.2, 0.8), b = 0.5, g = factor("low", levels(d$g)))
predict(fit, new)
#> [1] 0.3437324 0.6641378
predict(fit, new, interval = "credible", level = 0.9)
#>            fit     lower     upper
#> [1,] 0.3437324 0.2253157 0.4600493
#> [2,] 0.6641378 0.5421018 0.7852601
predict(fit, new, interval = "prediction", level = 0.9)
#>            fit     lower     upper
#> [1,] 0.3437324 0.1536057 0.5338050
#> [2,] 0.6641378 0.4705782 0.8587153
```

The figure follows the fit along `a` with `b` and `g` held fixed, the 90
per cent credible band around it and the noise-free truth dashed.

``` r

grid <- data.frame(
  a = seq(0, 1, length.out = 100), b = 0.5,
  g = factor("low", levels(d$g))
)
band <- predict(fit, grid, interval = "credible", level = 0.9)
grid$truth <- 3 * (grid$a - 0.4)^2 + 0.5 * grid$b

ggplot(cbind(grid, band), aes(a)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue",
              alpha = 0.3) +
  geom_line(aes(y = fit), colour = "steelblue") +
  geom_line(aes(y = truth), linetype = "dashed") +
  labs(y = "f(a, b = 0.5, g = low)")
```

![](thiessen_files/figure-html/transect-1.png)

`predict(type = "draws")` returns the per-draw values, one row per kept
draw, for any summary the two intervals do not give.

## The usual methods

[`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
[`residuals()`](https://rdrr.io/r/stats/residuals.html),
[`sigma()`](https://rdrr.io/r/stats/sigma.html),
[`nobs()`](https://rdrr.io/r/stats/nobs.html),
[`summary()`](https://rdrr.io/r/base/summary.html) and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) behave as they
do on any model object, and `update(fit, seed = 2)` refits with that one
argument replaced.

``` r

c(nobs = nobs(fit), sigma = sigma(fit),
  rmse = sqrt(mean(residuals(fit)^2)))
#>         nobs        sigma         rmse 
#> 200.00000000   0.09101508   0.06568429
summary(fit)
#> AddiVortes fit
#> Call: thiessen(formula = y ~ ., data = d, seed = 1)
#> gaussian model, 200 observations, 4 covariates
#> 200 tessellations, 4000 draws kept after 200 burn-in, thinning 1
#> 
#> Residuals:
#>        2.5%         25%         50%         75%       97.5% 
#> -0.12568090 -0.04316460  0.00488110  0.04043969  0.12285359 
#> 
#> sigma:
#>       2.5%        25%        50%        75%      97.5% 
#> 0.07845003 0.08621635 0.09081057 0.09541242 0.10474962 
#> 
#> In-sample RMSE 0.06568
#> 4 chains, largest R-hat 1.009, smallest effective sample size 618
```

[`sigma()`](https://rdrr.io/r/stats/sigma.html) is the posterior mean of
the residual standard deviation, to be read against the 0.1 the data
were made with. [`summary()`](https://rdrr.io/r/base/summary.html)
repeats the convergence line and adds the quantiles of the residuals and
of the posterior draws of sigma.

[`plot()`](https://rdrr.io/r/graphics/plot.default.html) traces the
per-draw sampler diagnostics, one panel per quantity and one line per
chain: the residual scale, the mean cells per tessellation and the mean
active covariates per tessellation. It is a convergence check rather
than a display of the fit.

``` r

plot(fit)
```

![](thiessen_files/figure-html/trace-1.png)

## Which covariates the ensemble uses

[`variable_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/variable_inclusion.md)
reports the share of the ensemble’s active dimensions falling on each
covariate.

``` r

variable_inclusion(fit)
#>         a         b      glow      gmid 
#> 0.2639085 0.2519900 0.2429079 0.2411935
```

Read that as where the ensemble spent its dimensions, not as variable
selection: at the default `omega` of `min(3, p)` every dimension is
always active when p is 3 or fewer, so the proportions are then uniform
by construction. Here p is 4 after the factor encoding, and the
informative covariates separate from the rest only weakly.

## Reproducibility

`seed = NULL`, the default, draws the chain’s seed from R’s stream, so
[`set.seed()`](https://rdrr.io/r/base/Random.html) governs; a whole
number reproduces the same draws for a given package version and
platform. The resolved seed is on the fit.

``` r

again <- thiessen(y ~ ., d, seed = 1)
identical(predict(fit, d), predict(again, d))
#> [1] TRUE
```

## Saving a fit

A fit is a plain R object holding the sampler state, so
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) writes one and a
later session reads it and predicts the same values without a refit.

``` r

path <- tempfile(fileext = ".rds")
saveRDS(fit, path)
identical(predict(readRDS(path), d), predict(fit, d))
#> [1] TRUE
```

## Progress

Nothing is printed while a fit runs. Progress is signalled with
progressr, so the session chooses whether and how to report it; choose a
handler once and it applies to every fit afterwards.

``` r

progressr::handlers(global = TRUE)
progressr::handlers("txtprogressbar")
fit <- thiessen(y ~ ., d, seed = 1)
```

The chunk is not evaluated because progressr renders nothing into a
static document. Reporting does not change the draws.

## Where next

- [Gaussian
  regression](https://l-thomson.github.io/thiessen/r/articles/gaussian.html),
  [Binary
  AddiVortes](https://l-thomson.github.io/thiessen/r/articles/binary-addivortes.html)
  and
  [H-AddiVortes](https://l-thomson.github.io/thiessen/r/articles/h-addivortes.html):
  one page per published model, with the likelihood, the priors and a
  worked example against a known truth. [Model
  description](https://l-thomson.github.io/thiessen/r/articles/model-description.md)
  holds the notation and the symbol-to-argument table.
- [Working with the
  posterior](https://l-thomson.github.io/thiessen/r/articles/posterior.md):
  the draws through posterior, bayesplot, loo and tidybayes.
- [Chains, convergence and
  compute](https://l-thomson.github.io/thiessen/r/articles/convergence.html):
  what the four chains monitor, when the fit warns, and what a fit
  costs.
- [The control
  surface](https://l-thomson.github.io/thiessen/r/articles/control-surface.md):
  every configuration group and the defaults the core resolves.
- [The sampler
  API](https://l-thomson.github.io/thiessen/r/articles/sampler-api.md):
  the Gibbs loop driven from R, so a model the package does not ship is
  built without Rust.
- [Troubleshooting](https://l-thomson.github.io/thiessen/r/articles/troubleshooting.md):
  symptom, then remedy.

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. *The Annals of Applied Statistics*
4(1), 266-298. <doi:10.1214/09-AOAS285>

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871. <doi:10.1080/10618600.2024.2414104>
