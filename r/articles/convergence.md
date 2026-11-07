# Chains, convergence and compute

A default fit runs four chains and pools their draws, so every fit
carries the convergence diagnostics of Vehtari and others (2021):
rank-normalised split R-hat and the bulk and tail effective sample
sizes. This page shows what is monitored, what a fit that fails the
check looks like beside one that passes, what the thresholds mean, and
what the chains cost in time.

``` r

library(thiessen)
library(posterior)
#> This is posterior version 1.7.0
#> 
#> Attaching package: 'posterior'
#> The following objects are masked from 'package:stats':
#> 
#>     mad, sd, var
#> The following objects are masked from 'package:base':
#> 
#>     %in%, match

worst <- function(fit) {
  with(fit$convergence, c(ess = min(ess_bulk, ess_tail), rhat = rhat))
}

set.seed(1)
n <- 200
x <- cbind(a = runif(n), b = runif(n))
y <- 2 * (x[, "a"] - 0.5)^2 + 0.5 * x[, "b"] + rnorm(n, sd = 0.1)
```

## What is monitored

The check runs over sigma and over the mean function at up to twenty
training rows, spread through the data. Each chain has its own seed,
derived from `seed` in the core, and each starts from the same initial
state, so the chains differ by their random streams alone. A fit warns
where R-hat exceeds 1.01 or an effective sample size falls below 400,
the thresholds Vehtari and others (2021) give, and
[`print()`](https://rdrr.io/r/base/print.html) and
[`summary()`](https://rdrr.io/r/base/summary.html) repeat the warning.

## A fit that warns, and one that does not

A schedule of 50 burn-in sweeps and 100 draws is too short for the
chains to agree:

``` r

short <- thiessen(x, y, thiessen_control(
  general_params = general_params(burn_in = 50, draws = 100)
), seed = 1)
#> Warning in thiessen(x, y, thiessen_control(general_params =
#> general_params(burn_in = 50, : The chains may not have converged: largest R-hat
#> 1.056 (threshold 1.01), smallest effective sample size 80 (threshold 400). Run
#> more draws or more chains.
```

The default schedule of 200 burn-in sweeps and 1000 draws per chain
passes on these data:

``` r

long <- thiessen(x, y, seed = 1)
long$convergence[c("rhat", "ess_bulk", "ess_tail")]
#> $rhat
#> [1] 1.00809
#> 
#> $ess_bulk
#> [1] 899.0285
#> 
#> $ess_tail
#> [1] 1877.808
```

The trace of sigma over the chains shows what the schedule buys: 100
draws per chain against 1000. Over the monitored quantities the short
fit reaches a smallest effective sample size of 80 against 400 and a
largest R-hat of 1.056 against 1.01; the default fit reaches 899 and
1.008, and its chains overlap.

``` r

bayesplot::mcmc_trace(as_draws_array(short), pars = "sigma")
```

![](convergence_files/figure-html/trace-short-1.png)

``` r

bayesplot::mcmc_trace(as_draws_array(long), pars = "sigma")
```

![](convergence_files/figure-html/trace-long-1.png)

[`summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html)
gives the per-variable numbers behind the check. Here, sigma and the
mean function at three training rows:

``` r

draws <- as_draws_df(long)
summarise_draws(
  subset_draws(draws, variable = c("sigma", "mu[1]", "mu[2]", "mu[3]")),
  "mean", "sd", "rhat", "ess_bulk", "ess_tail"
)
#> # A tibble: 4 × 6
#>   variable   mean      sd  rhat ess_bulk ess_tail
#>   <chr>     <dbl>   <dbl> <dbl>    <dbl>    <dbl>
#> 1 sigma    0.0937 0.00590  1.01     899.    1878.
#> 2 mu[1]    0.216  0.0444   1.00    1445.    2787.
#> 3 mu[2]    0.152  0.0407   1.00    2175.    3256.
#> 4 mu[3]    0.366  0.0490   1.00    1787.    2922.
```

## What to change when the fit warns

More draws per chain, `general_params(draws = )`, is the first remedy:
the effective sample size grows with the draws, and R-hat falls as the
chains have longer to mix. More chains add draws at the same cost per
draw. `thinning` reduces the stored draws without adding information, so
it is not a remedy for a low effective sample size.

The default schedule is short for a harder problem. On Friedman \#1
(Friedman 1991) with n = 200 and p = 10:

``` r

friedman <- function(n, p = 10, sd = 1) {
  x <- matrix(runif(n * p), n, p, dimnames = list(NULL, paste0("x", 1:p)))
  f <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
    10 * x[, 4] + 5 * x[, 5]
  list(x = x, y = f + rnorm(n, sd = sd))
}
hard <- friedman(200)
hard_fit <- thiessen(hard$x, hard$y, seed = 1)
#> Warning in thiessen(hard$x, hard$y, seed = 1): The chains may not have
#> converged: largest R-hat 1.062 (threshold 1.01), smallest effective sample size
#> 66 (threshold 400). Run more draws or more chains.
worst(hard_fit)
#>       ess      rhat 
#> 65.893610  1.061863
```

The [Gaussian
regression](https://l-thomson.github.io/thiessen/r/articles/gaussian.md)
page runs 3000 draws per chain on the same function, which clears both
thresholds.

A fit of one chain carries no diagnostics and says so:

``` r

one <- thiessen(x, y, seed = 1, chains = 1)
one
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, seed = 1, chains = 1)
#> gaussian model, 200 observations, 2 covariates
#> 200 tessellations, 1000 draws kept after 200 burn-in, thinning 1
#> In-sample RMSE 0.07918, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains
```

## Compute

Cost is linear in the sweeps and close to linear in the rows and in the
tessellation count. The chains run on `getOption("mc.cores", 1L)`
threads, read when the fit is called, as Stan’s interfaces do: a session
that sets nothing runs the four chains on one thread and pays four
chains; `options(mc.cores = 4)` runs them on four cores for the same
draws. The draws do not depend on the thread count.

Effective sample size per second is the number to compare schedules and
thread counts by, since it measures information gained rather than draws
stored. On the machine that built this page, the default fit on one, two
and four threads:

``` r

seconds <- function(threads) {
  elapsed <- system.time(
    fit <- thiessen(x, y, seed = 1, threads = threads)
  )[["elapsed"]]
  c(threads = threads, seconds = round(elapsed, 1),
    min_ess = round(fit$convergence$ess_bulk),
    ess_per_second = round(fit$convergence$ess_bulk / elapsed))
}
rbind(seconds(1), seconds(2), seconds(4))
#>      threads seconds min_ess ess_per_second
#> [1,]       1     3.2     899            281
#> [2,]       2     1.8     899            493
#> [3,]       4     1.7     899            529
```

Pooling the draws and the diagnostics run on one thread after the
sweeps, so the speed-up from more threads is less than the thread count.
[`predict()`](https://rdrr.io/r/stats/predict.html) splits its rows over
the fit’s thread count, or over its own `threads` argument.

Progress over the whole fit is signalled with progressr, so nothing is
printed unless a session chooses a handler; the [getting
started](https://l-thomson.github.io/thiessen/r/articles/thiessen.md)
page shows the two lines that set one.

## References

Friedman, J. H. (1991). Multivariate adaptive regression splines. *The
Annals of Statistics* 19(1), 1-67. <doi:10.1214/aos/1176347963>

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Buerkner, P.-C.
(2021). Rank-normalization, folding, and localization: an improved R-hat
for assessing convergence of MCMC. *Bayesian Analysis* 16(2), 667-718.
<doi:10.1214/20-BA1221>
