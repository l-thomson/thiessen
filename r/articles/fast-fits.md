# Fast fits and fast starts

[`xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes.md)
fits the AddiVortes model, a sum of Voronoi tessellations, by a fast
point fit in place of Markov chain Monte Carlo. The columns are screened
by mutual information with the response, the tessellations are built in
turn on the running residual, and each round replaces the share with the
smallest Bayes factor by fresh tessellations and refits the rest; a
final refit chooses the damping of the cell values by generalised
cross-validation. The fit has two uses: a fast point estimate, and the
starting state of the exact sampler, which changes where the chains
start and not the posterior they target. Both are experimental: they
need a build with the core’s `experimental` feature, and each chunk is
skipped in a build without it, so a rendering with no output below is
that, not a failure. The [table of experimental
items](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md)
gives the install command.

## A fast fit

Friedman’s first function, five relevant covariates among ten:

``` r

library(thiessen)
set.seed(1)
friedman <- function(n) {
  x <- matrix(runif(10 * n), n, 10)
  f <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
    10 * x[, 4] + 5 * x[, 5]
  list(x = x, y = f + rnorm(n), f = f)
}
train <- friedman(300)
test <- friedman(300)
fit <- xaddivortes(train$x, train$y, seed = 1)
fit
sqrt(mean((predict(fit, test$x) - test$f)^2))
```

[`xaddivortes_params()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_params.md)
holds the five settings and their defaults; the fit time grows linearly
with `rounds` (default 40).

``` r

round(sort(fit$fast$column_scores, decreasing = TRUE), 3)
```

## Intervals

`sigma(fit)` is the root of the training residual sum of squares over
the residual degrees of freedom of the final refit. Those degrees of
freedom do not count the choice of the cells, so the intervals of
`predict(fit, interval = "prediction")` can cover less than their level:

``` r

y_test <- test$f + rnorm(nrow(test$x))
naive <- predict(fit, test$x, interval = "prediction")
mean(y_test >= naive[, "lower"] & y_test <= naive[, "upper"])
```

A split-conformal interval (Lei and others, 2018) is calibrated without
that assumption: fit on one part of the data and widen the prediction by
a quantile of the absolute residuals on the other.

``` r

half <- sample(nrow(train$x), nrow(train$x) / 2)
proper <- xaddivortes(train$x[half, ], train$y[half], seed = 1)
scores <- abs(train$y[-half] - predict(proper, train$x[-half, ]))
k <- ceiling(0.95 * (length(scores) + 1))
width <- sort(scores)[k]
mean(abs(y_test - predict(proper, test$x)) <= width)
```

## Fast starts

`thiessen(start = "fast")` starts every chain from its own fast fit,
after which the configured burn-in and draws run. A fast start and a
warm start from a previous fit (`init`) cannot be combined.

``` r

control <- thiessen_control(
  general_params = general_params(burn_in = 50, draws = 100)
)
started <- thiessen(train$x, train$y, control, seed = 1, chains = 2,
                    start = "fast")
sqrt(mean((predict(started, test$x) - test$f)^2))
```

The schedule above is short so that the page builds quickly, and the fit
warns that the chains may not have converged. A fast start changes where
the chains start; the convergence checks apply as they do to a cold
start.

## References

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
(2018). Distribution-free predictive inference for regression. Journal
of the American Statistical Association 113(523), 1094-1111.
