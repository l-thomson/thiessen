# Central posterior predictive interval

The `rstantools` generic
[`predictive_interval()`](https://mc-stan.org/rstantools/reference/predictive_interval.html)
for a fit: the central interval of the posterior predictive distribution
at each row.

## Usage

``` r
# S3 method for class 'thiessen'
predictive_interval(object, prob = 0.9, newdata = NULL, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- prob:

  The mass of the interval. Default 0.9.

- newdata:

  New covariates, as
  [`predict.thiessen()`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md)
  takes them. `NULL`, the default, is the training rows.

- ...:

  Ignored.

## Value

A double matrix of one row per row of `newdata`, with the lower and
upper bounds as columns named by their percentiles.

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
head(predictive_interval(fit))
#>             5%       95%
#> [1,] 0.2842902 0.5409320
#> [2,] 0.4775706 0.7488575
#> [3,] 0.2700285 0.5204748
#> [4,] 0.4701944 0.7189686
#> [5,] 0.2165645 0.4598651
#> [6,] 0.4131914 0.6815822
```
