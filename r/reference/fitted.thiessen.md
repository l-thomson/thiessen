# Fitted values, residuals and observation count of a fitted model

The `stats` generics
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
[`residuals()`](https://rdrr.io/r/stats/residuals.html) and
[`nobs()`](https://rdrr.io/r/stats/nobs.html) for a fit, on the scale of
the response.

## Usage

``` r
# S3 method for class 'thiessen'
fitted(object, ...)

# S3 method for class 'thiessen'
residuals(object, ...)

# S3 method for class 'thiessen'
nobs(object, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

For [`fitted()`](https://rdrr.io/r/stats/fitted.values.html) and
[`residuals()`](https://rdrr.io/r/stats/residuals.html), a numeric
vector of length `nobs(object)`: the posterior mean of the response at
each training row, and the response less that mean. For
[`nobs()`](https://rdrr.io/r/stats/nobs.html), the number of
observations.

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
nobs(fit)
#> [1] 60
head(residuals(fit))
#> [1] 0.08337970 0.09654695 0.03354623 0.05462711 0.03440560 0.03597332
```
