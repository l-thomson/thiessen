# Residual standard deviation of a fitted model

The `stats` generic [`sigma()`](https://rdrr.io/r/stats/sigma.html) for
a fit: the posterior mean of the residual scale.

## Usage

``` r
# S3 method for class 'thiessen'
sigma(object, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

A single number: the posterior mean of sigma under a model with a global
residual scale (the Gaussian, tobit, AFT, interval-censored, Student-t
and Laplace models; under the last two the scale of the t or Laplace
errors), and 1 under the probit and ordinal models, whose latent scale
is fixed. The heteroscedastic model has no single residual scale; use
`predict(type = "variance")`.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
sigma(thiessen(x, y, control, seed = 1))
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
#> [1] 0.05985129
```
