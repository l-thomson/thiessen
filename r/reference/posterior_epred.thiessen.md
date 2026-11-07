# Draw from the posterior distribution of the expected response

The `rstantools` generic
[`posterior_epred()`](https://mc-stan.org/rstantools/reference/posterior_epred.html)
for a fit: the expected response for every kept draw.

## Usage

``` r
# S3 method for class 'thiessen'
posterior_epred(object, newdata = NULL, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- newdata:

  New covariates, as
  [`predict.thiessen()`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md)
  takes them. `NULL`, the default, is the training rows.

- ...:

  Ignored.

## Value

A double matrix, one row per kept draw and one column per row of
`newdata`: the mean of the response, the probability under the probit
model.

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
dim(posterior_epred(fit))
#> [1] 160  60
```
