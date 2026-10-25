# Predictions of an XAddiVortes fit

The prediction is the sum of the fitted cell values. The prediction
interval is the prediction plus and minus the normal quantile times
`sigma(object)`, the fit's noise estimate: a rough interval that takes
the fit as known, not a posterior interval (see
[`xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes.md)
for calibrated intervals).

## Usage

``` r
# S3 method for class 'xaddivortes'
predict(
  object,
  newdata = NULL,
  interval = c("none", "prediction"),
  level = 0.95,
  threads = NULL,
  ...
)
```

## Arguments

- object:

  An object of class `"xaddivortes"`.

- newdata:

  New covariates, as for
  [`predict.thiessen()`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md);
  `NULL` predicts at the training rows.

- interval:

  `"none"` or `"prediction"`.

- level:

  The interval level, in (0, 1).

- threads:

  Threads for this prediction; `NULL` takes the fit's.

- ...:

  Not used; must be empty.

## Value

A numeric vector, or with an interval a matrix with columns `fit`,
`lower` and `upper`.

## Examples

``` r
if (FALSE) { # core_experimental()
set.seed(1)
x <- matrix(runif(400), ncol = 2)
y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
fit <- xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
                   control = thiessen_control(tessellations = 20),
                   seed = 1)
head(predict(fit, x))
head(predict(fit, x, interval = "prediction"))
}
```
