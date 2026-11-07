# Posterior predictions from a fitted model

The posterior mean of the response, its draws, the latent mean or the
variance, at new rows or the training rows, with an optional credible or
predictive interval.

## Usage

``` r
# S3 method for class 'thiessen'
predict(
  object,
  newdata = NULL,
  type = c("mean", "draws", "latent", "variance", "probs"),
  interval = c("none", "credible", "prediction"),
  level = 0.95,
  threads = NULL,
  ...
)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- newdata:

  New covariates. A fit from a formula or a data frame takes a data
  frame, whose columns are matched to the fitted design by name and
  type; a fit from a matrix takes a numeric matrix with the fitted
  columns. `NULL`, the default, predicts at the training rows.

- type:

  The quantity: `"mean"`, the posterior mean of the response (the
  probability under the probit model, the expected category under the
  ordinal model, f(x) on the log-time scale under the AFT model);
  `"draws"`, that quantity for every kept draw; `"latent"`, the latent
  mean for every kept draw, f(x), or c + f(x) under the probit model;
  `"variance"`, the variance of y given f for every kept draw;
  `"probs"`, under the ordinal model only, the posterior-mean
  probability of each category, the
  [`MASS::polr`](https://rdrr.io/pkg/MASS/man/polr.html) name.

- interval:

  `"none"`, the default; `"credible"` for the interval of the posterior
  mean; `"prediction"` for the posterior predictive interval. Only with
  `type = "mean"`.

- level:

  The mass of a central interval. Default 0.95.

- threads:

  The number of threads, a whole number; `NULL`, the default, is the
  count the fit was made with.

- ...:

  Ignored.

## Value

For `type = "mean"` and `interval = "none"`, a numeric vector of length
`nrow(newdata)`; with an interval, a matrix of that many rows with
columns `fit`, `lower` and `upper`. For `type = "probs"`, a matrix of
one row per row of `newdata` and one column per category, named by the
levels of the response. For the other types, a matrix of one row per
kept draw and one column per row of `newdata`.

## Details

The rows of `newdata` are split over `threads` threads, each chunk
evaluated on a thread of its own; the values do not depend on the count.

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
head(predict(fit))
#> [1] 0.4166203 0.6201293 0.4009554 0.5988489 0.3391939 0.5588988
head(predict(fit, interval = "credible"))
#>            fit     lower     upper
#> [1,] 0.4166203 0.3283299 0.5179514
#> [2,] 0.6201293 0.5088993 0.7202284
#> [3,] 0.4009554 0.3016275 0.4812341
#> [4,] 0.5988489 0.5088993 0.6899766
#> [5,] 0.3391939 0.2652220 0.4342294
#> [6,] 0.5588988 0.4369092 0.6324378
```
