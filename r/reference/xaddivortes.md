# Fit XAddiVortes, the fast fit of AddiVortes

XAddiVortes is a fast point fit of the additive ensemble of Voronoi
tessellations of AddiVortes (Stone and Gosling, 2025), in place of
Markov chain Monte Carlo. The columns are screened by binned mutual
information with the response, and the tessellations are built in turn
on the running residual with damped, shrunk cell values. Each round then
replaces the share with the smallest log Bayes factor under the model's
prior by fresh tessellations and refits every cell value, and a final
refit chooses the damping by generalised cross-validation. The result is
a single point estimate, not a posterior sample. Starting the sampler
from the fit, `thiessen(start = "fast")`, gives posterior draws after
the burn-in.

## Usage

``` r
xaddivortes(x, ...)

# Default S3 method
xaddivortes(
  x,
  y,
  params = NULL,
  control = thiessen_control(),
  seed = NULL,
  threads = getOption("mc.cores", 1L),
  ...
)

# S3 method for class 'data.frame'
xaddivortes(
  x,
  y,
  params = NULL,
  control = thiessen_control(),
  seed = NULL,
  threads = getOption("mc.cores", 1L),
  ...
)

# S3 method for class 'formula'
xaddivortes(
  formula,
  data,
  params = NULL,
  control = thiessen_control(),
  seed = NULL,
  threads = getOption("mc.cores", 1L),
  ...
)
```

## Arguments

- x:

  A numeric matrix or data frame of covariates.

- ...:

  Not used; must be empty.

- y:

  The numeric response.

- params:

  The settings, from
  [`xaddivortes_params()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_params.md);
  `NULL` takes the defaults.

- control:

  The model configuration, from
  [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md);
  its `mean_params` set the number of tessellations and the covariate
  space. The sweep schedule is not used.

- seed:

  The seed, as for
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md).

- threads:

  Threads for prediction.

- formula:

  A formula, `response ~ covariates`.

- data:

  A data frame holding the variables of `formula`.

## Value

An object of class `c("xaddivortes", "thiessen")`: a
[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
fit with one draw, the fast fit, whose noise draw is the fit's sigma, so
every method of the fit uses that value. Its `fast` element is a list of
`sigma`, the noise estimate on the response's scale; `column_scores`,
the screening scores named by column; `train_score`, the training
residual root mean square after the build, after each round and after
the final refit; and `seconds`, the elapsed fit time.

## Details

The noise estimate `sigma(fit)` is the root of the training residual sum
of squares over the residual degrees of freedom of the final refit.
Those degrees of freedom do not count the choice of the cells, so the
intervals of `predict(fit, interval = "prediction")` can cover less than
their level;
[`vignette("fast-fits", package = "thiessen")`](https://l-thomson.github.io/thiessen/r/articles/fast-fits.md)
measures the coverage on held-out data. Split-conformal intervals around
the fit (Lei and others, 2018) are calibrated without that assumption:
fit on one part of the data and widen the prediction by the `level`
quantile of the absolute residuals on the other part.

The fit holds one draw, so it has no posterior to compare models by:
[`log_lik()`](https://mc-stan.org/rstantools/reference/log_lik.html),
and with it [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html)
and [`loo::waic()`](https://mc-stan.org/loo/reference/waic.html), refuse
it with an error of class `thiessen_error`.

The design and response are handled as by
[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md):
a matrix, a data frame or a formula, factors encoded as
treatment-contrast indicators. The model is the Gaussian one, so the
response must be numeric; a response or a `control` that selects another
family is an error.

XAddiVortes is experimental: it needs a build with the core's
`experimental` feature
([`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md))
and errors with the condition class `thiessen_requires_feature`
otherwise.

## References

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
(2018). Distribution-free predictive inference for regression. *Journal
of the American Statistical Association* 113(523), 1094-1111.
[doi:10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116)

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
[doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)

## See also

[`xaddivortes_params()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_params.md),
[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md),
[`predict.xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/predict.xaddivortes.md),
[`summary.xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/summary.xaddivortes.md)

## Examples

``` r
if (FALSE) { # core_experimental()
set.seed(1)
x <- matrix(runif(400), ncol = 2)
y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
control <- thiessen_control(tessellations = 20)
fit <- xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
                   control = control, seed = 1)
fit

frame <- data.frame(y = y, a = x[, 1], b = x[, 2])
xaddivortes(y ~ a + b, frame, control = control, seed = 1)
}
```
