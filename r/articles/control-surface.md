# The control surface

The configuration of a fit has four parts, and
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)
holds one of each. Every name here is a name in the core’s stored
configuration and in the Python package, so one description serves all
three surfaces. The outcome constructors are the exception: they carry
an `_outcome` suffix in R, and the name the configuration stores is
unchanged, so a fit saved by one surface is read by another. The full
mapping is
[`docs/parity.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/parity.md).

``` r

library(thiessen)

thiessen_control()
#> <thiessen_control>
#>   outcome         from the response
#>   mean_params     term_params(k = 3, lambda_c = 5)
#>   variance_params none (constant spread)
#>   general_params  general_params(burn_in = 200, draws = 1000, thinning = 1, prior_only = FALSE)
```

## The outcome family

`outcome` selects the observation model: `gaussian_outcome(nu, q)` for
continuous responses, carrying the sigma^2 prior’s degrees of freedom
and calibration quantile, and `probit_outcome(offset)` for binary ones.
The suffix keeps
[`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md)
from masking [`stats::gaussian()`](https://rdrr.io/r/stats/family.html),
which [`glm()`](https://rdrr.io/r/stats/glm.html) takes, and both
families carry it rather than only the one that would clash. Attaching
`variance_params` with a positive tessellation count extends the
Gaussian family to the heteroscedastic model. `outcome` left `NULL`, the
default, takes the family from the response at fit: a numeric vector the
Gaussian family and a two-level factor the probit family, with the
experimental families selected by their own response shapes (the
[`?experimental_outcomes`](https://l-thomson.github.io/thiessen/r/reference/experimental_outcomes.md)
lists them). A family named here is checked against the response and a
mismatch is an error naming both.

``` r

gaussian_outcome(nu = 3)
#> gaussian_outcome(nu = 3, q = 0.85)
probit_outcome()
#> probit_outcome()
```

## The ensembles

`mean_params` describes the ensemble behind the average and
`variance_params`, when given, the ensemble behind the spread. Each is a
[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)
group: the tessellation count, the cell-value prior spread `k`, the
cell-count prior rate `lambda_c`, and two nested groups.

``` r

term_params(tessellations = 200, k = 3, lambda_c = 25)
#> term_params(tessellations = 200, k = 3, lambda_c = 25)
```

[`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md)
is the covariate space: the `metric` of each column and the
centre-coordinate scale `sigma_c`. A default build reaches three
metrics, `"euclidean"`, `"categorical"` and a labelled sphere; the rest
the core carries are experimental and are rejected without the opt-in
build. A Euclidean column is min-max scaled to \[-0.5, 0.5\] over its
training range inside the sampler, which is what makes one distance
comparable across columns; a categorical or spherical column is not
scaled, and `sigma_c` is on the scaled coordinate, so 1 is the full
range of a Euclidean column.
[`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md)
is the covariate-inclusion prior: `omega / p` is the prior probability
of including a covariate, resolved to `min(3, p)` at fit when unset. The
two ensembles share one covariate space: set these on `mean_params` and
they apply to `variance_params` as well.

``` r

term_params(
  geometry = geometry_params(metric = list("euclidean", "categorical")),
  structure = structure_params(omega = 2)
)
#> term_params(k = 3, lambda_c = 5, geometry = geometry_params(metric = list("euclidean", "categorical"), sigma_c = 0.8), structure = structure_params(omega = 2))
```

## The sweep schedule

[`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md)
carries the run lengths: `burn_in`, `draws`, `thinning`, and
`prior_only`, which switches the likelihood off so the chain draws from
the prior. The chains and threads of a fit are arguments of
[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
rather than of the schedule, since they change nothing about the model.

``` r

general_params(burn_in = 500, draws = 2000, thinning = 2)
#> general_params(burn_in = 500, draws = 2000, thinning = 2, prior_only = FALSE)
```

## The shortcut, and validation

`thiessen_control(tessellations = 200)` sets the mean ensemble’s size
without spelling the group; it is the single number most fits tune, and
it is the only flat argument. Every value is validated by the core at
construction, with the reason kept:

``` r

thiessen_control(outcome = gaussian_outcome(q = 1.5))
#> Error in `thiessen_control()`:
#> ! invalid hyperparameter `q`: must be in the open interval (0, 1), got 1.5
thiessen_control(
  outcome = probit_outcome(),
  variance_params = term_params(tessellations = 40)
)
#> Error in `thiessen_control()`:
#> ! invalid hyperparameter `variance_params.tessellations`: a variance ensemble needs a sampled sigma^2 to carry, and the probit and ordinal latent scales are fixed at 1 for identification
```

An unknown name in any group is an ordinary unused-argument error, since
each group is a function with a fixed signature:

``` r

term_params(zeta = 1)
#> Error in `term_params()`:
#> ! unused argument (zeta = 1)
```

## Defaults

An argument left at its default gives the core’s default, so
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)
is the published configuration. The core resolves the configuration at
fit and the fit carries what it resolved, so the values below are read
from the core rather than repeated here.

``` r

n <- 40
x <- cbind(
  a = seq(0, 1, length.out = n),
  b = rep(c(0, 0.25, 0.5, 0.75), length.out = n),
  c = rep(c(0, 1), length.out = n)
)
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]

resolved <- thiessen(x, y, thiessen_control(
  general_params = general_params(burn_in = 1, draws = 1)
), seed = 1, chains = 1)$control
mean_params <- resolved$mean_params

c(
  tessellations = mean_params$tessellations,
  k = mean_params$k,
  lambda_c = mean_params$lambda_c,
  sigma_c = mean_params$geometry$sigma_c,
  omega = mean_params$structure$omega,
  nu = resolved$outcome$nu,
  q = resolved$outcome$q
)
#> tessellations             k      lambda_c       sigma_c         omega 
#>        200.00          3.00          5.00          0.80          3.00 
#>            nu             q 
#>          6.00          0.85
```

`omega` is the one that depends on the data, resolving to `min(3, p)`,
three of the three covariates above. The one default that departs from
the paper is `lambda_c`: Stone and Gosling (2025), s. 2.3, report 25,
and CRAN AddiVortes takes 5 from 0.6.8 onward; this package follows the
implementation, and `term_params(lambda_c = 25)` is the paper’s setting.
The [priors](https://l-thomson.github.io/thiessen/r/articles/priors.md)
page draws from the prior at three settings of each.
