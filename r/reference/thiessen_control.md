# Configuration of a fit

The configuration of Stone and Gosling (2025), s. 2, in the shape the
core stores it: an outcome family, one parameter group per ensemble, and
the sweep schedule. Each part has its own constructor with its own
documentation:
[`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md)
and
[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md)
for the family,
[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)
for an ensemble, and
[`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md)
for the schedule. An argument left at its default gives the core's
default, so `thiessen_control()` is the published configuration.

## Usage

``` r
thiessen_control(
  outcome = NULL,
  mean_params = term_params(),
  variance_params = NULL,
  general_params = NULL,
  tessellations = NULL
)
```

## Arguments

- outcome:

  The outcome family, from
  [`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md),
  [`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md)
  or one of the experimental constructors listed under Details. `NULL`,
  the default, takes the family the response selects.

- mean_params:

  The ensemble describing the average, from
  [`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md).

- variance_params:

  The ensemble describing the spread, from
  [`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md).
  `NULL`, the default, keeps the spread constant.

- general_params:

  The sweep schedule, from
  [`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md).
  `NULL`, the default, is
  [`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md).

- tessellations:

  Shortcut for the mean ensemble's size:
  `thiessen_control(tessellations = 200)` is
  `thiessen_control(mean_params = term_params(tessellations = 200))`. An
  error if `mean_params` also sets a count.

## Value

An object of class `"thiessen_control"` holding the four groups.

## Details

Attaching `variance_params` with a positive tessellation count selects
the heteroscedastic model, in which the residual variance varies with x;
it needs the Gaussian family with nu \> 2, and the paper's count is 40.
The two ensembles share one covariate space: `geometry` and `structure`
set on `mean_params` apply to `variance_params` as well.

One shortcut: `thiessen_control(tessellations = 200)` sets the mean
ensemble's size without spelling the group, since that count is the
single number most fits tune. Every other setting is named in its group.

`outcome` left `NULL` takes the family from the response at fit: a
numeric vector the Gaussian family, a two-level factor the probit
family, an ordered factor the ordinal family, and a
[`survival::Surv()`](https://rdrr.io/pkg/survival/man/Surv.html) the AFT
or the interval-censored family by its type. A named family is checked
against the response and a mismatch is an error naming both; see
[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md).

The models reachable with
[`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md)
and
[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md)
are the published method and follow semantic versioning. Everything else
the core crate adds sits behind its `experimental` Cargo feature, which
a released build does not enable, so a configuration or a saved fit
naming such an option is rejected with the condition class
`thiessen_requires_feature`. The families it gates each have a
constructor here
([`tobit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/tobit_outcome.md),
[`aft_outcome()`](https://l-thomson.github.io/thiessen/r/reference/aft_outcome.md),
[`interval_censored_outcome()`](https://l-thomson.github.io/thiessen/r/reference/interval_censored_outcome.md),
[`ordinal_outcome()`](https://l-thomson.github.io/thiessen/r/reference/ordinal_outcome.md),
[`student_t_outcome()`](https://l-thomson.github.io/thiessen/r/reference/student_t_outcome.md)
and
[`laplace_outcome()`](https://l-thomson.github.io/thiessen/r/reference/laplace_outcome.md)),
so naming one is portable, and a build accepting them is installed from
source with `THIESSEN_EXPERIMENTAL=1` in the environment;
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
reports the setting of the build in use. Each item graduates on its own
once calibrated and is then accepted as any other option, with no
separate opt-in. The table of experimental items and their status is
[`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).

The core's calibration suite covers the configurations listed in
[`docs/calibrated.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/calibrated.md);
component options are verified in isolation, and every other combination
of the documented options is valid to run and is not separately
verified.

## References

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
[doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)

## Examples

``` r
thiessen_control(tessellations = 50)
#> <thiessen_control>
#>   outcome         from the response
#>   mean_params     term_params(tessellations = 50, k = 3, lambda_c = 5)
#>   variance_params none (constant spread)
#>   general_params  general_params(burn_in = 200, draws = 1000, thinning = 1, prior_only = FALSE)

thiessen_control(outcome = probit_outcome())
#> <thiessen_control>
#>   outcome         probit_outcome()
#>   mean_params     term_params(k = 3, lambda_c = 5)
#>   variance_params none (constant spread)
#>   general_params  general_params(burn_in = 200, draws = 1000, thinning = 1, prior_only = FALSE)

thiessen_control(
  outcome = gaussian_outcome(nu = 10),
  mean_params = term_params(tessellations = 200, lambda_c = 25),
  variance_params = term_params(tessellations = 40),
  general_params = general_params(burn_in = 500, draws = 2000)
)
#> <thiessen_control>
#>   outcome         gaussian_outcome(nu = 10, q = 0.85)
#>   mean_params     term_params(tessellations = 200, k = 3, lambda_c = 25)
#>   variance_params term_params(tessellations = 40, k = 3, lambda_c = 5)
#>   general_params  general_params(burn_in = 500, draws = 2000, thinning = 1, prior_only = FALSE)
```
