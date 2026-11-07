# The interval-censored outcome family

**\[experimental\]**

## Usage

``` r
interval_censored_outcome(nu = 6, q = 0.85)
```

## Arguments

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

## Value

An object of class
`c("thiessen_interval_censored", "thiessen_outcome")`.

## Details

The interval-censoring observation scheme (Sun 2006) for a response
known only to lie between two row-specific bounds, an equal pair being
an exact value and an infinite endpoint one-sided censoring. The
censoring is taken as independent of the response, so the bounds enter
the likelihood only through the interval probability.

The bounds are data, not parameters: the response is a
[`survival::Surv()`](https://rdrr.io/pkg/survival/man/Surv.html) of type
`"interval2"`, `Surv(lower, upper, type = "interval2")`, in which an
`NA` bound is one-sided censoring and an equal pair an exact value; it
selects this family by itself.
[`predict()`](https://rdrr.io/r/stats/predict.html) and the fitted
values are the uncensored f(x), and
[`log_lik()`](https://mc-stan.org/rstantools/reference/log_lik.html) is
the interval likelihood of a `Surv` response.

## Experimental

This family is compiled only into a core built with its `experimental`
Cargo feature. The constructor exists in every build, so a script naming
the family is portable, but a fit or a validated configuration is
rejected with the condition class `thiessen_requires_feature` unless the
package was installed from source with `THIESSEN_EXPERIMENTAL=1` in the
environment;
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
reports the setting of the build in use. An experimental family sits
outside semantic versioning: its configuration and the values it draws
may change in any release. The table of experimental items and their
status is
[`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).

## References

Sun, J. (2006). *The Statistical Analysis of Interval-censored Failure
Time Data*. Springer.
[doi:10.1007/0-387-37119-2](https://doi.org/10.1007/0-387-37119-2)

## See also

[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
interval_censored_outcome()
#> interval_censored_outcome(nu = 6, q = 0.85)
```
