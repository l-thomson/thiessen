# The Student-t outcome family

**\[experimental\]**

## Usage

``` r
student_t_outcome(df = 4, nu = 6, q = 0.85)
```

## Arguments

- df:

  The error degrees of freedom: one value, the default being 4, or a
  grid of at least two strictly increasing values drawn over.

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

## Value

An object of class `c("thiessen_student_t", "thiessen_outcome")`.

## Details

The independent Student-t model of Geweke (1993) for a continuous
response with outliers: y = f(x) + e with e ~ t_df(0, sigma^2), drawn
through its scale-mixture representation. The degrees of freedom are
fixed at a value, or drawn each sweep over a grid carrying a uniform
prior; no continuous sampler over them exists, df being weakly
identified.

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

Geweke, J. (1993). Bayesian treatment of the independent Student-t
linear model. *Journal of Applied Econometrics* 8(S1), S19-S40.
[doi:10.1002/jae.3950080504](https://doi.org/10.1002/jae.3950080504)

## See also

[`laplace_outcome()`](https://l-thomson.github.io/thiessen/r/reference/laplace_outcome.md),
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
student_t_outcome(df = c(3, 6, 12))
#> student_t_outcome(df = c(3, 6, 12), nu = 6, q = 0.85)
```
