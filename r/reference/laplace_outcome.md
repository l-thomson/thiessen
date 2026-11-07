# The Laplace outcome family

**\[experimental\]**

## Usage

``` r
laplace_outcome(nu = 6, q = 0.85)
```

## Arguments

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

## Value

An object of class `c("thiessen_laplace", "thiessen_outcome")`.

## Details

The Laplace model for a continuous response with outliers: y = f(x) + e
with e ~ Laplace(0, sigma), drawn through the normal-exponential mixture
of Park and Casella (2008). The errors have exponential tails, so a wild
observation is discounted at rate 1/\|r\| against the Student-t model's
1/r^2.

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

Park, T. and Casella, G. (2008). The Bayesian lasso. *Journal of the
American Statistical Association* 103(482), 681-686.
[doi:10.1198/016214508000000337](https://doi.org/10.1198/016214508000000337)

## See also

[`student_t_outcome()`](https://l-thomson.github.io/thiessen/r/reference/student_t_outcome.md),
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
laplace_outcome()
#> laplace_outcome(nu = 6, q = 0.85)
```
