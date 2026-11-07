# The tobit outcome family

**\[experimental\]**

## Usage

``` r
tobit_outcome(lower = NULL, upper = NULL, nu = 6, q = 0.85)
```

## Arguments

- lower:

  The lower censoring limit. `NULL`, the default, is no lower limit.

- upper:

  The upper censoring limit. `NULL`, the default, is no upper limit.

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

## Value

An object of class `c("thiessen_tobit", "thiessen_outcome")`.

## Details

The type-I tobit model (Tobin 1958) for a response censored at known
limits: a response value equal to a limit is read as censored on that
side, and the latent value behind it is drawn by the augmentation of
Chib (1992). At least one limit is required, and a response value beyond
a limit is rejected at fit.

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

Tobin, J. (1958). Estimation of relationships for limited dependent
variables. *Econometrica* 26(1), 24-36.
[doi:10.2307/1907382](https://doi.org/10.2307/1907382)

## See also

[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
tobit_outcome(lower = 0)
#> tobit_outcome(lower = 0, nu = 6, q = 0.85)
```
