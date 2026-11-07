# Outcome families behind the core's experimental feature

The tobit, AFT, interval-censored, ordinal, Student-t and Laplace
families are compiled only into a core built with its `experimental`
feature. This page states the policy they share.

## Arguments

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

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
