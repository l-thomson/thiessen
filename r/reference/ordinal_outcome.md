# The ordinal outcome family

**\[experimental\]**

## Usage

``` r
ordinal_outcome(categories = NULL, offset = NULL, cutpoint_sd = 1)
```

## Arguments

- categories:

  Number of ordered categories K, at least 2. `NULL`, the default, is
  the number of levels of the response; a count that differs from it is
  an error.

- offset:

  The offset c. `NULL`, the default, resolves at fit to Phi^-1 of the
  share of rows above the first category.

- cutpoint_sd:

  Standard deviation of the N(0, cutpoint_sd^2) prior on the log-gaps
  between interior cutpoints. Default 1.

## Value

An object of class `c("thiessen_ordinal", "thiessen_outcome")`.

## Details

The ordered probit model of Albert and Chib (1993), P(y \<= k \| x) =
Phi(gamma\_(k+1) - c - f(x)), for an ordered factor response (the
[`MASS::polr`](https://rdrr.io/pkg/MASS/man/polr.html) convention), its
levels the categories 0 to K - 1 in order. The latent variance is fixed
at 1 and the first cutpoint at 0 for identification, and the interior
cutpoints are drawn on the log-gap scale of Albert and Chib (2001). At K
= 2 the model is
[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md).
[`predict()`](https://rdrr.io/r/stats/predict.html) is the expected
category on the code scale, and `predict(type = "probs")` the category
probabilities with the levels as column names.

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

Albert, J. H. and Chib, S. (2001). Sequential ordinal modeling with
applications to survival data. *Biometrics* 57(3), 829-836.
[doi:10.1111/j.0006-341X.2001.00829.x](https://doi.org/10.1111/j.0006-341X.2001.00829.x)

## See also

[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md),
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
ordinal_outcome(categories = 4)
#> ordinal_outcome(categories = 4, cutpoint_sd = 1)
```
