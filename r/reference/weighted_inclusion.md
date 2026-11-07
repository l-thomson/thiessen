# Fixed inclusion weights over the covariates

**\[experimental\]**

## Usage

``` r
weighted_inclusion(weights)
```

## Arguments

- weights:

  One non-negative finite weight per column of the encoded design, in
  column order, at least one positive.

## Value

An object of class
`c("thiessen_weighted", "thiessen_inclusion", "thiessen_option")`, for
the `inclusion` argument of
[`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md).

## Details

A fixed prior weight per column, the `cov_prior_vec` of bartMachine
(Kapelner and Bleich 2016): the prior on a subset of covariates given
its size is proportional to the product of the member weights, a
proposal picks the incoming covariate with probability proportional to
its weight, and a zero weight excludes the column. Equal weights are the
uniform prior and reproduce its draws exactly.

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

Kapelner, A. and Bleich, J. (2016). bartMachine: machine learning with
Bayesian additive regression trees. *Journal of Statistical Software*
70(4), 1-40.
[doi:10.18637/jss.v070.i04](https://doi.org/10.18637/jss.v070.i04)

## See also

[`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md),
[`dart_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/dart_inclusion.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
structure_params(inclusion = weighted_inclusion(c(2, 1, 1)))
#> structure_params(inclusion = weighted_inclusion(weights = c(2, 1, 1)))
```
