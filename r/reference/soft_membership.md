# Soft membership of observations in cells

**\[experimental\]**

## Usage

``` r
soft_membership(rate = 10)
```

## Arguments

- rate:

  Rate of the exponential prior on the bandwidth, on the scaled
  covariate space. Default 10, so the prior mean bandwidth is a tenth of
  a column's range.

## Value

An object of class
`c("thiessen_soft", "thiessen_membership", "thiessen_option")`, for the
`membership` argument of
[`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md).

## Details

Kernel-weighted membership, the softening of the tree split of Linero
and Yang (2018) carried to the Voronoi assignment: observation i takes
weight proportional to exp(-d^2 / (2 tau^2)) in each cell, normalised
over the tessellation's centres, with tau a per-tessellation bandwidth
under an exponential prior and updated by a Metropolis step. Constant
cell basis and constant spread only. The bandwidth draws are carried by
[`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html)
as `bandwidth[j]`.

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

Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles
that adapt to smoothness and sparsity. *Journal of the Royal Statistical
Society: Series B* 80(5), 1087-1110.
[doi:10.1111/rssb.12293](https://doi.org/10.1111/rssb.12293)

## See also

[`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
geometry_params(membership = soft_membership(rate = 10))
#> geometry_params(sigma_c = 0.8, membership = soft_membership(rate = 10))
```
