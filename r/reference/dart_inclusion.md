# The DART sparsity prior over the covariates

**\[experimental\]**

## Usage

``` r
dart_inclusion(a = 0.5, b = 1, rho = NULL)
```

## Arguments

- a:

  Beta shape a of the concentration prior. Default 0.5.

- b:

  Beta shape b of the concentration prior. Default 1.

- rho:

  The concentration scale rho. `NULL`, the default, resolves to the
  number of columns at fit.

## Value

An object of class
`c("thiessen_dart", "thiessen_inclusion", "thiessen_option")`, for the
`inclusion` argument of
[`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md).

## Details

The Dirichlet prior of Linero (2018) on the inclusion weights, as the
BART package ships it (`sparse = TRUE` with `a`, `b` and `rho`): the
weights are a sampled vector s ~ Dirichlet(theta / p) and the
concentration theta is drawn on a grid with lambda = theta / (theta +
rho) under a Beta(a, b) prior. The sampled weights and concentration are
carried by
[`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html)
as `inclusion_weight[j]` and `concentration`.

This is the prior of Linero (2018) carried over from trees. It is not
the Dirichlet AddiVortes of Stone and Gosling (in preparation), whose
subset prior draws covariates sequentially without replacement and whose
weights have an exact Gibbs update by latent augmentation.

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

Linero, A. R. (2018). Bayesian regression trees for high-dimensional
prediction and variable selection. *Journal of the American Statistical
Association* 113(522), 626-636.
[doi:10.1080/01621459.2016.1264957](https://doi.org/10.1080/01621459.2016.1264957)

## See also

[`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md),
[`weighted_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/weighted_inclusion.md),
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)

## Examples

``` r
if (FALSE) { # core_experimental()
structure_params(inclusion = dart_inclusion())
}
```
