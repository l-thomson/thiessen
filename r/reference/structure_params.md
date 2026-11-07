# The covariate-inclusion prior of the ensembles

Constructs the `structure` group of
[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md):
the prior on the number of covariates a tessellation uses and on which
it uses.

## Usage

``` r
structure_params(omega = NULL, inclusion = NULL)
```

## Arguments

- omega:

  Dimension-count prior parameter omega; omega / p is the prior
  probability of including a covariate. `NULL`, the default, resolves to
  min(3, p) at fit. Must satisfy 0 \< omega \<= p.

- inclusion:

  The prior weight of each covariate: `"uniform"`, the published prior,
  [`weighted_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/weighted_inclusion.md)
  or
  [`dart_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/dart_inclusion.md).
  `NULL`, the default, is `"uniform"`.

## Value

An object of class `"structure_params"`.

## See also

[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md),
[`weighted_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/weighted_inclusion.md),
[`dart_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/dart_inclusion.md)

## Examples

``` r
structure_params(omega = 2)
#> structure_params(omega = 2)
```
