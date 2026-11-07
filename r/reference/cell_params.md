# The within-cell response surface of the ensembles

Constructs the `cell` group of
[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md):
the value each cell of a tessellation holds.

## Usage

``` r
cell_params(basis = NULL)
```

## Arguments

- basis:

  The value a cell holds: `"constant"`, one value per cell, the
  published basis; or `"linear"`, a value that tilts across the cell,
  mu + beta' (x_A - c) over the active covariates centred at the cell's
  centre, with the slopes under the cell-value prior. The linear basis
  is compiled only into a core built with its `experimental` feature
  (see
  [experimental_outcomes](https://l-thomson.github.io/thiessen/r/reference/experimental_outcomes.md)
  for the policy), needs every column min-max scaled, and applies to the
  mean ensemble only. `NULL`, the default, is `"constant"`.

## Value

An object of class `"cell_params"`.

## See also

[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)

## Examples

``` r
cell_params(basis = "constant")
#> cell_params(basis = "constant")
```
