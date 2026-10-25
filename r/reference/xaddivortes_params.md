# XAddiVortes settings

The settings of
[`xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes.md),
and of the fast start of `thiessen(start = "fast")`. The defaults are
the core's, which
[`xaddivortes_defaults()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_defaults.md)
returns.

## Usage

``` r
xaddivortes_params(
  rounds = 40,
  max_columns = 3,
  centres = 64,
  value_scale = 0.5,
  replace_share = 0.3
)
```

## Arguments

- rounds:

  Replacement rounds after the build, a whole number from 0. The fit
  time grows linearly with it; with 0 the build goes straight to the
  final refit.

- max_columns:

  The most columns of a tessellation, a whole number from 1; capped at
  the number of columns.

- centres:

  The most centres of a tessellation, drawn at random training rows
  before empty cells are dropped; fresh tessellations take it or one of
  its halvings down to 4. A whole number from 1 to 65,535.

- value_scale:

  The share of the shrunk cell mean used as the cell value in the build
  and the rounds, in (0, 1\]; the final refit chooses from it up to 1.
  Smaller values fit more slowly and overfit less.

- replace_share:

  The share of tessellations replaced each round, in \[0, 1\]; 0 makes
  the rounds pure refits.

## Value

An object of class `"xaddivortes_params"`.

## See also

[`xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes.md),
[`xaddivortes_defaults()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_defaults.md)

## Examples

``` r
xaddivortes_params(rounds = 20, max_columns = 2)
#> xaddivortes_params(rounds = 20, max_columns = 2, centres = 64, value_scale = 0.5, replace_share = 0.3)
```
