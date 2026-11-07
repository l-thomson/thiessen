# The covariate space of the ensembles

Constructs the `geometry` group of
[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md):
the metric of each covariate column, the prior standard deviation of a
centre coordinate and the membership rule.

## Usage

``` r
geometry_params(
  metric = NULL,
  sigma_c = 0.8,
  membership = NULL,
  precision = NULL
)
```

## Arguments

- metric:

  The metric of each covariate column, in column order: a list whose
  entries are `"euclidean"`, `"categorical"`, or
  `list(spherical = list(sphere = i))` for one coordinate of the sphere
  labelled `i`, its latitudes first and its longitude last, in radians;
  or one of the experimental entries below. `NULL`, the default, is
  Euclidean on every column. Entries are matched case-sensitively, so
  `"Euclidean"` is rejected. Non-Euclidean columns are not scaled.

- sigma_c:

  Prior and proposal standard deviation sigma_c of a centre coordinate.
  A Euclidean column is min-max scaled to \[-0.5, 0.5\] over its
  training range inside the sampler and `sigma_c` is on that scale, so 1
  is the full range of a column. Default 0.8.

- membership:

  How an observation belongs to a tessellation's cells: `"hard"`, the
  published rule, or
  [`soft_membership()`](https://l-thomson.github.io/thiessen/r/reference/soft_membership.md).
  `NULL`, the default, is `"hard"`.

- precision:

  The precision matrix of the Mahalanobis metric, a square numeric
  matrix over the columns of the encoded design, required exactly when
  an entry of `metric` is `"mahalanobis"`; it is checked at fit to be
  symmetric and positive definite. `NULL`, the default, is none.
  Experimental, as the metric it serves.

## Value

An object of class `"geometry_params"`.

## Experimental metrics

The entries beyond `"euclidean"`, `"categorical"` and the sphere are
compiled only into a core built with its `experimental` feature (see
[experimental_outcomes](https://l-thomson.github.io/thiessen/r/reference/experimental_outcomes.md)
for the policy) and are named as the core stores them:
`list(minkowski = list(p = 3))` for the Minkowski distance of order p
\>= 1, `"manhattan"` for its order-1 case, `"cosine"` for the cosine
distance, `list(gower = list(kind = "numeric"))` or
`list(gower = list(kind = "categorical"))` for one column of the Gower
distance, and `"mahalanobis"` for the Mahalanobis distance under
`precision`. The Minkowski, Manhattan, cosine and Gower entries take a
`group` label (default 0), `list(cosine = list(group = 1))`, so the
columns sharing a label form one composite distance.

## See also

[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md),
[`soft_membership()`](https://l-thomson.github.io/thiessen/r/reference/soft_membership.md)

## Examples

``` r
geometry_params(metric = list("euclidean", "categorical"))
#> geometry_params(metric = list("euclidean", "categorical"), sigma_c = 0.8)
```
