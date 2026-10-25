# Print a parameter group

Print a parameter group

## Usage

``` r
# S3 method for class 'term_params'
print(x, ...)

# S3 method for class 'geometry_params'
print(x, ...)

# S3 method for class 'structure_params'
print(x, ...)

# S3 method for class 'cell_params'
print(x, ...)

# S3 method for class 'general_params'
print(x, ...)

# S3 method for class 'xaddivortes_params'
print(x, ...)
```

## Arguments

- x:

  A parameter group from
  [`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md),
  [`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md),
  [`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md),
  [`cell_params()`](https://l-thomson.github.io/thiessen/r/reference/cell_params.md),
  [`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md)
  or
  [`xaddivortes_params()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_params.md).

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
print(term_params(tessellations = 40))
#> term_params(tessellations = 40, k = 3, lambda_c = 5)
```
