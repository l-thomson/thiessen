# Print a component option

Prints the option and its settings on one line.

## Usage

``` r
# S3 method for class 'thiessen_option'
print(x, ...)
```

## Arguments

- x:

  An option from
  [`soft_membership()`](https://l-thomson.github.io/thiessen/r/reference/soft_membership.md),
  [`weighted_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/weighted_inclusion.md)
  or
  [`dart_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/dart_inclusion.md).

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
print(dart_inclusion())
#> dart_inclusion(a = 0.5, b = 1)
```
