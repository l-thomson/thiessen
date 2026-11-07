# Print a sampler

Prints the number of draws kept so far, or that the sampler is finished.

## Usage

``` r
# S3 method for class 'thiessen_sampler'
print(x, ...)
```

## Arguments

- x:

  An object of class `"thiessen_sampler"`.

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
design <- matrix(seq(0, 1, length.out = 40), ncol = 1)
print(thiessen_sampler(design, design[, 1]^2,
                       thiessen_control(tessellations = 10), seed = 1))
#> <thiessen_sampler> 0 draw(s) kept
```
