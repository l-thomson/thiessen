# Print an XAddiVortes fit

Print an XAddiVortes fit

## Usage

``` r
# S3 method for class 'xaddivortes'
print(x, ...)
```

## Arguments

- x:

  An object of class `"xaddivortes"`.

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
if (FALSE) { # core_experimental()
set.seed(1)
x <- matrix(runif(400), ncol = 2)
y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
print(xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
                  control = thiessen_control(tessellations = 20),
                  seed = 1))
}
```
