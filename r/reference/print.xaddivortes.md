# Print an XAddiVortes fit

Prints the call, the model and its dimensions, the fast-fit settings,
the in-sample error and the seed.

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
set.seed(1)
x <- matrix(runif(400), ncol = 2)
y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
print(xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
                  control = thiessen_control(tessellations = 20),
                  seed = 1))
#> XAddiVortes fast fit (experimental)
#> Call: xaddivortes(x = x, y = y, params = xaddivortes_params(rounds = 10),      control = thiessen_control(tessellations = 20), seed = 1)
#> gaussian model, 200 observations, 2 covariates
#> 20 tessellations, 10 rounds (0.00 s)
#> In-sample RMSE 0.0799, seed 1
```
