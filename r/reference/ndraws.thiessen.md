# Number of draws and chains of a fitted model

The `posterior` generics `ndraws()` and `nchains()` for a fit.

## Usage

``` r
# S3 method for class 'thiessen'
ndraws(x)

# S3 method for class 'thiessen'
nchains(x)
```

## Arguments

- x:

  An object of class `"thiessen"`.

## Value

The number of kept draws over every chain, and the number of chains the
fit ran.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
fit <- thiessen(x, y, control, seed = 1)
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
posterior::ndraws(fit)
#> [1] 160
```
