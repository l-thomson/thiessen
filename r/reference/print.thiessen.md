# Print a fitted model

Prints the call, the model and its dimensions, the sweep schedule, the
in-sample error, the seed and the convergence check.

## Usage

``` r
# S3 method for class 'thiessen'
print(x, ...)
```

## Arguments

- x:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
print(thiessen(x, y, control, seed = 1))
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 160 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.03107, seed 1
#> 4 chains, largest R-hat 1.777, smallest effective sample size 7
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
```
