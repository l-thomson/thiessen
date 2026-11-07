# Summarise a fitted model

Collects the model, the schedule, the in-sample error, residual and
sigma quantiles and the convergence diagnostics of a fit.

## Usage

``` r
# S3 method for class 'thiessen'
summary(object, ...)

# S3 method for class 'summary.thiessen'
print(x, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- ...:

  Ignored.

- x:

  An object of class `"summary.thiessen"`.

## Value

An object of class `"summary.thiessen"`: a list of the model, the
dimensions of the fit, the sweep schedule, the in-sample root mean
squared error, the quantiles of the residuals, the quantiles of the
posterior draws of sigma where the model has one, and the convergence
diagnostics where two or more chains ran.

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
summary(thiessen(x, y, control, seed = 1))
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 160 draws kept after 20 burn-in, thinning 1
#> 
#> Residuals:
#>         2.5%          25%          50%          75%        97.5% 
#> -0.031431959 -0.016728154 -0.008114297  0.014119159  0.090292511 
#> 
#> sigma:
#>       2.5%        25%        50%        75%      97.5% 
#> 0.04596719 0.05100876 0.05624163 0.06604908 0.08676376 
#> 
#> In-sample RMSE 0.03107
#> 4 chains, largest R-hat 1.777, smallest effective sample size 7
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
```
