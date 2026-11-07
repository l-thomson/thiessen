# The sweep schedule of a fit

Constructs the `general_params` group of
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md):
burn-in, kept draws, thinning, and whether the likelihood is switched
off.

## Usage

``` r
general_params(burn_in = 200, draws = 1000, thinning = 1, prior_only = FALSE)
```

## Arguments

- burn_in:

  Sweeps discarded before the kept draws. Default 200.

- draws:

  Posterior draws kept. Default 1000.

- thinning:

  Keep every `thinning`-th sweep after burn-in. Default 1.

- prior_only:

  Switch off the likelihood, so the chain draws from the prior and
  [`predict()`](https://rdrr.io/r/stats/predict.html) gives prior
  predictive draws. Default `FALSE`.

## Value

An object of class `"general_params"`.

## See also

[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)

## Examples

``` r
general_params(burn_in = 100, draws = 500)
#> general_params(burn_in = 100, draws = 500, thinning = 1, prior_only = FALSE)
```
