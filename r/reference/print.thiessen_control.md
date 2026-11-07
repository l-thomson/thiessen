# Print a control object

Prints the outcome family and each parameter group, one per line.

## Usage

``` r
# S3 method for class 'thiessen_control'
print(x, ...)
```

## Arguments

- x:

  An object of class `"thiessen_control"`.

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
print(thiessen_control(tessellations = 50))
#> <thiessen_control>
#>   outcome         from the response
#>   mean_params     term_params(tessellations = 50, k = 3, lambda_c = 5)
#>   variance_params none (constant spread)
#>   general_params  general_params(burn_in = 200, draws = 1000, thinning = 1, prior_only = FALSE)
```
