# Summarise an XAddiVortes fit

Summarise an XAddiVortes fit

## Usage

``` r
# S3 method for class 'xaddivortes'
summary(object, ...)

# S3 method for class 'summary.xaddivortes'
print(x, ...)
```

## Arguments

- object:

  An object of class `"xaddivortes"`.

- ...:

  Ignored.

- x:

  An object of class `"summary.xaddivortes"`.

## Value

An object of class `"summary.xaddivortes"`: a list of the call, the
model, the dimensions of the fit, the number of tessellations, the
`fast` element of the fit, the in-sample root mean squared error and the
column scores, highest first.

`x`, invisibly.

## Examples

``` r
if (FALSE) { # core_experimental()
set.seed(1)
x <- matrix(runif(400), ncol = 2)
y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
summary(xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
                    control = thiessen_control(tessellations = 20),
                    seed = 1))
}
```
