# Version of the core crate

Reports the version of the Rust core compiled into the package.

## Usage

``` r
core_version()
```

## Value

A character string: the semantic version of the Rust core the package
was built with, equal to the `Config/thiessen/core-version` field of the
package `DESCRIPTION`.

## Examples

``` r
core_version()
#> [1] "0.3.0"
```
