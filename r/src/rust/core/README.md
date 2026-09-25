# thiessen

[![CI](https://github.com/l-thomson/thiessen/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/l-thomson/thiessen/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/l-thomson/thiessen/branch/dev/graph/badge.svg)](https://codecov.io/gh/l-thomson/thiessen)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/l-thomson/thiessen/badge)](https://scorecard.dev/viewer/?uri=github.com/l-thomson/thiessen)

Bayesian additive Voronoi tessellation regression (AddiVortes): a Rust
core with Python and R packages.

> Status: pre-release. Nothing is published to crates.io, PyPI or
> CRAN yet. The name `thiessen` is a working name; crates.io, docs.rs,
> PyPI, CRAN and DOI badges join at first publication.

## About

`thiessen` implements the AddiVortes method (Stone and Gosling 2025,
JCGS 34(3):859-871, doi:10.1080/10618600.2024.2414104) and its published
variants, Binary AddiVortes (probit classification) and H-AddiVortes
(heteroscedastic variance). All credit for the method belongs to its
authors; the original R package is
[`AddiVortes`](https://github.com/johnpaulgosling/AddiVortes).

The model is `Y = sum_{j=1..m} g(x | T_j, M_j) + e` with
`e ~ N(0, sigma^2)`: a sum of `m` Voronoi tessellations, each
partitioning a random subspace of the covariates, explored by a Gibbs
backfitting sampler with Metropolis-Hastings moves on the tessellation
structure. Where BART-family packages partition with axis-aligned trees,
AddiVortes partitions with Voronoi cells, so a single component captures
interactions between its covariates without deep splits.

The intended users are statisticians and data scientists who would
otherwise reach for BART: the R and Python packages give the familiar
fit and predict surface, and the Rust core gives one audited, fast,
reproducible sampler underneath both.

## Install

Until the first release, each package installs from this repository
and compiles the Rust core, so each needs rustc 1.74 or later with Cargo.

Rust:

    cargo add thiessen --git https://github.com/l-thomson/thiessen

R (the [R package documentation](https://l-thomson.github.io/thiessen/r/)
lists the platform prerequisites):

```r
install.packages("remotes")
remotes::install_github("l-thomson/thiessen", subdir = "r")
```

Python ([documentation](https://l-thomson.github.io/thiessen/python/)):

    pip install "thiessen @ git+https://github.com/l-thomson/thiessen#subdirectory=python"

## Example

```rust
use thiessen::{fit, Config, Data};

fn main() -> thiessen::Result<()> {
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let x = Data::new(xs, n, 1)?;
    let config = Config::new().with_m(10).with_burn_in(20).with_draws(30);
    let model = fit(&config, &x, &y, 42)?;
    println!("{:?}", model.predict(&x)?);
    Ok(())
}
```

## Reproducibility

Same seed, same `thiessen` version and same target triple give identical
draws; the full contract is in the crate-root documentation
([crates/thiessen/src/lib.rs](crates/thiessen/src/lib.rs)).

## Models

`Config::with_outcome` selects the model: `Outcome::gaussian()` for
AddiVortes, `Outcome::probit()` for Binary AddiVortes, and the Gaussian
outcome with a variance ensemble (`Config::with_m_var` above 0) for
H-AddiVortes. Their statements, priors and parameter correspondence with CRAN
AddiVortes and the BART family are in [docs/models.md](docs/models.md).
Components and models beyond the published method are compiled only with
the `experimental` Cargo feature and are outside the semver promise; see
[docs/experimental.md](docs/experimental.md).

## Testing

The suite is layered from unit tests to simulation-based calibration and
comparison against the upstream R package; the layers, their limits and
the numbers are in [docs/testing.md](docs/testing.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Conduct is governed by
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md); security reports follow
[SECURITY.md](SECURITY.md).

## Citation

See [CITATION.cff](CITATION.cff); cite the software and the paper.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT licence ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
licence, shall be dual licensed as above, without any additional terms or
conditions.
