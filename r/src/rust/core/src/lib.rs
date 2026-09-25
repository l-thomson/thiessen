//! Bayesian additive Voronoi tessellations (AddiVortes; Stone and Gosling
//! 2025, JCGS 34(3):859-871): the Gaussian, probit and heteroscedastic
//! models, a Gibbs sampler with a step API, `fit` and `predict`. The model
//! is chosen by [`Config::outcome`] and the variance-ensemble count; each
//! model's statement, priors and
//! prediction semantics are in the [`models`] module documentation.
//!
//! # Example
//!
//! ```
//! use thiessen::{fit, Config, Data};
//!
//! # fn main() -> thiessen::Result<()> {
//! let n = 30;
//! let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
//! let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
//! let x = Data::new(xs, n, 1)?;
//!
//! let config = Config::new().with_m(10).with_burn_in(20).with_draws(30);
//! let model = fit(&config, &x, &y, 42)?;
//! let predictions = model.predict(&x)?;
//! assert_eq!(predictions.len(), n);
//! # Ok(())
//! # }
//! ```
//!
//! # Reproducibility
//!
//! The same seed, crate version and target triple give identical draws;
//! draws do not depend on thread count. Transcendental functions go through
//! `libm`, so the reference target `x86_64-unknown-linux-gnu` does not
//! drift with system libc releases. Distribution sampling is pinned to a
//! minor series of `rand_distr` and follows the value-stability policy of
//! `rand` (Rust Rand Book, Reproducibility chapter). Across targets results
//! are statistically equivalent and are compared by posterior summaries,
//! never draw by draw. Patch releases preserve sampled values for a fixed
//! seed; minor releases may change them and the changelog entry says
//! "Sampled values changed" with the reason. Fixed-seed chains are stored
//! under `tests/chains/` and checked bit-exact on the reference target;
//! other targets check posterior summaries against the stored chain within
//! Monte Carlo error. The testing strategy, from unit tests to
//! simulation-based calibration, is `docs/testing.md` in the repository.
//!
//! A [`Fitted`] reloads bit-exact from a serde format that restores each
//! `f64` exactly: a binary format such as MessagePack, or `serde_json` with
//! its `float_roundtrip` feature. Without that feature `serde_json` parses
//! to within one ULP, and a reloaded fit's state and predictions can differ
//! in the last bits.
//!
//! Configurations using options behind the `experimental` feature are
//! deterministic under the same contract and carry fixed-seed snapshots on
//! the reference target, but their sampled values may change in any
//! release, including a patch, with a changelog line. Enabling the feature
//! does not change the draws of a configuration that uses no experimental
//! option.
//!
//! # Stability
//!
//! The stable surface is the method as published: the models and
//! components of Stone and Gosling (2025) and of CRAN AddiVortes. It
//! follows semantic versioning. Everything else the crate adds is
//! experimental: compiled only with the Cargo feature `experimental`
//! (`thiessen = { version = "...", features = ["experimental"] }`), shown
//! on docs.rs with a feature banner, tested to the same standard, and
//! outside the semver promise, whether gated here or badged in a binding.
//! A configuration naming an experimental field or outcome variant fails
//! [`Config::validate`] in a build without the feature with
//! [`Error::RequiresFeature`], naming the item and the feature; the
//! entry points of the experimental models ([`fit_aft`],
//! [`fit_interval_censored`], [`Sampler::aft`],
//! [`Fitted::predict_category_probabilities`] and their siblings) are
//! present in every build and return the same error without it, so a
//! binding compiles against one surface. The table of experimental items
//! and their status is `docs/experimental.md` in the repository.
//!
//! An experimental item is stabilised when it has met its acceptance
//! criteria (the component conformance tests, or the model-grade battery
//! where it changes the posterior), has shipped behind the feature for at
//! least one minor release, has a page under `docs/` stating the model
//! and its calibration evidence, and has a stabilisation note in the
//! changelog. The stabilising pull request removes the gate, marks the
//! item's row in `docs/experimental.md` stabilised with the version, and
//! is a minor version bump. The method's authors are informed of a
//! stabilisation; their reply is not a gate. This rule is stated only
//! here; graduation is a pull request against it.
//!
//! The supported-likelihood boundary (which observation models the crate
//! can carry, and why) and the validation claim over the configuration
//! surface are the sections of those names in `docs/models.md`, with the
//! covered configurations listed in `docs/calibrated.md`; every other
//! combination of the documented options is valid to run and is not
//! separately verified.
//!
//! # Input data
//!
//! `x` is a numeric row-major matrix with at least one column; the response
//! is numeric. Rejected with an [`Error`], never repaired: missing or
//! non-finite values in `x` or `y`, a constant response, a constant column,
//! no columns, fewer than two rows, and a row count that differs between
//! `x` and `y`. Missing-value imputation is the caller's job. Duplicate
//! rows are valid data. A response lying exactly on a least-squares fit
//! of the design is valid; the sigma^2 prior then calibrates from the
//! response standard deviation. More columns than rows fits and returns
//! [`Warning::MoreFeaturesThanObservations`]. At predict the column count
//! must match the fitted model; an empty matrix is valid. Each column has
//! a [`Metric`] ([`GeometryParams::metric`]): Euclidean columns are min-max scaled
//! over their training range; spherical columns are coordinates in
//! radians, a sphere's latitudes then its longitude, and are not scaled;
//! categorical columns are integer level codes, not scaled, any
//! non-integer value rejected with [`Error::InvalidCategoryCode`]. A
//! categorical covariate reaches the crate either as d - 1 Euclidean
//! indicator columns (the encoding the Python and R packages apply by
//! default) or as one categorical column of codes. This section
//! corresponds to rOpenSci general standard G2 and Bayesian standard BS2.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![deny(rustdoc::broken_intra_doc_links)]

#[cfg(test)]
mod broken;
mod cells;
mod config;
mod data;
mod ensemble;
mod error;
mod fitted;
mod geometry;
mod maths;
pub mod models;
mod moves;
#[cfg_attr(not(test), allow(dead_code))]
mod outcome;
mod rng;
mod sampler;
mod scaler;
mod tessellation;
mod threads;

/// The crate version, as the bindings report it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "experimental")]
pub use config::{
    AftParams, Basis, DegreesOfFreedom, Inclusion, IntervalCensoredParams, LaplaceParams,
    Membership, OrdinalParams, StudentTParams, TobitParams,
};
pub use config::{
    CellParams, Config, GaussianParams, GeneralParams, GeometryParams, Outcome, ProbitParams,
    StructureParams, TermParams,
};
pub use data::{Data, Warning};
pub use error::{Error, Result};
pub use fitted::{Fitted, Interval, IntervalKind, Posterior};
#[cfg(feature = "experimental")]
pub use geometry::GowerKind;
pub use geometry::Metric;
pub use models::{
    fit, fit_aft, fit_aft_chains_with_threads, fit_chains, fit_chains_with_threads,
    fit_interval_censored, fit_interval_censored_chains_with_threads, fit_with_progress,
};
pub use rng::chain_seed;
pub use sampler::Sampler;
pub use scaler::Scaler;
pub use tessellation::Tessellation;
