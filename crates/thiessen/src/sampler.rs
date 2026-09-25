//! The Gibbs sampler with Metropolis-Hastings structural moves (Stone and
//! Gosling 2025, Algorithm 1): one sweep per `step`, the caller's loop
//! deciding burn-in and thinning through `keep`, and `finish` producing the
//! fitted model.
//!
//! The sampler composes a mean ensemble (Gaussian cell means, additive)
//! with an outcome model and, for the heteroscedastic model, a variance
//! ensemble (inverse-gamma cell values, multiplicative) whose product at
//! each observation is its variance, updated before the mean sweep (the
//! sweep order of the authors' code, HBART sweeping mean then variance).
//! The scale of the precisions follows the outcome's mode: a global
//! sigma^2 drawn from its inverse-gamma conditional under `Sampled`, the
//! fixed value under `Fixed` (the probit model's unit latent variance),
//! the ensemble's product when one is attached.

use crate::cells::{GaussianCells, InverseGammaCells};
use crate::config::Config;
use crate::config::Outcome as OutcomeConfig;
use crate::data::{self, Data, Warning};
use crate::ensemble::Ensemble;
use crate::error::{Error, Result};
use crate::fitted::{Fitted, Posterior};
use crate::geometry::Geometry;
use crate::maths;
#[cfg(feature = "experimental")]
use crate::models::aft::AftOutcome;
use crate::models::gaussian::GaussianOutcome;
#[cfg(feature = "experimental")]
use crate::models::interval_censored::IntervalCensoredOutcome;
#[cfg(feature = "experimental")]
use crate::models::laplace::LaplaceOutcome;
#[cfg(feature = "experimental")]
use crate::models::ordinal::OrdinalOutcome;
use crate::models::probit::ProbitOutcome;
#[cfg(feature = "experimental")]
use crate::models::student_t::StudentTOutcome;
#[cfg(feature = "experimental")]
use crate::models::tobit::TobitOutcome;
use crate::moves::Prior;
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::{self, Rng};
use crate::scaler::{self, Scaler};
use crate::tessellation::Tessellation;

/// The outcome model in force, one variant per shipped model; the
/// variance ensemble is not a variant here because it composes with any
/// outcome whose scale mode permits it.
#[derive(Debug, Clone)]
enum Outcome {
    Gaussian(GaussianOutcome),
    Probit(ProbitOutcome),
    #[cfg(feature = "experimental")]
    Tobit(TobitOutcome),
    #[cfg(feature = "experimental")]
    Aft(AftOutcome),
    #[cfg(feature = "experimental")]
    IntervalCensored(IntervalCensoredOutcome),
    #[cfg(feature = "experimental")]
    Ordinal(OrdinalOutcome),
    #[cfg(feature = "experimental")]
    StudentT(StudentTOutcome),
    #[cfg(feature = "experimental")]
    Laplace(LaplaceOutcome),
}

/// Dispatch one method over every variant.
macro_rules! each_outcome {
    ($self:expr, $outcome:ident => $body:expr) => {
        match $self {
            Outcome::Gaussian($outcome) => $body,
            Outcome::Probit($outcome) => $body,
            #[cfg(feature = "experimental")]
            Outcome::Tobit($outcome) => $body,
            #[cfg(feature = "experimental")]
            Outcome::Aft($outcome) => $body,
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored($outcome) => $body,
            #[cfg(feature = "experimental")]
            Outcome::Ordinal($outcome) => $body,
            #[cfg(feature = "experimental")]
            Outcome::StudentT($outcome) => $body,
            #[cfg(feature = "experimental")]
            Outcome::Laplace($outcome) => $body,
        }
    };
}

impl Outcome {
    fn as_probit(&self) -> Option<&ProbitOutcome> {
        match self {
            Outcome::Probit(outcome) => Some(outcome),
            _ => None,
        }
    }

    fn as_probit_mut(&mut self) -> Option<&mut ProbitOutcome> {
        match self {
            Outcome::Probit(outcome) => Some(outcome),
            _ => None,
        }
    }

    /// The response the in-sample fit is measured against, where the
    /// working response has been replaced by latents; `None` where the
    /// working response is the observed one.
    fn observed_response(&self) -> Option<&[f64]> {
        match self {
            Outcome::Probit(outcome) => Some(outcome.labels()),
            #[cfg(feature = "experimental")]
            Outcome::Tobit(outcome) => Some(outcome.observed()),
            #[cfg(feature = "experimental")]
            Outcome::Aft(outcome) => Some(outcome.observed()),
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(outcome) => Some(outcome.observed()),
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(outcome) => Some(outcome.labels()),
            Outcome::Gaussian(_) => None,
            #[cfg(feature = "experimental")]
            Outcome::StudentT(_) => None,
            #[cfg(feature = "experimental")]
            Outcome::Laplace(_) => None,
        }
    }

    #[cfg(feature = "experimental")]
    fn as_ordinal(&self) -> Option<&OrdinalOutcome> {
        match self {
            Outcome::Ordinal(outcome) => Some(outcome),
            _ => None,
        }
    }

    #[cfg(feature = "experimental")]
    fn as_ordinal_mut(&mut self) -> Option<&mut OrdinalOutcome> {
        match self {
            Outcome::Ordinal(outcome) => Some(outcome),
            _ => None,
        }
    }

    #[cfg(feature = "experimental")]
    fn as_student_t(&self) -> Option<&StudentTOutcome> {
        match self {
            Outcome::StudentT(outcome) => Some(outcome),
            _ => None,
        }
    }

    #[cfg(feature = "experimental")]
    fn as_aft_mut(&mut self) -> Option<&mut AftOutcome> {
        match self {
            Outcome::Aft(outcome) => Some(outcome),
            _ => None,
        }
    }

    #[cfg(feature = "experimental")]
    fn as_interval_censored_mut(&mut self) -> Option<&mut IntervalCensoredOutcome> {
        match self {
            Outcome::IntervalCensored(outcome) => Some(outcome),
            _ => None,
        }
    }
}

impl OutcomeModel for Outcome {
    fn required_data(&self) -> RequiredData {
        each_outcome!(self, outcome => outcome.required_data())
    }

    fn init(&mut self, y: &[f64]) {
        each_outcome!(self, outcome => outcome.init(y))
    }

    fn draw_extra(&mut self, y: &[f64], total: &[f64], precision: &[f64], rng: &mut Rng) {
        each_outcome!(self, outcome => outcome.draw_extra(y, total, precision, rng))
    }

    fn working_response(&mut self, total: &[f64], precision: &[f64], y: &mut [f64], rng: &mut Rng) {
        each_outcome!(self, outcome => outcome.working_response(total, precision, y, rng))
    }

    fn weights(&self) -> Option<&[f64]> {
        each_outcome!(self, outcome => outcome.weights())
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        each_outcome!(self, outcome => outcome.sigma2_mode())
    }

    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64> {
        each_outcome!(self, outcome => outcome.predictive_quantile(mean, sd, p))
    }
}

/// The sampler state for one chain. Construct with [`Sampler::new`], advance
/// with [`step`](Sampler::step), record draws with [`keep`](Sampler::keep),
/// and close with [`finish`](Sampler::finish).
///
/// The response is on the caller's scale through the affine map frozen at
/// construction; [`set_response`](Sampler::set_response) and
/// [`fitted_values`](Sampler::fitted_values) speak that scale. Under the
/// probit model the response is the labels and the fitted values are the
/// latent mean c + f(x). The sampler owns its RNG, seeded at construction;
/// a caller's loop consumes none of it.
#[derive(Debug, Clone)]
pub struct Sampler {
    rng: Rng,
    config: Config,
    scaler: Scaler,
    warnings: Vec<Warning>,
    /// Scaled design.
    x: Data,
    /// Scaled working response: the scaled y, or under the probit model the
    /// latent z - c.
    y: Vec<f64>,
    mean: Ensemble<GaussianCells>,
    outcome: Outcome,
    /// Global sigma^2, scaled space: drawn under the Sampled mode when no
    /// variance ensemble is attached; otherwise the initial value, unused.
    sigma_sq: f64,
    /// The variance ensemble carrying s^2(x) in place of the global
    /// sigma^2 (H-AddiVortes); `None` for a constant spread.
    variance: Option<Box<Ensemble<InverseGammaCells>>>,
    /// Per-observation precision of the mean ensemble's observations.
    precision: Vec<f64>,
    /// Calibrated scale of the prior sigma^2 ~ nu lambda / chi^2_nu.
    lambda: f64,
    kept: Posterior,
    /// The mean function at every training row summed over the kept
    /// draws, from the cached cell assignments at each `keep`; `None` once
    /// a kept draw had a soft tessellation, whose value is not a cell's.
    fit_sums: Option<FitSums>,
    /// The DART inclusion state, when the structure prior samples its
    /// weights.
    #[cfg(feature = "experimental")]
    dart: Option<Dart>,
    /// Test-only acceptance defect; `Breakage::None` on every constructed
    /// sampler.
    #[cfg(test)]
    pub(crate) breakage: crate::broken::Breakage,
}

/// Running sums over the kept draws of the per-draw quantity two
/// consumers average: `scaled` is what [`Sampler::finish`] averages for
/// its in-sample RMSE (f, or the probit or ordinal response of f + c),
/// and `response` is what [`Fitted::predict`] returns per draw at the
/// training rows (the same on the caller's scale).
#[derive(Debug, Clone)]
struct FitSums {
    scaled: Vec<f64>,
    response: Vec<f64>,
}

/// The sampled state and fixed grid of the DART inclusion prior: the
/// weight vector s, the concentration theta on the BART grid of
/// lambda = theta / (theta + rho), and the grid's log prior weights.
#[cfg(feature = "experimental")]
#[derive(Debug, Clone)]
struct Dart {
    /// theta at each grid point.
    grid_theta: Vec<f64>,
    /// ln Beta(a, b) kernel at each grid point.
    log_grid_prior: Vec<f64>,
    theta: f64,
    s: Vec<f64>,
}

#[cfg(feature = "experimental")]
impl Dart {
    const GRID: usize = 1000;

    /// The grid for (a, b, rho), theta drawn from its grid prior and s
    /// started uniform at 1 / p, as the BART package starts it. A prior
    /// draw of s is spiky when theta is small, and the structural moves
    /// pick columns in proportion to s, so a column starting near zero
    /// is never proposed and the chain sits there. Draw order: one
    /// uniform for theta.
    fn draw(a: f64, b: f64, rho: f64, p: usize, rng: &mut rng::Rng) -> Self {
        let k = Self::GRID;
        let mut grid_theta = Vec::with_capacity(k);
        let mut log_grid_prior = Vec::with_capacity(k);
        for i in 1..=k {
            let lambda = i as f64 / (k + 1) as f64;
            grid_theta.push(lambda * rho / (1.0 - lambda));
            log_grid_prior
                .push((a - 1.0) * maths::ln(lambda) + (b - 1.0) * maths::ln(1.0 - lambda));
        }
        let mut dart = Self {
            grid_theta,
            log_grid_prior,
            theta: 0.0,
            s: Vec::new(),
        };
        let index = rng::draw_discrete(&dart.log_grid_prior, rng);
        dart.theta = dart.grid_theta[index];
        dart.s = vec![1.0 / p as f64; p];
        dart
    }
}

/// A Dirichlet draw by normalised gammas, one gamma per coordinate.
/// A tiny shape makes the gamma underflow to zero; the draw is floored
/// at the smallest positive normal so the logs downstream stay finite.
#[cfg(feature = "experimental")]
fn draw_dirichlet(shapes: &[f64], rng: &mut rng::Rng) -> Vec<f64> {
    let draws: Vec<f64> = shapes
        .iter()
        .map(|&shape| rng::gamma(shape, 1.0, rng).max(f64::MIN_POSITIVE))
        .collect();
    let total: f64 = draws.iter().sum();
    draws.iter().map(|&g| g / total).collect()
}

/// The extra data channel of a gated outcome, alongside the response.
#[cfg(feature = "experimental")]
#[derive(Clone, Copy)]
enum Extra<'a> {
    None,
    Events(&'a [bool]),
    Bounds { lower: &'a [f64], upper: &'a [f64] },
}

/// The configuration names the interval-censored outcome, which the
/// bound entry points require.
#[cfg(feature = "experimental")]
fn require_interval_censored(config: &Config) -> Result<()> {
    if matches!(config.outcome, OutcomeConfig::IntervalCensored(_)) {
        Ok(())
    } else {
        Err(crate::error::invalid(
            "outcome",
            "the bound entry points need the interval_censored outcome",
        ))
    }
}

/// Validated working response of the bound pairs: one pair per row, no
/// NaN endpoint, lower <= upper, at most one infinite endpoint and a
/// finite value where the pair is equal. The working value completes
/// each interval: the exact value, the midpoint where both bounds are
/// finite, the finite endpoint of a one-sided interval.
#[cfg(feature = "experimental")]
fn interval_working(lower: &[f64], upper: &[f64]) -> Result<Vec<f64>> {
    if lower.len() != upper.len() {
        return Err(Error::BoundCountMismatch {
            lower: lower.len(),
            upper: upper.len(),
        });
    }
    lower
        .iter()
        .zip(upper)
        .enumerate()
        .map(|(row, (&lo, &hi))| {
            if lo.is_nan()
                || hi.is_nan()
                || lo > hi
                || (lo == hi && !lo.is_finite())
                || (lo == f64::NEG_INFINITY && hi == f64::INFINITY)
            {
                Err(Error::InvalidInterval { row })
            } else if lo == f64::NEG_INFINITY {
                Ok(hi)
            } else if hi == f64::INFINITY {
                Ok(lo)
            } else {
                Ok(0.5 * (lo + hi))
            }
        })
        .collect()
}

/// Validated ordinal codes: integers in 0..categories.
#[cfg(feature = "experimental")]
fn validate_ordinal_labels(y: &[f64], categories: usize) -> Result<()> {
    if let Some(row) = y
        .iter()
        .position(|&v| !(v.fract() == 0.0 && v >= 0.0 && v < categories as f64))
    {
        return Err(Error::InvalidOrdinalLabel { row, categories });
    }
    Ok(())
}

/// The configuration names the AFT outcome, which the survival entry
/// points require.
#[cfg(feature = "experimental")]
fn require_aft(config: &Config) -> Result<()> {
    if matches!(config.outcome, OutcomeConfig::Aft(_)) {
        Ok(())
    } else {
        Err(crate::error::invalid(
            "outcome",
            "the survival entry points need the aft outcome",
        ))
    }
}

/// Validated log times: one event flag per time, every time finite and
/// positive.
#[cfg(feature = "experimental")]
fn log_times(times: &[f64], events: &[bool]) -> Result<Vec<f64>> {
    if events.len() != times.len() {
        return Err(Error::EventCountMismatch {
            events: events.len(),
            times: times.len(),
        });
    }
    if let Some(row) = times.iter().position(|&t| !(t.is_finite() && t > 0.0)) {
        return Err(Error::InvalidSurvivalTime { row });
    }
    Ok(times.iter().map(|&t| maths::ln(t)).collect())
}

impl Sampler {
    /// A sampler over raw data, its RNG seeded from `seed`.
    ///
    /// # Errors
    ///
    /// [`Config::validate`], then the data checks: row counts, at least two
    /// observations, finite values, non-constant response and columns,
    /// omega <= p, `metric` naming every column with each sphere's
    /// longitude last, `InvalidCategoryCode` for a non-integer value in a
    /// categorical column, and under the probit model `InvalidLabel` for
    /// a response value outside {0, 1}.
    pub fn new(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Self> {
        Self::build(
            config,
            x,
            y,
            seed,
            None,
            #[cfg(feature = "experimental")]
            Extra::None,
        )
    }

    /// A sampler for the AFT model over raw data: `times` holds n
    /// positive event or censoring times and `events` one flag per row
    /// (true is an event, false right-censoring); the response the chain
    /// runs on is ln t. Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// [`Sampler::new`]; `InvalidHyperparameter` for an outcome other
    /// than `Outcome::Aft`,
    /// `EventCountMismatch`, `InvalidSurvivalTime`.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn aft(
        config: &Config,
        x: &Data,
        times: &[f64],
        events: &[bool],
        seed: u64,
    ) -> Result<Self> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::AFT.requires_feature());
        #[cfg(feature = "experimental")]
        {
            require_aft(config)?;
            let y = log_times(times, events)?;
            Self::build(config, x, &y, seed, None, Extra::Events(events))
        }
    }

    /// A sampler for the interval-censored model over raw data: `lower`
    /// and `upper` hold one pair of bounds per row on the response
    /// scale (an equal pair is an exact value, an infinite endpoint
    /// one-sided censoring); the response the chain runs on is the
    /// working completion of each interval. Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// [`Sampler::new`]; `InvalidHyperparameter` for an outcome other
    /// than `Outcome::IntervalCensored`,
    /// `BoundCountMismatch`, `InvalidInterval`.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn interval_censored(
        config: &Config,
        x: &Data,
        lower: &[f64],
        upper: &[f64],
        seed: u64,
    ) -> Result<Self> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::INTERVAL_CENSORED.requires_feature());
        #[cfg(feature = "experimental")]
        {
            require_interval_censored(config)?;
            let y = interval_working(lower, upper)?;
            Self::build(config, x, &y, seed, None, Extra::Bounds { lower, upper })
        }
    }

    /// The interval-censored model under the pinned prior, as
    /// [`pinned_prior`](Sampler::pinned_prior): the bounds are taken as
    /// already scaled. Experimental (`docs/experimental.md`).
    ///
    /// # Errors
    ///
    /// [`Sampler::interval_censored`], `InvalidHyperparameter` for a
    /// non-positive `lambda`.
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    pub fn pinned_prior_interval_censored(
        config: &Config,
        x: &Data,
        lower: &[f64],
        upper: &[f64],
        lambda: f64,
        seed: u64,
    ) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(crate::error::invalid(
                "lambda",
                format!("must be finite and positive, got {lambda}"),
            ));
        }
        require_interval_censored(config)?;
        let y = interval_working(lower, upper)?;
        Self::build(
            config,
            x,
            &y,
            seed,
            Some(lambda),
            Extra::Bounds { lower, upper },
        )
    }

    /// The AFT model under the pinned prior, as
    /// [`pinned_prior`](Sampler::pinned_prior): `times` are taken as
    /// exp of the already scaled log-time response. Experimental
    /// (`docs/experimental.md`).
    ///
    /// # Errors
    ///
    /// [`Sampler::aft`], `InvalidHyperparameter` for a non-positive
    /// `lambda`.
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    pub fn pinned_prior_aft(
        config: &Config,
        x: &Data,
        times: &[f64],
        events: &[bool],
        lambda: f64,
        seed: u64,
    ) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(crate::error::invalid(
                "lambda",
                format!("must be finite and positive, got {lambda}"),
            ));
        }
        require_aft(config)?;
        let y = log_times(times, events)?;
        Self::build(config, x, &y, seed, Some(lambda), Extra::Events(events))
    }

    /// A sampler whose prior is fixed by the caller: `x` and `y` are taken
    /// as already scaled, the affine maps are identity, and `lambda` is
    /// given rather than calibrated, so the prior does not depend on the
    /// data. Calibration tests (SBC, Geweke test) require this
    /// constructor; under [`Sampler::new`] the y-scaling and the lambda
    /// calibration make the prior a function of the data, which
    /// invalidates a joint-distribution test. Under the probit model
    /// `lambda` is not used; `offset` must then be set on the
    /// configuration, since Phi^-1(ybar) is a function of the data.
    ///
    /// # Errors
    ///
    /// [`Sampler::new`], `InvalidHyperparameter` for a non-positive
    /// `lambda`, and for a probit configuration without `offset`.
    pub fn pinned_prior(
        config: &Config,
        x: &Data,
        y: &[f64],
        lambda: f64,
        seed: u64,
    ) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(crate::error::invalid(
                "lambda",
                format!("must be finite and positive, got {lambda}"),
            ));
        }
        #[cfg(feature = "experimental")]
        let needs_offset = matches!(
            config.outcome,
            OutcomeConfig::Probit(_) | OutcomeConfig::Ordinal(_)
        );
        #[cfg(not(feature = "experimental"))]
        let needs_offset = matches!(config.outcome, OutcomeConfig::Probit(_));
        if needs_offset && config.offset().is_none() {
            return Err(crate::error::invalid(
                "offset",
                "must be set under the pinned prior of a model with an offset",
            ));
        }
        Self::build(
            config,
            x,
            y,
            seed,
            Some(lambda),
            #[cfg(feature = "experimental")]
            Extra::None,
        )
    }

    fn build(
        config: &Config,
        x: &Data,
        y: &[f64],
        seed: u64,
        pinned_lambda: Option<f64>,
        #[cfg(feature = "experimental")] extra: Extra<'_>,
    ) -> Result<Self> {
        config.validate()?;
        let p = x.n_cols();
        let omega = config.omega_for(p);
        data::validate_fit(x, y, omega)?;
        let geometry = Geometry::fit(&config.mean_params.geometry.metric, x)?;
        #[cfg(feature = "experimental")]
        let geometry = geometry.with_precision(config.mean_params.geometry.precision.as_deref())?;
        #[cfg(feature = "experimental")]
        let linear = matches!(config.mean_params.cell.basis, crate::config::Basis::Linear);
        #[cfg(not(feature = "experimental"))]
        let linear = false;
        #[cfg(feature = "experimental")]
        let soft_rate = match config.mean_params.geometry.membership {
            crate::config::Membership::Soft { rate } => Some(rate),
            crate::config::Membership::Hard => None,
        };
        #[cfg(not(feature = "experimental"))]
        let soft_rate = None;
        #[cfg(feature = "experimental")]
        if linear {
            if let Some(col) = (0..p).find(|&col| !geometry.scaled(col)) {
                return Err(crate::error::invalid(
                    "mean_params.cell.basis",
                    format!("the linear basis needs min-max scaled columns; column {col} is not"),
                ));
            }
        }
        let laws = geometry.laws(x, config.mean_params.geometry.sigma_c)?;
        if config.outcome.required_data() == RequiredData::Binary {
            if let Some(row) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
                return Err(Error::InvalidLabel { row });
            }
        }
        #[cfg(feature = "experimental")]
        if let OutcomeConfig::Ordinal(params) = &config.outcome {
            validate_ordinal_labels(y, params.categories)?;
        }
        #[cfg(feature = "experimental")]
        if matches!(config.outcome, OutcomeConfig::Aft(_)) && !matches!(extra, Extra::Events(_)) {
            return Err(crate::error::invalid(
                "outcome",
                "the aft outcome takes times and an event indicator; fit through \
                 fit_aft or Sampler::aft",
            ));
        }
        #[cfg(feature = "experimental")]
        if matches!(config.outcome, OutcomeConfig::IntervalCensored(_))
            && !matches!(extra, Extra::Bounds { .. })
        {
            return Err(crate::error::invalid(
                "outcome",
                "the interval_censored outcome takes bound vectors; fit through \
                 fit_interval_censored or Sampler::interval_censored",
            ));
        }
        #[cfg(feature = "experimental")]
        if let OutcomeConfig::Tobit(params) = &config.outcome {
            let beyond = |v: f64| {
                params.lower.is_some_and(|limit| v < limit)
                    || params.upper.is_some_and(|limit| v > limit)
            };
            if let Some(row) = y.iter().position(|&v| beyond(v)) {
                return Err(Error::ResponseBeyondLimit { row });
            }
        }
        let mut config = config.clone();
        config.resolve(omega);
        let warnings = data::fit_warnings(x);
        let n = x.n_rows();

        // Response scaling, lambda and the initial sigma^2 at the model
        // boundary: the probit and ordinal models keep the response
        // unscaled and their initial scale is the mode's fixed value.
        let fixed_scale = match config.outcome.sigma2_mode() {
            Sigma2Mode::Fixed(value) => value,
            Sigma2Mode::Sampled | Sigma2Mode::Absent => 1.0,
        };
        #[cfg(feature = "experimental")]
        let latent_label = |outcome: &OutcomeConfig| {
            matches!(
                outcome,
                OutcomeConfig::Probit(_) | OutcomeConfig::Ordinal(_)
            )
        };
        #[cfg(not(feature = "experimental"))]
        let latent_label = |outcome: &OutcomeConfig| matches!(outcome, OutcomeConfig::Probit(_));
        let (nu, q) = config.sigma2_prior();
        let (scaler, x_scaled, y_scaled, lambda, sigma_sq) = match (&config.outcome, pinned_lambda)
        {
            (outcome, Some(lambda)) if latent_label(outcome) => (
                Scaler::identity(p),
                x.clone(),
                y.to_vec(),
                lambda,
                fixed_scale,
            ),
            (outcome, None) if latent_label(outcome) => {
                let (scaler, x_scaled) = Scaler::fit_x(x, &geometry);
                (scaler, x_scaled, y.to_vec(), 1.0, fixed_scale)
            }
            // The censored outcomes compare or complete response values
            // against their bounds after the map, so the response
            // crosses by the same map as the bounds even where it is
            // the identity in exact arithmetic.
            #[cfg(feature = "experimental")]
            (OutcomeConfig::Tobit(_) | OutcomeConfig::IntervalCensored(_), Some(lambda)) => {
                let scaler = Scaler::identity(p);
                let y_scaled = y.iter().map(|&v| scaler.scale_y(v)).collect();
                (scaler, x.clone(), y_scaled, lambda, lambda)
            }
            (_, Some(lambda)) => (Scaler::identity(p), x.clone(), y.to_vec(), lambda, lambda),
            (_, None) => {
                let (scaler, x_scaled, y_scaled) = Scaler::fit(x, y, &geometry);
                let sigma_hat = scaler::sigma_hat(&x_scaled, &y_scaled);
                let lambda = scaler::calibrate_lambda(nu, q, sigma_hat);
                (scaler, x_scaled, y_scaled, lambda, sigma_hat * sigma_hat)
            }
        };
        let m = config.mean_tessellations();
        let half_width = match &config.outcome {
            OutcomeConfig::Probit(_) => ProbitOutcome::CELL_PRIOR_HALF_WIDTH,
            OutcomeConfig::Gaussian(_) => GaussianOutcome::CELL_PRIOR_HALF_WIDTH,
            #[cfg(not(feature = "experimental"))]
            OutcomeConfig::Gated(gated) => gated.unreachable(),
            #[cfg(feature = "experimental")]
            OutcomeConfig::Tobit(_) => TobitOutcome::CELL_PRIOR_HALF_WIDTH,
            #[cfg(feature = "experimental")]
            OutcomeConfig::Aft(_) => AftOutcome::CELL_PRIOR_HALF_WIDTH,
            #[cfg(feature = "experimental")]
            OutcomeConfig::IntervalCensored(_) => IntervalCensoredOutcome::CELL_PRIOR_HALF_WIDTH,
            #[cfg(feature = "experimental")]
            OutcomeConfig::Ordinal(_) => OrdinalOutcome::CELL_PRIOR_HALF_WIDTH,
            #[cfg(feature = "experimental")]
            OutcomeConfig::StudentT(_) => StudentTOutcome::CELL_PRIOR_HALF_WIDTH,
            #[cfg(feature = "experimental")]
            OutcomeConfig::Laplace(_) => LaplaceOutcome::CELL_PRIOR_HALF_WIDTH,
        };
        let sigma_mu_sq = scaler::sigma_mu_sq(half_width, config.mean_params.k, m);
        // Both slots declare identical geometry and structure, so the
        // variance prior differs from the mean prior only where its group
        // does.
        let mut rng = rng::chain_rng(seed);
        #[cfg(feature = "experimental")]
        let dart = match &config.mean_params.structure.inclusion {
            crate::config::Inclusion::Dart { a, b, rho } => {
                Some(Dart::draw(*a, *b, rho.unwrap_or(p as f64), p, &mut rng))
            }
            _ => None,
        };
        #[cfg(feature = "experimental")]
        let inclusion_weights = match &config.mean_params.structure.inclusion {
            crate::config::Inclusion::Uniform => None,
            crate::config::Inclusion::Weighted { weights } => {
                if weights.len() != p {
                    return Err(crate::error::invalid(
                        "mean_params.structure.inclusion",
                        format!(
                            "must weight every column: {} entries for p = {p} columns",
                            weights.len()
                        ),
                    ));
                }
                crate::moves::InclusionWeights::new(weights)
            }
            crate::config::Inclusion::Dart { .. } => Some(crate::moves::InclusionWeights::sampled(
                dart.as_ref().expect("dart state").s.clone(),
            )),
        };
        let prior = Prior {
            p,
            omega,
            lambda_c: config.mean_params.lambda_c,
            geometry,
            laws,
            #[cfg(feature = "experimental")]
            weights: inclusion_weights,
        };
        let mean_y = y_scaled.iter().sum::<f64>() / n as f64;

        // Initial state: m single-cell tessellations on one covariate each,
        // every cell mean ybar / m so the ensemble fit starts at ybar;
        // under the probit and ordinal models f starts at 0 and the
        // offset carries the mean.
        let (cell_value, total) = if latent_label(&config.outcome) {
            (0.0, 0.0)
        } else {
            (mean_y / m as f64, mean_y)
        };
        let mean = Ensemble::new(
            GaussianCells {
                sigma_mu_sq,
                linear,
            },
            prior.clone(),
            soft_rate,
            &x_scaled,
            m,
            cell_value,
            total,
            &mut rng,
        );
        let outcome = match &config.outcome {
            OutcomeConfig::Gaussian(_) => {
                let mut outcome = GaussianOutcome;
                outcome.init(&y_scaled);
                Outcome::Gaussian(outcome)
            }
            #[cfg(not(feature = "experimental"))]
            OutcomeConfig::Gated(gated) => gated.unreachable(),
            OutcomeConfig::Probit(_) => {
                let mut outcome = ProbitOutcome::new(config.offset(), mean_y);
                config.set_offset(outcome.offset());
                outcome.init(&y_scaled);
                Outcome::Probit(outcome)
            }
            // The offset and the initial cutpoints resolve from the
            // marginal category shares inside init.
            #[cfg(feature = "experimental")]
            OutcomeConfig::Ordinal(params) => {
                let mut outcome =
                    OrdinalOutcome::new(params.categories, config.offset(), params.cutpoint_sd);
                outcome.init(&y_scaled);
                config.set_offset(outcome.offset());
                Outcome::Ordinal(outcome)
            }
            // The limits cross to the scaled response space by the same
            // frozen affine map as the response, so a value at a limit
            // stays exactly at it.
            #[cfg(feature = "experimental")]
            OutcomeConfig::Tobit(params) => {
                let mut outcome = TobitOutcome::new(
                    params.lower.map(|v| scaler.scale_y(v)),
                    params.upper.map(|v| scaler.scale_y(v)),
                );
                outcome.init(&y_scaled);
                Outcome::Tobit(outcome)
            }
            // A censored row's truncation point is its own scaled log
            // time, so nothing beyond the response crosses the map.
            #[cfg(feature = "experimental")]
            OutcomeConfig::Aft(_) => {
                let Extra::Events(events) = extra else {
                    unreachable!("validated above");
                };
                let mut outcome = AftOutcome::new(events.to_vec());
                outcome.init(&y_scaled);
                Outcome::Aft(outcome)
            }
            // Weights are dimensionless, so nothing crosses the map.
            #[cfg(feature = "experimental")]
            OutcomeConfig::StudentT(params) => {
                let mut outcome =
                    StudentTOutcome::new(params.df.initial(), params.df.grid().to_vec());
                outcome.init(&y_scaled);
                Outcome::StudentT(outcome)
            }
            #[cfg(feature = "experimental")]
            OutcomeConfig::Laplace(_) => {
                let mut outcome = LaplaceOutcome::default();
                outcome.init(&y_scaled);
                Outcome::Laplace(outcome)
            }
            // The bounds cross to the scaled response space by the same
            // frozen affine map as the working response; an infinite
            // endpoint stays infinite.
            #[cfg(feature = "experimental")]
            OutcomeConfig::IntervalCensored(_) => {
                let Extra::Bounds { lower, upper } = extra else {
                    unreachable!("validated above");
                };
                let map = |&v: &f64| if v.is_finite() { scaler.scale_y(v) } else { v };
                let mut outcome = IntervalCensoredOutcome::new(
                    lower.iter().map(map).collect(),
                    upper.iter().map(map).collect(),
                );
                outcome.init(&y_scaled);
                Outcome::IntervalCensored(outcome)
            }
        };
        // H-AddiVortes: the variance ensemble carries the scale in place
        // of the global sigma^2; validation has already derived its
        // permission from the outcome's mode.
        let m_var = config.variance_tessellations();
        let variance = (m_var > 0).then(|| {
            let (nu_prime, lambda_prime) = scaler::variance_cell_prior(nu, lambda, m_var);
            let variance_prior = Prior {
                lambda_c: config.variance_params.lambda_c,
                ..prior
            };
            // Every variance cell starts at sigma^2 ^ (1 / m') so the
            // product starts at the initial sigma^2.
            Box::new(Ensemble::new(
                InverseGammaCells {
                    nu: nu_prime,
                    lambda: lambda_prime,
                    prior_only: config.general_params.prior_only,
                },
                variance_prior,
                None,
                &x_scaled,
                m_var,
                libm::pow(sigma_sq, 1.0 / m_var as f64),
                sigma_sq,
                &mut rng,
            ))
        });
        // Zero precision removes the likelihood from every conditional:
        // the integrated-likelihood terms of the acceptance ratio vanish
        // and the cell means are drawn from N(0, sigma_mu^2).
        let precision = if config.general_params.prior_only {
            vec![0.0; n]
        } else {
            vec![1.0 / sigma_sq; n]
        };
        // Under a censored outcome the vector starts at the working
        // response, censored rows at a value inside their bounds, a
        // valid latent initialisation; under the probit and ordinal
        // models at 0, refreshed before anything conditions on it.
        let y = if latent_label(&config.outcome) {
            vec![0.0; n]
        } else {
            y_scaled
        };

        Ok(Self {
            rng,
            config,
            scaler,
            warnings,
            x: x_scaled,
            y,
            mean,
            outcome,
            sigma_sq,
            variance,
            precision,
            lambda,
            kept: Posterior::empty(),
            fit_sums: Some(FitSums {
                scaled: vec![0.0; n],
                response: vec![0.0; n],
            }),
            #[cfg(feature = "experimental")]
            dart,
            #[cfg(test)]
            breakage: crate::broken::Breakage::None,
        })
    }

    /// One sweep: the noise update (sigma^2 | rest; the latent response |
    /// rest; or the variance ensemble | rest), then for each mean
    /// tessellation in turn a structural move, accept or reject, and the
    /// cell means | rest.
    pub fn step(&mut self) {
        self.update_noise(true);
        self.mean.sweep(
            &self.x,
            &self.y,
            &self.precision,
            &mut self.rng,
            #[cfg(test)]
            self.breakage,
        );
        #[cfg(feature = "experimental")]
        self.update_inclusion();
    }

    /// The DART updates: s | dims, theta by a Metropolis step whose
    /// Dirichlet(theta / p + counts) proposal leaves the subset-prior
    /// normalisers e_d in the ratio, then theta | s exactly on its grid.
    /// Draw order: p gammas, the acceptance uniform, the grid uniform.
    #[cfg(feature = "experimental")]
    fn update_inclusion(&mut self) {
        let Some(dart) = &mut self.dart else {
            return;
        };
        let p = dart.s.len();
        let mut counts = vec![0.0_f64; p];
        let mut dim_counts: Vec<usize> = Vec::new();
        let mut tally = |tessellations: &[crate::tessellation::Tessellation]| {
            for t in tessellations {
                dim_counts.push(t.n_dims());
                for &dim in t.dims() {
                    counts[dim] += 1.0;
                }
            }
        };
        tally(self.mean.tessellations());
        if let Some(variance) = &self.variance {
            tally(variance.tessellations());
        }
        let shapes: Vec<f64> = counts.iter().map(|&m| dart.theta / p as f64 + m).collect();
        let proposed = draw_dirichlet(&shapes, &mut self.rng);
        let current = crate::moves::InclusionWeights::sampled(dart.s.clone());
        let candidate = crate::moves::InclusionWeights::sampled(proposed.clone());
        let log_alpha: f64 = dim_counts
            .iter()
            .map(|&d| current.log_e(d) - candidate.log_e(d))
            .sum();
        #[cfg(test)]
        let log_alpha = if self.breakage == crate::broken::Breakage::DroppedSubsetNormaliser {
            0.0
        } else {
            log_alpha
        };
        let accepted = maths::ln(rng::uniform(&mut self.rng)) < log_alpha;
        if accepted {
            dart.s = proposed;
        }
        let weights = if accepted { candidate } else { current };
        self.mean.set_inclusion_weights(weights.clone());
        if let Some(variance) = &mut self.variance {
            variance.set_inclusion_weights(weights);
        }
        let sum_ln_s: f64 = dart.s.iter().map(|&v| maths::ln(v)).sum();
        let log_density: Vec<f64> = dart
            .grid_theta
            .iter()
            .zip(&dart.log_grid_prior)
            .map(|(&theta, &log_prior)| {
                let c = theta / p as f64;
                log_prior + maths::lgamma(theta) - p as f64 * maths::lgamma(c)
                    + (c - 1.0) * sum_ln_s
            })
            .collect();
        dart.theta = dart.grid_theta[rng::draw_discrete(&log_density, &mut self.rng)];
    }

    /// The DART inclusion state, (weights, concentration), when the
    /// structure prior samples its weights; `None` in a build without the
    /// feature.
    pub fn inclusion_state(&self) -> Option<(&[f64], f64)> {
        #[cfg(not(feature = "experimental"))]
        return None;
        #[cfg(feature = "experimental")]
        {
            self.dart
                .as_ref()
                .map(|dart| (dart.s.as_slice(), dart.theta))
        }
    }

    /// The scale and working-response update; `structural` enables the
    /// variance ensemble's structural moves, which the conjugate sweep of
    /// the known-answer tests disables.
    fn update_noise(&mut self, structural: bool) {
        let prior_only = self.config.general_params.prior_only;
        // The variance ensemble carries the scale: its update, on the
        // residuals e = y - F, replaces the global sigma^2 draw
        // (H-AddiVortes has no global sigma^2).
        if let Some(variance) = &mut self.variance {
            if !prior_only {
                // The latent refresh runs before the ensemble update,
                // with the standing precisions, so everything after it
                // conditions on latents drawn from their conditional; a
                // response replacement is thereby repaired before it is
                // conditioned on.
                self.outcome
                    .draw_extra(&self.y, self.mean.total(), &self.precision, &mut self.rng);
                self.outcome.working_response(
                    self.mean.total(),
                    &self.precision,
                    &mut self.y,
                    &mut self.rng,
                );
            }
            let residuals: Vec<f64> = self
                .y
                .iter()
                .zip(self.mean.total())
                .map(|(y, f)| y - f)
                .collect();
            if structural {
                variance.sweep(
                    &self.x,
                    &residuals,
                    &self.precision,
                    &mut self.rng,
                    #[cfg(test)]
                    self.breakage,
                );
            } else {
                #[cfg(test)]
                variance.conjugate_sweep(&self.x, &residuals, &self.precision, &mut self.rng);
                #[cfg(not(test))]
                unreachable!("the conjugate sweep exists only under test");
            }
            if !prior_only {
                for (w, &s) in self.precision.iter_mut().zip(variance.total()) {
                    *w = 1.0 / s;
                }
            }
            return;
        }
        match self.outcome.sigma2_mode() {
            // sigma^2 | y, F ~ Inv-Gamma((nu + n) / 2,
            // (nu lambda + sum w_i r_i^2) / 2) with r = y - F and w the
            // outcome's weights (unit where it has none), drawn by the
            // kernel; under prior-only sampling the prior
            // Inv-Gamma(nu / 2, nu lambda / 2). The latent refresh runs
            // first, with the standing precisions, so sigma^2 and the
            // ensemble condition on latents drawn from their
            // conditional; a response replacement is thereby repaired
            // before it is conditioned on.
            Sigma2Mode::Sampled => {
                self.outcome
                    .draw_extra(&self.y, self.mean.total(), &self.precision, &mut self.rng);
                if !prior_only {
                    self.outcome.working_response(
                        self.mean.total(),
                        &self.precision,
                        &mut self.y,
                        &mut self.rng,
                    );
                }
                let (nu, _) = self.config.sigma2_prior();
                let (shape, scale) = if prior_only {
                    (0.5 * nu, 2.0 / (nu * self.lambda))
                } else {
                    let residuals = self.y.iter().zip(self.mean.total());
                    let rss: f64 = match self.outcome.weights() {
                        None => residuals.map(|(y, f)| (y - f) * (y - f)).sum(),
                        Some(weights) => residuals
                            .zip(weights)
                            .map(|((y, f), w)| w * (y - f) * (y - f))
                            .sum(),
                    };
                    let n = self.y.len();
                    (0.5 * (nu + n as f64), 2.0 / (nu * self.lambda + rss))
                };
                self.sigma_sq = 1.0 / rng::gamma(shape, scale, &mut self.rng);
                if !prior_only {
                    let scale_precision = 1.0 / self.sigma_sq;
                    match self.outcome.weights() {
                        None => self.precision.iter_mut().for_each(|w| *w = scale_precision),
                        Some(weights) => {
                            for (w, &weight) in self.precision.iter_mut().zip(weights) {
                                *w = weight * scale_precision;
                            }
                        }
                    }
                }
            }
            // The scale is fixed and the precisions stand; the outcome
            // refreshes its latent response.
            Sigma2Mode::Fixed(_) => {
                if prior_only {
                    return;
                }
                #[cfg(all(test, feature = "experimental"))]
                if let Outcome::Ordinal(outcome) = &mut self.outcome {
                    outcome.drop_cutpoint_prior =
                        self.breakage == crate::broken::Breakage::DroppedCutpointPrior;
                }
                self.outcome
                    .draw_extra(&self.y, self.mean.total(), &self.precision, &mut self.rng);
                self.outcome.working_response(
                    self.mean.total(),
                    &self.precision,
                    &mut self.y,
                    &mut self.rng,
                );
            }
            Sigma2Mode::Absent => {}
        }
    }

    /// Advance every sampler through `burn_in` sweeps and then `draws`
    /// kept draws of `thinning` sweeps each, the samplers spread over at
    /// most `threads` threads. Each sampler's draws are those of the same
    /// calls made on it alone, on any thread count.
    pub fn advance_all(
        samplers: &mut [&mut Sampler],
        burn_in: usize,
        draws: usize,
        thinning: usize,
        threads: usize,
    ) {
        crate::threads::spread(samplers, threads, |sampler| {
            for _ in 0..burn_in {
                sampler.step();
            }
            for _ in 0..draws {
                for _ in 0..thinning {
                    sampler.step();
                }
                sampler.keep();
            }
        });
    }

    /// Record the current state as a posterior draw.
    pub fn keep(&mut self) {
        let sigma_sq = (self.outcome.sigma2_mode().samples_global_sigma_sq()
            && self.variance.is_none())
        .then_some(self.sigma_sq);
        let variance = self.variance.as_ref().map(|v| v.tessellations().to_vec());
        #[cfg(feature = "experimental")]
        let cutpoints = self
            .outcome
            .as_ordinal()
            .map(|o| o.free_cutpoints())
            .filter(|free| !free.is_empty())
            .map(<[f64]>::to_vec);
        #[cfg(not(feature = "experimental"))]
        let cutpoints = None;
        #[cfg(feature = "experimental")]
        let df = self
            .outcome
            .as_student_t()
            .filter(|outcome| outcome.grid_sampled())
            .map(StudentTOutcome::df);
        #[cfg(not(feature = "experimental"))]
        let df = None;
        let inclusion = self.inclusion_state().map(|(s, theta)| (s.to_vec(), theta));
        self.kept.push(
            sigma_sq,
            self.mean.tessellations().to_vec(),
            variance,
            cutpoints,
            df,
            inclusion,
        );
        self.accumulate_fit();
    }

    /// Add this draw's mean function at every training row to the fit
    /// sums, each row's value being the sum over the tessellations in
    /// order of the cell the cached assignment places it in, which is the
    /// evaluation `Fitted::predict` makes on the kept draw. A soft
    /// tessellation's value is a kernel over every centre, so a kept soft
    /// draw retires the sums.
    fn accumulate_fit(&mut self) {
        let Some(mut sums) = self.fit_sums.take() else {
            return;
        };
        if self.mean.tessellations().iter().any(|t| t.tau.is_some()) {
            return;
        }
        let offset = self.offset();
        let probit = self.outcome.as_probit().is_some();
        #[cfg(feature = "experimental")]
        let ordinal = self.outcome.as_ordinal();
        #[cfg(feature = "experimental")]
        let free_cutpoints: &[f64] = ordinal.map_or(&[], OrdinalOutcome::free_cutpoints);
        #[cfg(feature = "experimental")]
        let ordinal = ordinal.is_some();
        #[cfg(not(feature = "experimental"))]
        let (ordinal, free_cutpoints): (bool, &[f64]) = (false, &[]);
        let expected_category = |latent: f64| {
            maths::normal_cdf(latent)
                + free_cutpoints
                    .iter()
                    .map(|&g| maths::normal_cdf(latent - g))
                    .sum::<f64>()
        };
        let tessellations = self.mean.tessellations();
        let assignments = self.mean.assignments();
        for i in 0..self.y.len() {
            let row = self.x.row(i);
            let mut f = 0.0;
            for (t, a) in tessellations.iter().zip(assignments) {
                f += t.value_in_cell(a.cells[i], row);
            }
            sums.scaled[i] += if ordinal {
                expected_category(f + offset)
            } else if probit {
                maths::normal_cdf(f + offset)
            } else {
                f
            };
            let latent = self.scaler.unscale_y(f) + offset;
            sums.response[i] += if ordinal {
                expected_category(latent)
            } else if probit {
                maths::normal_cdf(latent)
            } else {
                latent
            };
        }
        self.fit_sums = Some(sums);
    }

    /// The posterior mean of [`Fitted::predict`] at every training row
    /// times the kept draw count, from the fit sums; `None` when a kept
    /// draw retired them.
    pub(crate) fn fit_sum_response(&self) -> Option<&[f64]> {
        self.fit_sums.as_ref().map(|sums| sums.response.as_slice())
    }

    /// The current interior cutpoints of the ordinal model, increasing;
    /// empty at K = 2 and under another outcome, and in a build without
    /// the feature. Experimental (`docs/experimental.md`).
    pub fn cutpoints(&self) -> &[f64] {
        #[cfg(not(feature = "experimental"))]
        return &[];
        #[cfg(feature = "experimental")]
        {
            self.outcome
                .as_ordinal()
                .map(OrdinalOutcome::free_cutpoints)
                .unwrap_or(&[])
        }
    }

    /// The current error degrees of freedom of the Student-t outcome;
    /// `None` under another outcome, and in a build without the feature.
    /// Experimental (`docs/experimental.md`).
    pub fn student_df(&self) -> Option<f64> {
        #[cfg(not(feature = "experimental"))]
        return None;
        #[cfg(feature = "experimental")]
        {
            self.outcome.as_student_t().map(StudentTOutcome::df)
        }
    }

    /// Number of draws kept so far.
    pub fn n_kept(&self) -> usize {
        self.kept.n_draws()
    }

    /// Replace the response (caller scale; labels in {0, 1} under the
    /// probit model), keeping the tessellations, the cell values and
    /// sigma^2. The next sweep conditions on the new response.
    ///
    /// # Errors
    ///
    /// `RowCountMismatch`, `NonFiniteResponse`, `InvalidLabel`.
    pub fn set_response(&mut self, y: &[f64]) -> Result<()> {
        if y.len() != self.y.len() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: self.y.len(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteResponse { row });
        }
        if let Some(outcome) = self.outcome.as_probit_mut() {
            if let Some(row) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
                return Err(Error::InvalidLabel { row });
            }
            outcome.set_labels(y);
            return Ok(());
        }
        // The offset and the cutpoints stand across a replacement, so
        // valid codes suffice: an empty category only removes likelihood
        // terms.
        #[cfg(feature = "experimental")]
        if let OutcomeConfig::Ordinal(params) = &self.config.outcome {
            validate_ordinal_labels(y, params.categories)?;
            if let Some(outcome) = self.outcome.as_ordinal_mut() {
                outcome.set_labels(y);
            }
            return Ok(());
        }
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, OutcomeConfig::Aft(_)) {
            return Err(crate::error::invalid(
                "outcome",
                "the aft outcome replaces its response through set_aft_response",
            ));
        }
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, OutcomeConfig::IntervalCensored(_)) {
            return Err(crate::error::invalid(
                "outcome",
                "the interval_censored outcome replaces its response through \
                 set_interval_censored_response",
            ));
        }
        #[cfg(feature = "experimental")]
        if let OutcomeConfig::Tobit(params) = &self.config.outcome {
            let beyond = |v: f64| {
                params.lower.is_some_and(|limit| v < limit)
                    || params.upper.is_some_and(|limit| v > limit)
            };
            if let Some(row) = y.iter().position(|&v| beyond(v)) {
                return Err(Error::ResponseBeyondLimit { row });
            }
        }
        for (slot, &v) in self.y.iter_mut().zip(y) {
            *slot = self.scaler.scale_y(v);
        }
        // A latent-response outcome rereads censoring from the new
        // response; a no-op elsewhere.
        self.outcome.init(&self.y);
        Ok(())
    }

    /// Replace the AFT model's times and event indicator, keeping the
    /// tessellations, the cell values and sigma^2; the latents reset to
    /// the observed log times and the next sweep's refresh redraws the
    /// censored ones from their conditional. Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for an outcome other than
    /// `Outcome::Aft`; `RowCountMismatch`,
    /// `EventCountMismatch`, `InvalidSurvivalTime`.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn set_aft_response(&mut self, times: &[f64], events: &[bool]) -> Result<()> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::AFT.requires_feature());
        #[cfg(feature = "experimental")]
        {
            require_aft(&self.config)?;
            if times.len() != self.y.len() {
                return Err(Error::RowCountMismatch {
                    y_len: times.len(),
                    x_rows: self.y.len(),
                });
            }
            let y = log_times(times, events)?;
            for (slot, v) in self.y.iter_mut().zip(y) {
                *slot = self.scaler.scale_y(v);
            }
            if let Some(outcome) = self.outcome.as_aft_mut() {
                outcome.set_events(events);
            }
            self.outcome.init(&self.y);
            Ok(())
        }
    }

    /// Replace the interval-censored model's bound pairs, keeping the
    /// tessellations, the cell values and sigma^2; the latents reset to
    /// the working completions and the next sweep's refresh redraws the
    /// censored ones from their conditional. Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for an outcome other than
    /// `Outcome::IntervalCensored`;
    /// `RowCountMismatch`, `BoundCountMismatch`, `InvalidInterval`.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn set_interval_censored_response(&mut self, lower: &[f64], upper: &[f64]) -> Result<()> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::INTERVAL_CENSORED.requires_feature());
        #[cfg(feature = "experimental")]
        {
            require_interval_censored(&self.config)?;
            if lower.len() != self.y.len() {
                return Err(Error::RowCountMismatch {
                    y_len: lower.len(),
                    x_rows: self.y.len(),
                });
            }
            let y = interval_working(lower, upper)?;
            for (slot, v) in self.y.iter_mut().zip(y) {
                *slot = self.scaler.scale_y(v);
            }
            let map = |&v: &f64| {
                if v.is_finite() {
                    self.scaler.scale_y(v)
                } else {
                    v
                }
            };
            let (lower, upper) = (
                lower.iter().map(map).collect(),
                upper.iter().map(map).collect(),
            );
            if let Some(outcome) = self.outcome.as_interval_censored_mut() {
                outcome.set_bounds(lower, upper);
            }
            self.outcome.init(&self.y);
            Ok(())
        }
    }

    /// The current mean function at the training rows, caller scale: f(x_i),
    /// or c + f(x_i) under the probit model.
    pub fn fitted_values(&self) -> Vec<f64> {
        let offset = self.offset();
        self.mean
            .total()
            .iter()
            .map(|&f| self.scaler.unscale_y(f) + offset)
            .collect()
    }

    /// The response the in-sample fit is measured against, caller scale:
    /// the observed values where the working response is replaced by
    /// latents (the labels under the probit and ordinal models, the log
    /// times under the AFT model, the working completions under the
    /// tobit and interval-censored models), the working response
    /// otherwise. The target of [`finish`](Self::finish)'s in-sample
    /// RMSE.
    pub fn training_response(&self) -> Vec<f64> {
        self.outcome
            .observed_response()
            .unwrap_or(&self.y)
            .iter()
            .map(|&v| self.scaler.unscale_y(v))
            .collect()
    }

    /// The variance of y_i given f at each training row, caller scale:
    /// sigma^2 under the Gaussian model, 1 under the probit model (the
    /// latent scale), s^2(x_i) under the heteroscedastic model.
    pub fn noise_variances(&self) -> Vec<f64> {
        let n = self.y.len();
        let range_sq = self.scaler.y_range() * self.scaler.y_range();
        if let Some(variance) = &self.variance {
            return variance.total().iter().map(|s| s * range_sq).collect();
        }
        match self.outcome.sigma2_mode() {
            Sigma2Mode::Sampled => vec![self.sigma_sq * range_sq; n],
            Sigma2Mode::Fixed(value) => vec![value; n],
            Sigma2Mode::Absent => vec![f64::NAN; n],
        }
    }

    /// The current global sigma^2, scaled space: the Gaussian model's draw,
    /// or 1 under the probit and heteroscedastic models, whose variance is
    /// fixed at 1 on the latent scale and carried by the variance ensemble
    /// respectively.
    pub fn sigma_sq(&self) -> f64 {
        if self.variance.is_some() {
            return 1.0;
        }
        match self.outcome.sigma2_mode() {
            Sigma2Mode::Sampled => self.sigma_sq,
            Sigma2Mode::Fixed(value) => value,
            Sigma2Mode::Absent => 1.0,
        }
    }

    /// lambda of the sigma^2 prior nu lambda / chi^2_nu: calibrated from
    /// the data at construction, or the value given to
    /// [`pinned_prior`](Sampler::pinned_prior). The heteroscedastic model
    /// derives its per-cell lambda' = lambda^(1 / m') from it; the probit
    /// model does not use it.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The current mean tessellations, scaled space.
    pub fn tessellations(&self) -> &[Tessellation] {
        self.mean.tessellations()
    }

    /// The current variance tessellations, scaled space; empty outside the
    /// heteroscedastic model.
    pub fn variance_tessellations(&self) -> &[Tessellation] {
        match &self.variance {
            Some(v) => v.tessellations(),
            None => &[],
        }
    }

    /// The scaling frozen at construction.
    pub fn scaler(&self) -> &Scaler {
        &self.scaler
    }

    /// The configuration, with omega (and under the probit model the
    /// offset) resolved.
    pub fn config(&self) -> &Config {
        &self.config
    }

    fn offset(&self) -> f64 {
        #[cfg(feature = "experimental")]
        if let Some(outcome) = self.outcome.as_ordinal() {
            return outcome.offset();
        }
        self.outcome.as_probit().map_or(0.0, ProbitOutcome::offset)
    }

    /// The fitted model from the kept draws, with the in-sample RMSE of
    /// this chain from one prediction pass over the training rows. A fit
    /// that pools chains, or wants the fitted values, goes through
    /// [`Fitted::pool_samplers`] instead, which makes that pass once for
    /// every chain.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for `draws` when nothing was kept.
    pub fn finish(self) -> Result<Fitted> {
        if self.kept.n_draws() == 0 {
            return Err(crate::error::invalid("draws", "no draws were kept"));
        }
        let n = self.y.len();
        let mean_prediction = match &self.fit_sums {
            Some(sums) => sums.scaled.clone(),
            None => self.evaluated_fit_sum(),
        };
        let n_draws = self.kept.n_draws() as f64;
        let range = self.scaler.y_range();
        let target: &[f64] = self.outcome.observed_response().unwrap_or(&self.y);
        let in_sample_rmse = (mean_prediction
            .iter()
            .zip(target)
            .map(|(f, y)| {
                let r = (f / n_draws - y) * range;
                r * r
            })
            .sum::<f64>()
            / n as f64)
            .sqrt();
        self.into_fitted(in_sample_rmse)
    }

    /// The fit sum of [`FitSums::scaled`] by evaluating every kept draw at
    /// every training row, for a chain whose sums were retired.
    fn evaluated_fit_sum(&self) -> Vec<f64> {
        let n = self.y.len();
        // The posterior-mean prediction on the response scale: f,
        // Phi(c + f) under the probit model, or the expected category
        // sum_k Phi(c + f - gamma_k) under the ordinal model.
        let offset = self.offset();
        let probit = self.outcome.as_probit().is_some();
        #[cfg(feature = "experimental")]
        let ordinal = self.outcome.as_ordinal().is_some();
        #[cfg(not(feature = "experimental"))]
        let ordinal = false;
        let mut mean_prediction = vec![0.0; n];
        let geometry = self.mean.geometry();
        for (d, draw) in self.kept.tessellations().iter().enumerate() {
            #[cfg(feature = "experimental")]
            let free_cutpoints: &[f64] = if ordinal {
                self.kept.cutpoints().get(d).map_or(&[], Vec::as_slice)
            } else {
                &[]
            };
            #[cfg(not(feature = "experimental"))]
            let free_cutpoints: &[f64] = &[];
            let _ = d;
            for (i, slot) in mean_prediction.iter_mut().enumerate() {
                let row = self.x.row(i);
                let f: f64 = draw.iter().map(|t| t.value_at(row, geometry)).sum();
                *slot += if ordinal {
                    let latent = f + offset;
                    maths::normal_cdf(latent)
                        + free_cutpoints
                            .iter()
                            .map(|&g| maths::normal_cdf(latent - g))
                            .sum::<f64>()
                } else if probit {
                    maths::normal_cdf(f + offset)
                } else {
                    f
                };
            }
        }
        mean_prediction
    }

    /// The kept draws as a fitted model carrying `in_sample_rmse`.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for `draws` when nothing was kept.
    pub(crate) fn into_fitted(self, in_sample_rmse: f64) -> Result<Fitted> {
        if self.kept.n_draws() == 0 {
            return Err(crate::error::invalid("draws", "no draws were kept"));
        }
        let categories = self.mean.geometry().categories().to_vec();
        Ok(Fitted::new(
            self.config,
            self.scaler,
            self.kept,
            self.warnings,
            in_sample_rmse,
            categories,
        ))
    }
}

#[cfg(test)]
impl Sampler {
    /// One sweep with the structural moves disabled: the noise update, then
    /// every mean tessellation's cell means | rest. On a fixed tessellation
    /// the chain is the conjugate Gibbs sampler of the known-answer tests.
    pub(crate) fn conjugate_sweep(&mut self) {
        self.update_noise(false);
        self.mean
            .conjugate_sweep(&self.x, &self.y, &self.precision, &mut self.rng);
    }

    /// Replace mean tessellation `j` by `t`, whose cell means must be zero,
    /// and reset the running fit to zero.
    pub(crate) fn fix_mean_tessellation(&mut self, j: usize, t: Tessellation) {
        debug_assert!(t.mus.iter().all(|&m| m == 0.0));
        let x = self.x.clone();
        self.mean.set_tessellation(j, &x, t, 0.0);
    }

    /// The cell index of every training row under mean tessellation `j`.
    pub(crate) fn mean_cells(&self, j: usize) -> &[usize] {
        &self.mean.assignments()[j].cells
    }

    pub(crate) fn mean_sigma_mu_sq(&self) -> f64 {
        self.mean.family().sigma_mu_sq
    }

    pub(crate) fn variance_ensemble(&self) -> Option<&Ensemble<InverseGammaCells>> {
        self.variance.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit;
    use crate::tessellation::Assignment;

    fn toy(n: usize) -> (Data, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
        let x = Data::new(xs, n, 1).unwrap();
        (x, y)
    }

    fn labels(n: usize) -> (Data, Vec<f64>) {
        let (x, y) = toy(n);
        let labels = y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        (x, labels)
    }

    fn small() -> Config {
        Config::new().with_m(10).with_burn_in(20).with_draws(30)
    }

    #[test]
    fn same_seed_gives_identical_draws() {
        let (x, y) = toy(30);
        let a = fit(&small(), &x, &y, 42).unwrap();
        let b = fit(&small(), &x, &y, 42).unwrap();
        let c = fit(&small(), &x, &y, 43).unwrap();
        assert_eq!(a.posterior(), b.posterior());
        assert_ne!(a.posterior(), c.posterior());
    }

    #[test]
    fn fit_equals_the_explicit_loop() {
        let (x, y) = toy(30);
        let config = small().with_thinning(2);
        let fitted = fit(&config, &x, &y, 7).unwrap();
        let mut sampler = Sampler::new(&config, &x, &y, 7).unwrap();
        for _ in 0..20 {
            sampler.step();
        }
        for _ in 0..30 {
            sampler.step();
            sampler.step();
            sampler.keep();
        }
        let looped = sampler.finish().unwrap();
        assert_eq!(fitted.posterior(), looped.posterior());
        assert_eq!(fitted.in_sample_rmse(), looped.in_sample_rmse());
    }

    #[test]
    fn running_fit_matches_recomputation() {
        let (x, y) = toy(40);
        let mut sampler = Sampler::new(&small(), &x, &y, 3).unwrap();
        for _ in 0..15 {
            sampler.step();
            let g = sampler.mean.geometry();
            for i in 0..40 {
                let row = sampler.x.row(i);
                let sum: f64 = sampler
                    .tessellations()
                    .iter()
                    .map(|t| t.value_at(row, g))
                    .sum();
                assert!((sum - sampler.mean.total()[i]).abs() < 1e-9);
            }
            for (t, a) in sampler
                .tessellations()
                .iter()
                .zip(sampler.mean.assignments())
            {
                assert_eq!(*a, Assignment::full(&sampler.x, t, g));
            }
        }
    }

    #[test]
    fn running_variance_product_matches_recomputation() {
        let (x, y) = toy(40);
        let config = small().with_m_var(6);
        let mut sampler = Sampler::new(&config, &x, &y, 3).unwrap();
        for _ in 0..15 {
            sampler.step();
            let variance = sampler.variance_ensemble().unwrap();
            let g = variance.geometry();
            for i in 0..40 {
                let row = sampler.x.row(i);
                let product: f64 = variance
                    .tessellations()
                    .iter()
                    .map(|t| t.value_at(row, g))
                    .product();
                assert!((product - variance.total()[i]).abs() < 1e-9 * product);
                assert!((1.0 / variance.total()[i] - sampler.precision[i]).abs() < 1e-9);
            }
            for (t, a) in variance.tessellations().iter().zip(variance.assignments()) {
                assert_eq!(*a, Assignment::full(&sampler.x, t, g));
            }
        }
    }

    #[test]
    fn set_response_and_fitted_values_on_the_caller_scale() {
        let (x, y) = toy(30);
        let config = small();
        let mut plain = Sampler::new(&config, &x, &y, 1).unwrap();
        let mut same = Sampler::new(&config, &x, &y, 1).unwrap();
        let mut shifted = Sampler::new(&config, &x, &y, 1).unwrap();
        same.set_response(&y).unwrap();
        let up: Vec<f64> = y.iter().map(|v| v + 0.5).collect();
        shifted.set_response(&up).unwrap();
        for _ in 0..100 {
            plain.step();
            same.step();
            shifted.step();
        }
        assert_eq!(plain.fitted_values(), same.fitted_values());
        let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let (plain_mean, shifted_mean) =
            (mean(plain.fitted_values()), mean(shifted.fitted_values()));
        assert!(
            shifted_mean - plain_mean > 0.3,
            "{plain_mean} {shifted_mean}"
        );
        let mut sampler = plain;
        assert!(matches!(
            sampler.set_response(&[1.0]),
            Err(Error::RowCountMismatch { .. })
        ));
        assert!(matches!(
            sampler.set_response(&vec![f64::NAN; 30]),
            Err(Error::NonFiniteResponse { row: 0 })
        ));
    }

    #[test]
    fn probit_boundary_and_state() {
        let (x, labels) = labels(30);
        let config = small().with_outcome(OutcomeConfig::probit());
        let mut bad = labels.clone();
        bad[3] = 0.5;
        assert_eq!(
            Sampler::new(&config, &x, &bad, 1).unwrap_err(),
            Error::InvalidLabel { row: 3 }
        );
        assert!(Sampler::pinned_prior(&config, &x, &labels, 1.0, 1).is_err());
        assert!(
            Sampler::pinned_prior(&config.clone().with_offset(0.0), &x, &labels, 1.0, 1).is_ok()
        );

        let mut sampler = Sampler::new(&config, &x, &labels, 1).unwrap();
        let share = labels.iter().sum::<f64>() / 30.0;
        let offset = sampler.config().offset().unwrap();
        assert!((maths::normal_cdf(offset) - share).abs() < 1e-9);
        // The response is unscaled and f starts at 0, so the fitted values
        // start at the offset.
        assert_eq!(sampler.scaler().y_range(), 1.0);
        assert!(sampler.fitted_values().iter().all(|&v| v == offset));
        assert_eq!(sampler.sigma_sq(), 1.0);
        assert_eq!(sampler.noise_variances(), vec![1.0; 30]);
        assert!(sampler.variance_tessellations().is_empty());
        assert_eq!(
            sampler.mean.family().sigma_mu_sq,
            (3.0 / (3.0 * 10f64.sqrt())).powi(2)
        );
        for _ in 0..50 {
            sampler.step();
        }
        // The latent response has the sign of its label.
        for (z, &label) in sampler.y.iter().zip(&labels) {
            assert_eq!((z + offset > 0.0), label == 1.0);
        }
        assert_eq!(
            sampler.set_response(&vec![2.0; 30]).unwrap_err(),
            Error::InvalidLabel { row: 0 }
        );
        sampler.keep();
        let fitted = sampler.finish().unwrap();
        assert!(fitted.posterior().sigma_sq().is_empty());
        assert!(fitted.in_sample_rmse() < 0.5);
    }

    #[test]
    fn heteroscedastic_state() {
        let (x, y) = toy(30);
        let config = small().with_m_var(4);
        let mut sampler = Sampler::new(&config, &x, &y, 1).unwrap();
        assert_eq!(sampler.variance_tessellations().len(), 4);
        assert_eq!(sampler.sigma_sq(), 1.0);
        let family = sampler.variance_ensemble().unwrap().family();
        let (nu, lambda) = scaler::variance_cell_prior(6.0, sampler.lambda(), 4);
        assert_eq!((family.nu, family.lambda), (nu, lambda));
        // The product starts at sigma_hat^2 and the precision at its inverse.
        let initial = sampler.noise_variances()[0] / sampler.scaler().y_range().powi(2);
        assert!((1.0 / initial - sampler.precision[0]).abs() < 1e-12);
        for _ in 0..30 {
            sampler.step();
        }
        sampler.keep();
        assert!(sampler
            .noise_variances()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let fitted = sampler.finish().unwrap();
        assert!(fitted.posterior().sigma_sq().is_empty());
        assert_eq!(fitted.posterior().variance_tessellations()[0].len(), 4);
    }

    #[test]
    fn finish_needs_a_kept_draw() {
        let (x, y) = toy(10);
        let sampler = Sampler::new(&small(), &x, &y, 1).unwrap();
        assert!(sampler.finish().is_err());
    }

    #[test]
    fn omega_is_resolved_on_the_sampler_config() {
        let (x, y) = toy(10);
        let sampler = Sampler::new(&small(), &x, &y, 1).unwrap();
        assert_eq!(sampler.config().mean_params.structure.omega, Some(1.0));
    }

    #[test]
    fn recovers_a_smooth_function() {
        let n = 200;
        let mut rng = rng::chain_rng(99);
        let xs: Vec<f64> = (0..n).map(|_| rng::uniform(&mut rng)).collect();
        let f = |v: f64| (2.0 * std::f64::consts::PI * v).sin();
        let y: Vec<f64> = xs
            .iter()
            .map(|&v| f(v) + 0.1 * rng::standard_normal(&mut rng))
            .collect();
        let x = Data::new(xs.clone(), n, 1).unwrap();
        let config = Config::new().with_m(50).with_burn_in(200).with_draws(200);
        let fitted = fit(&config, &x, &y, 5).unwrap();
        let pred = fitted.predict(&x).unwrap();
        let rmse = (xs
            .iter()
            .zip(&pred)
            .map(|(&v, &p)| (f(v) - p).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 0.15, "rmse {rmse}");
        let sigma = fitted.sigma();
        let mean_sigma = sigma.iter().sum::<f64>() / sigma.len() as f64;
        assert!((0.05..0.2).contains(&mean_sigma), "sigma {mean_sigma}");
    }

    /// Per-cell count, sum and sum of squares of the response under a
    /// fixed assignment.
    struct FixedCells {
        n: Vec<f64>,
        s1: Vec<f64>,
        s2: Vec<f64>,
    }

    impl FixedCells {
        fn accumulate(cells: &[usize], y: &[f64], b: usize) -> Self {
            let mut out = Self {
                n: vec![0.0; b],
                s1: vec![0.0; b],
                s2: vec![0.0; b],
            };
            for (&cell, &v) in cells.iter().zip(y) {
                out.n[cell] += 1.0;
                out.s1[cell] += v;
                out.s2[cell] += v * v;
            }
            out
        }
    }

    /// Posterior means of sigma^2 and of every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2.
    ///
    /// With mu integrated out cell k contributes, up to constants,
    ///
    /// ```text
    /// ln N(y_k; 0, sigma^2 I + sigma_mu^2 J) =
    ///   -[ (n_k - 1) ln sigma^2 + ln(sigma^2 + n_k sigma_mu^2)
    ///      + (s2_k - sigma_mu^2 s1_k^2 / (sigma^2 + n_k sigma_mu^2)) / sigma^2 ] / 2,
    /// ```
    ///
    /// the prior is Inv-Gamma(nu / 2, nu lambda / 2), and
    /// E[mu_k | sigma^2, y] = sigma_mu^2 s1_k / (sigma^2 + n_k sigma_mu^2).
    fn quadrature_reference(
        stats: &FixedCells,
        nu: f64,
        lambda: f64,
        sigma_mu_sq: f64,
    ) -> (f64, Vec<f64>) {
        let b = stats.n.len();
        let (a, scale) = (0.5 * nu, 0.5 * nu * lambda);
        let steps = 40_000;
        let (lo, hi) = (lambda.ln() - 10.0, lambda.ln() + 10.0);
        let mut log_weights = Vec::with_capacity(steps + 1);
        let mut grid = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = lo + (hi - lo) * i as f64 / steps as f64;
            let sigma_sq = t.exp();
            let mut lp = -(a + 1.0) * t - scale / sigma_sq + t;
            for k in 0..b {
                let den = sigma_sq + stats.n[k] * sigma_mu_sq;
                lp += -0.5
                    * ((stats.n[k] - 1.0) * t
                        + den.ln()
                        + (stats.s2[k] - sigma_mu_sq * stats.s1[k] * stats.s1[k] / den) / sigma_sq);
            }
            grid.push(sigma_sq);
            log_weights.push(lp);
        }
        let max = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|lp| (lp - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mean_sigma_sq: f64 = weights.iter().zip(&grid).map(|(w, s)| w * s).sum::<f64>() / total;
        let mean_mus: Vec<f64> = (0..b)
            .map(|k| {
                weights
                    .iter()
                    .zip(&grid)
                    .map(|(w, s)| w * sigma_mu_sq * stats.s1[k] / (s + stats.n[k] * sigma_mu_sq))
                    .sum::<f64>()
                    / total
            })
            .collect();
        (mean_sigma_sq, mean_mus)
    }

    fn batch_means_mcse(values: &[f64]) -> (f64, f64) {
        let batches = 200;
        let size = values.len() / batches;
        let means: Vec<f64> = (0..batches)
            .map(|k| values[k * size..(k + 1) * size].iter().sum::<f64>() / size as f64)
            .collect();
        let mean = means.iter().sum::<f64>() / batches as f64;
        let var =
            means.iter().map(|m| (m - mean) * (m - mean)).sum::<f64>() / (batches as f64 - 1.0);
        (mean, (var / batches as f64).sqrt())
    }

    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % 13) as f64 / 26.0 - 0.25)
            .collect();
        let x = Data::new(xs, n, 1).unwrap();
        let lambda = 0.02;
        // With m' = 1 and a single-cell variance tessellation the
        // heteroscedastic model is the Gaussian model: its one cell value
        // is sigma^2 under the same inverse-gamma prior.
        let configs = [
            Config::new().with_m(1),
            Config::new().with_m(1).with_m_var(1),
        ];
        for (config, (centres, seed)) in configs
            .iter()
            .flat_map(|c| [(vec![-0.35, 0.0, 0.3], 21_u64), (vec![0.0], 22)].map(|case| (c, case)))
        {
            let b = centres.len();
            let mut sampler = Sampler::pinned_prior(config, &x, &y, lambda, seed).unwrap();
            let fixed = Tessellation {
                centres,
                dims: vec![0],
                mus: vec![0.0; b],
                betas: Vec::new(),
                tau: None,
            };
            sampler.fix_mean_tessellation(0, fixed);

            let stats = FixedCells::accumulate(sampler.mean_cells(0), &y, b);
            assert!(stats.n.iter().all(|&c| c > 0.0));
            let sigma_mu_sq = sampler.mean_sigma_mu_sq();
            let (ref_sigma_sq, ref_mus) =
                quadrature_reference(&stats, sampler.config.sigma2_prior().0, lambda, sigma_mu_sq);

            for _ in 0..200 {
                sampler.conjugate_sweep();
            }
            let kept = 40_000;
            let mut sigma_sq = Vec::with_capacity(kept);
            let mut mus: Vec<Vec<f64>> = vec![Vec::with_capacity(kept); b];
            for _ in 0..kept {
                sampler.conjugate_sweep();
                sigma_sq.push(sampler.noise_variances()[0]);
                for (k, series) in mus.iter_mut().enumerate() {
                    series.push(sampler.tessellations()[0].mus[k]);
                }
            }
            let (mean, mcse) = batch_means_mcse(&sigma_sq);
            assert!(
                (mean - ref_sigma_sq).abs() < 4.0 * mcse,
                "sigma^2 {mean} vs {ref_sigma_sq} +- {mcse}"
            );
            for (k, series) in mus.iter().enumerate() {
                let (mean, mcse) = batch_means_mcse(series);
                assert!(
                    (mean - ref_mus[k]).abs() < 4.0 * mcse,
                    "mu_{k} {mean} vs {} +- {mcse}",
                    ref_mus[k]
                );
            }
        }
    }
}
