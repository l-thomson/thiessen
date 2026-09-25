//! Model configuration (`Config`): the outcome model and the parameter
//! groups, plain data with serde, `Default`, consuming `with_*` setters
//! and a data-free `validate()`.

use crate::error::{invalid, Result};
use crate::geometry::Metric;
use crate::outcome::{RequiredData, Sigma2Mode};

/// The observation model and its own parameters, including the sigma^2
/// settings: what generated y, as opposed to the ensembles that describe
/// its average and spread.
///
/// Serialises externally tagged in snake case, so a JSON configuration
/// reads `{"outcome": {"probit": {}}}`. Validity is derived from the
/// outcome's scale mode: a variance ensemble may attach exactly where
/// sigma^2 is sampled.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(not(feature = "experimental"), serde(from = "StableOutcome"))]
pub enum Outcome {
    /// y = f(x) + e, e ~ N(0, sigma^2) (Stone and Gosling 2025). With
    /// `variance_params.tessellations` above 0 the spread is the
    /// variance ensemble's product s^2(x) in place of the global sigma^2
    /// (H-AddiVortes).
    Gaussian(GaussianParams),
    /// y in {0, 1} with P(y = 1 | x) = Phi(c + f(x)), fitted by Albert
    /// and Chib (1993) augmentation with unit latent variance (Binary
    /// AddiVortes).
    Probit(ProbitParams),
    /// y = max(lower, min(upper, y*)) with y* = f(x) + e, e ~ N(0,
    /// sigma^2), the type-I tobit model: a response that takes a known
    /// limit's value whenever the latent value lies beyond it, fitted by
    /// Chib (1992) augmentation. Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    Tobit(TobitParams),
    /// ln T = f(x) + e, e ~ N(0, sigma^2), the lognormal accelerated
    /// failure time model for a right-censored time-to-event response
    /// (Wei 1992; the BART package's `abart`), fitted by censored-data
    /// augmentation on the log scale. The event times and the event
    /// indicator are data: the model is fitted through
    /// [`fit_aft`](crate::fit_aft) or [`Sampler::aft`](crate::Sampler::aft).
    /// Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    Aft(AftParams),
    /// y* = f(x) + e, e ~ N(0, sigma^2), observed only as a pair of
    /// bounds [l_i, u_i] per row (an equal pair is an exact value, an
    /// infinite endpoint one-sided censoring), fitted by censored-data
    /// augmentation with a two-sided truncated draw. The bounds are
    /// data: the model is fitted through
    /// [`fit_interval_censored`](crate::fit_interval_censored) or
    /// [`Sampler::interval_censored`](crate::Sampler::interval_censored).
    /// Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    IntervalCensored(IntervalCensoredParams),
    /// y in {0, ..., K - 1} ordered, with P(y <= k | x) =
    /// Phi(gamma_{k+1} - c - f(x)): the ordinal probit model of Albert
    /// and Chib (1993, s. 5), fitted by latent augmentation with the
    /// interior cutpoints sampled by the Cowles (1996) blocked collapsed
    /// move. The latent variance is 1 and the first cutpoint 0 for
    /// identification. Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    Ordinal(OrdinalParams),
    /// y = f(x) + e, e ~ sigma t_df: the independent Student-t model of
    /// Geweke (1993), fitted as a scale mixture of normals with
    /// per-observation Gamma weights. The error degrees of freedom are
    /// fixed (default 4) or drawn over a declared grid by their exact
    /// discrete conditional. Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    StudentT(StudentTParams),
    /// y = f(x) + e, e ~ Laplace(0, sigma): errors with exponential
    /// tails, fitted as a scale mixture of normals with per-observation
    /// inverse-Gaussian weights (Park and Casella 2008); no parameters
    /// of its own. Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    Laplace(LaplaceParams),
    /// An outcome this build gates; [`Config::validate`] reports it.
    #[cfg(not(feature = "experimental"))]
    #[doc(hidden)]
    #[serde(skip)]
    Gated(Gated),
}

/// The stable variants plus the gated names, so a build without the
/// feature carries a gated outcome through to [`Config::validate`]
/// rather than failing inside the deserialiser, which has no channel for
/// a typed error.
#[cfg(not(feature = "experimental"))]
#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum StableOutcome {
    Gaussian(GaussianParams),
    Probit(ProbitParams),
    Tobit(serde::de::IgnoredAny),
    Aft(serde::de::IgnoredAny),
    IntervalCensored(serde::de::IgnoredAny),
    Ordinal(serde::de::IgnoredAny),
    StudentT(serde::de::IgnoredAny),
    Laplace(serde::de::IgnoredAny),
}

#[cfg(not(feature = "experimental"))]
impl From<StableOutcome> for Outcome {
    fn from(outcome: StableOutcome) -> Self {
        match outcome {
            StableOutcome::Gaussian(params) => Outcome::Gaussian(params),
            StableOutcome::Probit(params) => Outcome::Probit(params),
            StableOutcome::Tobit(_) => Outcome::Gated(Gated::TOBIT),
            StableOutcome::Aft(_) => Outcome::Gated(Gated::AFT),
            StableOutcome::IntervalCensored(_) => Outcome::Gated(Gated::INTERVAL_CENSORED),
            StableOutcome::Ordinal(_) => Outcome::Gated(Gated::ORDINAL),
            StableOutcome::StudentT(_) => Outcome::Gated(Gated::STUDENT_T),
            StableOutcome::Laplace(_) => Outcome::Gated(Gated::LAPLACE),
        }
    }
}

/// A configuration value this build gates, carried from the deserialiser
/// to [`Config::validate`], which reports it as
/// [`Error::RequiresFeature`](crate::Error::RequiresFeature). serde's
/// error channel is a string, so a value rejected inside `Deserialize`
/// reaches a caller untyped; deserialising states the shape and
/// `validate` states the policy, as they already do for every other
/// field. Present only without the feature, and never serialised: a
/// configuration holding one does not pass `validate`.
#[cfg(not(feature = "experimental"))]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gated(pub(crate) &'static str);

#[cfg(not(feature = "experimental"))]
impl Gated {
    pub(crate) const TOBIT: Self = Gated("the `tobit` outcome");
    pub(crate) const AFT: Self = Gated("the `aft` outcome");
    pub(crate) const INTERVAL_CENSORED: Self = Gated("the `interval_censored` outcome");
    pub(crate) const ORDINAL: Self = Gated("the `ordinal` outcome");
    pub(crate) const STUDENT_T: Self = Gated("the `student_t` outcome");
    pub(crate) const LAPLACE: Self = Gated("the `laplace` outcome");

    /// The item, as [`Error::RequiresFeature`](crate::Error::RequiresFeature)
    /// names it.
    pub(crate) fn requires_feature(self) -> crate::error::Error {
        crate::error::Error::RequiresFeature {
            item: self.0.into(),
            feature: "experimental",
        }
    }

    /// The arm of a match on a gated value past [`Config::validate`],
    /// which rejects it.
    pub(crate) fn unreachable(self) -> ! {
        unreachable!("{} is rejected by Config::validate", self.0)
    }
}

impl Default for Outcome {
    fn default() -> Self {
        Outcome::Gaussian(GaussianParams::default())
    }
}

impl Outcome {
    /// The Gaussian outcome with its default sigma^2 prior.
    pub fn gaussian() -> Self {
        Outcome::Gaussian(GaussianParams::default())
    }

    /// The probit outcome with the offset resolved from the data at fit.
    pub fn probit() -> Self {
        Outcome::Probit(ProbitParams::default())
    }

    /// The tobit outcome with the given censoring limits and the default
    /// sigma^2 prior; at least one limit is required at validation.
    /// Experimental.
    #[cfg(feature = "experimental")]
    pub fn tobit(lower: Option<f64>, upper: Option<f64>) -> Self {
        Outcome::Tobit(TobitParams {
            lower,
            upper,
            ..TobitParams::default()
        })
    }

    /// The AFT outcome with the default sigma^2 prior on the log scale;
    /// the times and the event indicator are given at fit. Experimental.
    #[cfg(feature = "experimental")]
    pub fn aft() -> Self {
        Outcome::Aft(AftParams::default())
    }

    /// The interval-censored outcome with the default sigma^2 prior;
    /// the bound pairs are given at fit. Experimental.
    #[cfg(feature = "experimental")]
    pub fn interval_censored() -> Self {
        Outcome::IntervalCensored(IntervalCensoredParams::default())
    }

    /// The ordinal outcome over `categories` ordered categories, with
    /// the offset resolved from the data at fit and the default
    /// cutpoint prior. Experimental.
    #[cfg(feature = "experimental")]
    pub fn ordinal(categories: usize) -> Self {
        Outcome::Ordinal(OrdinalParams {
            categories,
            ..OrdinalParams::default()
        })
    }

    /// The Student-t outcome with the error degrees of freedom fixed at
    /// `df` and the default sigma^2 prior. Experimental.
    #[cfg(feature = "experimental")]
    pub fn student_t(df: f64) -> Self {
        Outcome::StudentT(StudentTParams {
            df: DegreesOfFreedom::Fixed(df),
            ..StudentTParams::default()
        })
    }

    /// The Student-t outcome with the error degrees of freedom drawn
    /// over `grid` (uniform prior, exact discrete conditional) and the
    /// default sigma^2 prior. Experimental.
    #[cfg(feature = "experimental")]
    pub fn student_t_grid(grid: Vec<f64>) -> Self {
        Outcome::StudentT(StudentTParams {
            df: DegreesOfFreedom::Grid(grid),
            ..StudentTParams::default()
        })
    }

    /// The Laplace outcome with the default sigma^2 prior. Experimental.
    #[cfg(feature = "experimental")]
    pub fn laplace() -> Self {
        Outcome::Laplace(LaplaceParams::default())
    }

    /// Every outcome family this build carries, at its defaults. A
    /// binding reads the family names and their parameters from here,
    /// so neither keeps a list of its own.
    pub fn catalogue() -> Vec<Self> {
        #[cfg(not(feature = "experimental"))]
        let gated: [Self; 0] = [];
        #[cfg(feature = "experimental")]
        let gated = [
            Outcome::Tobit(TobitParams::default()),
            Outcome::Aft(AftParams::default()),
            Outcome::IntervalCensored(IntervalCensoredParams::default()),
            Outcome::Ordinal(OrdinalParams::default()),
            Outcome::StudentT(StudentTParams::default()),
            Outcome::Laplace(LaplaceParams::default()),
        ];
        [Outcome::gaussian(), Outcome::probit()]
            .into_iter()
            .chain(gated)
            .collect()
    }

    /// What the outcome does with sigma^2; scale validity derives from
    /// this value, never from a per-outcome table.
    pub(crate) fn sigma2_mode(&self) -> Sigma2Mode {
        match self {
            Outcome::Gaussian(_) => Sigma2Mode::Sampled,
            Outcome::Probit(_) => Sigma2Mode::Fixed(1.0),
            #[cfg(not(feature = "experimental"))]
            Outcome::Gated(gated) => gated.unreachable(),
            #[cfg(feature = "experimental")]
            Outcome::Tobit(_) => Sigma2Mode::Sampled,
            #[cfg(feature = "experimental")]
            Outcome::Aft(_) => Sigma2Mode::Sampled,
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(_) => Sigma2Mode::Sampled,
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(_) => Sigma2Mode::Fixed(1.0),
            #[cfg(feature = "experimental")]
            Outcome::StudentT(_) => Sigma2Mode::Sampled,
            #[cfg(feature = "experimental")]
            Outcome::Laplace(_) => Sigma2Mode::Sampled,
        }
    }

    /// The response contract the outcome imposes at fit.
    pub(crate) fn required_data(&self) -> RequiredData {
        match self {
            Outcome::Gaussian(_) => RequiredData::Continuous,
            Outcome::Probit(_) => RequiredData::Binary,
            #[cfg(not(feature = "experimental"))]
            Outcome::Gated(gated) => gated.unreachable(),
            #[cfg(feature = "experimental")]
            Outcome::Tobit(_) => RequiredData::Continuous,
            #[cfg(feature = "experimental")]
            Outcome::Aft(_) => RequiredData::Continuous,
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(_) => RequiredData::Continuous,
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(_) => RequiredData::Ordinal,
            #[cfg(feature = "experimental")]
            Outcome::StudentT(_) => RequiredData::Continuous,
            #[cfg(feature = "experimental")]
            Outcome::Laplace(_) => RequiredData::Continuous,
        }
    }
}

/// The Gaussian outcome's parameters: the sigma^2 prior, folded onto the
/// outcome because they are facts about the observation model.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GaussianParams {
    /// sigma^2 prior degrees of freedom nu. Default 6. A variance
    /// ensemble requires nu > 2.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self { nu: 6.0, q: 0.85 }
    }
}

/// The probit outcome's parameters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProbitParams {
    /// The offset c in P(y = 1 | x) = Phi(c + f(x)). `None` resolves to
    /// Phi^-1(ybar) at fit (the BART package's `binaryOffset`); the
    /// resolved value is stored on the fitted model. Default `None`.
    pub offset: Option<f64>,
}

/// The tobit outcome's parameters: the censoring limits and the sigma^2
/// prior, folded onto the outcome because they are facts about the
/// observation model. Experimental (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TobitParams {
    /// The lower censoring limit: a response value equal to it is read
    /// as censored below. `None` is no lower limit. At least one limit
    /// is required at validation.
    pub lower: Option<f64>,
    /// The upper censoring limit: a response value equal to it is read
    /// as censored above. `None` is no upper limit.
    pub upper: Option<f64>,
    /// sigma^2 prior degrees of freedom nu, as the Gaussian outcome's.
    /// Default 6. A variance ensemble requires nu > 2.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

#[cfg(feature = "experimental")]
impl Default for TobitParams {
    fn default() -> Self {
        let defaults = GaussianParams::default();
        Self {
            lower: None,
            upper: None,
            nu: defaults.nu,
            q: defaults.q,
        }
    }
}

/// The AFT outcome's parameters: the sigma^2 prior on the log-time
/// scale, folded onto the outcome because they are facts about the
/// observation model; the times and the event indicator are data, not
/// parameters. Experimental (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AftParams {
    /// sigma^2 prior degrees of freedom nu on the log scale. Default 6.
    /// A variance ensemble requires nu > 2.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

#[cfg(feature = "experimental")]
impl Default for AftParams {
    fn default() -> Self {
        let defaults = GaussianParams::default();
        Self {
            nu: defaults.nu,
            q: defaults.q,
        }
    }
}

/// The interval-censored outcome's parameters: the sigma^2 prior,
/// folded onto the outcome because it is a fact about the observation
/// model; the bound pairs are data, not parameters. Experimental
/// (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntervalCensoredParams {
    /// sigma^2 prior degrees of freedom nu, as the Gaussian outcome's.
    /// Default 6. A variance ensemble requires nu > 2.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

#[cfg(feature = "experimental")]
impl Default for IntervalCensoredParams {
    fn default() -> Self {
        let defaults = GaussianParams::default();
        Self {
            nu: defaults.nu,
            q: defaults.q,
        }
    }
}

/// The ordinal outcome's parameters: the category count, the offset and
/// the cutpoint prior. The latent variance is fixed at 1 and the first
/// cutpoint at 0 for identification. Experimental
/// (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrdinalParams {
    /// Number of ordered categories K, at least 2; the response holds
    /// integer codes 0 to K - 1.
    pub categories: usize,
    /// The offset c in P(y <= k | x) = Phi(gamma_{k+1} - c - f(x)).
    /// `None` resolves to Phi^-1(share of y >= 1) at fit, the probit
    /// rule at K = 2; the resolved value is stored on the fitted model.
    /// Default `None`.
    pub offset: Option<f64>,
    /// Standard deviation of the independent N(0, cutpoint_sd^2) prior
    /// on the log-gaps ln(gamma_k - gamma_{k-1}) of the interior
    /// cutpoints (Albert and Chib 2001). Default 1.
    pub cutpoint_sd: f64,
}

#[cfg(feature = "experimental")]
impl Default for OrdinalParams {
    fn default() -> Self {
        Self {
            categories: 2,
            offset: None,
            cutpoint_sd: 1.0,
        }
    }
}

/// The error degrees of freedom of the Student-t outcome: a fixed value,
/// or a grid carrying a uniform prior and drawn each sweep by its exact
/// discrete conditional. Serialises untagged, so a JSON configuration
/// reads `"df": 4.0` or `"df": [3.0, 6.0, 12.0]`. No continuous-df
/// sampler exists: df is weakly identified and random-walk samplers over
/// it mix poorly, while the grid conditional is exact. Experimental
/// (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum DegreesOfFreedom {
    /// df fixed at the value.
    Fixed(f64),
    /// df uniform over the grid a priori: at least two values, strictly
    /// increasing.
    Grid(Vec<f64>),
}

#[cfg(feature = "experimental")]
impl DegreesOfFreedom {
    /// The chain's initial df: the fixed value, or the grid's middle
    /// entry. Called on a validated configuration, whose grid is
    /// non-empty.
    pub(crate) fn initial(&self) -> f64 {
        match self {
            DegreesOfFreedom::Fixed(df) => *df,
            DegreesOfFreedom::Grid(grid) => grid[grid.len() / 2],
        }
    }

    /// The smallest admissible df: the fixed value, or the grid's first
    /// entry, the grid being increasing.
    pub(crate) fn minimum(&self) -> f64 {
        match self {
            DegreesOfFreedom::Fixed(df) => *df,
            DegreesOfFreedom::Grid(grid) => grid[0],
        }
    }

    /// The grid; empty for a fixed df.
    pub(crate) fn grid(&self) -> &[f64] {
        match self {
            DegreesOfFreedom::Fixed(_) => &[],
            DegreesOfFreedom::Grid(grid) => grid,
        }
    }
}

/// The Student-t outcome's parameters: the error degrees of freedom and
/// the sigma^2 prior, folded onto the outcome because they are facts
/// about the observation model. `nu` and `q` are the sigma^2 prior as on
/// the Gaussian outcome; the error degrees of freedom sit on `df`.
/// Experimental (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StudentTParams {
    /// The error degrees of freedom: a fixed value (default 4), or a
    /// grid of at least two strictly increasing values with a uniform
    /// prior.
    pub df: DegreesOfFreedom,
    /// sigma^2 prior degrees of freedom nu, as the Gaussian outcome's.
    /// Default 6.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

#[cfg(feature = "experimental")]
impl Default for StudentTParams {
    fn default() -> Self {
        let defaults = GaussianParams::default();
        Self {
            df: DegreesOfFreedom::Fixed(4.0),
            nu: defaults.nu,
            q: defaults.q,
        }
    }
}

/// The Laplace outcome's parameters: the sigma^2 prior, folded onto the
/// outcome because it is a fact about the observation model; the model
/// has no parameters of its own. Experimental (`docs/experimental.md`).
#[cfg(feature = "experimental")]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LaplaceParams {
    /// sigma^2 prior degrees of freedom nu, as the Gaussian outcome's.
    /// Default 6.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

#[cfg(feature = "experimental")]
impl Default for LaplaceParams {
    fn default() -> Self {
        let defaults = GaussianParams::default();
        Self {
            nu: defaults.nu,
            q: defaults.q,
        }
    }
}

/// One term group: the ensemble describing the average (`mean_params`) or
/// the spread (`variance_params`). Written once and instantiated per
/// slot, so a new option reaches both ensembles at once.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TermParams {
    /// Number of tessellations in the ensemble. `None` resolves at fit to
    /// 200 on the mean slot and to 0 on the variance slot; 0 on the
    /// variance slot is a constant spread, above 0 is H-AddiVortes (the
    /// paper's count is 40).
    pub tessellations: Option<usize>,
    /// Cell-value prior spread k: sigma_mu = w / (k sqrt m) with the
    /// half-width w the outcome model owns. Default 3. The variance
    /// ensemble's inverse-gamma cells do not use it.
    pub k: f64,
    /// Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). Default
    /// 5, following AddiVortes >= 0.6.8; the paper reports 25
    /// ([`Config::paper`]).
    pub lambda_c: f64,
    /// The covariate space: the metric of each column and the
    /// centre-coordinate law's scale.
    pub geometry: GeometryParams,
    /// Which covariates a tessellation may use.
    pub structure: StructureParams,
    /// The within-cell response surface.
    pub cell: CellParams,
}

impl Default for TermParams {
    fn default() -> Self {
        Self {
            tessellations: None,
            k: 3.0,
            lambda_c: 5.0,
            geometry: GeometryParams::default(),
            structure: StructureParams::default(),
            cell: CellParams::default(),
        }
    }
}

impl TermParams {
    /// The count in force on the mean slot: the field, or 200.
    pub(crate) fn mean_tessellations(&self) -> usize {
        self.tessellations.unwrap_or(200)
    }

    /// The count in force on the variance slot: the field, or 0.
    pub(crate) fn variance_tessellations(&self) -> usize {
        self.tessellations.unwrap_or(0)
    }
}

/// The covariate space of one term group.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeometryParams {
    /// The metric of each covariate column, one entry per column in
    /// column order; empty, the default, is [`Metric::Euclidean`] on
    /// every column. Checked against the design at fit.
    pub metric: Vec<Metric>,
    /// Centre-coordinate prior and proposal standard deviation sigma_c
    /// (scaled space). Default 0.8.
    pub sigma_c: f64,
    /// How an observation belongs to a tessellation's cells. Default
    /// hard, the published rule. Every other value is experimental
    /// (`docs/experimental.md`).
    #[serde(skip_serializing_if = "Membership::is_hard")]
    pub membership: Membership,
    /// The precision matrix of the Mahalanobis metric, row-major p x p
    /// over the encoded design; required exactly when a column's metric
    /// is the Mahalanobis one. Checked at fit: symmetric, positive
    /// definite. The active-subspace distance uses the principal
    /// submatrix on the active columns, which is not the conditional
    /// precision. The field has no published value, so any value is
    /// experimental (`docs/experimental.md`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<Vec<f64>>,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            metric: Vec::new(),
            sigma_c: 0.8,
            membership: Membership::Hard,
            precision: None,
        }
    }
}

/// How an observation belongs to a tessellation's cells,
/// [`GeometryParams::membership`].
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(not(feature = "experimental"), serde(from = "StableMembership"))]
#[non_exhaustive]
pub enum Membership {
    /// Every observation belongs wholly to its nearest centre, the
    /// published rule. The default.
    #[default]
    Hard,
    /// A membership rule this build gates; [`Config::validate`] reports
    /// it.
    #[cfg(not(feature = "experimental"))]
    #[doc(hidden)]
    #[serde(skip)]
    Gated(Gated),
    /// Kernel-weighted membership (SBART's softening of the tree split,
    /// Linero and Yang 2018, carried to the Voronoi assignment):
    /// observation i takes weight w_ik proportional to
    /// exp(-d_ik^2 / (2 tau^2)) in cell k, normalised over the
    /// tessellation's centres, with d_ik^2 the squared distance of the
    /// active metrics and tau a per-tessellation bandwidth,
    /// tau ~ Exponential(`rate`), updated by a Metropolis step on
    /// ln tau. The tessellation's value at x is the weighted sum of its
    /// cell values; the cell values are drawn jointly from the
    /// b-dimensional conjugate normal, and the structural moves
    /// integrate them out jointly. The empty-cell rule still counts
    /// nearest-centre members. Constant cell basis and constant spread
    /// only: the linear basis has no derived weighted update, and the
    /// variance ensemble's inverse-gamma cells have no closed-form
    /// weighted conditional. Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    Soft {
        /// Rate of the exponential bandwidth prior, on the scaled
        /// covariate space. Default 10, so the prior mean bandwidth is
        /// a tenth of a column's range (the SoftBart `tau_rate`
        /// default).
        #[serde(default = "default_soft_rate")]
        rate: f64,
    },
}

#[cfg(feature = "experimental")]
fn default_soft_rate() -> f64 {
    10.0
}

impl Membership {
    /// Whether this is the default, for compact serialisation.
    fn is_hard(&self) -> bool {
        matches!(self, Membership::Hard)
    }

    /// The soft membership with the default bandwidth prior.
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    pub fn soft() -> Self {
        Membership::Soft {
            rate: default_soft_rate(),
        }
    }
}

/// The published rule plus the gated names; as [`StableOutcome`].
#[cfg(not(feature = "experimental"))]
#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum StableMembership {
    Hard,
    Soft(serde::de::IgnoredAny),
}

#[cfg(not(feature = "experimental"))]
impl From<StableMembership> for Membership {
    fn from(membership: StableMembership) -> Self {
        match membership {
            StableMembership::Hard => Membership::Hard,
            StableMembership::Soft(_) => Membership::Gated(Gated("`soft` membership")),
        }
    }
}

/// The covariate-inclusion prior over the columns of one term group:
/// which covariates a tessellation may use, and with what prior weight.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(not(feature = "experimental"), serde(from = "StableInclusion"))]
#[non_exhaustive]
pub enum Inclusion {
    /// Dimensions drawn uniformly, the published prior. The default.
    #[default]
    Uniform,
    /// An inclusion prior this build gates; [`Config::validate`] reports
    /// it.
    #[cfg(not(feature = "experimental"))]
    #[doc(hidden)]
    #[serde(skip)]
    Gated(Gated),
    /// A fixed weight per column (bartMachine `cov_prior_vec`, Kapelner
    /// and Bleich 2016): the subset prior given the dimension count is
    /// proportional to the product of the member weights, proposals pick
    /// the incoming covariate with probability proportional to its
    /// weight, and a zero weight excludes the column. Weights are
    /// non-negative and finite with at least one positive, checked with
    /// the hyperparameters; the length is checked against the design at
    /// fit. Equal weights are the uniform prior and take its code path,
    /// so they reproduce the default draws exactly. Nothing is sampled:
    /// the sampler's conditional updates are untouched. Experimental
    /// (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    Weighted {
        /// One weight per column, in column order.
        weights: Vec<f64>,
    },
    /// The DART sparsity prior (Linero 2018), as the BART package ships
    /// it (`sparse = TRUE` with `a`, `b`, `rho`): the weights are a
    /// sampled vector s ~ Dirichlet(theta / p), the subset prior given
    /// the dimension count is proportional to the product of member
    /// weights, and the concentration theta is sampled on the BART grid,
    /// lambda = theta / (theta + rho) uniform over 1000 points of (0, 1)
    /// with prior weights Beta(a, b); the grid is the prior, not an
    /// approximation of one. s starts uniform at 1 / p and is updated by
    /// a Metropolis step whose Dirichlet(theta / p + counts) proposal
    /// leaves exactly the subset-prior normalisers in the ratio. In the
    /// API this is a component of the term group; in validation it is
    /// model-grade, because the sampled weights change the posterior.
    /// Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    Dart {
        /// Beta shape a of the concentration prior. Default 0.5.
        #[serde(default = "default_dart_a")]
        a: f64,
        /// Beta shape b of the concentration prior. Default 1.
        #[serde(default = "default_dart_b")]
        b: f64,
        /// The concentration scale rho; `None` resolves to p at fit.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rho: Option<f64>,
    },
}

#[cfg(feature = "experimental")]
fn default_dart_a() -> f64 {
    0.5
}

#[cfg(feature = "experimental")]
fn default_dart_b() -> f64 {
    1.0
}

impl Inclusion {
    /// Whether this is the default, for compact serialisation.
    fn is_uniform(&self) -> bool {
        matches!(self, Inclusion::Uniform)
    }
}

/// The published prior plus the gated names; as [`StableOutcome`].
#[cfg(not(feature = "experimental"))]
#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum StableInclusion {
    Uniform,
    Weighted(serde::de::IgnoredAny),
    Dart(serde::de::IgnoredAny),
}

#[cfg(not(feature = "experimental"))]
impl From<StableInclusion> for Inclusion {
    fn from(inclusion: StableInclusion) -> Self {
        match inclusion {
            StableInclusion::Uniform => Inclusion::Uniform,
            StableInclusion::Weighted(_) => Inclusion::Gated(Gated("`weighted` inclusion")),
            StableInclusion::Dart(_) => Inclusion::Gated(Gated("`dart` inclusion")),
        }
    }
}

/// The covariate-inclusion prior of one term group.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StructureParams {
    /// The covariate-inclusion prior. Default uniform, the published
    /// prior. Every other value is experimental
    /// (`docs/experimental.md`).
    #[serde(skip_serializing_if = "Inclusion::is_uniform")]
    pub inclusion: Inclusion,
    /// Dimension-count prior parameter omega; omega / p is the prior
    /// probability of including a covariate. `None` resolves to
    /// min(3, p) at fit. Must satisfy 0 < omega <= p; at omega = p the
    /// dimension count saturates at p.
    pub omega: Option<f64>,
}

/// The within-cell response surface of one term group. Carries no options
/// yet: every cell holds one constant value, the paper's basis.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CellParams {
    /// The within-cell response surface. Default constant, the
    /// published basis. Every other value is experimental
    /// (`docs/experimental.md`).
    #[serde(skip_serializing_if = "Basis::is_constant")]
    pub basis: Basis,
}

/// The within-cell response surface of one term group,
/// [`CellParams::basis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(not(feature = "experimental"), serde(from = "StableBasis"))]
#[non_exhaustive]
pub enum Basis {
    /// One value per cell, the published model. The default.
    #[default]
    Constant,
    /// A cell basis this build gates; [`Config::validate`] reports it.
    #[cfg(not(feature = "experimental"))]
    #[doc(hidden)]
    #[serde(skip)]
    Gated(Gated),
    /// The cell value tilts across the region: mu + beta' (x_A - c) over
    /// the active covariates, centred at the cell's centre, so mu keeps
    /// its role as the level there. Slopes take the cell-value prior
    /// N(0, sigma_mu^2) coordinate-wise; the cell update draws
    /// (mu, beta) jointly from the conjugate normal and the structural
    /// moves integrate them out jointly. Needs every column min-max
    /// scaled (checked at fit): the offsets are only comparable on the
    /// scaled space. Mean slot only; the variance ensemble's
    /// inverse-gamma cells keep the constant basis. Experimental
    /// (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    Linear,
}

impl Basis {
    /// Whether this is the default, for compact serialisation.
    fn is_constant(&self) -> bool {
        matches!(self, Basis::Constant)
    }
}

/// The published basis plus the gated names; as [`StableOutcome`].
#[cfg(not(feature = "experimental"))]
#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum StableBasis {
    Constant,
    Linear,
}

#[cfg(not(feature = "experimental"))]
impl From<StableBasis> for Basis {
    fn from(basis: StableBasis) -> Self {
        match basis {
            StableBasis::Constant => Basis::Constant,
            StableBasis::Linear => Basis::Gated(Gated("the `linear` cell basis")),
        }
    }
}

/// The sweep schedule and everything that belongs to no ensemble.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralParams {
    /// Burn-in sweeps discarded by `fit`. Default 200.
    pub burn_in: usize,
    /// Posterior draws kept by `fit`. Default 1000.
    pub draws: usize,
    /// Thinning interval: `fit` keeps every `thinning`-th sweep after
    /// burn-in. Default 1.
    pub thinning: usize,
    /// Sample from the prior: the likelihood is switched off, so the
    /// chain draws sigma^2, the tessellations and the cell values from
    /// the prior and `predict` on the fitted model gives prior predictive
    /// draws (brms `sample_prior = "only"`). The response still fixes the
    /// scaling and the lambda calibration, so the prior sampled is the
    /// prior a fit on the same data would use, and the empty-cell
    /// rejection stays. Default false.
    pub prior_only: bool,
}

impl Default for GeneralParams {
    fn default() -> Self {
        Self {
            burn_in: 200,
            draws: 1000,
            thinning: 1,
            prior_only: false,
        }
    }
}

/// Configuration of an AddiVortes fit: the outcome model, one term group
/// per ensemble, and the sweep schedule (Stone and Gosling 2025, s. 2).
///
/// Every field has a default; unset JSON fields take it; unknown fields
/// are rejected. The seed is not part of the configuration; it is an
/// argument to [`fit`](crate::fit) and [`Sampler::new`](crate::Sampler::new).
///
/// Setters never panic or clamp; [`validate`](Config::validate) checks
/// every field in force and `fit` calls it first. The geometry and
/// structure setters write both slots, since the ensembles must declare
/// identical geometry while per-ensemble geometry awaits its
/// identification argument.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ConfigParts")]
#[non_exhaustive]
pub struct Config {
    /// The observation model, carrying its own parameters and its sigma^2
    /// settings.
    pub outcome: Outcome,
    /// The ensemble describing the average.
    pub mean_params: TermParams,
    /// The ensemble describing the spread; a resolved count of 0, the
    /// default, is a constant spread.
    pub variance_params: TermParams,
    /// Burn-in, draws, thinning, prior-only sampling.
    pub general_params: GeneralParams,
}

impl Config {
    /// The defaults ([`Default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// The paper's settings: the defaults with lambda_c = 25 on both
    /// slots (Stone and Gosling 2025, s. 2.3).
    pub fn paper() -> Self {
        Self::default().with_lambda_c(25.0)
    }

    /// The observation model.
    #[must_use]
    pub fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Mean-ensemble size m.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.mean_params.tessellations = Some(m);
        self
    }

    /// Variance-ensemble size m'; above 0 is H-AddiVortes.
    #[must_use]
    pub fn with_m_var(mut self, m_var: usize) -> Self {
        self.variance_params.tessellations = Some(m_var);
        self
    }

    /// sigma^2 prior degrees of freedom nu of an outcome with a sampled
    /// sigma^2; no effect under another outcome.
    #[must_use]
    pub fn with_nu(mut self, nu: f64) -> Self {
        match &mut self.outcome {
            Outcome::Gaussian(params) => params.nu = nu,
            #[cfg(feature = "experimental")]
            Outcome::Tobit(params) => params.nu = nu,
            #[cfg(feature = "experimental")]
            Outcome::Aft(params) => params.nu = nu,
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(params) => params.nu = nu,
            #[cfg(feature = "experimental")]
            Outcome::StudentT(params) => params.nu = nu,
            #[cfg(feature = "experimental")]
            Outcome::Laplace(params) => params.nu = nu,
            _ => {}
        }
        self
    }

    /// sigma^2 prior calibration quantile q of an outcome with a sampled
    /// sigma^2; no effect under another outcome.
    #[must_use]
    pub fn with_q(mut self, q: f64) -> Self {
        match &mut self.outcome {
            Outcome::Gaussian(params) => params.q = q,
            #[cfg(feature = "experimental")]
            Outcome::Tobit(params) => params.q = q,
            #[cfg(feature = "experimental")]
            Outcome::Aft(params) => params.q = q,
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(params) => params.q = q,
            #[cfg(feature = "experimental")]
            Outcome::StudentT(params) => params.q = q,
            #[cfg(feature = "experimental")]
            Outcome::Laplace(params) => params.q = q,
            _ => {}
        }
        self
    }

    /// The offset c of an outcome with one (the probit and ordinal
    /// models); no effect under another outcome.
    #[must_use]
    pub fn with_offset(mut self, offset: f64) -> Self {
        match &mut self.outcome {
            Outcome::Probit(params) => params.offset = Some(offset),
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(params) => params.offset = Some(offset),
            _ => {}
        }
        self
    }

    /// Cutpoint prior standard deviation of the ordinal outcome; no
    /// effect under another outcome. Experimental.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn with_cutpoint_sd(mut self, cutpoint_sd: f64) -> Self {
        if let Outcome::Ordinal(params) = &mut self.outcome {
            params.cutpoint_sd = cutpoint_sd;
        }
        self
    }

    /// Cell-value prior spread k, both slots.
    #[must_use]
    pub fn with_k(mut self, k: f64) -> Self {
        self.mean_params.k = k;
        self.variance_params.k = k;
        self
    }

    /// Cell-count prior rate lambda_c, both slots.
    #[must_use]
    pub fn with_lambda_c(mut self, lambda_c: f64) -> Self {
        self.mean_params.lambda_c = lambda_c;
        self.variance_params.lambda_c = lambda_c;
        self
    }

    /// Centre-coordinate standard deviation sigma_c, both slots.
    #[must_use]
    pub fn with_sigma_c(mut self, sigma_c: f64) -> Self {
        self.mean_params.geometry.sigma_c = sigma_c;
        self.variance_params.geometry.sigma_c = sigma_c;
        self
    }

    /// The metric of each covariate column, both slots.
    #[must_use]
    pub fn with_metric(mut self, metric: Vec<Metric>) -> Self {
        self.mean_params.geometry.metric.clone_from(&metric);
        self.variance_params.geometry.metric = metric;
        self
    }

    /// The Mahalanobis precision matrix, row-major p x p over the
    /// encoded design, both slots. Experimental.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn with_precision(mut self, precision: Vec<f64>) -> Self {
        self.mean_params.geometry.precision = Some(precision.clone());
        self.variance_params.geometry.precision = Some(precision);
        self
    }

    /// The membership rule, both slots. Experimental.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn with_membership(mut self, membership: Membership) -> Self {
        self.mean_params.geometry.membership = membership;
        self.variance_params.geometry.membership = membership;
        self
    }

    /// The covariate-inclusion prior, both slots. Experimental.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn with_inclusion(mut self, inclusion: Inclusion) -> Self {
        self.mean_params.structure.inclusion = inclusion.clone();
        self.variance_params.structure.inclusion = inclusion;
        self
    }

    /// The within-cell basis of the mean ensemble; the variance
    /// ensemble's cells keep the constant basis. Experimental.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn with_basis(mut self, basis: Basis) -> Self {
        self.mean_params.cell.basis = basis;
        self
    }

    /// Dimension-count prior parameter omega, both slots.
    #[must_use]
    pub fn with_omega(mut self, omega: f64) -> Self {
        self.mean_params.structure.omega = Some(omega);
        self.variance_params.structure.omega = Some(omega);
        self
    }

    /// Burn-in sweeps.
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.general_params.burn_in = burn_in;
        self
    }

    /// Posterior draws kept.
    #[must_use]
    pub fn with_draws(mut self, draws: usize) -> Self {
        self.general_params.draws = draws;
        self
    }

    /// Thinning interval.
    #[must_use]
    pub fn with_thinning(mut self, thinning: usize) -> Self {
        self.general_params.thinning = thinning;
        self
    }

    /// Prior-only sampling.
    #[must_use]
    pub fn with_prior_only(mut self, prior_only: bool) -> Self {
        self.general_params.prior_only = prior_only;
        self
    }

    /// The fitted model's name: the outcome's, or "heteroscedastic" for
    /// the Gaussian outcome with a variance ensemble attached (the paper
    /// names of the shipped models). An experimental outcome keeps its
    /// own name whether or not a variance ensemble is attached.
    ///
    /// # Panics
    ///
    /// On an outcome this build gates, which [`validate`](Self::validate)
    /// reports first.
    pub fn model_name(&self) -> &'static str {
        match (&self.outcome, self.variance_tessellations()) {
            (Outcome::Probit(_), _) => "probit",
            (Outcome::Gaussian(_), 0) => "gaussian",
            (Outcome::Gaussian(_), _) => "heteroscedastic",
            #[cfg(not(feature = "experimental"))]
            (Outcome::Gated(gated), _) => gated.unreachable(),
            #[cfg(feature = "experimental")]
            (Outcome::Tobit(_), _) => "tobit",
            #[cfg(feature = "experimental")]
            (Outcome::Aft(_), _) => "aft",
            #[cfg(feature = "experimental")]
            (Outcome::IntervalCensored(_), _) => "interval_censored",
            #[cfg(feature = "experimental")]
            (Outcome::Ordinal(_), _) => "ordinal",
            #[cfg(feature = "experimental")]
            (Outcome::StudentT(_), _) => "student_t",
            #[cfg(feature = "experimental")]
            (Outcome::Laplace(_), _) => "laplace",
        }
    }

    /// The mean-ensemble size in force: the field, or 200.
    pub fn mean_tessellations(&self) -> usize {
        self.mean_params.mean_tessellations()
    }

    /// The variance-ensemble size in force: the field, or 0.
    pub fn variance_tessellations(&self) -> usize {
        self.variance_params.variance_tessellations()
    }

    /// The offset c, where the outcome has one (the probit and ordinal
    /// models).
    pub fn offset(&self) -> Option<f64> {
        match &self.outcome {
            Outcome::Probit(params) => params.offset,
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(params) => params.offset,
            _ => None,
        }
    }

    /// The resolved offset, written back at fit.
    pub(crate) fn set_offset(&mut self, offset: f64) {
        match &mut self.outcome {
            Outcome::Probit(params) => params.offset = Some(offset),
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(params) => params.offset = Some(offset),
            _ => {}
        }
    }

    /// The sigma^2 prior (nu, q) of an outcome that samples one; the
    /// defaults where the outcome carries no sigma^2 prior, on paths
    /// that never read them.
    pub(crate) fn sigma2_prior(&self) -> (f64, f64) {
        match &self.outcome {
            Outcome::Gaussian(params) => (params.nu, params.q),
            #[cfg(not(feature = "experimental"))]
            Outcome::Gated(gated) => gated.unreachable(),
            #[cfg(feature = "experimental")]
            Outcome::Tobit(params) => (params.nu, params.q),
            #[cfg(feature = "experimental")]
            Outcome::Aft(params) => (params.nu, params.q),
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(params) => (params.nu, params.q),
            #[cfg(feature = "experimental")]
            Outcome::StudentT(params) => (params.nu, params.q),
            #[cfg(feature = "experimental")]
            Outcome::Laplace(params) => (params.nu, params.q),
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(_) => {
                let defaults = GaussianParams::default();
                (defaults.nu, defaults.q)
            }
            Outcome::Probit(_) => {
                let defaults = GaussianParams::default();
                (defaults.nu, defaults.q)
            }
        }
    }

    /// The omega in force for p covariates: the mean slot's field, or
    /// min(3, p).
    pub(crate) fn omega_for(&self, p: usize) -> f64 {
        self.mean_params
            .structure
            .omega
            .unwrap_or_else(|| 3.0_f64.min(p as f64))
    }

    /// The first value this build gates, if any: the outcome, then each
    /// term group's metric entries, membership, precision, inclusion
    /// prior and cell basis. `precision` has no published value, so any
    /// value is gated.
    #[cfg(not(feature = "experimental"))]
    fn gated(&self) -> Option<Gated> {
        fn in_term(params: &TermParams) -> Option<Gated> {
            params
                .geometry
                .metric
                .iter()
                .find_map(|metric| match metric {
                    crate::geometry::Metric::Gated(gated) => Some(*gated),
                    _ => None,
                })
                .or(match params.geometry.membership {
                    Membership::Gated(gated) => Some(gated),
                    Membership::Hard => None,
                })
                .or_else(|| {
                    params
                        .geometry
                        .precision
                        .is_some()
                        .then_some(Gated("`geometry.precision`"))
                })
                .or(match params.structure.inclusion {
                    Inclusion::Gated(gated) => Some(gated),
                    Inclusion::Uniform => None,
                })
                .or(match params.cell.basis {
                    Basis::Gated(gated) => Some(gated),
                    Basis::Constant => None,
                })
        }
        match self.outcome {
            Outcome::Gated(gated) => Some(gated),
            _ => in_term(&self.mean_params).or_else(|| in_term(&self.variance_params)),
        }
    }

    /// The resolved counts and omega, written back at fit.
    pub(crate) fn resolve(&mut self, omega: f64) {
        self.mean_params.tessellations = Some(self.mean_tessellations());
        self.variance_params.tessellations = Some(self.variance_tessellations());
        self.mean_params.structure.omega = Some(omega);
        self.variance_params.structure.omega = Some(omega);
    }

    /// Data-free validation of every field in force.
    ///
    /// # Errors
    ///
    /// `RequiresFeature` naming the first value this build gates;
    /// `InvalidHyperparameter` naming the field. The omega <= p check and
    /// the `metric` length check need the data and run at the fit
    /// boundary.
    pub fn validate(&self) -> Result<()> {
        #[cfg(not(feature = "experimental"))]
        if let Some(gated) = self.gated() {
            return Err(gated.requires_feature());
        }
        if self.mean_tessellations() < 1 {
            return Err(invalid("mean_params.tessellations", "must be at least 1"));
        }
        validate_term("mean_params", &self.mean_params)?;
        #[cfg(feature = "experimental")]
        if matches!(
            self.mean_params.geometry.membership,
            Membership::Soft { .. }
        ) && self.mean_params.cell.basis != Basis::Constant
        {
            return Err(invalid(
                "mean_params.cell.basis",
                "soft membership takes the constant basis; the weighted linear \
                 update is not derived",
            ));
        }
        match &self.outcome {
            #[cfg(feature = "experimental")]
            Outcome::Laplace(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
            }
            #[cfg(feature = "experimental")]
            Outcome::StudentT(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
                match &params.df {
                    DegreesOfFreedom::Fixed(df) => {
                        if !(df.is_finite() && *df > 0.0) {
                            return Err(invalid(
                                "df",
                                format!("must be finite and positive, got {df}"),
                            ));
                        }
                    }
                    DegreesOfFreedom::Grid(grid) => {
                        if grid.len() < 2 {
                            return Err(invalid(
                                "df",
                                "a grid needs at least two values; a single value is \
                                 the fixed form",
                            ));
                        }
                        if let Some(v) = grid.iter().find(|v| !(v.is_finite() && **v > 0.0)) {
                            return Err(invalid(
                                "df",
                                format!("grid values must be finite and positive, got {v}"),
                            ));
                        }
                        if grid.windows(2).any(|pair| pair[0] >= pair[1]) {
                            return Err(invalid("df", "grid values must be strictly increasing"));
                        }
                    }
                }
            }
            #[cfg(feature = "experimental")]
            Outcome::Aft(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
            }
            #[cfg(feature = "experimental")]
            Outcome::IntervalCensored(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
            }
            Outcome::Gaussian(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
            }
            Outcome::Probit(params) => {
                if let Some(c) = params.offset {
                    if !c.is_finite() {
                        return Err(invalid("offset", format!("must be finite, got {c}")));
                    }
                }
            }
            #[cfg(not(feature = "experimental"))]
            Outcome::Gated(gated) => gated.unreachable(),
            #[cfg(feature = "experimental")]
            Outcome::Ordinal(params) => {
                if params.categories < 2 {
                    return Err(invalid(
                        "categories",
                        format!("must be at least 2, got {}", params.categories),
                    ));
                }
                if let Some(c) = params.offset {
                    if !c.is_finite() {
                        return Err(invalid("offset", format!("must be finite, got {c}")));
                    }
                }
                if !(params.cutpoint_sd.is_finite() && params.cutpoint_sd > 0.0) {
                    return Err(invalid(
                        "cutpoint_sd",
                        format!("must be finite and positive, got {}", params.cutpoint_sd),
                    ));
                }
            }
            #[cfg(feature = "experimental")]
            Outcome::Tobit(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
                match (params.lower, params.upper) {
                    (None, None) => {
                        return Err(invalid(
                            "lower",
                            "the tobit outcome needs at least one censoring limit; \
                             the gaussian outcome is the uncensored model",
                        ));
                    }
                    (lower, upper) => {
                        for (name, limit) in [("lower", lower), ("upper", upper)] {
                            if let Some(v) = limit {
                                if !v.is_finite() {
                                    return Err(invalid(name, format!("must be finite, got {v}")));
                                }
                            }
                        }
                        if let (Some(lo), Some(hi)) = (lower, upper) {
                            if lo >= hi {
                                return Err(invalid(
                                    "lower",
                                    format!("must lie below upper, got {lo} and {hi}"),
                                ));
                            }
                        }
                    }
                }
            }
        }
        let m_var = self.variance_tessellations();
        if m_var > 0 {
            if !self.outcome.sigma2_mode().permits_variance_ensemble() {
                return Err(invalid(
                    "variance_params.tessellations",
                    "a variance ensemble needs a sampled sigma^2 to carry, and the \
                     probit and ordinal latent scales are fixed at 1 for identification",
                ));
            }
            #[cfg(feature = "experimental")]
            if matches!(self.outcome, Outcome::StudentT(_) | Outcome::Laplace(_)) {
                return Err(invalid(
                    "variance_params.tessellations",
                    "a scale-mixture outcome's weights (student_t, laplace) and a \
                     variance ensemble both model per-observation dispersion; the \
                     combination awaits its identification argument",
                ));
            }
            let (nu, _) = self.sigma2_prior();
            if nu <= 2.0 {
                return Err(invalid(
                    "nu",
                    format!("must exceed 2 under a variance ensemble, got {nu}"),
                ));
            }
            #[cfg(feature = "experimental")]
            if matches!(
                self.mean_params.geometry.membership,
                Membership::Soft { .. }
            ) {
                return Err(invalid(
                    "mean_params.geometry.membership",
                    "soft membership needs a constant spread; the inverse-gamma \
                     cells of a variance ensemble have no closed-form weighted \
                     conditional",
                ));
            }
            validate_term("variance_params", &self.variance_params)?;
            if self.variance_params.geometry != self.mean_params.geometry {
                return Err(invalid(
                    "variance_params.geometry",
                    "must equal mean_params.geometry; per-ensemble geometry awaits \
                     its identification argument",
                ));
            }
            if self.variance_params.structure != self.mean_params.structure {
                return Err(invalid(
                    "variance_params.structure",
                    "must equal mean_params.structure; per-ensemble structure awaits \
                     its identification argument",
                ));
            }
            #[cfg(feature = "experimental")]
            if self.variance_params.cell.basis != Basis::Constant {
                return Err(invalid(
                    "variance_params.cell.basis",
                    "the variance ensemble's inverse-gamma cells take the constant basis",
                ));
            }
        }
        if self.general_params.draws < 1 {
            return Err(invalid("draws", "must be at least 1"));
        }
        if self.general_params.thinning < 1 {
            return Err(invalid("thinning", "must be at least 1"));
        }
        Ok(())
    }
}

/// The deserialisation shape of [`Config`]: the four groups plus catchers
/// for the flat pre-reshape field names, so a saved flat configuration
/// fails with an error naming the replacement rather than an unknown-field
/// message.
#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigParts {
    outcome: Outcome,
    mean_params: TermParams,
    variance_params: TermParams,
    general_params: GeneralParams,
    #[serde(rename = "model")]
    legacy_model: Option<serde::de::IgnoredAny>,
    #[serde(rename = "m")]
    legacy_m: Option<serde::de::IgnoredAny>,
    #[serde(rename = "nu")]
    legacy_nu: Option<serde::de::IgnoredAny>,
    #[serde(rename = "q")]
    legacy_q: Option<serde::de::IgnoredAny>,
    #[serde(rename = "k")]
    legacy_k: Option<serde::de::IgnoredAny>,
    #[serde(rename = "sigma_c")]
    legacy_sigma_c: Option<serde::de::IgnoredAny>,
    #[serde(rename = "omega")]
    legacy_omega: Option<serde::de::IgnoredAny>,
    #[serde(rename = "lambda_c")]
    legacy_lambda_c: Option<serde::de::IgnoredAny>,
    #[serde(rename = "burn_in")]
    legacy_burn_in: Option<serde::de::IgnoredAny>,
    #[serde(rename = "draws")]
    legacy_draws: Option<serde::de::IgnoredAny>,
    #[serde(rename = "thinning")]
    legacy_thinning: Option<serde::de::IgnoredAny>,
    #[serde(rename = "prior_only")]
    legacy_prior_only: Option<serde::de::IgnoredAny>,
    #[serde(rename = "offset")]
    legacy_offset: Option<serde::de::IgnoredAny>,
    #[serde(rename = "m_var")]
    legacy_m_var: Option<serde::de::IgnoredAny>,
    #[serde(rename = "metric")]
    legacy_metric: Option<serde::de::IgnoredAny>,
}

impl TryFrom<ConfigParts> for Config {
    type Error = crate::error::Error;

    fn try_from(parts: ConfigParts) -> Result<Self> {
        let legacy = parts.legacy_model.is_some()
            || parts.legacy_m.is_some()
            || parts.legacy_nu.is_some()
            || parts.legacy_q.is_some()
            || parts.legacy_k.is_some()
            || parts.legacy_sigma_c.is_some()
            || parts.legacy_omega.is_some()
            || parts.legacy_lambda_c.is_some()
            || parts.legacy_burn_in.is_some()
            || parts.legacy_draws.is_some()
            || parts.legacy_thinning.is_some()
            || parts.legacy_prior_only.is_some()
            || parts.legacy_offset.is_some()
            || parts.legacy_m_var.is_some()
            || parts.legacy_metric.is_some();
        if legacy {
            return Err(invalid(
                "model",
                "the flat configuration is replaced by `outcome`, `mean_params`,                  `variance_params` and `general_params`; `model` is the `outcome`                  variant, with a variance ensemble as `variance_params.tessellations`",
            ));
        }
        Ok(Self {
            outcome: parts.outcome,
            mean_params: parts.mean_params,
            variance_params: parts.variance_params,
            general_params: parts.general_params,
        })
    }
}

fn positive(name: &str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(
            name,
            format!("must be finite and positive, got {value}"),
        ))
    }
}

fn validate_term(slot: &str, params: &TermParams) -> Result<()> {
    positive(&format!("{slot}.k"), params.k)?;
    positive(&format!("{slot}.lambda_c"), params.lambda_c)?;
    positive(&format!("{slot}.geometry.sigma_c"), params.geometry.sigma_c)?;
    #[cfg(feature = "experimental")]
    if let Membership::Soft { rate } = params.geometry.membership {
        positive(&format!("{slot}.geometry.membership.rate"), rate)?;
    }
    #[cfg(feature = "experimental")]
    match &params.structure.inclusion {
        Inclusion::Uniform => {}
        Inclusion::Weighted { weights } => {
            let name = format!("{slot}.structure.inclusion");
            if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
                return Err(invalid(&name, "weights must be finite and non-negative"));
            }
            if !weights.iter().any(|w| *w > 0.0) {
                return Err(invalid(&name, "at least one weight must be positive"));
            }
        }
        Inclusion::Dart { a, b, rho } => {
            positive(&format!("{slot}.structure.inclusion.a"), *a)?;
            positive(&format!("{slot}.structure.inclusion.b"), *b)?;
            if let Some(rho) = rho {
                positive(&format!("{slot}.structure.inclusion.rho"), *rho)?;
            }
        }
    }
    #[cfg(feature = "experimental")]
    for kind in &params.geometry.metric {
        if let Metric::Minkowski { p, .. } = *kind {
            if !(p.is_finite() && p >= 1.0) {
                return Err(invalid(
                    &format!("{slot}.geometry.metric"),
                    format!("the Minkowski order must be at least 1, got {p}"),
                ));
            }
        }
    }
    if let Some(omega) = params.structure.omega {
        positive(&format!("{slot}.structure.omega"), omega)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn rejects(config: Config, field: &str) {
        assert!(matches!(
            config.validate(),
            Err(Error::InvalidHyperparameter { ref name, .. }) if name == field
        ));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn a_minkowski_order_below_one_is_rejected() {
        for p in [0.5, f64::NAN, f64::INFINITY] {
            rejects(
                Config::new().with_metric(vec![Metric::Minkowski { p, group: 0 }]),
                "mean_params.geometry.metric",
            );
        }
        assert!(Config::new()
            .with_metric(vec![Metric::Minkowski { p: 1.0, group: 0 }])
            .validate()
            .is_ok());
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn inclusion_weights_are_validated() {
        for weights in [vec![1.0, -0.5], vec![0.0, 0.0], vec![1.0, f64::NAN]] {
            rejects(
                Config::new().with_inclusion(Inclusion::Weighted { weights }),
                "mean_params.structure.inclusion",
            );
        }
        assert!(Config::new()
            .with_inclusion(Inclusion::Weighted {
                weights: vec![1.0, 0.0]
            })
            .validate()
            .is_ok());
    }

    #[test]
    fn the_catalogue_names_each_family_once() {
        let names: Vec<String> = Outcome::catalogue()
            .iter()
            .map(|outcome| {
                let tagged = serde_json::to_value(outcome).unwrap();
                tagged.as_object().unwrap().keys().next().unwrap().clone()
            })
            .collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();

        assert_eq!(unique.len(), names.len(), "{names:?}");
        #[cfg(not(feature = "experimental"))]
        assert_eq!(names, ["gaussian", "probit"]);
        #[cfg(feature = "experimental")]
        assert_eq!(names.len(), 8, "{names:?}");
    }

    #[test]
    fn defaults_validate_and_resolve() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.mean_tessellations(), 200);
        assert_eq!(config.variance_tessellations(), 0);
        assert_eq!(config.model_name(), "gaussian");
        assert_eq!(Config::paper().mean_params.lambda_c, 25.0);
        assert_eq!(Config::paper().variance_params.lambda_c, 25.0);
    }

    #[test]
    fn every_field_in_force_is_checked() {
        rejects(Config::new().with_m(0), "mean_params.tessellations");
        rejects(Config::new().with_nu(0.0), "nu");
        rejects(Config::new().with_q(1.0), "q");
        rejects(Config::new().with_k(f64::NAN), "mean_params.k");
        rejects(
            Config::new().with_sigma_c(-1.0),
            "mean_params.geometry.sigma_c",
        );
        rejects(Config::new().with_omega(0.0), "mean_params.structure.omega");
        rejects(
            Config::new().with_lambda_c(f64::INFINITY),
            "mean_params.lambda_c",
        );
        rejects(Config::new().with_draws(0), "draws");
        rejects(Config::new().with_thinning(0), "thinning");
    }

    #[test]
    fn outcome_specific_fields_are_checked_only_under_their_outcome() {
        let probit = Config::new().with_outcome(Outcome::probit());
        rejects(probit.clone().with_offset(f64::NAN), "offset");
        assert!(probit.clone().with_offset(0.3).validate().is_ok());
        // Setters for another outcome's parameters have no effect.
        assert_eq!(probit.clone().with_nu(0.0), probit);
        assert_eq!(Config::new().with_offset(0.3), Config::new());
    }

    #[test]
    fn the_variance_ensemble_derives_its_validity_from_the_mode() {
        let hetero = Config::new().with_m_var(4);
        assert!(hetero.validate().is_ok());
        assert_eq!(hetero.model_name(), "heteroscedastic");
        rejects(hetero.clone().with_nu(2.0), "nu");
        rejects(
            Config::new().with_outcome(Outcome::probit()).with_m_var(4),
            "variance_params.tessellations",
        );
        let message = Config::new()
            .with_outcome(Outcome::probit())
            .with_m_var(4)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(message.contains("identification"), "{message}");
        // A zero-count variance slot is not validated.
        assert!(Config::new()
            .with_outcome(Outcome::probit())
            .with_m_var(0)
            .validate()
            .is_ok());
    }

    #[test]
    fn the_slots_must_share_geometry_and_structure_when_attached() {
        let mut config = Config::new().with_m_var(4);
        config.variance_params.geometry.sigma_c = 0.5;
        rejects(config, "variance_params.geometry");
        let mut config = Config::new().with_m_var(4);
        config.variance_params.structure.omega = Some(1.0);
        rejects(config, "variance_params.structure");
        let mut detached = Config::new();
        detached.variance_params.geometry.sigma_c = 0.5;
        assert!(detached.validate().is_ok());
    }

    #[test]
    fn a_flat_configuration_names_the_replacement() {
        let err = serde_json::from_str::<Config>(r#"{"model": "gaussian", "m": 5}"#).unwrap_err();
        assert!(err.to_string().contains("`outcome`"), "{err}");
        let err = serde_json::from_str::<Config>(r#"{"m_var": 3}"#).unwrap_err();
        assert!(err.to_string().contains("variance_params"), "{err}");
        assert!(serde_json::from_str::<Config>(r#"{"unheard_of": 1}"#)
            .unwrap_err()
            .to_string()
            .contains("unheard_of"));
    }

    #[test]
    fn serde_round_trip_partial_and_unknown_field() {
        let config = Config::new().with_m(20).with_omega(1.5);
        let json = serde_json::to_string(&config).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back, config);
        let partial: Config =
            serde_json::from_str(r#"{"mean_params": {"tessellations": 7}}"#).unwrap();
        assert_eq!(partial, Config::new().with_m(7));
        // A partially specified variance slot keeps the slot's resolution.
        let partial: Config = serde_json::from_str(r#"{"variance_params": {"k": 2.0}}"#).unwrap();
        assert_eq!(partial.variance_tessellations(), 0);
        assert!(serde_json::from_str::<Config>(r#"{"lambda_C": 5}"#).is_err());
        assert!(serde_json::from_str::<Config>(r#"{"mean_params": {"lambda_C": 5}}"#).is_err());
    }

    #[test]
    fn outcome_serialises_externally_tagged_in_snake_case() {
        let probit: Config =
            serde_json::from_str(r#"{"outcome": {"probit": {"offset": 0.3}}}"#).unwrap();
        assert_eq!(probit.offset(), Some(0.3));
        let json = serde_json::to_string(&probit).unwrap();
        assert!(
            json.contains(r#""outcome":{"probit":{"offset":0.3}}"#),
            "{json}"
        );
        let gaussian = serde_json::to_string(&Config::new()).unwrap();
        assert!(
            gaussian.contains(r#""outcome":{"gaussian":{"nu":6.0,"q":0.85}}"#),
            "{gaussian}"
        );
        assert!(serde_json::from_str::<Config>(r#"{"outcome": {"cauchy": {}}}"#).is_err());
    }

    #[test]
    fn metric_serialises_by_name() {
        let config =
            Config::new().with_metric(vec![Metric::Euclidean, Metric::Spherical { sphere: 1 }]);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains(r#""metric":["euclidean",{"spherical":{"sphere":1}}]"#));
        assert_eq!(serde_json::from_str::<Config>(&json).unwrap(), config);
    }

    #[test]
    fn omega_default_is_min_three_p() {
        let config = Config::default();
        assert_eq!(config.omega_for(1), 1.0);
        assert_eq!(config.omega_for(2), 2.0);
        assert_eq!(config.omega_for(10), 3.0);
        assert_eq!(config.with_omega(0.5).omega_for(10), 0.5);
    }

    #[test]
    fn resolve_writes_the_counts_and_omega_back() {
        let mut config = Config::new().with_m_var(4);
        config.resolve(2.5);
        assert_eq!(config.mean_params.tessellations, Some(200));
        assert_eq!(config.variance_params.tessellations, Some(4));
        assert_eq!(config.mean_params.structure.omega, Some(2.5));
        assert_eq!(config.variance_params.structure.omega, Some(2.5));
    }
}
