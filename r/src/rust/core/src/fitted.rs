//! The fitted model: the kept posterior draws, the scaling, and the
//! prediction surface, with the semantics of each method per outcome
//! model.

use crate::config::Config;
use crate::config::Outcome;
#[cfg(feature = "experimental")]
use crate::config::{Inclusion, Membership, StudentTParams};
use crate::data::{self, Data, Warning};
use crate::error::{Error, Result};
use crate::geometry::Geometry;
use crate::maths;
use crate::sampler::Sampler;
use crate::scaler::{Scaler, ScalerParts};
use crate::tessellation::{Tessellation, TessellationParts};
use crate::threads;

/// The kept posterior draws, scaled space: the m mean tessellations per
/// draw; sigma^2 per draw under the Gaussian model; the m' variance
/// tessellations per draw under the heteroscedastic model; the interior
/// cutpoints per draw under the ordinal model above two categories; the
/// error degrees of freedom per draw under the Student-t model with a
/// grid; the inclusion weights and their concentration per draw under
/// the DART inclusion prior.
///
/// The soft-membership bandwidth is not here: it belongs to a
/// tessellation and is kept on it, one per tessellation per draw.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "PosteriorParts")]
pub struct Posterior {
    sigma_sq: Vec<f64>,
    tessellations: Vec<Vec<Tessellation>>,
    variance_tessellations: Vec<Vec<Tessellation>>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    cutpoints: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    dfs: Vec<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    inclusion_weights: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    concentration: Vec<f64>,
}

impl Posterior {
    pub(crate) fn empty() -> Self {
        Self {
            sigma_sq: Vec::new(),
            tessellations: Vec::new(),
            variance_tessellations: Vec::new(),
            cutpoints: Vec::new(),
            dfs: Vec::new(),
            inclusion_weights: Vec::new(),
            concentration: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        sigma_sq: Option<f64>,
        tessellations: Vec<Tessellation>,
        variance_tessellations: Option<Vec<Tessellation>>,
        cutpoints: Option<Vec<f64>>,
        df: Option<f64>,
        inclusion: Option<(Vec<f64>, f64)>,
    ) {
        self.sigma_sq.extend(sigma_sq);
        self.tessellations.push(tessellations);
        self.variance_tessellations.extend(variance_tessellations);
        self.cutpoints.extend(cutpoints);
        self.dfs.extend(df);
        if let Some((weights, concentration)) = inclusion {
            self.inclusion_weights.push(weights);
            self.concentration.push(concentration);
        }
    }

    pub(crate) fn push_state(&mut self, state: crate::state::EnsembleState) {
        let (tessellations, variance, sigma_sq, cutpoints, df, inclusion) = state.into_parts();
        self.push(
            sigma_sq,
            tessellations,
            (!variance.is_empty()).then_some(variance),
            (!cutpoints.is_empty()).then_some(cutpoints),
            df,
            inclusion,
        );
    }

    /// The state of kept draw `draw`.
    #[cfg_attr(not(feature = "experimental"), allow(dead_code))]
    pub(crate) fn state(&self, draw: usize) -> crate::state::EnsembleState {
        crate::state::EnsembleState::new(
            self.tessellations[draw].clone(),
            self.variance_tessellations
                .get(draw)
                .cloned()
                .unwrap_or_default(),
            self.sigma_sq.get(draw).copied(),
            self.cutpoints.get(draw).cloned().unwrap_or_default(),
            self.dfs.get(draw).copied(),
            self.inclusion_weights
                .get(draw)
                .map(|s| (s.clone(), self.concentration[draw])),
        )
    }

    /// Number of kept draws.
    pub fn n_draws(&self) -> usize {
        self.tessellations.len()
    }

    /// sigma^2 per draw, scaled space; empty outside the Gaussian model.
    pub fn sigma_sq(&self) -> &[f64] {
        &self.sigma_sq
    }

    /// The m mean tessellations of each draw.
    pub fn tessellations(&self) -> &[Vec<Tessellation>] {
        &self.tessellations
    }

    /// The m' variance tessellations of each draw; empty outside the
    /// heteroscedastic model.
    pub fn variance_tessellations(&self) -> &[Vec<Tessellation>] {
        &self.variance_tessellations
    }

    /// The interior cutpoints of each draw, increasing; empty outside
    /// the ordinal model and at two categories, where none is sampled.
    pub fn cutpoints(&self) -> &[Vec<f64>] {
        &self.cutpoints
    }

    /// The error degrees of freedom of each draw; empty outside the
    /// Student-t model with a grid, where none is sampled.
    pub fn dfs(&self) -> &[f64] {
        &self.dfs
    }

    /// The sampled inclusion weight of each covariate, in column order,
    /// per draw; empty outside the DART inclusion prior. Each draw sums
    /// to one.
    pub fn inclusion_weights(&self) -> &[Vec<f64>] {
        &self.inclusion_weights
    }

    /// The Dirichlet concentration theta of each draw; empty outside the
    /// DART inclusion prior.
    pub fn concentration(&self) -> &[f64] {
        &self.concentration
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        self.sigma_sq.extend_from_slice(&other.sigma_sq);
        self.tessellations.extend_from_slice(&other.tessellations);
        self.variance_tessellations
            .extend_from_slice(&other.variance_tessellations);
        self.cutpoints.extend_from_slice(&other.cutpoints);
        self.dfs.extend_from_slice(&other.dfs);
        self.inclusion_weights
            .extend_from_slice(&other.inclusion_weights);
        self.concentration.extend_from_slice(&other.concentration);
    }
}

#[derive(serde::Deserialize)]
struct PosteriorParts {
    sigma_sq: Vec<f64>,
    tessellations: Vec<Vec<TessellationParts>>,
    #[serde(default)]
    variance_tessellations: Vec<Vec<TessellationParts>>,
    #[serde(default)]
    cutpoints: Vec<Vec<f64>>,
    #[serde(default)]
    dfs: Vec<f64>,
    #[serde(default)]
    inclusion_weights: Vec<Vec<f64>>,
    #[serde(default)]
    concentration: Vec<f64>,
}

impl TryFrom<PosteriorParts> for Posterior {
    type Error = Error;

    fn try_from(parts: PosteriorParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        let PosteriorParts {
            sigma_sq,
            tessellations,
            variance_tessellations,
            cutpoints,
            dfs,
            inclusion_weights,
            concentration,
        } = parts;
        let tessellations = tessellation_draws(tessellations)?;
        let variance_tessellations = tessellation_draws(variance_tessellations)?;
        let n_draws = tessellations.len();
        if n_draws == 0 {
            return Err(bad("posterior needs at least one draw"));
        }
        if !(sigma_sq.is_empty() || sigma_sq.len() == n_draws) {
            return Err(bad("sigma^2 draws must be absent or one per draw"));
        }
        if sigma_sq.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(bad("sigma^2 draws must be finite and positive"));
        }
        let m = tessellations[0].len();
        if m == 0 || tessellations.iter().any(|d| d.len() != m) {
            return Err(bad(
                "every draw must hold the same positive number of tessellations",
            ));
        }
        if !(variance_tessellations.is_empty() || variance_tessellations.len() == n_draws) {
            return Err(bad(
                "variance tessellations must be absent or one set per draw",
            ));
        }
        if let Some(first) = variance_tessellations.first() {
            let m_var = first.len();
            if m_var == 0 || variance_tessellations.iter().any(|d| d.len() != m_var) {
                return Err(bad(
                    "every draw must hold the same positive number of variance tessellations",
                ));
            }
            if variance_tessellations
                .iter()
                .flatten()
                .any(|t| t.mus().iter().any(|v| *v <= 0.0))
            {
                return Err(bad("variance cell values must be positive"));
            }
        }
        if !(cutpoints.is_empty() || cutpoints.len() == n_draws) {
            return Err(bad("cutpoints must be absent or one set per draw"));
        }
        if let Some(first) = cutpoints.first() {
            let width = first.len();
            if width == 0 || cutpoints.iter().any(|c| c.len() != width) {
                return Err(bad(
                    "every draw must hold the same positive number of cutpoints",
                ));
            }
            for draw in &cutpoints {
                let mut previous = 0.0;
                for &g in draw {
                    if !(g.is_finite() && g > previous) {
                        return Err(bad("cutpoints must be finite, positive and increasing"));
                    }
                    previous = g;
                }
            }
        }
        if !(dfs.is_empty() || dfs.len() == n_draws) {
            return Err(bad(
                "degrees-of-freedom draws must be absent or one per draw",
            ));
        }
        if dfs.iter().any(|df| !(df.is_finite() && *df > 0.0)) {
            return Err(bad("degrees-of-freedom draws must be finite and positive"));
        }
        if !(inclusion_weights.is_empty() || inclusion_weights.len() == n_draws) {
            return Err(bad("inclusion weights must be absent or one set per draw"));
        }
        if concentration.len() != inclusion_weights.len() {
            return Err(bad(
                "inclusion weights and their concentration must be kept together",
            ));
        }
        if let Some(first) = inclusion_weights.first() {
            let p = first.len();
            if p == 0 || inclusion_weights.iter().any(|w| w.len() != p) {
                return Err(bad(
                    "every draw must hold the same positive number of inclusion weights",
                ));
            }
            for draw in &inclusion_weights {
                if draw.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
                    return Err(bad("inclusion weights must be finite and non-negative"));
                }
                if (draw.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
                    return Err(bad("inclusion weights must sum to one"));
                }
            }
        }
        if concentration
            .iter()
            .any(|theta| !(theta.is_finite() && *theta > 0.0))
        {
            return Err(bad("concentration draws must be finite and positive"));
        }
        Ok(Self {
            sigma_sq,
            tessellations,
            variance_tessellations,
            cutpoints,
            dfs,
            inclusion_weights,
            concentration,
        })
    }
}

/// Each draw's tessellations from their saved parts.
fn tessellation_draws(draws: Vec<Vec<TessellationParts>>) -> Result<Vec<Vec<Tessellation>>> {
    draws
        .into_iter()
        .map(|draw| draw.into_iter().map(Tessellation::try_from).collect())
        .collect()
}

/// A fitted model: the configuration, the scaling, the kept draws and the
/// fit-time warnings. Serialises through serde; loading validates the
/// payload against the model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "FittedParts")]
pub struct Fitted {
    config: Config,
    scaler: Scaler,
    posterior: Posterior,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
    /// Levels of each categorical column; empty for the other columns.
    categories: Vec<Vec<f64>>,
    /// The draws each chain kept, in pooled order; empty for a save from
    /// before fits kept the count per chain.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    chain_draws: Vec<usize>,
    /// The thread count of the predictions; an execution setting, not
    /// part of the model, so not persisted.
    #[serde(skip)]
    threads: usize,
}

impl PartialEq for Fitted {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.scaler == other.scaler
            && self.posterior == other.posterior
            && self.warnings == other.warnings
            && self.in_sample_rmse == other.in_sample_rmse
            && self.categories == other.categories
            && self.chain_draws == other.chain_draws
    }
}

#[derive(serde::Deserialize)]
struct FittedParts {
    config: Config,
    scaler: ScalerParts,
    posterior: PosteriorParts,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
    #[serde(default)]
    categories: Vec<Vec<f64>>,
    #[serde(default)]
    chain_draws: Vec<usize>,
}

impl TryFrom<FittedParts> for Fitted {
    type Error = Error;

    fn try_from(parts: FittedParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        let FittedParts {
            config,
            scaler,
            posterior,
            warnings,
            in_sample_rmse,
            categories,
            chain_draws,
        } = parts;
        config.validate()?;
        let scaler = Scaler::try_from(scaler)?;
        let posterior = Posterior::try_from(posterior)?;
        let p = scaler.n_cols();
        // A save from before categorical levels were stored carries none.
        let categories = if categories.is_empty() {
            vec![Vec::new(); p]
        } else {
            categories
        };
        let geometry =
            Geometry::with_categories(&config.mean_params.geometry.metric, p, &categories)?;
        #[cfg(feature = "experimental")]
        geometry.with_precision(config.mean_params.geometry.precision.as_deref())?;
        #[cfg(not(feature = "experimental"))]
        drop(geometry);
        let uses_covariates = |draws: &[Vec<Tessellation>]| {
            draws
                .iter()
                .flatten()
                .all(|t| t.dims().iter().all(|&d| d < p))
        };
        for draw in posterior.tessellations() {
            if draw.len() != config.mean_tessellations() {
                return Err(bad("draws do not hold m tessellations"));
            }
        }
        if !uses_covariates(posterior.tessellations())
            || !uses_covariates(posterior.variance_tessellations())
        {
            return Err(bad(
                "a tessellation uses a covariate the scaler does not have",
            ));
        }
        let n_draws = posterior.n_draws();
        let has_ensemble = config.variance_tessellations() > 0;
        let has_global_sigma_sq =
            config.outcome.sigma2_mode().samples_global_sigma_sq() && !has_ensemble;
        if (posterior.sigma_sq().len() == n_draws) != has_global_sigma_sq {
            return Err(bad(
                "sigma^2 draws are present exactly where the scale is sampled globally",
            ));
        }
        #[cfg(feature = "experimental")]
        let expects_cutpoints = matches!(
            &config.outcome,
            Outcome::Ordinal(params) if params.categories > 2
        );
        #[cfg(not(feature = "experimental"))]
        let expects_cutpoints = false;
        if (posterior.cutpoints().len() == n_draws) != expects_cutpoints {
            return Err(bad(
                "cutpoint draws are present exactly under the ordinal model above two categories",
            ));
        }
        #[cfg(feature = "experimental")]
        if let Outcome::Ordinal(params) = &config.outcome {
            if params.categories > 2
                && posterior
                    .cutpoints()
                    .iter()
                    .any(|draw| draw.len() != params.categories - 2)
            {
                return Err(bad("each draw holds the K - 2 interior cutpoints"));
            }
        }
        #[cfg(feature = "experimental")]
        let expects_dfs = matches!(
            &config.outcome,
            Outcome::StudentT(params) if !params.df.grid().is_empty()
        );
        #[cfg(not(feature = "experimental"))]
        let expects_dfs = false;
        if (posterior.dfs().len() == n_draws) != expects_dfs {
            return Err(bad(
                "degrees-of-freedom draws are present exactly under the student_t \
                 model with a grid",
            ));
        }
        #[cfg(feature = "experimental")]
        let expects_inclusion = matches!(
            config.mean_params.structure.inclusion,
            Inclusion::Dart { .. }
        );
        #[cfg(not(feature = "experimental"))]
        let expects_inclusion = false;
        if (posterior.inclusion_weights().len() == n_draws) != expects_inclusion {
            return Err(bad(
                "inclusion-weight draws are present exactly under the dart inclusion prior",
            ));
        }
        if posterior
            .inclusion_weights()
            .iter()
            .any(|draw| draw.len() != p)
        {
            return Err(bad("each draw holds one inclusion weight per covariate"));
        }
        #[cfg(feature = "experimental")]
        let expects_bandwidth = matches!(
            config.mean_params.geometry.membership,
            Membership::Soft { .. }
        );
        #[cfg(not(feature = "experimental"))]
        let expects_bandwidth = false;
        if posterior
            .tessellations()
            .iter()
            .flatten()
            .any(|t| t.bandwidth().is_some() != expects_bandwidth)
        {
            return Err(bad(
                "a bandwidth is kept on every tessellation exactly under soft membership",
            ));
        }
        if (posterior.variance_tessellations().len() == n_draws) != has_ensemble {
            return Err(bad(
                "variance tessellations are present exactly under a variance ensemble",
            ));
        }
        if has_ensemble
            && posterior.variance_tessellations()[0].len() != config.variance_tessellations()
        {
            return Err(bad(
                "draws do not hold the variance-ensemble tessellation count",
            ));
        }
        if matches!(config.outcome, Outcome::Probit(_)) {
            match config.offset() {
                Some(c) if c.is_finite() => {}
                _ => return Err(bad("a probit fit carries a finite offset")),
            }
            if scaler.y_range() != 1.0 || scaler.y_min() != -0.5 {
                return Err(bad("a probit fit leaves the response unscaled"));
            }
        }
        if !in_sample_rmse.is_finite() {
            return Err(bad("in-sample RMSE must be finite"));
        }
        if !chain_draws.is_empty()
            && (chain_draws.contains(&0) || chain_draws.iter().sum::<usize>() != n_draws)
        {
            return Err(bad("the chains' draw counts do not add up to the draws"));
        }
        Ok(Self {
            config,
            scaler,
            posterior,
            warnings,
            in_sample_rmse,
            categories,
            chain_draws,
            threads: 1,
        })
    }
}

/// The central interval [`Fitted::predict_with_interval`] returns beside
/// the posterior mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalKind {
    /// The credible interval of the mean function
    /// ([`Fitted::credible_interval`]).
    Credible,
    /// The posterior predictive interval for a new observation
    /// ([`Fitted::prediction_interval`]).
    Prediction,
}

/// A central credible interval for the mean function at one row.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Interval {
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
}

impl Fitted {
    pub(crate) fn new(
        config: Config,
        scaler: Scaler,
        posterior: Posterior,
        warnings: Vec<Warning>,
        in_sample_rmse: f64,
        categories: Vec<Vec<f64>>,
    ) -> Self {
        let chain_draws = vec![posterior.n_draws()];
        Self {
            config,
            scaler,
            posterior,
            warnings,
            in_sample_rmse,
            categories,
            chain_draws,
            threads: 1,
        }
    }

    /// The model of a saved payload, read through `deserializer` in
    /// whatever format it speaks. The [`Deserialize`](serde::Deserialize)
    /// impl runs the same checks but reports every failure through the
    /// format's error type, as text; here the payload's shape is the
    /// format's to report and each invariant of a saved model keeps its
    /// own error.
    ///
    /// # Errors
    ///
    /// `InvalidSavedModel` for a payload that does not parse or breaks an
    /// invariant; [`Config::validate`]'s errors for its configuration,
    /// `RequiresFeature` among them in a build without the feature the
    /// fit used.
    pub fn load<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self> {
        let parts =
            <FittedParts as serde::Deserialize>::deserialize(deserializer).map_err(|error| {
                Error::InvalidSavedModel {
                    reason: error.to_string(),
                }
            })?;
        Self::try_from(parts)
    }

    /// The number of threads the predictions run on: the rows of a design
    /// are split into as many contiguous chunks, each evaluated on a
    /// thread of its own, never more than the parallelism available to
    /// the process. One unless [`set_threads`](Self::set_threads) or
    /// [`fit_chains_with_threads`](crate::fit_chains_with_threads) set it;
    /// not persisted with the model. The predicted values do not depend on
    /// it: every row is evaluated by the same operations in the same
    /// order whichever chunk holds it.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Set the thread count of the predictions; zero counts as one.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// Run `fill` over the rows of `x` in at most [`threads`](Self::threads)
    /// contiguous chunks, each on a thread of its own, with `out[d]`
    /// split at the chunk boundaries: `fill(chunk, pieces)` receives the
    /// chunk's rows and, per draw, its slice of `out[d]`.
    fn over_row_chunks(
        &self,
        x: &Data,
        out: &mut [Vec<f64>],
        fill: impl Fn(&Data, &mut [&mut [f64]]) + Sync,
    ) {
        let n = x.n_rows();
        let threads = self.threads.clamp(1, n.max(1));
        if threads == 1 {
            let mut pieces: Vec<&mut [f64]> = out.iter_mut().map(Vec::as_mut_slice).collect();
            fill(x, &mut pieces);
            return;
        }
        let per = n.div_ceil(threads);
        let chunks: Vec<Data> = (0..n)
            .step_by(per)
            .map(|start| x.rows(start..(start + per).min(n)))
            .collect();
        let mut pieces: Vec<Vec<&mut [f64]>> = chunks
            .iter()
            .map(|_| Vec::with_capacity(out.len()))
            .collect();
        for row in out.iter_mut() {
            let mut rest = row.as_mut_slice();
            for (chunk, piece) in chunks.iter().zip(&mut pieces) {
                let (head, tail) = rest.split_at_mut(chunk.n_rows());
                piece.push(head);
                rest = tail;
            }
        }
        let fill = &fill;
        std::thread::scope(|scope| {
            for (chunk, mut piece) in chunks.iter().zip(pieces) {
                scope.spawn(move || fill(chunk, &mut piece));
            }
        });
    }

    /// The fitted model's name: "gaussian", "probit", or "heteroscedastic"
    /// for the Gaussian outcome with its variance ensemble attached.
    pub fn model_name(&self) -> &'static str {
        self.config.model_name()
    }

    /// The observation model the draws were fitted under.
    pub fn outcome(&self) -> &Outcome {
        &self.config.outcome
    }

    /// Whether the spread is the variance ensemble's product s^2(x).
    pub fn has_variance_ensemble(&self) -> bool {
        self.config.variance_tessellations() > 0
    }

    /// The kept draws of `fits`, in chain order, as one fitted model.
    ///
    /// # Arguments
    ///
    /// `fits` are chains of the same model fitted to `x` and `y`, keyed by
    /// [`chain_seed`](crate::chain_seed). The in-sample RMSE is that of the
    /// pooled posterior mean.
    ///
    /// # Errors
    ///
    /// `MismatchedChains` for an empty `fits` or chains whose
    /// configuration, scaling or category levels differ;
    /// `RowCountMismatch`; the [`predict`](Self::predict) errors.
    pub fn pool(fits: &[Self], x: &Data, y: &[f64]) -> Result<Self> {
        Self::pooled(fits, x, y, None).map(|(fitted, _)| fitted)
    }

    /// The kept draws of `samplers`, in chain order, as one fitted model,
    /// with the posterior mean at the rows of `x`: the fitted values.
    ///
    /// # Arguments
    ///
    /// `samplers` are chains of the same model constructed on `x` and
    /// `y`, keyed by [`chain_seed`](crate::chain_seed), each with at least
    /// one kept draw. The fitted values and the in-sample RMSE come from
    /// the sums each sampler accumulated at `keep`, so no prediction pass
    /// runs: for one chain the values equal [`pool`](Self::pool) of its
    /// [`Sampler::finish`] result bit for bit; for several chains the
    /// per-chain sums are added before the division, which can differ
    /// from the pooled pass in the last bit. A chain that kept a soft
    /// draw has no sums, and the pooled model then predicts once over
    /// `x`.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for `draws` when a chain kept nothing; the
    /// [`pool`](Self::pool) errors.
    pub fn pool_samplers(samplers: Vec<Sampler>, x: &Data, y: &[f64]) -> Result<(Self, Vec<f64>)> {
        let n_draws: usize = samplers.iter().map(Sampler::n_kept).sum();
        let mean = samplers
            .iter()
            .try_fold(vec![0.0; x.n_rows()], |mut acc, sampler| {
                let sum = sampler.fit_sum_response()?;
                if sum.len() != acc.len() {
                    return None;
                }
                for (a, v) in acc.iter_mut().zip(sum) {
                    *a += v;
                }
                Some(acc)
            })
            .map(|acc| {
                acc.into_iter()
                    .map(|sum| sum / n_draws as f64)
                    .collect::<Vec<f64>>()
            });
        let chains = samplers
            .into_iter()
            .map(|sampler| sampler.into_fitted(0.0))
            .collect::<Result<Vec<_>>>()?;
        Self::pooled(&chains, x, y, mean)
    }

    /// The pooled model and the posterior mean at the rows of `x`, from
    /// which its in-sample RMSE is taken: `mean` when the caller has it,
    /// otherwise one prediction pass.
    fn pooled(
        fits: &[Self],
        x: &Data,
        y: &[f64],
        mean: Option<Vec<f64>>,
    ) -> Result<(Self, Vec<f64>)> {
        let mismatch = |reason: &str| Error::MismatchedChains {
            reason: reason.into(),
        };
        let (first, rest) = fits.split_first().ok_or_else(|| mismatch("no chains"))?;
        for fit in rest {
            if fit.config != first.config {
                return Err(mismatch("configurations differ"));
            }
            if fit.scaler != first.scaler || fit.categories != first.categories {
                return Err(mismatch("chains were fitted to different data"));
            }
        }
        if y.len() != x.n_rows() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: x.n_rows(),
            });
        }
        let mut posterior = Posterior::empty();
        let mut warnings: Vec<Warning> = Vec::new();
        let mut chain_draws = Vec::new();
        for fit in fits {
            posterior.extend(&fit.posterior);
            if fit.chain_draws.is_empty() {
                chain_draws.push(fit.posterior.n_draws());
            } else {
                chain_draws.extend_from_slice(&fit.chain_draws);
            }
            for warning in &fit.warnings {
                if !warnings.contains(warning) {
                    warnings.push(*warning);
                }
            }
        }
        let mut pooled = Self::new(
            first.config.clone(),
            first.scaler.clone(),
            posterior,
            warnings,
            0.0,
            first.categories.clone(),
        );
        pooled.chain_draws = chain_draws;
        let mean = match mean {
            Some(mean) => {
                data::validate_predict(x, pooled.scaler.n_cols())?;
                mean
            }
            None => pooled.predict(x)?,
        };
        let n = y.len() as f64;
        pooled.in_sample_rmse = (mean
            .iter()
            .zip(y)
            .map(|(f, y)| (f - y) * (f - y))
            .sum::<f64>()
            / n)
            .sqrt();
        Ok((pooled, mean))
    }

    fn not_applicable(&self, method: &str) -> Error {
        Error::NotApplicable {
            method: method.into(),
            model: self.config.model_name().into(),
        }
    }

    /// Posterior mean at each row of `x`, caller scale: of f(x), or of
    /// P(y = 1 | x) = Phi(c + f(x)) under the probit model. Under the
    /// tobit and interval-censored models the quantity is the uncensored
    /// f(x), the latent mean; under the AFT model it is f(x) on the
    /// log-time scale (the BART package's `yhat` convention for `abart`);
    /// under the ordinal model it is the expected category
    /// E[y | x] = sum_{k >= 1} Phi(c + f(x) - gamma_k).
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict(&self, x: &Data) -> Result<Vec<f64>> {
        Ok(column_means(&self.predict_draws(x)?, x.n_rows()))
    }

    /// The quantity of [`predict`](Self::predict) at each row of `x` for
    /// every kept draw, draw-major (`n_draws` by `n_rows`), caller scale.
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict_draws(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        let mut latent = self.predict_latent(x)?;
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            for draw in &mut latent {
                for v in draw {
                    *v = maths::normal_cdf(*v);
                }
            }
        }
        // The expected category sum_{k >= 1} Phi(c + f - gamma_k) with
        // the draw's own cutpoints, gamma_1 = 0.
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Ordinal(_)) {
            for (d, draw) in latent.iter_mut().enumerate() {
                let free = self
                    .posterior
                    .cutpoints()
                    .get(d)
                    .map_or(&[][..], Vec::as_slice);
                for v in draw {
                    let l = *v;
                    *v = maths::normal_cdf(l)
                        + free.iter().map(|&g| maths::normal_cdf(l - g)).sum::<f64>();
                }
            }
        }
        Ok(latent)
    }

    /// The mean function at each row of `x` for every kept draw, draw-major
    /// (`n_draws` by `n_rows`), caller scale: f(x), or the latent mean
    /// c + f(x) under the probit model.
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict_latent(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        data::validate_predict(x, self.scaler.n_cols())?;
        let geometry = self.geometry()?;
        geometry.check_codes(x)?;
        let x_scaled = self.scaler.scale_x(x);
        let n = x.n_rows();
        let offset = self.offset();
        let draws = self.posterior.tessellations();
        let mut out = vec![vec![0.0; n]; draws.len()];
        self.over_row_chunks(&x_scaled, &mut out, |chunk, pieces| {
            let mut keys = Vec::new();
            for (draw, sums) in draws.iter().zip(pieces.iter_mut()) {
                for t in draw {
                    t.for_each_value(chunk, &geometry, &mut keys, |i, v| sums[i] += v);
                }
                for sum in sums.iter_mut() {
                    *sum = self.scaler.unscale_y(*sum) + offset;
                }
            }
        });
        Ok(out)
    }

    /// The column structure of the fit, from the configuration and the
    /// stored categorical levels.
    fn geometry(&self) -> Result<Geometry> {
        let geometry = Geometry::with_categories(
            &self.config.mean_params.geometry.metric,
            self.scaler.n_cols(),
            &self.categories,
        )?;
        #[cfg(feature = "experimental")]
        let geometry =
            geometry.with_precision(self.config.mean_params.geometry.precision.as_deref())?;
        Ok(geometry)
    }

    /// The variance of y given f at each row of `x` for every kept draw,
    /// draw-major (`n_draws` by `n_rows`), caller scale: sigma_d^2 under
    /// the Gaussian model (constant across rows); s_d^2(x), the product of
    /// the variance tessellations, under the heteroscedastic model (the
    /// square of `rbart`'s `sdraws`); the error variance
    /// sigma_d^2 df_d / (df_d - 2) under the Student-t model and
    /// 2 sigma_d^2 under the Laplace model.
    ///
    /// # Errors
    ///
    /// `NotApplicable` under the probit model, and under the Student-t
    /// model where the configuration admits df <= 2, whose t has no
    /// variance; `FeatureCountMismatch`, `NonFiniteFeature`,
    /// `InvalidCategoryCode`.
    pub fn predict_variance(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        data::validate_predict(x, self.scaler.n_cols())?;
        let n = x.n_rows();
        let range_sq = self.scaler.y_range() * self.scaler.y_range();
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            return Err(self.not_applicable("predict_variance"));
        }
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Ordinal(_)) {
            return Err(self.not_applicable("predict_variance"));
        }
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Laplace(_)) {
            return Ok(self
                .posterior
                .sigma_sq()
                .iter()
                .map(|s| vec![2.0 * s * range_sq; n])
                .collect());
        }
        #[cfg(feature = "experimental")]
        if let Outcome::StudentT(params) = &self.config.outcome {
            if params.df.minimum() <= 2.0 {
                return Err(self.not_applicable("predict_variance"));
            }
            return Ok(self
                .posterior
                .sigma_sq()
                .iter()
                .enumerate()
                .map(|(d, s)| {
                    let df = self.student_df(params, d);
                    vec![s * range_sq * df / (df - 2.0); n]
                })
                .collect());
        }
        if self.has_variance_ensemble() {
            let geometry = self.geometry()?;
            geometry.check_codes(x)?;
            let x_scaled = self.scaler.scale_x(x);
            let draws = self.posterior.variance_tessellations();
            let mut out = vec![vec![1.0; n]; draws.len()];
            self.over_row_chunks(&x_scaled, &mut out, |chunk, pieces| {
                let mut keys = Vec::new();
                for (draw, products) in draws.iter().zip(pieces.iter_mut()) {
                    for t in draw {
                        t.for_each_value(chunk, &geometry, &mut keys, |i, v| products[i] *= v);
                    }
                    for p in products.iter_mut() {
                        *p *= range_sq;
                    }
                }
            });
            return Ok(out);
        }
        Ok(self
            .posterior
            .sigma_sq()
            .iter()
            .map(|s| vec![s * range_sq; n])
            .collect())
    }

    /// Posterior quantiles of the quantity of [`predict`](Self::predict) at
    /// each row of `x` for each probability in `probs`, row-major
    /// (`n_rows` by `probs.len()`), caller scale; type 7 interpolation over
    /// the kept draws.
    ///
    /// # Errors
    ///
    /// `InvalidProbability` for a probability outside (0, 1) or an empty
    /// `probs`; the predict errors.
    pub fn predict_quantiles(&self, x: &Data, probs: &[f64]) -> Result<Vec<f64>> {
        check_probabilities(probs)?;
        let per_draw = self.predict_draws(x)?;
        Ok(quantiles_from_draws(
            &per_draw,
            x.n_rows(),
            probs,
            self.threads,
        ))
    }

    /// Central credible interval for the quantity of
    /// [`predict`](Self::predict) at each row of `x` at `level` (the
    /// (1 - level) / 2 and (1 + level) / 2 posterior quantiles); on the
    /// probability scale under the probit model.
    ///
    /// # Errors
    ///
    /// `InvalidProbability` for `level` outside (0, 1); the predict errors.
    pub fn credible_interval(&self, x: &Data, level: f64) -> Result<Vec<Interval>> {
        check_probability(level)?;
        let per_draw = self.predict_draws(x)?;
        Ok(credible_from_draws(
            &per_draw,
            x.n_rows(),
            level,
            self.threads,
        ))
    }

    /// Central posterior predictive interval for a new observation at each
    /// row of `x` at `level`: the quantiles of the equal-weight mixture over
    /// kept draws of N(f_d(x), s_d^2(x)), found by bisection on the mixture
    /// CDF. Under the tobit model the predictive is censored, so the ends
    /// are clamped to the limits (censoring is monotone, which makes the
    /// clamp the exact quantile). Under the Student-t model the mixture
    /// components are f_d(x) + sigma_d t_{df_d}, and under the Laplace
    /// model Laplace(f_d(x), sigma_d), the model's own predictive.
    ///
    /// # Errors
    ///
    /// `NotApplicable` under the probit model, which has no continuous
    /// predictive distribution; `InvalidProbability` for `level` outside
    /// (0, 1); the predict errors.
    pub fn prediction_interval(&self, x: &Data, level: f64) -> Result<Vec<Interval>> {
        check_probability(level)?;
        self.prediction_interval_applies()?;
        let per_draw = self.predict_draws(x)?;
        self.prediction_from_draws(x, &per_draw, level)
    }

    /// The posterior mean of [`predict`](Self::predict) and the central
    /// `kind` interval at `level` at each row of `x`, from one traversal of
    /// the kept draws: the values [`predict`](Self::predict) and the
    /// interval method return when called separately.
    ///
    /// # Errors
    ///
    /// Those of [`predict`](Self::predict), [`credible_interval`](Self::credible_interval)
    /// and [`prediction_interval`](Self::prediction_interval).
    pub fn predict_with_interval(
        &self,
        x: &Data,
        kind: IntervalKind,
        level: f64,
    ) -> Result<(Vec<f64>, Vec<Interval>)> {
        check_probability(level)?;
        if kind == IntervalKind::Prediction {
            self.prediction_interval_applies()?;
        }
        let per_draw = self.predict_draws(x)?;
        let n = x.n_rows();
        let mean = column_means(&per_draw, n);
        let intervals = match kind {
            IntervalKind::Credible => credible_from_draws(&per_draw, n, level, self.threads),
            IntervalKind::Prediction => self.prediction_from_draws(x, &per_draw, level)?,
        };
        Ok((mean, intervals))
    }

    /// `NotApplicable` under the models whose predictive has no interval on
    /// the response scale.
    fn prediction_interval_applies(&self) -> Result<()> {
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            return Err(self.not_applicable("prediction_interval"));
        }
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Ordinal(_)) {
            return Err(self.not_applicable("prediction_interval"));
        }
        Ok(())
    }

    /// The central predictive interval at `level` at each row of `x`, from
    /// the per-draw means `per_draw` at those rows.
    fn prediction_from_draws(
        &self,
        x: &Data,
        per_draw: &[Vec<f64>],
        level: f64,
    ) -> Result<Vec<Interval>> {
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Laplace(_)) {
            let range = self.scaler.y_range();
            let sigmas: Vec<f64> = self
                .posterior
                .sigma_sq()
                .iter()
                .map(|s| s.sqrt() * range)
                .collect();
            let tail = 0.5 * (1.0 - level);
            let mut out = vec![Interval::default(); x.n_rows()];
            threads::spread_rows(&mut out, self.threads, |start, chunk| {
                let mut fits = vec![0.0; per_draw.len()];
                for (offset, interval) in chunk.iter_mut().enumerate() {
                    for (fit, draw) in fits.iter_mut().zip(per_draw) {
                        *fit = draw[start + offset];
                    }
                    let cdf = |t: f64| laplace_mixture_cdf(&fits, &sigmas, t);
                    *interval = Interval {
                        lower: heavy_mixture_quantile(&fits, &sigmas, tail, cdf),
                        upper: heavy_mixture_quantile(&fits, &sigmas, 1.0 - tail, cdf),
                    };
                }
            });
            return Ok(out);
        }
        #[cfg(feature = "experimental")]
        if let Outcome::StudentT(params) = &self.config.outcome {
            let range = self.scaler.y_range();
            let sigmas: Vec<f64> = self
                .posterior
                .sigma_sq()
                .iter()
                .map(|s| s.sqrt() * range)
                .collect();
            let dfs: Vec<f64> = (0..per_draw.len())
                .map(|d| self.student_df(params, d))
                .collect();
            let tail = 0.5 * (1.0 - level);
            let mut out = vec![Interval::default(); x.n_rows()];
            threads::spread_rows(&mut out, self.threads, |start, chunk| {
                let mut fits = vec![0.0; per_draw.len()];
                for (offset, interval) in chunk.iter_mut().enumerate() {
                    for (fit, draw) in fits.iter_mut().zip(per_draw) {
                        *fit = draw[start + offset];
                    }
                    let cdf = |t: f64| student_mixture_cdf(&fits, &sigmas, &dfs, t);
                    *interval = Interval {
                        lower: heavy_mixture_quantile(&fits, &sigmas, tail, cdf),
                        upper: heavy_mixture_quantile(&fits, &sigmas, 1.0 - tail, cdf),
                    };
                }
            });
            return Ok(out);
        }
        let variances = self.predict_variance(x)?;
        let tail = 0.5 * (1.0 - level);
        let mut out = vec![Interval::default(); x.n_rows()];
        threads::spread_rows(&mut out, self.threads, |start, chunk| {
            let mut fits = vec![0.0; per_draw.len()];
            let mut sigmas = vec![0.0; per_draw.len()];
            for (offset, interval) in chunk.iter_mut().enumerate() {
                let row = start + offset;
                for ((fit, sigma), (draw, variance)) in fits
                    .iter_mut()
                    .zip(&mut sigmas)
                    .zip(per_draw.iter().zip(&variances))
                {
                    *fit = draw[row];
                    *sigma = variance[row].sqrt();
                }
                *interval = Interval {
                    lower: mixture_quantile(&fits, &sigmas, tail),
                    upper: mixture_quantile(&fits, &sigmas, 1.0 - tail),
                };
            }
        });
        #[cfg(feature = "experimental")]
        if let Outcome::Tobit(params) = &self.config.outcome {
            let lo = params.lower.unwrap_or(f64::NEG_INFINITY);
            let hi = params.upper.unwrap_or(f64::INFINITY);
            for interval in &mut out {
                interval.lower = interval.lower.clamp(lo, hi);
                interval.upper = interval.upper.clamp(lo, hi);
            }
        }
        Ok(out)
    }

    /// Pointwise log-likelihood per draw, draw-major (`n_draws` by
    /// `n_rows`): ln N(y_i | f_d(x_i), s_d^2(x_i)), or under the probit
    /// model y_i ln p_d(x_i) + (1 - y_i) ln(1 - p_d(x_i)) with p_d the
    /// draw's P(y = 1 | x). Under the tobit model a row at a limit takes
    /// its censored term, ln Phi((lower - f_d) / s_d) or
    /// ln Phi((f_d - upper) / s_d), and the Normal log density otherwise.
    /// Under the ordinal model the term is the ordinal likelihood,
    /// ln(Phi(gamma_{y+1} - c - f_d) - Phi(gamma_y - c - f_d)) with the
    /// draw's own cutpoints. Under the Student-t model the term is the
    /// location-scale t log density with the draw's scale sigma_d and
    /// degrees of freedom df_d, and under the Laplace model the Laplace
    /// log density with the draw's scale. `NotApplicable` under the AFT
    /// model, whose
    /// pointwise likelihood needs the event indicator
    /// (`log_likelihood_survival`), and under the interval-censored
    /// model, whose pointwise likelihood needs the bounds
    /// (`log_likelihood_interval_censored`); both methods are compiled
    /// with the `experimental` feature.
    ///
    /// # Errors
    ///
    /// `RowCountMismatch`, `NonFiniteResponse`, `InvalidLabel` under the
    /// probit model, `ResponseBeyondLimit` under the tobit model,
    /// `InvalidOrdinalLabel` under the ordinal model; the predict
    /// errors.
    pub fn log_likelihood(&self, x: &Data, y: &[f64]) -> Result<Vec<Vec<f64>>> {
        #[cfg(feature = "experimental")]
        if matches!(
            self.config.outcome,
            Outcome::Aft(_) | Outcome::IntervalCensored(_)
        ) {
            return Err(self.not_applicable("log_likelihood"));
        }
        if y.len() != x.n_rows() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: x.n_rows(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteResponse { row });
        }
        #[cfg(feature = "experimental")]
        if let Outcome::Ordinal(params) = &self.config.outcome {
            let categories = params.categories;
            if let Some(row) = y
                .iter()
                .position(|&v| !(v.fract() == 0.0 && v >= 0.0 && v < categories as f64))
            {
                return Err(Error::InvalidOrdinalLabel { row, categories });
            }
            let latent = self.predict_latent(x)?;
            return Ok(latent
                .iter()
                .enumerate()
                .map(|(d, fits)| {
                    let free = self
                        .posterior
                        .cutpoints()
                        .get(d)
                        .map_or(&[][..], Vec::as_slice);
                    let gamma = |k: usize| if k == 1 { 0.0 } else { free[k - 2] };
                    y.iter()
                        .zip(fits)
                        .map(|(&yi, &l)| {
                            let k = yi as usize;
                            let above = if k == categories - 1 {
                                1.0
                            } else {
                                maths::normal_cdf(gamma(k + 1) - l)
                            };
                            let below = if k == 0 {
                                0.0
                            } else {
                                maths::normal_cdf(gamma(k) - l)
                            };
                            maths::ln(above - below)
                        })
                        .collect()
                })
                .collect());
        }
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            if let Some(row) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
                return Err(Error::InvalidLabel { row });
            }
            let per_draw = self.predict_draws(x)?;
            return Ok(per_draw
                .iter()
                .map(|probs| {
                    y.iter()
                        .zip(probs)
                        .map(|(&yi, &p)| {
                            if yi == 1.0 {
                                maths::ln(p)
                            } else {
                                maths::ln(1.0 - p)
                            }
                        })
                        .collect()
                })
                .collect());
        }
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Laplace(_)) {
            let per_draw = self.predict_draws(x)?;
            let range = self.scaler.y_range();
            return Ok(per_draw
                .iter()
                .enumerate()
                .map(|(d, fits)| {
                    let scale = self.posterior.sigma_sq()[d].sqrt() * range;
                    let ln_norm = maths::ln(2.0 * scale);
                    y.iter()
                        .zip(fits)
                        .map(|(&yi, &fit)| -ln_norm - (yi - fit).abs() / scale)
                        .collect()
                })
                .collect());
        }
        #[cfg(feature = "experimental")]
        if let Outcome::StudentT(params) = &self.config.outcome {
            let per_draw = self.predict_draws(x)?;
            let range_sq = self.scaler.y_range() * self.scaler.y_range();
            let ln_pi = maths::ln(std::f64::consts::PI);
            return Ok(per_draw
                .iter()
                .enumerate()
                .map(|(d, fits)| {
                    let sigma_sq = self.posterior.sigma_sq()[d] * range_sq;
                    let df = self.student_df(params, d);
                    let ln_c = maths::lgamma(0.5 * (df + 1.0))
                        - maths::lgamma(0.5 * df)
                        - 0.5 * (maths::ln(df) + ln_pi + maths::ln(sigma_sq));
                    y.iter()
                        .zip(fits)
                        .map(|(&yi, &fit)| {
                            let z_sq = (yi - fit) * (yi - fit) / (df * sigma_sq);
                            ln_c - 0.5 * (df + 1.0) * maths::ln(1.0 + z_sq)
                        })
                        .collect()
                })
                .collect());
        }
        let per_draw = self.predict_draws(x)?;
        let variances = self.predict_variance(x)?;
        let ln_2pi = maths::ln(2.0 * std::f64::consts::PI);
        #[cfg(feature = "experimental")]
        if let Outcome::Tobit(params) = &self.config.outcome {
            let (lower, upper) = (params.lower, params.upper);
            let beyond = |v: f64| {
                lower.is_some_and(|limit| v < limit) || upper.is_some_and(|limit| v > limit)
            };
            if let Some(row) = y.iter().position(|&v| beyond(v)) {
                return Err(Error::ResponseBeyondLimit { row });
            }
            return Ok(per_draw
                .iter()
                .zip(&variances)
                .map(|(fits, variance)| {
                    y.iter()
                        .zip(fits.iter().zip(variance))
                        .map(|(&yi, (&fit, &var))| {
                            let sd = var.sqrt();
                            if lower == Some(yi) {
                                maths::ln(maths::normal_cdf((yi - fit) / sd))
                            } else if upper == Some(yi) {
                                maths::ln(maths::normal_cdf((fit - yi) / sd))
                            } else {
                                let z = (yi - fit) * (yi - fit) / var;
                                -0.5 * (ln_2pi + maths::ln(var) + z)
                            }
                        })
                        .collect()
                })
                .collect());
        }
        Ok(per_draw
            .iter()
            .zip(&variances)
            .map(|(fits, variance)| {
                y.iter()
                    .zip(fits.iter().zip(variance))
                    .map(|(&yi, (&fit, &var))| {
                        let z = (yi - fit) * (yi - fit) / var;
                        -0.5 * (ln_2pi + maths::ln(var) + z)
                    })
                    .collect()
            })
            .collect())
    }

    /// Pointwise survival log-likelihood per draw under the AFT model,
    /// draw-major (`n_draws` by `n_rows`): ln N(ln t_i; f_d(x_i),
    /// s_d^2(x_i)) at an event and ln Phi((f_d(x_i) - ln t_i) / s_d(x_i))
    /// at a censored row, the log-time density and the survival
    /// probability of the lognormal AFT model. Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// `NotApplicable` under another model; `RowCountMismatch`,
    /// `EventCountMismatch`, `InvalidSurvivalTime`; the predict errors.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn log_likelihood_survival(
        &self,
        x: &Data,
        times: &[f64],
        events: &[bool],
    ) -> Result<Vec<Vec<f64>>> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::AFT.requires_feature());
        #[cfg(feature = "experimental")]
        {
            if !matches!(self.config.outcome, Outcome::Aft(_)) {
                return Err(self.not_applicable("log_likelihood_survival"));
            }
            if times.len() != x.n_rows() {
                return Err(Error::RowCountMismatch {
                    y_len: times.len(),
                    x_rows: x.n_rows(),
                });
            }
            if events.len() != times.len() {
                return Err(Error::EventCountMismatch {
                    events: events.len(),
                    times: times.len(),
                });
            }
            if let Some(row) = times.iter().position(|&t| !(t.is_finite() && t > 0.0)) {
                return Err(Error::InvalidSurvivalTime { row });
            }
            let per_draw = self.predict_draws(x)?;
            let variances = self.predict_variance(x)?;
            let ln_2pi = maths::ln(2.0 * std::f64::consts::PI);
            let log_times: Vec<f64> = times.iter().map(|&t| maths::ln(t)).collect();
            Ok(per_draw
                .iter()
                .zip(&variances)
                .map(|(fits, variance)| {
                    log_times
                        .iter()
                        .zip(events)
                        .zip(fits.iter().zip(variance))
                        .map(|((&v, &event), (&fit, &var))| {
                            if event {
                                let z = (v - fit) * (v - fit) / var;
                                -0.5 * (ln_2pi + maths::ln(var) + z)
                            } else {
                                maths::ln(maths::normal_cdf((fit - v) / var.sqrt()))
                            }
                        })
                        .collect()
                })
                .collect())
        }
    }

    /// Pointwise interval log-likelihood per draw under the
    /// interval-censored model, draw-major (`n_draws` by `n_rows`):
    /// ln N(l_i; f_d(x_i), s_d^2(x_i)) at an exact row (l_i = u_i) and
    /// ln(Phi((u_i - f_d) / s_d) - Phi((l_i - f_d) / s_d)) at a censored
    /// one, an infinite endpoint dropping its term. Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// `NotApplicable` under another model; `RowCountMismatch`,
    /// `BoundCountMismatch`, `InvalidInterval`; the predict errors.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn log_likelihood_interval_censored(
        &self,
        x: &Data,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::INTERVAL_CENSORED.requires_feature());
        #[cfg(feature = "experimental")]
        {
            if !matches!(self.config.outcome, Outcome::IntervalCensored(_)) {
                return Err(self.not_applicable("log_likelihood_interval_censored"));
            }
            if lower.len() != x.n_rows() {
                return Err(Error::RowCountMismatch {
                    y_len: lower.len(),
                    x_rows: x.n_rows(),
                });
            }
            if upper.len() != lower.len() {
                return Err(Error::BoundCountMismatch {
                    lower: lower.len(),
                    upper: upper.len(),
                });
            }
            if let Some(row) = lower.iter().zip(upper).position(|(&lo, &hi)| {
                lo.is_nan()
                    || hi.is_nan()
                    || lo > hi
                    || (lo == hi && !lo.is_finite())
                    || (lo == f64::NEG_INFINITY && hi == f64::INFINITY)
            }) {
                return Err(Error::InvalidInterval { row });
            }
            let per_draw = self.predict_draws(x)?;
            let variances = self.predict_variance(x)?;
            let ln_2pi = maths::ln(2.0 * std::f64::consts::PI);
            Ok(per_draw
                .iter()
                .zip(&variances)
                .map(|(fits, variance)| {
                    lower
                        .iter()
                        .zip(upper)
                        .zip(fits.iter().zip(variance))
                        .map(|((&lo, &hi), (&fit, &var))| {
                            if lo == hi {
                                let z = (lo - fit) * (lo - fit) / var;
                                -0.5 * (ln_2pi + maths::ln(var) + z)
                            } else {
                                let sd = var.sqrt();
                                let above = if hi == f64::INFINITY {
                                    1.0
                                } else {
                                    maths::normal_cdf((hi - fit) / sd)
                                };
                                let below = if lo == f64::NEG_INFINITY {
                                    0.0
                                } else {
                                    maths::normal_cdf((lo - fit) / sd)
                                };
                                maths::ln(above - below)
                            }
                        })
                        .collect()
                })
                .collect())
        }
    }

    /// Posterior-mean category probabilities under the ordinal model,
    /// row-major (`n_rows` by `categories`): the average over kept draws
    /// of P(y = k | x) = Phi(gamma_{k+1} - c - f_d(x)) -
    /// Phi(gamma_k - c - f_d(x)) with the draw's own cutpoints.
    /// Experimental
    /// (`docs/experimental.md`); `RequiresFeature` in a build without the
    /// feature.
    ///
    /// # Errors
    ///
    /// `NotApplicable` under another model; the predict errors.
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn predict_category_probabilities(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        #[cfg(not(feature = "experimental"))]
        return Err(crate::config::Gated::ORDINAL.requires_feature());
        #[cfg(feature = "experimental")]
        {
            let Outcome::Ordinal(params) = &self.config.outcome else {
                return Err(self.not_applicable("predict_category_probabilities"));
            };
            let categories = params.categories;
            let latent = self.predict_latent(x)?;
            let mut out = vec![vec![0.0; categories]; x.n_rows()];
            for (d, fits) in latent.iter().enumerate() {
                let free = self
                    .posterior
                    .cutpoints()
                    .get(d)
                    .map_or(&[][..], Vec::as_slice);
                for (row, &l) in fits.iter().enumerate() {
                    let mut previous = 0.0;
                    for (k, slot) in out[row].iter_mut().enumerate() {
                        let cumulative = if k == categories - 1 {
                            1.0
                        } else {
                            let g = if k == 0 { 0.0 } else { free[k - 1] };
                            maths::normal_cdf(g - l)
                        };
                        *slot += cumulative - previous;
                        previous = cumulative;
                    }
                }
            }
            let n_draws = latent.len() as f64;
            for row in &mut out {
                for p in row {
                    *p /= n_draws;
                }
            }
            Ok(out)
        }
    }

    /// The interior cutpoints of each kept draw under the ordinal model,
    /// increasing, latent scale; empty under another model and at two
    /// categories, where none is sampled, and in a build without the
    /// feature. Experimental (`docs/experimental.md`).
    pub fn cutpoint_draws(&self) -> &[Vec<f64>] {
        self.posterior.cutpoints()
    }

    /// The sampled inclusion weight of each covariate, in column order,
    /// per kept draw under the DART inclusion prior; empty under another
    /// inclusion prior. Each draw sums to one.
    ///
    /// This is the prior weight s the sampler drew, the quantity Linero
    /// (2018) reports; it is not
    /// [`variable_inclusion_proportions`](Self::variable_inclusion_proportions),
    /// which counts the usage the tessellations realised.
    /// Experimental (`docs/experimental.md`).
    pub fn inclusion_weight_draws(&self) -> &[Vec<f64>] {
        self.posterior.inclusion_weights()
    }

    /// The Dirichlet concentration theta of each kept draw under the DART
    /// inclusion prior; empty under another inclusion prior. One value
    /// per draw, not one per covariate. Experimental
    /// (`docs/experimental.md`).
    pub fn concentration_draws(&self) -> &[f64] {
        self.posterior.concentration()
    }

    /// The soft-membership kernel bandwidth of each mean tessellation,
    /// one row per kept draw, on the scaled covariate space its prior is
    /// on; empty under hard membership, and in a build without the
    /// feature. Experimental (`docs/experimental.md`).
    pub fn bandwidth_draws(&self) -> Vec<Vec<f64>> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().filter_map(Tessellation::bandwidth).collect())
            .filter(|draw: &Vec<f64>| !draw.is_empty())
            .collect()
    }

    /// sigma per kept draw under a model with a global sampled sigma^2
    /// (the Gaussian, tobit, Student-t and Laplace models; under the
    /// scale-mixture models the scale of the t or Laplace, not the error
    /// standard deviation),
    /// caller scale: sqrt(sigma^2) times the training range of the
    /// response. Empty under the probit model (unit latent variance) and
    /// under a variance ensemble
    /// ([`predict_variance`](Self::predict_variance) gives s^2(x)).
    pub fn sigma(&self) -> Vec<f64> {
        let range = self.scaler.y_range();
        self.posterior
            .sigma_sq()
            .iter()
            .map(|s| s.sqrt() * range)
            .collect()
    }

    /// The probit offset c; 0 under the other models.
    fn offset(&self) -> f64 {
        self.config.offset().unwrap_or(0.0)
    }

    /// The error degrees of freedom of draw `d` under the Student-t
    /// model: the draw's own value under a grid, the fixed value
    /// otherwise.
    #[cfg(feature = "experimental")]
    fn student_df(&self, params: &StudentTParams, d: usize) -> f64 {
        self.posterior
            .dfs()
            .get(d)
            .copied()
            .unwrap_or_else(|| params.df.initial())
    }

    /// Mean number of cells per mean tessellation, one value per kept draw.
    pub fn cell_counts(&self) -> Vec<f64> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().map(|t| t.n_cells() as f64).sum::<f64>() / draw.len() as f64)
            .collect()
    }

    /// Mean number of active covariates per mean tessellation, one value
    /// per kept draw.
    pub fn dimension_counts(&self) -> Vec<f64> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().map(|t| t.n_dims() as f64).sum::<f64>() / draw.len() as f64)
            .collect()
    }

    /// Share of active mean-tessellation dimensions over all kept draws
    /// that fall on each covariate; sums to 1 (Chipman, George and
    /// McCulloch 2010, s. 5.1).
    pub fn variable_inclusion_proportions(&self) -> Vec<f64> {
        let mut counts = vec![0u64; self.scaler.n_cols()];
        let mut total = 0u64;
        for draw in self.posterior.tessellations() {
            for t in draw {
                for &dim in t.dims() {
                    counts[dim] += 1;
                    total += 1;
                }
            }
        }
        counts
            .iter()
            .map(|&c| c as f64 / total.max(1) as f64)
            .collect()
    }

    /// Number of kept draws.
    pub fn n_draws(&self) -> usize {
        self.posterior.n_draws()
    }

    /// The kept draws, scaled space.
    pub fn posterior(&self) -> &Posterior {
        &self.posterior
    }

    /// The chain state at kept draw `draw` (0-based, in pooled chain
    /// order), from which [`Sampler::start_from`](crate::Sampler::start_from)
    /// continues a chain. Experimental (`docs/experimental.md`).
    ///
    /// # Errors
    ///
    /// `RequiresFeature` in a build without the feature;
    /// `InvalidHyperparameter` for `init_draw` at or beyond
    /// [`n_draws`](Self::n_draws).
    #[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
    pub fn state(&self, draw: usize) -> Result<crate::state::EnsembleState> {
        #[cfg(not(feature = "experimental"))]
        return Err(Error::RequiresFeature {
            item: "Fitted::state".into(),
            feature: "experimental",
        });
        #[cfg(feature = "experimental")]
        {
            let n = self.posterior.n_draws();
            if draw >= n {
                return Err(crate::error::invalid(
                    "init_draw",
                    format!("draw {draw} of a fit with {n} draws"),
                ));
            }
            Ok(self.posterior.state(draw))
        }
    }

    /// The draws each chain kept, in pooled order; empty for a save from
    /// before fits kept the count per chain.
    #[cfg_attr(not(feature = "experimental"), allow(dead_code))]
    pub(crate) fn chain_draws(&self) -> &[usize] {
        &self.chain_draws
    }

    /// The levels of each categorical column; empty for the other
    /// columns.
    #[cfg_attr(not(feature = "experimental"), allow(dead_code))]
    pub(crate) fn categories(&self) -> &[Vec<f64>] {
        &self.categories
    }

    /// The scaling the model was fitted with.
    pub fn scaler(&self) -> &Scaler {
        &self.scaler
    }

    /// The configuration the model was fitted with, omega resolved (and the
    /// offset, under the probit model).
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Fit-time warnings.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Root mean square of the posterior-mean prediction against the
    /// training response, caller scale; under the probit model the root
    /// Brier score of the predicted probabilities against the labels;
    /// under the tobit model the target is the observed response,
    /// censored rows at their limits.
    pub fn in_sample_rmse(&self) -> f64 {
        self.in_sample_rmse
    }
}

fn column_means(per_draw: &[Vec<f64>], n: usize) -> Vec<f64> {
    let n_draws = per_draw.len() as f64;
    let mut means = vec![0.0; n];
    for draw in per_draw {
        for (mean, value) in means.iter_mut().zip(draw) {
            *mean += value;
        }
    }
    for mean in &mut means {
        *mean /= n_draws;
    }
    means
}

/// `InvalidProbability` for an empty `probs` or one outside (0, 1).
fn check_probabilities(probs: &[f64]) -> Result<()> {
    if probs.is_empty() {
        return Err(Error::InvalidProbability { value: f64::NAN });
    }
    probs.iter().try_for_each(|&p| check_probability(p))
}

/// Posterior quantiles at each of `n` rows for each of `probs`, row-major,
/// by type 7 interpolation over the per-draw values.
fn quantiles_from_draws(
    per_draw: &[Vec<f64>],
    n: usize,
    probs: &[f64],
    threads: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; n * probs.len()];
    let mut rows: Vec<&mut [f64]> = out.chunks_mut(probs.len().max(1)).collect();
    rows.truncate(n);
    threads::spread_rows(&mut rows, threads, |start, chunk| {
        let mut sorted = vec![0.0; per_draw.len()];
        for (offset, row) in chunk.iter_mut().enumerate() {
            for (slot, draw) in sorted.iter_mut().zip(per_draw) {
                *slot = draw[start + offset];
            }
            sorted.sort_by(f64::total_cmp);
            for (slot, &p) in row.iter_mut().zip(probs) {
                *slot = maths::quantile_sorted(&sorted, p);
            }
        }
    });
    out
}

/// The central credible interval at `level` at each of `n` rows from the
/// per-draw values.
fn credible_from_draws(
    per_draw: &[Vec<f64>],
    n: usize,
    level: f64,
    threads: usize,
) -> Vec<Interval> {
    let tail = 0.5 * (1.0 - level);
    quantiles_from_draws(per_draw, n, &[tail, 1.0 - tail], threads)
        .chunks_exact(2)
        .map(|pair| Interval {
            lower: pair[0],
            upper: pair[1],
        })
        .collect()
}

fn check_probability(p: f64) -> Result<()> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(Error::InvalidProbability { value: p })
    }
}

/// CDF at `t` of the equal-weight mixture of N(fit_d, sigma_d^2).
fn mixture_cdf(fits: &[f64], sigmas: &[f64], t: f64) -> f64 {
    let sum: f64 = fits
        .iter()
        .zip(sigmas)
        .map(|(&fit, &sigma)| maths::normal_cdf((t - fit) / sigma))
        .sum();
    sum / fits.len() as f64
}

/// CDF at `t` of the equal-weight mixture of fit_d + sigma_d t_{df_d}.
#[cfg(feature = "experimental")]
fn student_mixture_cdf(fits: &[f64], sigmas: &[f64], dfs: &[f64], t: f64) -> f64 {
    let sum: f64 = fits
        .iter()
        .zip(sigmas.iter().zip(dfs))
        .map(|(&fit, (&sigma, &df))| maths::student_t_cdf((t - fit) / sigma, df))
        .sum();
    sum / fits.len() as f64
}

/// CDF at `t` of the equal-weight mixture of Laplace(fit_d, sigma_d).
#[cfg(feature = "experimental")]
fn laplace_mixture_cdf(fits: &[f64], sigmas: &[f64], t: f64) -> f64 {
    let sum: f64 = fits
        .iter()
        .zip(sigmas)
        .map(|(&fit, &sigma)| maths::laplace_cdf((t - fit) / sigma))
        .sum();
    sum / fits.len() as f64
}

/// Quantile `p` of a mixture with heavier-than-Gaussian tails by
/// bisection on its `cdf`, the bracket of the fits padded by sigma_max
/// and doubled outward until it covers p: polynomial and exponential
/// tails outrun any fixed pad.
#[cfg(feature = "experimental")]
fn heavy_mixture_quantile(fits: &[f64], sigmas: &[f64], p: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sigma_max = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        lo = lo.min(fit);
        hi = hi.max(fit);
        sigma_max = sigma_max.max(sigma);
    }
    let (mut lo, mut hi) = (lo - sigma_max, hi + sigma_max);
    while cdf(lo) > p {
        lo -= hi - lo;
    }
    while cdf(hi) < p {
        hi += hi - lo;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Quantile `p` of the mixture by bisection on [`mixture_cdf`] over a
/// bracket of the fits padded by 39 sigma_max (where Phi is 0 or 1 to
/// double precision).
fn mixture_quantile(fits: &[f64], sigmas: &[f64], p: f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sigma_max = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        lo = lo.min(fit);
        hi = hi.max(fit);
        sigma_max = sigma_max.max(sigma);
    }
    let pad = 39.0 * sigma_max;
    let (mut lo, mut hi) = (lo - pad, hi + pad);
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if mixture_cdf(fits, sigmas, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit;

    fn data() -> (Data, Vec<f64>) {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
        (Data::new(xs, n, 1).unwrap(), y)
    }

    fn small() -> Config {
        Config::new().with_m(10).with_burn_in(20).with_draws(30)
    }

    fn fitted() -> (Fitted, Data, Vec<f64>) {
        let (x, y) = data();
        (fit(&small(), &x, &y, 42).unwrap(), x, y)
    }

    fn fitted_probit() -> (Fitted, Data, Vec<f64>) {
        let (x, y) = data();
        let labels: Vec<f64> = y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        let config = small().with_outcome(Outcome::probit());
        (fit(&config, &x, &labels, 42).unwrap(), x, labels)
    }

    fn fitted_heteroscedastic() -> (Fitted, Data, Vec<f64>) {
        let (x, y) = data();
        let config = small().with_m_var(5);
        (fit(&config, &x, &y, 42).unwrap(), x, y)
    }

    #[test]
    fn prediction_surface_shapes_and_order() {
        let (model, x, y) = fitted();
        assert_eq!(model.model_name(), "gaussian");
        let mean = model.predict(&x).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        assert_eq!(mean.len(), 30);
        assert_eq!((draws.len(), draws[0].len()), (30, 30));
        assert_eq!(model.predict_latent(&x).unwrap(), draws);
        let variance = model.predict_variance(&x).unwrap();
        let sigma = model.sigma();
        for (row, &s) in variance.iter().zip(&sigma) {
            assert!(row.iter().all(|&v| (v - s * s).abs() < 1e-12 * s * s));
        }
        let q = model.predict_quantiles(&x, &[0.1, 0.5, 0.9]).unwrap();
        assert_eq!(q.len(), 90);
        for row in 0..30 {
            assert!(q[row * 3] <= q[row * 3 + 1] && q[row * 3 + 1] <= q[row * 3 + 2]);
        }
        let ci = model.credible_interval(&x, 0.9).unwrap();
        let pi = model.prediction_interval(&x, 0.9).unwrap();
        for (c, p) in ci.iter().zip(&pi) {
            assert!(p.lower <= c.lower && c.upper <= p.upper);
        }
        let ll = model.log_likelihood(&x, &y).unwrap();
        assert_eq!((ll.len(), ll[0].len()), (30, 30));
        assert!(ll.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(model.sigma().len(), 30);
        assert_eq!(model.cell_counts().len(), 30);
        assert_eq!(model.dimension_counts(), vec![1.0; 30]);
        assert_eq!(model.variable_inclusion_proportions(), vec![1.0]);
        assert!(model.in_sample_rmse() < 0.5);
        assert!(model.warnings().is_empty());
    }

    #[test]
    fn probit_surface() {
        let (model, x, labels) = fitted_probit();
        assert_eq!(model.model_name(), "probit");
        let probs = model.predict(&x).unwrap();
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
        // The mean probability over the draws is Phi of the latent, draw by
        // draw.
        let latent = model.predict_latent(&x).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        for (l, d) in latent.iter().flatten().zip(draws.iter().flatten()) {
            assert!((maths::normal_cdf(*l) - d).abs() < 1e-15);
        }
        let offset = model.config().offset().unwrap();
        assert!(offset.is_finite());
        assert!(model.sigma().is_empty());
        assert!(matches!(
            model.predict_variance(&x),
            Err(Error::NotApplicable { ref method, ref model }) if method == "predict_variance" && model == "probit"
        ));
        assert!(matches!(
            model.prediction_interval(&x, 0.9),
            Err(Error::NotApplicable { .. })
        ));
        let ci = model.credible_interval(&x, 0.9).unwrap();
        assert!(ci.iter().all(|c| 0.0 <= c.lower && c.upper <= 1.0));
        let ll = model.log_likelihood(&x, &labels).unwrap();
        assert!(ll.iter().flatten().all(|v| v.is_finite() && *v <= 0.0));
        assert_eq!(
            model.log_likelihood(&x, &vec![0.5; 30]).unwrap_err(),
            Error::InvalidLabel { row: 0 }
        );
        // Root Brier score: strictly below the constant-probability score.
        let share = labels.iter().sum::<f64>() / 30.0;
        assert!(model.in_sample_rmse() < (share * (1.0 - share)).sqrt());
    }

    #[test]
    fn heteroscedastic_surface() {
        let (model, x, y) = fitted_heteroscedastic();
        assert_eq!(model.model_name(), "heteroscedastic");
        assert!(model.sigma().is_empty());
        let variance = model.predict_variance(&x).unwrap();
        assert_eq!((variance.len(), variance[0].len()), (30, 30));
        assert!(variance.iter().flatten().all(|v| v.is_finite() && *v > 0.0));
        // s^2 is the product of the variance cell values times the range squared.
        let draw0 = &model.posterior().variance_tessellations()[0];
        let x_scaled = model.scaler().scale_x(&x);
        let g = model.geometry().unwrap();
        let product: f64 = draw0
            .iter()
            .map(|t| t.value_at(x_scaled.row(4), &g))
            .product();
        let range = model.scaler().y_range();
        assert!((variance[0][4] - product * range * range).abs() < 1e-12 * variance[0][4]);
        let pi = model.prediction_interval(&x, 0.9).unwrap();
        let ci = model.credible_interval(&x, 0.9).unwrap();
        for (c, p) in ci.iter().zip(&pi) {
            assert!(p.lower <= c.lower && c.upper <= p.upper);
        }
        let ll = model.log_likelihood(&x, &y).unwrap();
        assert!(ll.iter().flatten().all(|v| v.is_finite()));
        let expected = -0.5
            * ((2.0 * std::f64::consts::PI * variance[0][4]).ln()
                + (y[4] - model.predict_draws(&x).unwrap()[0][4]).powi(2) / variance[0][4]);
        assert!((ll[0][4] - expected).abs() < 1e-12);
    }

    #[test]
    fn predict_errors() {
        let (model, _, y) = fitted();
        let wrong = Data::new(vec![0.0, 1.0], 1, 2).unwrap();
        assert!(matches!(
            model.predict(&wrong),
            Err(Error::FeatureCountMismatch {
                expected: 1,
                found: 2
            })
        ));
        let x = Data::new(vec![0.5], 1, 1).unwrap();
        assert!(matches!(
            model.predict_quantiles(&x, &[1.5]),
            Err(Error::InvalidProbability { .. })
        ));
        assert!(model.predict_quantiles(&x, &[]).is_err());
        assert!(model.credible_interval(&x, 0.0).is_err());
        assert!(matches!(
            model.log_likelihood(&x, &y),
            Err(Error::RowCountMismatch { .. })
        ));
        let empty = Data::new(vec![], 0, 1).unwrap();
        assert!(model.predict(&empty).unwrap().is_empty());
    }

    #[test]
    fn serde_round_trip_preserves_predictions() {
        for (model, x, _) in [fitted(), fitted_probit(), fitted_heteroscedastic()] {
            let json = serde_json::to_string(&model).unwrap();
            let loaded: Fitted = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded, model);
            assert_eq!(loaded.predict(&x).unwrap(), model.predict(&x).unwrap());
            let corrupt = json.replace("\"in_sample_rmse\":", "\"in_sample_rmse\":1e999,\"x\":");
            assert!(serde_json::from_str::<Fitted>(&corrupt).is_err());
        }
    }

    #[test]
    fn loading_validates_the_model_specific_parts() {
        let (gaussian, _, _) = fitted();
        let (probit, _, _) = fitted_probit();
        let (hetero, _, _) = fitted_heteroscedastic();
        let json = |m: &Fitted| serde_json::to_string(m).unwrap();
        // A Gaussian payload relabelled probit: sigma draws present, no offset.
        let gaussian_outcome = r#""outcome":{"gaussian":{"nu":6.0,"q":0.85}}"#;
        let relabelled =
            json(&gaussian).replace(gaussian_outcome, r#""outcome":{"probit":{"offset":0.0}}"#);
        assert_ne!(relabelled, json(&gaussian));
        assert!(serde_json::from_str::<Fitted>(&relabelled).is_err());
        // A probit payload relabelled Gaussian: no sigma draws.
        let offset = probit.config().offset().unwrap();
        let probit_outcome = format!(r#""outcome":{{"probit":{{"offset":{offset:?}}}}}"#);
        let relabelled = json(&probit).replace(&probit_outcome, gaussian_outcome);
        assert_ne!(relabelled, json(&probit));
        assert!(serde_json::from_str::<Fitted>(&relabelled).is_err());
        // A heteroscedastic payload with the wrong variance count.
        let wrong = json(&hetero).replace("\"tessellations\":5", "\"tessellations\":6");
        assert_ne!(wrong, json(&hetero));
        assert!(serde_json::from_str::<Fitted>(&wrong).is_err());
        // A Gaussian payload without the variance field (a pre-0.2 save).
        let old = json(&gaussian).replace(",\"variance_tessellations\":[]", "");
        assert_ne!(old, json(&gaussian));
        assert_eq!(serde_json::from_str::<Fitted>(&old).unwrap(), gaussian);
    }

    #[test]
    fn mixture_quantile_of_one_normal() {
        let q = mixture_quantile(&[1.0], &[2.0], 0.975);
        assert!((q - (1.0 + 2.0 * 1.959_963_984_540_054)).abs() < 1e-9);
    }
}
