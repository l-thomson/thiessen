//! XAddiVortes: a fast point fit of the AddiVortes ensemble.
//!
//! [`Sampler::xaddivortes`](crate::Sampler::xaddivortes) replaces the
//! chain's cold start by the fit, so the fit serves two uses: on its own,
//! the prediction being the sum of the fitted cell values; or as the
//! initial state from which the exact sampler runs on, which changes the
//! chain's starting point and not its stationary distribution. The fit is
//! a single point estimate, not a posterior sample: [`FastFit::sigma`] is
//! the in-sample residual root mean square, which understates the noise
//! of an overfitting fit, most on small data, so intervals built from it
//! are a rough guide; split-conformal intervals (Lei and others, 2018)
//! around the fit are calibrated without that assumption.
//!
//! The fit, in order:
//!
//! 1. Column scores, from the response: each column's binned
//!    mutual information with the response (equal-width column bins over
//!    the column's training range, rank bins of the response with tied
//!    values sharing a bin), less its expected value under independence,
//!    plus a floor; and for every pair among the
//!    [`pair_screen`](XAddiVortesParams::pair_screen) best columns its
//!    interaction information plus a floor. Optionally
//!    ([`rescore`](XAddiVortesParams::rescore)) they are scored again from
//!    the residual after every round.
//! 2. Build the m tessellations in turn: the first
//!    [`main_effect_share`](XAddiVortesParams::main_effect_share) on one
//!    column drawn by score, the rest on
//!    [`max_columns`](XAddiVortesParams::max_columns) columns (a pair drawn
//!    by pair score, then any further columns by score);
//!    [`centres`](XAddiVortesParams::centres) centres at random training
//!    rows, empty cells dropped; cell values
//!    `value_scale * sum(r) / (count + sigma^2 / sigma_mu^2)` of the
//!    running residual r, which they are then subtracted from.
//! 3. Rounds, while the budget lasts: rank the tessellations by
//!    [`worth`](XAddiVortesParams::worth) (by default their gain at the
//!    last refit), replace the
//!    [`replace_share`](XAddiVortesParams::replace_share) with the least
//!    worth by the best of [`tries`](XAddiVortesParams::tries) fresh ones
//!    with the same number of columns, then refit every tessellation's
//!    cell values to its partial residual. By default
//!    ([`update_sigma`](XAddiVortesParams::update_sigma)) the noise variance
//!    in the shrinkage is re-estimated from the residual after each round.
//!
//! The budget is in approximate sweep-equivalents of the exact sampler on
//! the same data, through a deterministic count of operations per row
//! (not a clock), so a seed reproduces its fit. The count is an
//! approximation: the time a sweep-equivalent takes varies with the data
//! and the settings, and fixed costs such as copying the data are not
//! counted, so they dominate small budgets.
//!
//! Experimental (`docs/experimental.md`): the types exist in every build,
//! the fit only with the `experimental` feature.
//!
//! # References
//!
//! - Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
//!   (2018). Distribution-free predictive inference for regression.
//!   Journal of the American Statistical Association 113(523), 1094-1111.
//! - Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
//!   Voronoi tessellations. Journal of Computational and Graphical
//!   Statistics 34(3), 859-871.

use serde::{Deserialize, Serialize};

use crate::error::{invalid, Result};

/// How a round ranks the tessellations it may replace.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Worth {
    /// The sum over training rows of the tessellation's cell value squared:
    /// how much it moves the fit.
    SquaredValues,
    /// The tessellation's gain at its last refit,
    /// `sum_k S_k^2 / (n_k + sigma^2 / sigma_mu^2)` over its cells, with
    /// `S_k` the partial residual summed over cell k: how much of the
    /// residual it explains.
    #[default]
    Gain,
}

/// Settings of the XAddiVortes fast fit. The defaults were chosen on
/// public regression benchmark datasets and synthetic designs; `budget`,
/// `max_columns` and the ensemble size (the configuration's number of
/// tessellations) are the settings most users change.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct XAddiVortesParams {
    /// The compute budget B, in approximate sweeps of the exact sampler on
    /// the same data (default 50). More budget means more replacement
    /// rounds; the fit plans against
    /// [`PLANNING_MARGIN`](Self::PLANNING_MARGIN) times B. Scoring
    /// the columns is always done and costs about
    /// `(ln n + 2p + 3 S (S - 1) / 2) * 0.15 / m` sweeps for a pair screen
    /// of S, roughly 115 / m sweeps at the default S = 23. A budget that this
    /// cost and the build exhaust leaves tessellations at zero, which
    /// [`FastFit::zeroed`] reports and the fit records as a warning: at the
    /// default budget, every tessellation is zeroed for m of 3 or fewer.
    /// Positive and finite.
    pub budget: f64,
    /// The number of columns of the tessellations that are not main
    /// effects (default 2). 1 makes every tessellation a main effect. At
    /// least 1; capped at the number of columns.
    pub max_columns: usize,
    /// Centres drawn per tessellation, at random training rows, before
    /// empty cells are dropped (default 11). From 1 to 65,535.
    pub centres: usize,
    /// The share of the shrunk cell mean used as the cell value, nu
    /// (default 0.5). Smaller values fit more slowly and overfit less.
    /// In (0, 1].
    pub value_scale: f64,
    /// The share of tessellations on one column, built first (default
    /// 0.2). In [0, 1].
    pub main_effect_share: f64,
    /// The share of tessellations replaced each round, rho (default 0.3).
    /// In [0, 1]; 0 makes the rounds pure refits.
    pub replace_share: f64,
    /// The most replacement rounds (default 200); the budget usually stops
    /// the fit first.
    pub max_rounds: usize,
    /// The number of best-scoring columns whose pairs are scored for the
    /// two-column tessellations (default 23). At least 2. The budget is
    /// charged for `pair_screen (pair_screen - 1) / 2` pairs whatever the
    /// column count, so with few columns a large value spends budget on
    /// pairs that do not exist, and a value whose pairs cost more than the
    /// budget leaves every tessellation at zero.
    pub pair_screen: usize,
    /// Equal-width bins of a column, over its training range, for its
    /// score (default 8). From 2 to 255.
    pub column_bins: usize,
    /// Rank bins of the response for the scores (default 4); tied values
    /// share a bin. From 2 to 255.
    pub response_bins: usize,
    /// The floor added to every score, as a share of the mean score
    /// (default 0.02), so that every column and pair can be drawn. At
    /// least 0.
    pub score_floor: f64,
    /// Fresh candidates drawn per replacement, the one with the highest
    /// gain kept (default 4). At least 1.
    pub tries: usize,
    /// Re-score the columns from the residual after every round instead
    /// of scoring them once from the response (default false).
    pub rescore: bool,
    /// Refit passes over all tessellations per round (default 2). At
    /// least 1.
    pub refit_passes: usize,
    /// Re-estimate the noise variance used in the shrinkage from the
    /// residual after every round (default true; false keeps the chain's
    /// initial estimate throughout the fit).
    pub update_sigma: bool,
    /// How a round ranks the tessellations it may replace (default
    /// [`Worth::Gain`]).
    pub worth: Worth,
    /// Keep a replacement only when it lowers the training residual sum of
    /// squares, and a refit only when it does not raise it (default false).
    /// With it, the training error never rises from round to round; each
    /// check is charged to the budget and planned for.
    pub accept_if_better: bool,
}

impl Default for XAddiVortesParams {
    fn default() -> Self {
        Self {
            budget: 50.0,
            max_columns: 2,
            centres: 11,
            value_scale: 0.5,
            main_effect_share: 0.2,
            replace_share: 0.3,
            max_rounds: 200,
            pair_screen: 23,
            column_bins: 8,
            response_bins: 4,
            score_floor: 0.02,
            tries: 4,
            rescore: false,
            refit_passes: 2,
            update_sigma: true,
            worth: Worth::Gain,
            accept_if_better: false,
        }
    }
}

macro_rules! setter {
    ($(#[$doc:meta] $setter:ident => $name:ident: $ty:ty),* $(,)?) => {
        $(
            #[$doc]
            #[must_use]
            pub fn $setter(mut self, value: $ty) -> Self {
                self.$name = value;
                self
            }
        )*
    };
}

impl XAddiVortesParams {
    /// The share of the budget the fit plans against: its count of
    /// operations is approximate, so it plans to finish early.
    pub const PLANNING_MARGIN: f64 = 0.8;

    setter! {
        /// Set [`budget`](Self::budget).
        with_budget => budget: f64,
        /// Set [`max_columns`](Self::max_columns).
        with_max_columns => max_columns: usize,
        /// Set [`centres`](Self::centres).
        with_centres => centres: usize,
        /// Set [`value_scale`](Self::value_scale).
        with_value_scale => value_scale: f64,
        /// Set [`main_effect_share`](Self::main_effect_share).
        with_main_effect_share => main_effect_share: f64,
        /// Set [`replace_share`](Self::replace_share).
        with_replace_share => replace_share: f64,
        /// Set [`max_rounds`](Self::max_rounds).
        with_max_rounds => max_rounds: usize,
        /// Set [`pair_screen`](Self::pair_screen).
        with_pair_screen => pair_screen: usize,
        /// Set [`column_bins`](Self::column_bins).
        with_column_bins => column_bins: usize,
        /// Set [`response_bins`](Self::response_bins).
        with_response_bins => response_bins: usize,
        /// Set [`score_floor`](Self::score_floor).
        with_score_floor => score_floor: f64,
        /// Set [`tries`](Self::tries).
        with_tries => tries: usize,
        /// Set [`rescore`](Self::rescore).
        with_rescore => rescore: bool,
        /// Set [`refit_passes`](Self::refit_passes).
        with_refit_passes => refit_passes: usize,
        /// Set [`update_sigma`](Self::update_sigma).
        with_update_sigma => update_sigma: bool,
        /// Set [`worth`](Self::worth).
        with_worth => worth: Worth,
        /// Set [`accept_if_better`](Self::accept_if_better).
        with_accept_if_better => accept_if_better: bool,
    }

    /// Check every setting is in its range.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter`, naming the field as `xaddivortes.<name>`.
    pub fn validate(&self) -> Result<()> {
        let bad = |name: &str, why: String| Err(invalid(&format!("xaddivortes.{name}"), why));
        if !(self.budget.is_finite() && self.budget > 0.0) {
            return bad(
                "budget",
                format!("must be positive and finite, not {}", self.budget),
            );
        }
        if !(self.value_scale > 0.0 && self.value_scale <= 1.0) {
            return bad(
                "value_scale",
                format!("must be in (0, 1], not {}", self.value_scale),
            );
        }
        for (name, v) in [
            ("main_effect_share", self.main_effect_share),
            ("replace_share", self.replace_share),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return bad(name, format!("must be in [0, 1], not {v}"));
            }
        }
        if !(self.score_floor.is_finite() && self.score_floor >= 0.0) {
            return bad(
                "score_floor",
                format!("must be at least 0, not {}", self.score_floor),
            );
        }
        for (name, v, least) in [
            ("max_columns", self.max_columns, 1),
            ("centres", self.centres, 1),
            ("pair_screen", self.pair_screen, 2),
            ("column_bins", self.column_bins, 2),
            ("response_bins", self.response_bins, 2),
            ("tries", self.tries, 1),
            ("refit_passes", self.refit_passes, 1),
        ] {
            if v < least {
                return bad(name, format!("must be at least {least}, not {v}"));
            }
        }
        if self.centres > usize::from(u16::MAX) {
            return bad(
                "centres",
                format!("must be at most {}, not {}", u16::MAX, self.centres),
            );
        }
        for (name, v) in [
            ("column_bins", self.column_bins),
            ("response_bins", self.response_bins),
        ] {
            if v > 255 {
                return bad(name, format!("must be at most 255, not {v}"));
            }
        }
        Ok(())
    }
}

/// What [`Sampler::xaddivortes`](crate::Sampler::xaddivortes) returns
/// besides the new chain state.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastFit {
    /// The root mean square of the training residual of the fit, on the
    /// response's own scale; the noise draw of the model
    /// [`fit_xaddivortes`] returns. An in-sample estimate, not a posterior
    /// quantity: it understates the noise when the fit overfits, most on
    /// small data.
    pub sigma: f64,
    /// Replacement rounds run.
    pub rounds: usize,
    /// The work the fit charged, in sweeps (its count of operations).
    pub planned_sweeps: f64,
    /// The final score of each column: its mutual information with the
    /// response (or with the residual under
    /// [`rescore`](XAddiVortesParams::rescore)) less its expected value
    /// under independence, floored.
    pub column_scores: Vec<f64>,
    /// Tessellations left at zero because the budget ran out while
    /// building, and not replaced by a later round.
    pub zeroed: usize,
    /// The root mean square of the training residual after the build and
    /// after each round, on the response's own scale, in the form of
    /// scikit-learn's `GradientBoostingRegressor.train_score_`. Not charged
    /// to the budget.
    #[serde(default)]
    pub train_score: Vec<f64>,
}

/// The fast fit as a fitted model: a sampler on the chain-0 seed of
/// `seed` ([`chain_seed`](crate::chain_seed)`(seed, 0)`), the fit, kept as
/// the model's single draw. The prediction is the sum of the fitted cell
/// values, and the draw's noise variance is the square of
/// [`FastFit::sigma`], so every interval method of the model uses that
/// one value. Experimental: `RequiresFeature` without the feature.
///
/// # Errors
///
/// As [`Sampler::new`](crate::Sampler::new) and
/// [`Sampler::xaddivortes`](crate::Sampler::xaddivortes).
pub fn fit_xaddivortes(
    config: &crate::Config,
    x: &crate::Data,
    y: &[f64],
    seed: u64,
    params: &XAddiVortesParams,
) -> Result<(crate::Fitted, FastFit)> {
    let mut sampler = crate::Sampler::new(config, x, y, crate::chain_seed(seed, 0))?;
    let fast = sampler.xaddivortes(params)?;
    sampler.keep();
    Ok((sampler.finish()?, fast))
}

/// MCMC from fast starts (the warm start): `n_chains` chains seeded by
/// [`chain_seed`](crate::chain_seed)`(seed, k)` as in
/// [`fit_chains_with_threads`](crate::fit_chains_with_threads), each
/// started from its own fast fit under `params` instead of the cold
/// start, then the configured schedule (burn-in, draws, thinning) over at
/// most `n_threads` threads, the draws pooled in chain order. The second
/// value is the posterior mean at the rows of `x`. The fast fits run over
/// the same threads; each chain's fit draws from its own generator, so
/// the draws do not depend on `n_threads`. Experimental: `RequiresFeature`
/// without the feature.
///
/// # Errors
///
/// `InvalidHyperparameter` for no chains; as
/// [`Sampler::xaddivortes`](crate::Sampler::xaddivortes) otherwise.
pub fn fit_chains_from_fast(
    config: &crate::Config,
    x: &crate::Data,
    y: &[f64],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
    params: &XAddiVortesParams,
) -> Result<(crate::Fitted, Vec<f64>)> {
    if n_chains == 0 {
        return Err(invalid("n_chains", "a fit needs at least one chain"));
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = (config, x, y, seed, n_threads, params);
        Err(crate::Error::RequiresFeature {
            item: "fit_chains_from_fast".into(),
            feature: "experimental",
        })
    }
    #[cfg(feature = "experimental")]
    {
        let mut samplers = (0..n_chains)
            .map(|k| crate::Sampler::new(config, x, y, crate::chain_seed(seed, k)))
            .collect::<Result<Vec<_>>>()?;
        let mut chains: Vec<&mut crate::Sampler> = samplers.iter_mut().collect();
        crate::Sampler::xaddivortes_all(&mut chains, params, n_threads)?;
        run_from_fast(samplers, config, x, y, n_threads)
    }
}

#[cfg(feature = "experimental")]
fn run_from_fast(
    mut samplers: Vec<crate::Sampler>,
    config: &crate::Config,
    x: &crate::Data,
    y: &[f64],
    n_threads: usize,
) -> Result<(crate::Fitted, Vec<f64>)> {
    let schedule = &config.general_params;
    let mut chains: Vec<&mut crate::Sampler> = samplers.iter_mut().collect();
    crate::Sampler::advance_all(
        &mut chains,
        schedule.burn_in,
        schedule.draws,
        schedule.thinning,
        n_threads,
    );
    let (mut fitted, values) = crate::Fitted::pool_samplers(samplers, x, y)?;
    fitted.set_threads(n_threads);
    Ok((fitted, values))
}

#[cfg(feature = "experimental")]
pub(crate) use algorithm::{fit, Inputs};

#[cfg(feature = "experimental")]
mod algorithm {
    use super::{Worth, XAddiVortesParams};
    use crate::data::Data;
    use crate::geometry::Geometry;
    use crate::rng::{uniform, uniform_index, Rng};
    use crate::tessellation::{Assignment, Tessellation};

    /// What the fit reads, borrowed from the sampler.
    pub(crate) struct Inputs<'a> {
        /// Training covariates, scaled space.
        pub x: &'a Data,
        /// Response, scaled space.
        pub y: &'a [f64],
        /// The noise variance of the shrinkage (the chain's current one).
        pub sigma_sq: f64,
        /// The prior variance of one cell value.
        pub sigma_mu_sq: f64,
        /// The metric of every column.
        pub geometry: &'a Geometry,
        /// The cold start; its length is m, and a slot the budget never
        /// reaches keeps its shape with value zero.
        pub current: &'a [Tessellation],
    }

    /// What the fit returns.
    pub(crate) struct Outcome {
        pub tessellations: Vec<Tessellation>,
        /// Mean square of the final residual, scaled space.
        pub resid_ms: f64,
        pub rounds: usize,
        pub planned_sweeps: f64,
        pub column_scores: Vec<f64>,
        pub zeroed: usize,
        /// Root mean square of the residual after the build and after each
        /// round, scaled space.
        pub train_rms: Vec<f64>,
    }

    // The count's unit is one distance term (one centre on one column) for
    // one row. Every fit at a given budget depends on these constants.

    /// One sweep of the exact sampler, per row and tessellation.
    const SWEEP_OPS: f64 = 10.0;
    /// The fixed part of one pass over the rows, per row.
    const SCAN_OVERHEAD: f64 = 1.5;
    /// Building a tessellation (assignment, statistics and values), in
    /// plain passes on top of its centre scan.
    const BUILD_PASSES: f64 = 20.0;
    /// Scoring one screened pair, in plain passes; the columns cost
    /// `ln n + 2p` plain passes besides.
    const SCORE_PASSES_PER_PAIR: f64 = 3.0;

    /// The deterministic count: one sweep of the exact sampler costs
    /// `SWEEP_OPS` per row and tessellation; a pass over a tessellation
    /// with `cells` centres on `dims` columns costs `cells * dims +
    /// SCAN_OVERHEAD` per row.
    struct Cost {
        n: f64,
        sweep: f64,
        cap: f64,
        used: f64,
    }

    impl Cost {
        fn new(n: usize, m: usize, budget: f64) -> Self {
            let n = n as f64;
            let sweep = SWEEP_OPS * n * m as f64;
            Self {
                n,
                sweep,
                cap: (XAddiVortesParams::PLANNING_MARGIN * budget) * sweep,
                used: 0.0,
            }
        }

        /// A pass over `cells` centres on `dims` columns, in sweeps.
        fn pass_sweeps(&self, cells: f64, dims: f64) -> f64 {
            self.n * (cells * dims + SCAN_OVERHEAD) / self.sweep
        }

        fn charge_pass(&mut self, cells: usize, dims: usize) {
            self.used += self.n * ((cells * dims) as f64 + SCAN_OVERHEAD);
        }

        /// `factor` plain passes (no centres).
        fn charge_plain(&mut self, factor: f64) {
            self.used += factor * self.n * SCAN_OVERHEAD;
        }

        fn sweeps_left(&self) -> f64 {
            (self.cap - self.used) / self.sweep
        }

        fn sweeps_used(&self) -> f64 {
            self.used / self.sweep
        }
    }

    /// Column scores: single (length p) and the screened pairs, sparse and
    /// in increasing `lo * p + hi` order, so a draw over them equals a
    /// draw over the dense p by p array with zeros elsewhere.
    struct Scores {
        single: Vec<f64>,
        pairs: Vec<(usize, usize, f64)>,
    }

    /// Equal-width bins of every column over its training range; a constant
    /// column has one bin.
    fn column_bins(x: &Data, g: usize) -> Vec<Vec<u8>> {
        let (n, p) = (x.n_rows(), x.n_cols());
        let mut out = vec![vec![0u8; n]; p];
        for (q, bins) in out.iter_mut().enumerate() {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in 0..n {
                let v = x.row(i)[q];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let width = hi - lo;
            if width.is_nan() || width <= 0.0 {
                continue;
            }
            for (i, b) in bins.iter_mut().enumerate() {
                let v = x.row(i)[q];
                *b = (((v - lo) / width * g as f64).floor().max(0.0) as usize).min(g - 1) as u8;
            }
        }
        out
    }

    /// Rank bins of `r`: rows sorted by value, then `rank * rb / n`, every
    /// run of tied values taking the bin of its lowest rank, so the bins do
    /// not depend on the row order.
    fn rank_bins(r: &[f64], rb: usize) -> Vec<u8> {
        let n = r.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| r[a].total_cmp(&r[b]));
        let mut out = vec![0u8; n];
        let mut bin = 0;
        for (k, &i) in order.iter().enumerate() {
            if k == 0 || r[i].total_cmp(&r[order[k - 1]]).is_ne() {
                bin = (k * rb / n).min(rb - 1);
            }
            out[i] = bin as u8;
        }
        out
    }

    /// The plug-in mutual information of a binning (`nb` bins) with the
    /// rank bins `rbin` (`rb` bins), less its expected value under
    /// independence, `(kx - 1)(ky - 1) / 2n` over the `kx` and `ky`
    /// occupied bins: `2n` times the plug-in estimate is the G statistic,
    /// asymptotically chi-squared with that many degrees of freedom when
    /// the binnings are independent.
    fn mutual_information(bins: impl Fn(usize) -> usize, nb: usize, rbin: &[u8], rb: usize) -> f64 {
        let n = rbin.len();
        let nf = n as f64;
        let mut joint = vec![0.0; nb * rb];
        let mut px = vec![0.0; nb];
        let mut pr = vec![0.0; rb];
        for (i, &r) in rbin.iter().enumerate() {
            let b = bins(i);
            joint[b * rb + r as usize] += 1.0;
            px[b] += 1.0;
            pr[r as usize] += 1.0;
        }
        let mut v = 0.0;
        for b in 0..nb {
            for r in 0..rb {
                let c = joint[b * rb + r];
                if c > 0.0 {
                    v += c / nf * (c * nf / (px[b] * pr[r])).ln();
                }
            }
        }
        let occupied = |counts: &[f64]| counts.iter().filter(|&&c| c > 0.0).count();
        let (kx, ky) = (occupied(&px), occupied(&pr));
        v - (kx.saturating_sub(1) * ky.saturating_sub(1)) as f64 / (2.0 * nf)
    }

    fn scores(bins: &[Vec<u8>], r: &[f64], s: &XAddiVortesParams) -> Scores {
        let p = bins.len();
        let (g, rb) = (s.column_bins, s.response_bins);
        let rbin = rank_bins(r, rb);
        let raw: Vec<f64> = bins
            .iter()
            .map(|b| mutual_information(|i| b[i] as usize, g, &rbin, rb))
            .collect();
        let floor = s.score_floor * raw.iter().map(|v| v.max(0.0)).sum::<f64>() / p as f64 + 1e-9;
        let single: Vec<f64> = raw.iter().map(|v| v.max(0.0) + floor).collect();
        let mut cand: Vec<usize> = (0..p).collect();
        cand.sort_by(|&a, &b| single[b].total_cmp(&single[a]));
        cand.truncate(s.pair_screen);
        let mut pairs = Vec::with_capacity(cand.len() * cand.len().saturating_sub(1) / 2);
        let mut total = 0.0;
        for (ia, &a) in cand.iter().enumerate() {
            for &b in &cand[ia + 1..] {
                let (lo, hi) = (a.min(b), a.max(b));
                let (bl, bh) = (&bins[lo], &bins[hi]);
                let joint =
                    mutual_information(|i| bl[i] as usize * g + bh[i] as usize, g * g, &rbin, rb);
                let inter = (joint - raw[lo] - raw[hi]).max(0.0);
                pairs.push((lo, hi, inter));
                total += inter;
            }
        }
        let k = cand.len();
        let eps = s.score_floor * total / ((k * k.saturating_sub(1) / 2).max(1)) as f64 + 1e-9;
        for e in &mut pairs {
            e.2 += eps;
        }
        pairs.sort_by_key(|&(lo, hi, _)| lo * p + hi);
        Scores { single, pairs }
    }

    /// An index drawn in proportion to nonnegative `weights`, at least one
    /// positive (the last positive one on rounding overflow).
    fn draw_weighted(weights: impl Iterator<Item = f64> + Clone, rng: &mut Rng) -> usize {
        let total: f64 = weights.clone().sum();
        let mut u = uniform(rng) * total;
        let mut last = 0;
        for (k, w) in weights.enumerate() {
            if u < w {
                return k;
            }
            u -= w;
            if w > 0.0 {
                last = k;
            }
        }
        last
    }

    /// One fresh tessellation's columns (one uniform draw for a main effect
    /// or a pair, one more per further column) and `centres` centres at
    /// random training rows.
    fn fresh(
        x: &Data,
        d: usize,
        sc: &Scores,
        centres: usize,
        rng: &mut Rng,
    ) -> (Vec<usize>, Vec<f64>) {
        let p = sc.single.len();
        let mut dims = if d == 1 || p < 2 || sc.pairs.is_empty() {
            vec![draw_weighted(sc.single.iter().copied(), rng)]
        } else {
            let k = draw_weighted(sc.pairs.iter().map(|e| e.2), rng);
            vec![sc.pairs[k].0, sc.pairs[k].1]
        };
        while dims.len() < d.min(p) {
            let w = sc
                .single
                .iter()
                .enumerate()
                .map(|(q, &v)| if dims.contains(&q) { 0.0 } else { v });
            dims.push(draw_weighted(w, rng));
        }
        let n = x.n_rows();
        let mut c = Vec::with_capacity(centres * dims.len());
        for _ in 0..centres {
            let row = x.row(uniform_index(n, rng));
            c.extend(dims.iter().map(|&q| row[q]));
        }
        (dims, c)
    }

    fn tessellation(centres: Vec<f64>, dims: Vec<usize>, cells: usize) -> Tessellation {
        Tessellation {
            centres,
            dims,
            mus: vec![0.0; cells],
            betas: Vec::new(),
            tau: None,
        }
    }

    /// The cell index of a row: at most `u16::MAX` centres per
    /// tessellation (validated), so a cell index takes 2 bytes per row
    /// instead of 8.
    type Cell = u16;

    /// Buffers reused across every assignment of a fit.
    #[derive(Default)]
    struct Workspace {
        scratch: Vec<f64>,
        assignment: Assignment,
    }

    impl Workspace {
        /// The cell of every training row under `t`.
        fn cells(&mut self, x: &Data, t: &Tessellation, geometry: &Geometry) -> Vec<Cell> {
            Assignment::full_into(x, t, geometry, &mut self.scratch, &mut self.assignment);
            self.assignment.cells.iter().map(|&k| k as Cell).collect()
        }
    }

    /// The cell of every row under every tessellation, in one block of
    /// m by n indices: one allocation, returned whole when the fit ends.
    struct CellTable {
        n: usize,
        data: Vec<Cell>,
    }

    impl CellTable {
        fn new(m: usize, n: usize) -> Self {
            Self {
                n,
                data: vec![0; m * n],
            }
        }

        fn get(&self, j: usize) -> &[Cell] {
            &self.data[j * self.n..(j + 1) * self.n]
        }

        fn set(&mut self, j: usize, cells: &[Cell]) {
            self.data[j * self.n..(j + 1) * self.n].copy_from_slice(cells);
        }
    }

    /// A tessellation on `dims` with the given centres, every empty cell
    /// dropped, and the cell of every training row.
    fn nonempty(
        x: &Data,
        geometry: &Geometry,
        centres: Vec<f64>,
        dims: Vec<usize>,
        ws: &mut Workspace,
    ) -> (Tessellation, Vec<Cell>) {
        let d = dims.len();
        let b = centres.len() / d;
        let t = tessellation(centres, dims, b);
        let c = ws.cells(x, &t, geometry);
        let mut used = vec![false; b];
        for &k in &c {
            used[k as usize] = true;
        }
        if used.iter().all(|&u| u) {
            return (t, c);
        }
        let kept: Vec<usize> = (0..b).filter(|&k| used[k]).collect();
        let centres = kept
            .iter()
            .flat_map(|&k| t.centres[k * d..(k + 1) * d].to_vec())
            .collect();
        let t = tessellation(centres, t.dims, kept.len());
        let c = ws.cells(x, &t, geometry);
        (t, c)
    }

    /// Set `t`'s cell values to `rate` times the shrunk cell means of
    /// `resid`; returns the gain.
    fn refit(resid: &[f64], t: &mut Tessellation, cells: &[Cell], rate: f64, shrink: f64) -> f64 {
        let b = t.mus.len();
        let (mut sum, mut count) = (vec![0.0; b], vec![0.0; b]);
        for (i, &k) in cells.iter().enumerate() {
            sum[k as usize] += resid[i];
            count[k as usize] += 1.0;
        }
        let gain = (0..b).map(|k| sum[k] * sum[k] / (count[k] + shrink)).sum();
        for k in 0..b {
            t.mus[k] = rate * sum[k] / (count[k] + shrink);
        }
        gain
    }

    fn add(resid: &mut [f64], t: &Tessellation, cells: &[Cell]) {
        for (r, &q) in resid.iter_mut().zip(cells) {
            *r += t.mus[q as usize];
        }
    }

    /// How much subtracting `t`'s values from `resid` lowers its sum of
    /// squares: `sum(2 r mu - mu^2)` over the rows.
    fn rss_drop(resid: &[f64], t: &Tessellation, cells: &[Cell]) -> f64 {
        resid
            .iter()
            .zip(cells)
            .map(|(&r, &q)| {
                let mu = t.mus[q as usize];
                2.0 * r * mu - mu * mu
            })
            .sum()
    }

    fn mean_square(resid: &[f64]) -> f64 {
        resid.iter().map(|r| r * r).sum::<f64>() / resid.len() as f64
    }

    fn rms(resid: &[f64]) -> f64 {
        mean_square(resid).sqrt()
    }

    fn subtract(resid: &mut [f64], t: &Tessellation, cells: &[Cell]) {
        for (r, &q) in resid.iter_mut().zip(cells) {
            *r -= t.mus[q as usize];
        }
    }

    fn zeroed(t: &Tessellation) -> Tessellation {
        let mut t = t.clone();
        t.mus.iter_mut().for_each(|v| *v = 0.0);
        t
    }

    /// The fit. `s` must have passed [`XAddiVortesParams::validate`].
    pub(crate) fn fit(inp: &Inputs<'_>, s: &XAddiVortesParams, rng: &mut Rng) -> Outcome {
        let x = inp.x;
        let (n, p, m) = (x.n_rows(), x.n_cols(), inp.current.len());
        let mut cost = Cost::new(n, m, s.budget);
        let bins = column_bins(x, s.column_bins);
        let mut sc = scores(&bins, inp.y, s);
        let screen = s.pair_screen as f64;
        let pairs_screened = screen * (screen - 1.0) / 2.0;
        let score_passes =
            (n as f64).ln() + 2.0 * p as f64 + SCORE_PASSES_PER_PAIR * pairs_screened;
        cost.charge_plain(score_passes);

        let mut sigma_sq = inp.sigma_sq;
        let mut shrink = sigma_sq / inp.sigma_mu_sq;
        let mains = (m as f64 * s.main_effect_share).floor() as usize;
        let replace = (m as f64 * s.replace_share).floor() as usize;
        let build = cost.pass_sweeps(s.centres as f64, s.max_columns as f64)
            + BUILD_PASSES * cost.pass_sweeps(1.0, 0.0);

        let mut resid = inp.y.to_vec();
        let mut ts: Vec<Tessellation> = Vec::with_capacity(m);
        let mut cells = CellTable::new(m, n);
        let mut ws = Workspace::default();
        let mut gains: Vec<f64> = Vec::with_capacity(m);
        // A slot the build never reaches stays at zero, is not refitted, and
        // is the first a round replaces (its worth is zero).
        let mut built = vec![true; m];
        for j in 0..m {
            let d = if j < mains { 1 } else { s.max_columns }.min(p);
            if cost.sweeps_left() <= build {
                built[j..].iter_mut().for_each(|b| *b = false);
                ts.extend(inp.current[j..].iter().map(zeroed));
                gains.extend((j..m).map(|_| 0.0));
                break;
            }
            let (dims, centres) = fresh(x, d, &sc, s.centres, rng);
            cost.charge_pass(s.centres, d);
            cost.charge_plain(BUILD_PASSES);
            let (mut t, c) = nonempty(x, inp.geometry, centres, dims, &mut ws);
            let gain = refit(&resid, &mut t, &c, s.value_scale, shrink);
            subtract(&mut resid, &t, &c);
            ts.push(t);
            cells.set(j, &c);
            gains.push(gain);
        }

        let mut train_rms = vec![rms(&resid)];
        let mut rounds = 0;
        loop {
            let pass_cost: f64 = s.refit_passes as f64
                * ts.iter()
                    .zip(&built)
                    .filter(|(_, &b)| b)
                    .map(|(t, _)| cost.pass_sweeps(t.mus.len() as f64, 0.0))
                    .sum::<f64>();
            let rank_cost = cost.pass_sweeps(1.0, 0.0);
            let mut round_cost = (replace * s.tries) as f64 * build + pass_cost + rank_cost;
            if s.accept_if_better {
                round_cost +=
                    2.0 * (replace + s.refit_passes * m) as f64 * cost.pass_sweeps(1.0, 0.0);
            }
            if s.rescore {
                round_cost += score_passes * cost.pass_sweeps(1.0, 0.0);
            }
            if rounds >= s.max_rounds || cost.sweeps_left() <= round_cost {
                break;
            }
            let worth: Vec<f64> = match s.worth {
                Worth::SquaredValues => (0..m)
                    .map(|j| {
                        cells
                            .get(j)
                            .iter()
                            .map(|&q| ts[j].mus[q as usize] * ts[j].mus[q as usize])
                            .sum::<f64>()
                    })
                    .collect(),
                Worth::Gain => gains.clone(),
            };
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&a, &b| worth[a].total_cmp(&worth[b]));
            cost.charge_pass(1, 0);
            for &j in &order[..replace] {
                add(&mut resid, &ts[j], cells.get(j));
                let d = ts[j].dims.len();
                let mut best: Option<(Tessellation, Vec<Cell>, f64)> = None;
                for _ in 0..s.tries {
                    let (dims, centres) = fresh(x, d, &sc, s.centres, rng);
                    cost.charge_pass(s.centres, dims.len());
                    cost.charge_plain(BUILD_PASSES);
                    let (mut t, c) = nonempty(x, inp.geometry, centres, dims, &mut ws);
                    let gain = refit(&resid, &mut t, &c, s.value_scale, shrink);
                    if best.as_ref().map_or(true, |b| gain > b.2) {
                        best = Some((t, c, gain));
                    }
                }
                let (t, c, gain) = best.expect("tries is at least 1");
                if s.accept_if_better {
                    cost.charge_plain(2.0);
                    if rss_drop(&resid, &ts[j], cells.get(j)) >= rss_drop(&resid, &t, &c) {
                        subtract(&mut resid, &ts[j], cells.get(j));
                        continue;
                    }
                }
                subtract(&mut resid, &t, &c);
                ts[j] = t;
                cells.set(j, &c);
                gains[j] = gain;
                built[j] = true;
            }
            for _ in 0..s.refit_passes {
                for j in (0..m).filter(|&j| built[j]) {
                    add(&mut resid, &ts[j], cells.get(j));
                    cost.charge_pass(ts[j].mus.len(), 0);
                    let old = s.accept_if_better.then(|| (ts[j].mus.clone(), gains[j]));
                    let before = old.as_ref().map(|_| rss_drop(&resid, &ts[j], cells.get(j)));
                    gains[j] = refit(&resid, &mut ts[j], cells.get(j), s.value_scale, shrink);
                    if let (Some((mus, gain)), Some(before)) = (old, before) {
                        cost.charge_plain(2.0);
                        if before > rss_drop(&resid, &ts[j], cells.get(j)) {
                            ts[j].mus = mus;
                            gains[j] = gain;
                        }
                    }
                    subtract(&mut resid, &ts[j], cells.get(j));
                }
            }
            if s.update_sigma {
                sigma_sq = resid.iter().map(|r| r * r).sum::<f64>() / n as f64;
                shrink = sigma_sq / inp.sigma_mu_sq;
            }
            if s.rescore {
                sc = scores(&bins, &resid, s);
                cost.charge_plain(score_passes);
            }
            rounds += 1;
            train_rms.push(rms(&resid));
        }

        Outcome {
            tessellations: ts,
            resid_ms: mean_square(&resid),
            rounds,
            planned_sweeps: cost.sweeps_used(),
            column_scores: sc.single,
            zeroed: built.iter().filter(|&&b| !b).count(),
            train_rms,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rank_bins_follow_rank_and_tied_values_share_a_bin() {
            assert_eq!(rank_bins(&[3.0, 1.0, 2.0, 0.0], 4), vec![3, 1, 2, 0]);
            assert_eq!(rank_bins(&[1.0, 1.0, 0.0, 2.0], 2), vec![0, 0, 0, 1]);
            assert_eq!(rank_bins(&[0.0, 1.0, 0.0, 1.0], 4), vec![0, 2, 0, 2]);
        }

        #[test]
        fn scores_do_not_depend_on_the_row_order() {
            let n = 200;
            let rows: Vec<[f64; 2]> = (0..n)
                .map(|i| [(i % 17) as f64, ((i * 7919) % 101) as f64])
                .collect();
            let y: Vec<f64> = rows.iter().map(|r| f64::from(r[0] > 8.0)).collect();
            let order: Vec<usize> = (0..n).map(|i| (i * 37) % n).collect();
            let permuted_rows: Vec<[f64; 2]> = order.iter().map(|&i| rows[i]).collect();
            let permuted_y: Vec<f64> = order.iter().map(|&i| y[i]).collect();
            let s = XAddiVortesParams::default();
            let a = scores(&column_bins(&Data::from_rows(&rows).unwrap(), 8), &y, &s);
            let b = scores(
                &column_bins(&Data::from_rows(&permuted_rows).unwrap(), 8),
                &permuted_y,
                &s,
            );
            for (u, v) in a.single.iter().zip(&b.single) {
                assert!((u - v).abs() < 1e-12, "{u} vs {v}");
            }
        }

        #[test]
        fn column_bins_are_equal_width_over_the_training_range() {
            let x = Data::from_rows(&[
                [-0.5, 3.0, 7.0],
                [0.0, 5.0, 7.0],
                [0.5, 4.0, 7.0],
                [0.49, 3.5, 7.0],
            ])
            .unwrap();
            let b = column_bins(&x, 8);
            assert_eq!(b[0], vec![0, 4, 7, 7]);
            assert_eq!(b[1], vec![0, 7, 4, 2]);
            assert_eq!(b[2], vec![0, 0, 0, 0], "a constant column has one bin");
        }

        #[test]
        fn mutual_information_of_a_perfect_binning_is_ln_bins_less_the_correction() {
            let n = 400;
            let rbin: Vec<u8> = (0..n).map(|i| (i * 4 / n) as u8).collect();
            let mi = mutual_information(|i| rbin[i] as usize, 4, &rbin, 4);
            let want = 4f64.ln() - (3 * 3) as f64 / (2.0 * n as f64);
            assert!((mi - want).abs() < 1e-12, "{mi} vs {want}");
            let constant = mutual_information(|_| 0, 4, &rbin, 4);
            assert!(constant.abs() < 1e-12);
        }

        #[test]
        fn the_correction_counts_occupied_bins_only() {
            let n = 400;
            // A binary response fills 2 of the 4 rank bins.
            let rbin: Vec<u8> = (0..n).map(|i| if i < n / 2 { 0 } else { 2 }).collect();
            let mi = mutual_information(|i| usize::from(rbin[i] > 0), 4, &rbin, 4);
            let want = 2f64.ln() - 1.0 / (2.0 * n as f64);
            assert!((mi - want).abs() < 1e-12, "{mi} vs {want}");
        }

        #[test]
        fn scores_favour_the_informative_column_and_its_pair() {
            let n = 400;
            let rows: Vec<[f64; 3]> = (0..n)
                .map(|i| {
                    [
                        i as f64 / n as f64 - 0.5,
                        ((i * 7919) % n) as f64 / n as f64 - 0.5,
                        ((i * 104_729) % n) as f64 / n as f64 - 0.5,
                    ]
                })
                .collect();
            let y: Vec<f64> = rows.iter().map(|r| r[0]).collect();
            let x = Data::from_rows(&rows).unwrap();
            let s = XAddiVortesParams::default();
            let sc = scores(&column_bins(&x, 8), &y, &s);
            assert!(sc.single[0] > 10.0 * sc.single[1] && sc.single[0] > 10.0 * sc.single[2]);
            assert!(
                sc.single.iter().all(|&v| v > 0.0),
                "the floor keeps every column drawable"
            );
            assert_eq!(sc.pairs.len(), 3);
            let keys: Vec<usize> = sc.pairs.iter().map(|e| e.0 * 3 + e.1).collect();
            assert!(
                keys.windows(2).all(|w| w[0] < w[1]),
                "pairs in increasing lo * p + hi order"
            );
            assert!(sc.pairs.iter().all(|e| e.0 < e.1 && e.2 > 0.0));
        }

        #[test]
        fn pair_screen_limits_the_pairs() {
            let rows: Vec<[f64; 5]> = (0..50)
                .map(|i| {
                    [
                        i as f64,
                        (i % 3) as f64,
                        (i % 5) as f64,
                        (i % 7) as f64,
                        (i % 11) as f64,
                    ]
                })
                .collect();
            let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
            let x = Data::from_rows(&rows).unwrap();
            let sc = scores(
                &column_bins(&x, 8),
                &y,
                &XAddiVortesParams::default().with_pair_screen(3),
            );
            assert_eq!(sc.pairs.len(), 3);
        }

        #[test]
        fn a_sparse_draw_equals_the_dense_draw() {
            use crate::rng::chain_rng;
            let dense = [0.0, 0.3, 0.0, 0.0, 1.2, 0.0, 0.5, 0.0];
            let sparse: Vec<(usize, f64)> = dense
                .iter()
                .copied()
                .enumerate()
                .filter(|e| e.1 > 0.0)
                .collect();
            for seed in 0..200 {
                let (mut a, mut b) = (chain_rng(seed), chain_rng(seed));
                let k = draw_weighted(dense.iter().copied(), &mut a);
                let j = draw_weighted(sparse.iter().map(|e| e.1), &mut b);
                assert_eq!(k, sparse[j].0);
            }
        }

        #[test]
        fn cell_values_are_the_scaled_shrunk_means() {
            let mut t = crate::tessellation::Tessellation {
                centres: vec![0.0, 1.0],
                dims: vec![0],
                mus: vec![0.0, 0.0],
                betas: Vec::new(),
                tau: None,
            };
            let resid = [1.0, 3.0, -2.0];
            let cells = [0, 0, 1];
            let gain = refit(&resid, &mut t, &cells, 0.5, 1.0);
            assert!((t.mus[0] - 0.5 * 4.0 / 3.0).abs() < 1e-15);
            assert!((t.mus[1] - 0.5 * -2.0 / 2.0).abs() < 1e-15);
            assert!((gain - (16.0 / 3.0 + 4.0 / 2.0)).abs() < 1e-12);
        }

        #[test]
        fn empty_cells_are_dropped() {
            let x = Data::from_rows(&[[0.0], [0.1], [0.9]]).unwrap();
            let geometry = crate::geometry::Geometry::euclidean(1);
            // Centre 0.5 is nearest to no row once 0.05 and 0.95 are present.
            let (t, c) = nonempty(
                &x,
                &geometry,
                vec![0.05, 0.5, 0.95],
                vec![0],
                &mut Workspace::default(),
            );
            assert_eq!(t.centres, vec![0.05, 0.95]);
            assert_eq!(c, vec![0, 0, 1]);
        }

        #[test]
        fn the_cost_model_counts_passes_in_sweeps() {
            let mut c = Cost::new(100, 10, 20.0);
            // One sweep is 10 per row and tessellation: 10 * 100 * 10.
            assert!((c.sweeps_left() - 0.8 * 20.0).abs() < 1e-12);
            c.charge_pass(16, 2);
            assert!((c.sweeps_used() - 100.0 * 33.5 / 10_000.0).abs() < 1e-12);
            assert!((c.pass_sweeps(16.0, 2.0) - 100.0 * 33.5 / 10_000.0).abs() < 1e-12);
            c.charge_plain(2.0);
            assert!((c.sweeps_used() - 100.0 * (33.5 + 3.0) / 10_000.0).abs() < 1e-12);
        }
    }
}
