//! XAddiVortes: a fast point fit of the AddiVortes ensemble. The method is
//! documented on [`XAddiVortesParams`], which the public documentation
//! renders.
//!
//! Experimental (`docs/experimental.md`): the types exist in every build,
//! the fit only with the `experimental` feature.

use serde::{Deserialize, Serialize};

use crate::error::{invalid, Result};

/// Settings of XAddiVortes, a fast point fit of the AddiVortes ensemble.
///
/// [`Sampler::xaddivortes`](crate::Sampler::xaddivortes) replaces the
/// chain's cold start by the fit, so the fit serves two uses: on its own,
/// the prediction being the sum of the fitted cell values; or as the
/// initial state from which the exact sampler runs on, which changes the
/// chain's starting point and not its stationary distribution. The fit is
/// a single point estimate, not a posterior sample.
///
/// The fit, in order:
///
/// 1. Screen the columns: each column's binned mutual information with
///    the response (8 equal-width bins of the column over its training
///    range, 4 equal-count rank bins of the response, tied responses
///    ranked in row order), less its expected value under independence,
///    clamped at zero, plus a floor of 2% of the mean score.
/// 2. Build the m tessellations in turn. Each takes a shape, a number of
///    columns from 1 to [`max_columns`](Self::max_columns) and a number of
///    centres from [`centres`](Self::centres) and its halvings down to 4,
///    both uniformly; its columns are drawn without replacement in
///    proportion to their screening scores and its centres at random
///    training rows; empty cells are dropped; its cell values are
///    `value_scale * S_k / (n_k + sigma^2 / sigma_mu^2)`, with `S_k` the
///    running residual summed over cell k of `n_k` rows, and are then
///    subtracted from the residual. Every tessellation's cell values are
///    then refitted to its partial residual, twice over.
/// 3. [`rounds`](Self::rounds) times: replace the `rho m` tessellations
///    (rounded to a whole number), rho the
///    [`replace_share`](Self::replace_share), with the smallest log Bayes
///    factor at the last refit (below) by fresh ones; refit every
///    tessellation's cell values to its partial residual, twice over; set
///    the noise variance to the residual sum of squares over the residual
///    degrees of freedom of those refits (below). A fresh tessellation
///    draws its number of columns, its
///    number of centres and each of its columns in proportion to one plus
///    the number of tessellations with that value at the start of the
///    round: the posterior mean of the proportions under a uniform
///    Dirichlet prior given those counts (compare Linero, 2018).
/// 4. The final refit: refit every tessellation's cell values ten times
///    over at each damping from `value_scale` to 1 in three equal steps,
///    in place of `value_scale`, and keep the damping with the smallest
///    generalised cross-validation score `RSS / (1 - 1.4 tr(H) / n)^2`
///    (Craven and Wahba, 1979), the degrees of freedom inflated by 1.4 as
///    Kim and Gu (2004) suggest; H maps a response to the fitted values of
///    those refits from zero on the final cells.
///
/// The log Bayes factor of a tessellation is the log ratio of the
/// integrated likelihoods of the model with and without it, its cell
/// values integrated over their N(0, sigma_mu^2) prior given the other
/// tessellations (Chipman and others, 2010):
/// `S_k^2 / (2 sigma^2 (n_k + lambda)) - ln(1 + n_k / lambda) / 2` summed
/// over its cells, with `S_k` its partial residual summed over cell k and
/// `lambda = sigma^2 / sigma_mu^2`. The second term charges every cell, so
/// a tessellation with more cells must explain more of the residual to be
/// kept.
///
/// The residual degrees of freedom of refits on fixed cells are
/// `n - 2 tr(H) + tr(H'H)`, at least 1, for the smoother H of those
/// refits from zero (Hastie and Tibshirani, 1990, section 3.5), the traces
/// estimated from two random sign vectors (Hutchinson, 1989). They do not
/// count the degrees of freedom spent choosing the cells, so the noise
/// estimates can still understate the noise; split-conformal intervals
/// (Lei and others, 2018) around the fit are calibrated without that
/// assumption. [`FastFit::sigma`] is the root of the final residual sum of
/// squares over the kept final refit's residual degrees of freedom.
///
/// The work is `O(n m (rho C D + 1) + rho m D p)` per round for n rows, p
/// columns, m tessellations, C centres and D columns, and `O(n m)` per
/// pass of the final refit. Besides the design, the fit holds the cell of
/// every row under every tessellation, one byte each (two above 256
/// centres), and at most 12 values per row in the final refit.
///
/// Experimental (`docs/experimental.md`): the type exists in every build,
/// the fit only with the `experimental` feature.
///
/// # References
///
/// - Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
///   Bayesian additive regression trees. Annals of Applied Statistics
///   4(1), 266-298.
/// - Craven, P. and Wahba, G. (1979). Smoothing noisy data with spline
///   functions. Numerische Mathematik 31(4), 377-403.
/// - Hastie, T. J. and Tibshirani, R. J. (1990). Generalized Additive
///   Models. Chapman and Hall, London.
/// - Hutchinson, M. F. (1989). A stochastic estimator of the trace of the
///   influence matrix for Laplacian smoothing splines. Communications in
///   Statistics: Simulation and Computation 18(3), 1059-1076.
/// - Kim, Y.-J. and Gu, C. (2004). Smoothing spline Gaussian regression:
///   more scalable computation via efficient approximation. Journal of the
///   Royal Statistical Society Series B 66(2), 337-356.
/// - Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
///   (2018). Distribution-free predictive inference for regression.
///   Journal of the American Statistical Association 113(523), 1094-1111.
/// - Linero, A. R. (2018). Bayesian regression trees for high-dimensional
///   prediction and variable selection. Journal of the American
///   Statistical Association 113(522), 626-636.
/// - Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
///   Voronoi tessellations. Journal of Computational and Graphical
///   Statistics 34(3), 859-871.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct XAddiVortesParams {
    /// Replacement rounds after the build (default 40). The work grows
    /// linearly with it. With 0 the build goes straight to the final
    /// refit.
    pub rounds: usize,
    /// The most columns of a tessellation (default 3); capped at the
    /// number of columns. At least 1.
    pub max_columns: usize,
    /// The most centres of a tessellation, drawn at random training rows
    /// before empty cells are dropped (default 64); fresh tessellations
    /// take it or one of its halvings down to 4. From 1 to 65,535.
    pub centres: usize,
    /// The share of the shrunk cell mean used as the cell value, nu, in
    /// the build and the rounds (default 0.5); the final refit chooses
    /// from it up to 1. Smaller values fit more slowly and overfit less.
    /// In (0, 1].
    pub value_scale: f64,
    /// The share of tessellations replaced each round, rho (default 0.3):
    /// `rho m` of the m, rounded to the nearest whole number. In [0, 1]; 0
    /// makes the rounds pure refits.
    pub replace_share: f64,
}

impl Default for XAddiVortesParams {
    fn default() -> Self {
        Self {
            rounds: 40,
            max_columns: 3,
            centres: 64,
            value_scale: 0.5,
            replace_share: 0.3,
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
    setter! {
        /// Set [`rounds`](Self::rounds).
        with_rounds => rounds: usize,
        /// Set [`max_columns`](Self::max_columns).
        with_max_columns => max_columns: usize,
        /// Set [`centres`](Self::centres).
        with_centres => centres: usize,
        /// Set [`value_scale`](Self::value_scale).
        with_value_scale => value_scale: f64,
        /// Set [`replace_share`](Self::replace_share).
        with_replace_share => replace_share: f64,
    }

    /// Check every setting is in its range.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter`, naming the field as `xaddivortes.<name>`.
    pub fn validate(&self) -> Result<()> {
        let bad = |name: &str, why: String| Err(invalid(&format!("xaddivortes.{name}"), why));
        if !(self.value_scale > 0.0 && self.value_scale <= 1.0) {
            return bad(
                "value_scale",
                format!("must be in (0, 1], not {}", self.value_scale),
            );
        }
        if !(0.0..=1.0).contains(&self.replace_share) {
            return bad(
                "replace_share",
                format!("must be in [0, 1], not {}", self.replace_share),
            );
        }
        for (name, v) in [("max_columns", self.max_columns), ("centres", self.centres)] {
            if v < 1 {
                return bad(name, format!("must be at least 1, not {v}"));
            }
        }
        if self.centres > usize::from(u16::MAX) {
            return bad(
                "centres",
                format!("must be at most {}, not {}", u16::MAX, self.centres),
            );
        }
        Ok(())
    }
}

/// What [`Sampler::xaddivortes`](crate::Sampler::xaddivortes) returns
/// besides the new chain state.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastFit {
    /// The noise standard deviation on the response's own scale: the
    /// root of the training residual sum of squares over its residual
    /// degrees of freedom ([`XAddiVortesParams`]); the noise draw of the
    /// model [`fit_xaddivortes`] returns. A point estimate, not a
    /// posterior quantity.
    pub sigma: f64,
    /// Each column's screening score: its binned mutual information with
    /// the response less its expected value under independence, clamped
    /// at zero, plus the floor.
    pub column_scores: Vec<f64>,
    /// The root mean square of the training residual after the build,
    /// after each round and after the final refit, on the response's own
    /// scale, in the form of scikit-learn's
    /// `GradientBoostingRegressor.train_score_`.
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
    let mut sampler = crate::Sampler::unassigned(config, x, y, crate::chain_seed(seed, 0))?;
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
            .map(|k| crate::Sampler::unassigned(config, x, y, crate::chain_seed(seed, k)))
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
    use super::XAddiVortesParams;
    use crate::data::Data;
    use crate::geometry::Geometry;
    use crate::rng::{uniform, uniform_index, Rng};
    use crate::tessellation::{nearest_blocks, Tessellation};

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
        /// The number of tessellations.
        pub m: usize,
    }

    /// What the fit returns.
    pub(crate) struct Outcome {
        pub tessellations: Vec<Tessellation>,
        /// The final residual sum of squares over its residual degrees of
        /// freedom, scaled space.
        pub sigma_sq: f64,
        pub column_scores: Vec<f64>,
        /// Root mean square of the residual after the build, after each
        /// round and after the final refit, scaled space.
        pub train_rms: Vec<f64>,
        /// The cell of every training row under each tessellation.
        pub cells: FitCells,
    }

    /// Equal-width bins of a column for its screening score.
    const SCREEN_COLUMN_BINS: usize = 8;
    /// Equal-count rank bins of the response for the screening scores.
    const SCREEN_RESPONSE_BINS: usize = 4;
    /// The floor added to every screening score, as a share of the mean.
    const SCREEN_FLOOR: f64 = 0.02;
    /// Refit passes over all tessellations per round.
    const REFIT_PASSES: usize = 2;
    /// Refit passes at each candidate damping of the final refit.
    const FINAL_PASSES: usize = 10;
    /// Random sign vectors for each trace estimate.
    const PROBES: usize = 2;
    /// The factor on the degrees of freedom in the generalised
    /// cross-validation score (Kim and Gu, 2004).
    const GCV_INFLATION: f64 = 1.4;

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

    /// Rank bins of `r`: rows sorted by value (ties in row order), then
    /// `rank * rb / n`.
    fn rank_bins(r: &[f64], rb: usize) -> Vec<u8> {
        let n = r.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| r[a].total_cmp(&r[b]));
        let mut out = vec![0u8; n];
        for (k, &i) in order.iter().enumerate() {
            out[i] = ((k * rb / n).min(rb - 1)) as u8;
        }
        out
    }

    /// The plug-in mutual information of a binning (`nb` bins) with the
    /// rank bins `rbin` (`rb` bins, each taken to hold `n / rb` rows), less
    /// its expected value under independence, `(kx - 1)(rb - 1) / 2n` over
    /// the `kx` occupied column bins: `2n` times the plug-in estimate is
    /// the G statistic, asymptotically chi-squared with that many degrees
    /// of freedom when the binnings are independent.
    fn mutual_information(bins: impl Fn(usize) -> usize, nb: usize, rbin: &[u8], rb: usize) -> f64 {
        let n = rbin.len();
        let nf = n as f64;
        let mut joint = vec![0.0; nb * rb];
        let mut px = vec![0.0; nb];
        for (i, &r) in rbin.iter().enumerate() {
            let b = bins(i);
            joint[b * rb + r as usize] += 1.0;
            px[b] += 1.0;
        }
        let pr = nf / rb as f64;
        let mut v = 0.0;
        let mut kx = 0usize;
        for b in 0..nb {
            for r in 0..rb {
                let c = joint[b * rb + r];
                if c > 0.0 {
                    v += c / nf * crate::maths::ln(c * nf / (px[b] * pr));
                }
            }
            if px[b] > 0.0 {
                kx += 1;
            }
        }
        v - (kx.saturating_sub(1) * (rb - 1)) as f64 / (2.0 * nf)
    }

    /// The screening score of every column against `y`, floored so that
    /// every column can be drawn.
    fn screening_scores(x: &Data, y: &[f64]) -> Vec<f64> {
        let (g, rb) = (SCREEN_COLUMN_BINS, SCREEN_RESPONSE_BINS);
        let rbin = rank_bins(y, rb);
        let raw: Vec<f64> = column_bins(x, g)
            .iter()
            .map(|b| mutual_information(|i| b[i] as usize, g, &rbin, rb))
            .collect();
        let floor =
            SCREEN_FLOOR * raw.iter().map(|v| v.max(0.0)).sum::<f64>() / raw.len() as f64 + 1e-9;
        raw.iter().map(|v| v.max(0.0) + floor).collect()
    }

    /// One plus the number of tessellations on each of `p` columns.
    fn usage_weights(ts: &[Tessellation], p: usize) -> Vec<f64> {
        let mut w = vec![1.0; p];
        ts.iter().flat_map(|t| &t.dims).for_each(|&q| w[q] += 1.0);
        w
    }

    /// The centre counts a fresh tessellation can take: `centres` and its
    /// successive halvings (rounded down) while they are at least 4, in
    /// increasing order.
    fn centre_ladder(centres: usize) -> Vec<usize> {
        let mut ladder = vec![centres];
        while ladder[ladder.len() - 1] / 2 >= 4 {
            ladder.push(ladder[ladder.len() - 1] / 2);
        }
        ladder.reverse();
        ladder
    }

    /// One plus the number of `shapes` taking each of `len` values of
    /// `pick`.
    fn shape_weights(
        shapes: &[(usize, usize)],
        len: usize,
        pick: fn(&(usize, usize)) -> usize,
    ) -> Vec<f64> {
        let mut w = vec![1.0; len];
        shapes.iter().for_each(|e| w[pick(e)] += 1.0);
        w
    }

    /// The log Bayes factor of a tessellation against none, its values
    /// integrated over their N(0, sigma_mu^2) prior given the other
    /// tessellations: `gain / (2 sigma^2) - sum_k ln(1 + n_k / lambda) / 2`
    /// with `lambda = sigma^2 / sigma_mu^2` (Chipman and others, 2010,
    /// section 3.1).
    fn log_bayes_factor(gain: f64, count: &[f64], sigma_sq: f64, shrink: f64) -> f64 {
        gain / (2.0 * sigma_sq)
            - 0.5
                * count
                    .iter()
                    .map(|&c| crate::maths::ln_1p(c / shrink))
                    .sum::<f64>()
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

    /// One fresh tessellation's columns, `d` of them drawn without
    /// replacement in proportion to `weights` (one uniform each), and
    /// `centres` centres at random training rows.
    fn fresh(
        x: &Data,
        d: usize,
        weights: &[f64],
        centres: usize,
        rng: &mut Rng,
    ) -> (Vec<usize>, Vec<f64>) {
        let mut dims: Vec<usize> = Vec::with_capacity(d);
        while dims.len() < d.min(weights.len()) {
            let w = weights
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

    /// A row's cell index as the fit stores it, one per row and
    /// tessellation: `u8` when every tessellation has at most 256 centres,
    /// `u16` up to the validated `u16::MAX`.
    pub(crate) trait Cell: Copy + Default {
        /// `k`, which must fit.
        fn of(k: usize) -> Self;
        fn wrap(table: CellTable<Self>) -> FitCells;
        fn index(self) -> usize;
    }

    impl Cell for u8 {
        fn of(k: usize) -> Self {
            k as u8
        }

        fn wrap(table: CellTable<Self>) -> FitCells {
            FitCells::Narrow(table)
        }

        fn index(self) -> usize {
            usize::from(self)
        }
    }

    impl Cell for u16 {
        fn of(k: usize) -> Self {
            k as u16
        }

        fn wrap(table: CellTable<Self>) -> FitCells {
            FitCells::Wide(table)
        }

        fn index(self) -> usize {
            usize::from(self)
        }
    }

    /// The cells a fit ends with, at the width it stored them.
    pub(crate) enum FitCells {
        Narrow(CellTable<u8>),
        Wide(CellTable<u16>),
    }

    impl FitCells {
        /// The cell of every row under tessellation `j`, into `out`.
        pub(crate) fn fill(&self, j: usize, out: &mut Vec<usize>) {
            out.clear();
            match self {
                FitCells::Narrow(t) => out.extend(t.get(j).iter().map(|c| c.index())),
                FitCells::Wide(t) => out.extend(t.get(j).iter().map(|c| c.index())),
            }
        }
    }

    /// Buffers reused across every assignment of a fit.
    #[derive(Default)]
    struct Workspace<C> {
        /// The cells of the latest assignment.
        cells: Vec<C>,
    }

    impl<C: Cell> Workspace<C> {
        /// The cell of every training row under `t` into `self.cells`: the
        /// nearest centre, the lowest index on ties, from the keys
        /// [`Assignment::full`](crate::tessellation::Assignment) forms, in
        /// the same order, one centre at a time.
        fn assign(&mut self, x: &Data, t: &Tessellation, geometry: &Geometry) {
            let n = x.n_rows();
            self.cells.clear();
            if !geometry.is_plain() {
                self.cells
                    .extend((0..n).map(|i| C::of(t.nearest(x.row(i), geometry).0)));
                return;
            }
            self.cells.reserve(n);
            let cells = &mut self.cells;
            nearest_blocks(x, t, |_, block| {
                cells.extend(block.iter().map(|&k| C::of(k as usize)));
            });
        }
    }

    /// The cell of every row under every tessellation, in one block of
    /// m by n indices: one allocation, returned whole when the fit ends.
    pub(crate) struct CellTable<C> {
        n: usize,
        data: Vec<C>,
    }

    impl<C: Cell> CellTable<C> {
        fn new(m: usize, n: usize) -> Self {
            Self {
                n,
                data: vec![C::default(); m * n],
            }
        }

        fn get(&self, j: usize) -> &[C] {
            &self.data[j * self.n..(j + 1) * self.n]
        }

        fn set(&mut self, j: usize, cells: &[C]) {
            self.data[j * self.n..(j + 1) * self.n].copy_from_slice(cells);
        }
    }

    /// A tessellation on `dims` with the given centres, every empty cell
    /// dropped; the cell of every training row is left in `ws.cells`.
    fn nonempty<C: Cell>(
        x: &Data,
        geometry: &Geometry,
        centres: Vec<f64>,
        dims: Vec<usize>,
        ws: &mut Workspace<C>,
    ) -> Tessellation {
        let d = dims.len();
        let b = centres.len() / d;
        let t = tessellation(centres, dims, b);
        ws.assign(x, &t, geometry);
        let mut used = vec![false; b];
        for &k in &ws.cells {
            used[k.index()] = true;
        }
        if used.iter().all(|&u| u) {
            return t;
        }
        let kept: Vec<usize> = (0..b).filter(|&k| used[k]).collect();
        let centres = kept
            .iter()
            .flat_map(|&k| t.centres[k * d..(k + 1) * d].to_vec())
            .collect();
        // A centre nearest to no row is the lowest-index nearest of none,
        // so dropping it moves no row: the kept cells are renumbered.
        let mut renumber = vec![0; b];
        for (new, &k) in kept.iter().enumerate() {
            renumber[k] = new;
        }
        for c in &mut ws.cells {
            *c = C::of(renumber[c.index()]);
        }
        tessellation(centres, t.dims, kept.len())
    }

    /// The number of rows in each of `b` cells.
    fn cell_counts<C: Cell>(cells: &[C], b: usize) -> Vec<f64> {
        let mut count = vec![0.0; b];
        cells.iter().for_each(|&q| count[q.index()] += 1.0);
        count
    }

    /// Set `t`'s cell values to `rate` times the shrunk cell means of
    /// `resid`, given the cells' row counts; returns the gain.
    fn refit<C: Cell>(
        resid: &[f64],
        t: &mut Tessellation,
        cells: &[C],
        count: &[f64],
        rate: f64,
        shrink: f64,
    ) -> f64 {
        let mut sum = vec![0.0; t.mus.len()];
        for (&r, &k) in resid.iter().zip(cells) {
            sum[k.index()] += r;
        }
        let gain = (0..sum.len())
            .map(|k| sum[k] * sum[k] / (count[k] + shrink))
            .sum();
        for (k, mu) in t.mus.iter_mut().enumerate() {
            *mu = rate * sum[k] / (count[k] + shrink);
        }
        gain
    }

    fn add<C: Cell>(resid: &mut [f64], t: &Tessellation, cells: &[C]) {
        for (r, &q) in resid.iter_mut().zip(cells) {
            *r += t.mus[q.index()];
        }
    }

    fn mean_square(resid: &[f64]) -> f64 {
        resid.iter().map(|r| r * r).sum::<f64>() / resid.len() as f64
    }

    fn rms(resid: &[f64]) -> f64 {
        mean_square(resid).sqrt()
    }

    fn subtract<C: Cell>(resid: &mut [f64], t: &Tessellation, cells: &[C]) {
        for (r, &q) in resid.iter_mut().zip(cells) {
            *r -= t.mus[q.index()];
        }
    }

    /// [`PROBES`] vectors of `n` random signs.
    fn sign_vectors(n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        (0..PROBES)
            .map(|_| {
                (0..n)
                    .map(|_| if uniform(rng) < 0.5 { -1.0 } else { 1.0 })
                    .collect()
            })
            .collect()
    }

    /// One refit pass for several right-hand sides on the same cells: in
    /// turn, every tessellation's values become `rate` times the shrunk
    /// cell means of its partial residual, the residual following. `k =
    /// rates.len()` right-hand sides are interleaved row by row in `resid`
    /// (`resid[i * k + q]`) and each tessellation's values cell by cell in
    /// `values[j]` (`values[j][c * k + q]`); right-hand side q is refitted
    /// at damping `rates[q]`, and `gains`, when given, takes each
    /// tessellation's gain on right-hand side 0. A cell's partial-residual
    /// sum is its residual sum plus its count times its old value, and the
    /// change of one tessellation's values is applied to the residuals in
    /// the same pass over the rows that sums the next tessellation's cells.
    #[allow(clippy::too_many_arguments)]
    fn refit_pass_many<C: Cell>(
        resid: &mut [f64],
        values: &mut [Vec<f64>],
        cells: &CellTable<C>,
        counts: &[Vec<f64>],
        rates: &[f64],
        shrink: f64,
        mut gains: Option<&mut [f64]>,
    ) {
        let (m, k) = (values.len(), rates.len());
        if m == 0 {
            return;
        }
        let mut sum = vec![0.0; values[0].len()];
        for (row, &c) in resid.chunks_exact(k).zip(cells.get(0)) {
            let at = c.index() * k;
            for (s, &r) in sum[at..at + k].iter_mut().zip(row) {
                *s += r;
            }
        }
        let (mut next, mut delta) = (Vec::new(), Vec::new());
        for j in 0..m {
            delta.clear();
            let mut gain = 0.0;
            for (c, &count) in counts[j].iter().enumerate() {
                for (q, &rate) in rates.iter().enumerate() {
                    let v = &mut values[j][c * k + q];
                    let partial = sum[c * k + q] + count * *v;
                    if q == 0 {
                        gain += partial * partial / (count + shrink);
                    }
                    let value = rate * partial / (count + shrink);
                    delta.push(value - *v);
                    *v = value;
                }
            }
            if let Some(gains) = gains.as_deref_mut() {
                gains[j] = gain;
            }
            let here = cells.get(j);
            if j + 1 < m {
                next.clear();
                next.resize(values[j + 1].len(), 0.0);
                for ((row, &c), &d) in resid.chunks_exact_mut(k).zip(here).zip(cells.get(j + 1)) {
                    let (at, to) = (c.index() * k, d.index() * k);
                    for ((r, &dv), s) in row
                        .iter_mut()
                        .zip(&delta[at..at + k])
                        .zip(&mut next[to..to + k])
                    {
                        *r -= dv;
                        *s += *r;
                    }
                }
                std::mem::swap(&mut sum, &mut next);
            } else {
                for (row, &c) in resid.chunks_exact_mut(k).zip(here) {
                    let at = c.index() * k;
                    for (r, &dv) in row.iter_mut().zip(&delta[at..at + k]) {
                        *r -= dv;
                    }
                }
            }
        }
    }

    /// The traces `tr(H)` and `tr(H'H)`, each the mean over the probe
    /// vectors z of `z'Hz` and `|Hz|^2` (Hutchinson, 1989), of the smoother
    /// H that maps a response to the fitted values of `passes` refit
    /// passes from zero on the given cells at damping `rate`, computed from
    /// `resid`, the residuals of those refits of the probes in the layout of
    /// [`refit_pass_many`], `k` right-hand sides of which `probes` are the
    /// columns `first..first + probes.len()`.
    fn traces(resid: &[f64], k: usize, first: usize, probes: &[Vec<f64>]) -> (f64, f64) {
        let (mut t1, mut t2) = (0.0, 0.0);
        for (q, z) in probes.iter().enumerate() {
            for (&zi, row) in z.iter().zip(resid.chunks_exact(k)) {
                let f = zi - row[first + q];
                t1 += zi * f;
                t2 += f * f;
            }
        }
        let count = probes.len() as f64;
        (t1 / count, t2 / count)
    }

    /// [`REFIT_PASSES`] refit passes of every tessellation at damping
    /// `rate`, `ts` and `resid` updated and `gains` set, with `probes`
    /// refitted from zero as further right-hand sides of the same passes.
    /// Returns the residual degrees of freedom `n - 2 tr(H) + tr(H'H)`, at
    /// least 1, of those refits (Hastie and Tibshirani, 1990, section 3.5),
    /// H's traces estimated from the probes, when there are any.
    #[allow(clippy::too_many_arguments)]
    fn refits<C: Cell>(
        resid: &mut [f64],
        ts: &mut [Tessellation],
        cells: &CellTable<C>,
        counts: &[Vec<f64>],
        rate: f64,
        shrink: f64,
        gains: &mut [f64],
        probes: &[Vec<f64>],
    ) -> Option<f64> {
        let n = resid.len();
        let k = 1 + probes.len();
        let mut many = Vec::with_capacity(n * k);
        for (i, &r) in resid.iter().enumerate() {
            many.push(r);
            many.extend(probes.iter().map(|z| z[i]));
        }
        let mut values: Vec<Vec<f64>> = ts
            .iter()
            .map(|t| {
                t.mus
                    .iter()
                    .flat_map(|&mu| std::iter::once(mu).chain(std::iter::repeat(0.0).take(k - 1)))
                    .collect()
            })
            .collect();
        let rates = vec![rate; k];
        for _ in 0..REFIT_PASSES {
            refit_pass_many(
                &mut many,
                &mut values,
                cells,
                counts,
                &rates,
                shrink,
                Some(gains),
            );
        }
        for (t, v) in ts.iter_mut().zip(&values) {
            for (cell, mu) in t.mus.iter_mut().enumerate() {
                *mu = v[cell * k];
            }
        }
        for (r, row) in resid.iter_mut().zip(many.chunks_exact(k)) {
            *r = row[0];
        }
        (!probes.is_empty()).then(|| {
            let (t1, t2) = traces(&many, k, 1, probes);
            (n as f64 - 2.0 * t1 + t2).max(1.0)
        })
    }

    /// The final refit: [`FINAL_PASSES`] refit passes at each damping from
    /// `value_scale` to 1 in three equal steps, the one kept that minimises
    /// the generalised cross-validation score `RSS / (1 - gamma tr(H) / n)^2`
    /// (Craven and Wahba, 1979) with `gamma` [`GCV_INFLATION`], H the
    /// smoother of those refits from zero, its trace estimated from
    /// [`PROBES`] random sign vectors shared by the dampings. Every damping
    /// and probe is one right-hand side of [`refit_pass_many`]. `ts` and
    /// `resid` take the kept values and residual. Returns the kept fit's
    /// residual degrees of freedom `n - 2 tr(H) + tr(H'H)` (Hastie and
    /// Tibshirani, 1990, section 3.5), at least 1.
    #[allow(clippy::too_many_arguments)]
    fn final_refit<C: Cell>(
        ts: &mut [Tessellation],
        resid: &mut [f64],
        cells: &CellTable<C>,
        counts: &[Vec<f64>],
        value_scale: f64,
        shrink: f64,
        rng: &mut Rng,
    ) -> f64 {
        let n = resid.len();
        let probes = sign_vectors(n, rng);
        let dampings: Vec<f64> = if value_scale < 1.0 {
            (0..4)
                .map(|k| value_scale + (1.0 - value_scale) * k as f64 / 3.0)
                .collect()
        } else {
            vec![1.0]
        };
        let (d, pr) = (dampings.len(), probes.len());
        // Right-hand side c < d is the residual refitted at damping c; d + c
        // * pr + q is probe q from zero at damping c.
        let mut rates = dampings.clone();
        for &rate in &dampings {
            rates.extend(std::iter::repeat(rate).take(pr));
        }
        let k = rates.len();
        let mut many = Vec::with_capacity(n * k);
        for (i, &r) in resid.iter().enumerate() {
            many.extend(std::iter::repeat(r).take(d));
            for _ in 0..d {
                many.extend(probes.iter().map(|z| z[i]));
            }
        }
        let mut values: Vec<Vec<f64>> = ts
            .iter()
            .map(|t| {
                t.mus
                    .iter()
                    .flat_map(|&mu| {
                        std::iter::repeat(mu)
                            .take(d)
                            .chain(std::iter::repeat(0.0).take(d * pr))
                    })
                    .collect()
            })
            .collect();
        for _ in 0..FINAL_PASSES {
            refit_pass_many(&mut many, &mut values, cells, counts, &rates, shrink, None);
        }
        let mut best: Option<(f64, usize, f64)> = None;
        for c in 0..d {
            let rss = many.chunks_exact(k).map(|row| row[c] * row[c]).sum::<f64>();
            let (t1, t2) = traces(&many, k, d + c * pr, &probes);
            let free = 1.0 - GCV_INFLATION * t1 / n as f64;
            let score = if free > 0.0 {
                rss / (free * free)
            } else {
                f64::INFINITY
            };
            let dof = (n as f64 - 2.0 * t1 + t2).max(1.0);
            if best.map_or(true, |b| score < b.0) {
                best = Some((score, c, dof));
            }
        }
        let (_, c, dof) = best.expect("at least one damping");
        for (t, v) in ts.iter_mut().zip(&values) {
            for (cell, mu) in t.mus.iter_mut().enumerate() {
                *mu = v[cell * k + c];
            }
        }
        for (r, row) in resid.iter_mut().zip(many.chunks_exact(k)) {
            *r = row[c];
        }
        dof
    }

    /// The fit. `s` must have passed [`XAddiVortesParams::validate`].
    pub(crate) fn fit(inp: &Inputs<'_>, s: &XAddiVortesParams, rng: &mut Rng) -> Outcome {
        if s.centres <= 256 {
            fit_with::<u8>(inp, s, rng)
        } else {
            fit_with::<u16>(inp, s, rng)
        }
    }

    fn fit_with<C: Cell>(inp: &Inputs<'_>, s: &XAddiVortesParams, rng: &mut Rng) -> Outcome {
        let x = inp.x;
        let (n, p, m) = (x.n_rows(), x.n_cols(), inp.m);
        let column_counts: Vec<usize> = (1..=s.max_columns.min(p)).collect();
        let centre_counts = centre_ladder(s.centres);
        let column_scores = screening_scores(x, inp.y);
        let mut sigma_sq = inp.sigma_sq;
        let mut shrink = sigma_sq / inp.sigma_mu_sq;
        let replace = ((m as f64 * s.replace_share).round() as usize).min(m);

        let mut resid = inp.y.to_vec();
        let mut ts: Vec<Tessellation> = Vec::with_capacity(m);
        let mut shapes: Vec<(usize, usize)> = Vec::with_capacity(m);
        let mut cells = CellTable::<C>::new(m, n);
        let mut ws = Workspace::<C>::default();
        let mut gains: Vec<f64> = Vec::with_capacity(m);
        let mut counts: Vec<Vec<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            let shape = (
                draw_weighted(column_counts.iter().map(|_| 1.0), rng),
                draw_weighted(centre_counts.iter().map(|_| 1.0), rng),
            );
            let (dims, centres) = fresh(
                x,
                column_counts[shape.0],
                &column_scores,
                centre_counts[shape.1],
                rng,
            );
            let mut t = nonempty(x, inp.geometry, centres, dims, &mut ws);
            let count = cell_counts(&ws.cells, t.mus.len());
            gains.push(refit(
                &resid,
                &mut t,
                &ws.cells,
                &count,
                s.value_scale,
                shrink,
            ));
            subtract(&mut resid, &t, &ws.cells);
            cells.set(j, &ws.cells);
            counts.push(count);
            shapes.push(shape);
            ts.push(t);
        }

        refits(
            &mut resid,
            &mut ts,
            &cells,
            &counts,
            s.value_scale,
            shrink,
            &mut gains,
            &[],
        );
        let worth_of =
            |gains: &[f64], counts: &[Vec<f64>], sigma_sq: f64, shrink: f64| -> Vec<f64> {
                gains
                    .iter()
                    .zip(counts)
                    .map(|(&g, c)| log_bayes_factor(g, c, sigma_sq, shrink))
                    .collect()
            };
        let mut worth = worth_of(&gains, &counts, sigma_sq, shrink);
        let mut train_rms = vec![rms(&resid)];
        for _ in 0..s.rounds {
            let weights = usage_weights(&ts, p);
            let by_columns = shape_weights(&shapes, column_counts.len(), |e| e.0);
            let by_centres = shape_weights(&shapes, centre_counts.len(), |e| e.1);
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&a, &b| worth[a].total_cmp(&worth[b]));
            for &j in &order[..replace] {
                add(&mut resid, &ts[j], cells.get(j));
                shapes[j] = (
                    draw_weighted(by_columns.iter().copied(), rng),
                    draw_weighted(by_centres.iter().copied(), rng),
                );
                let (dims, centres) = fresh(
                    x,
                    column_counts[shapes[j].0],
                    &weights,
                    centre_counts[shapes[j].1],
                    rng,
                );
                let mut t = nonempty(x, inp.geometry, centres, dims, &mut ws);
                counts[j] = cell_counts(&ws.cells, t.mus.len());
                gains[j] = refit(&resid, &mut t, &ws.cells, &counts[j], s.value_scale, shrink);
                subtract(&mut resid, &t, &ws.cells);
                cells.set(j, &ws.cells);
                ts[j] = t;
            }
            let probes = sign_vectors(n, rng);
            let dof = refits(
                &mut resid,
                &mut ts,
                &cells,
                &counts,
                s.value_scale,
                shrink,
                &mut gains,
                &probes,
            )
            .expect("probes give the degrees of freedom");
            worth = worth_of(&gains, &counts, sigma_sq, shrink);
            let rss = resid.iter().map(|r| r * r).sum::<f64>();
            // A residual of exactly zero says nothing about the noise.
            if rss > 0.0 {
                sigma_sq = rss / dof;
                shrink = sigma_sq / inp.sigma_mu_sq;
            }
            train_rms.push(rms(&resid));
        }

        let dof = final_refit(
            &mut ts,
            &mut resid,
            &cells,
            &counts,
            s.value_scale,
            shrink,
            rng,
        );
        let sigma_sq = resid.iter().map(|r| r * r).sum::<f64>() / dof;
        train_rms.push(rms(&resid));
        Outcome {
            tessellations: ts,
            sigma_sq,
            column_scores,
            train_rms,
            cells: C::wrap(cells),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rank_bins_follow_rank_with_ties_in_row_order() {
            assert_eq!(rank_bins(&[3.0, 1.0, 2.0, 0.0], 4), vec![3, 1, 2, 0]);
            assert_eq!(rank_bins(&[1.0, 1.0, 0.0, 2.0], 2), vec![0, 1, 0, 1]);
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
        fn the_centre_ladder_halves_down_to_four() {
            assert_eq!(centre_ladder(64), vec![4, 8, 16, 32, 64]);
            assert_eq!(centre_ladder(30), vec![7, 15, 30]);
            assert_eq!(centre_ladder(8), vec![4, 8]);
            assert_eq!(centre_ladder(7), vec![7]);
            assert_eq!(centre_ladder(1), vec![1]);
        }

        #[test]
        fn shape_weights_are_one_plus_the_counts() {
            let shapes = [(0, 2), (1, 2), (0, 0)];
            assert_eq!(shape_weights(&shapes, 3, |e| e.0), vec![3.0, 2.0, 1.0]);
            assert_eq!(shape_weights(&shapes, 3, |e| e.1), vec![2.0, 1.0, 3.0]);
        }

        #[test]
        fn the_log_bayes_factor_charges_every_cell() {
            let lbf = log_bayes_factor(8.0, &[3.0, 1.0], 2.0, 1.0);
            let want = 8.0 / 4.0 - 0.5 * (4f64.ln() + 2f64.ln());
            assert!((lbf - want).abs() < 1e-12, "{lbf} vs {want}");
        }

        /// One tessellation's refits from zero settle after one pass at
        /// `H = rate * blockdiag(11' / (n_k + lambda))`; with the unit
        /// vectors as probes the trace estimates are exact.
        #[test]
        fn traces_match_one_tessellation_exactly() {
            let cells_of: [u8; 5] = [0, 0, 1, 1, 1];
            let mut table = CellTable::<u8>::new(1, 5);
            table.set(0, &cells_of);
            let counts = vec![vec![2.0, 3.0]];
            let (rate, shrink) = (0.5, 0.25);
            let probes: Vec<Vec<f64>> = (0..5)
                .map(|i| (0..5).map(|j| f64::from(u8::from(i == j))).collect())
                .collect();
            let mut resid: Vec<f64> = (0..5)
                .flat_map(|i| probes.iter().map(move |z| z[i]))
                .collect();
            let mut values = vec![vec![0.0; 2 * 5]];
            for _ in 0..3 {
                refit_pass_many(
                    &mut resid,
                    &mut values,
                    &table,
                    &counts,
                    &[rate; 5],
                    shrink,
                    None,
                );
            }
            let (t1, t2) = traces(&resid, 5, 0, &probes);
            let tr: f64 = counts[0].iter().map(|&c| rate * c / (c + shrink)).sum();
            let tr2: f64 = counts[0]
                .iter()
                .map(|&c| (rate * c / (c + shrink)).powi(2))
                .sum();
            assert!((5.0 * t1 - tr).abs() < 1e-12, "{} vs {tr}", 5.0 * t1);
            assert!((5.0 * t2 - tr2).abs() < 1e-12, "{} vs {tr2}", 5.0 * t2);
        }

        /// Every right-hand side of `refit_pass_many` follows the textbook
        /// refit, each tessellation in turn added back to the residual,
        /// refitted to it and subtracted again, and the gains are that
        /// refit's on the first right-hand side.
        #[test]
        fn many_right_hand_sides_follow_the_textbook_refit() {
            let n = 9;
            let layout: [&[u8]; 3] = [
                &[0, 1, 0, 1, 2, 2, 0, 1, 2],
                &[0, 0, 0, 1, 1, 1, 1, 0, 0],
                &[3, 2, 1, 0, 3, 2, 1, 0, 3],
            ];
            let mut table = CellTable::<u8>::new(3, n);
            let mut counts = Vec::new();
            for (j, c) in layout.iter().enumerate() {
                table.set(j, c);
                counts.push(cell_counts(c, usize::from(*c.iter().max().unwrap()) + 1));
            }
            let (rates, shrink) = ([0.5, 0.8], 0.3);
            let start: Vec<Vec<f64>> = vec![
                (0..n).map(|i| (i as f64 * 0.7).sin()).collect(),
                (0..n).map(|i| (i as f64 * 1.3).cos() - 0.2).collect(),
            ];
            let mut many: Vec<f64> = (0..n)
                .flat_map(|i| start.iter().map(move |r| r[i]))
                .collect();
            let mut values: Vec<Vec<f64>> = counts.iter().map(|c| vec![0.1; c.len() * 2]).collect();
            let mut gains = vec![0.0; 3];
            for _ in 0..3 {
                refit_pass_many(
                    &mut many,
                    &mut values,
                    &table,
                    &counts,
                    &rates,
                    shrink,
                    Some(&mut gains),
                );
            }
            for (q, &rate) in rates.iter().enumerate() {
                let mut r = start[q].clone();
                let mut mus: Vec<Vec<f64>> = counts.iter().map(|c| vec![0.1; c.len()]).collect();
                let mut want_gains = vec![0.0; 3];
                for _ in 0..3 {
                    for (j, c) in layout.iter().enumerate() {
                        for (ri, &cell) in r.iter_mut().zip(c.iter()) {
                            *ri += mus[j][usize::from(cell)];
                        }
                        let mut sum = vec![0.0; mus[j].len()];
                        for (&ri, &cell) in r.iter().zip(c.iter()) {
                            sum[usize::from(cell)] += ri;
                        }
                        want_gains[j] = 0.0;
                        for (cell, mu) in mus[j].iter_mut().enumerate() {
                            let denominator = counts[j][cell] + shrink;
                            want_gains[j] += sum[cell] * sum[cell] / denominator;
                            *mu = rate * sum[cell] / denominator;
                        }
                        for (ri, &cell) in r.iter_mut().zip(c.iter()) {
                            *ri -= mus[j][usize::from(cell)];
                        }
                    }
                }
                for i in 0..n {
                    assert!((many[i * 2 + q] - r[i]).abs() < 1e-12);
                }
                for (j, m) in mus.iter().enumerate() {
                    for (cell, &mu) in m.iter().enumerate() {
                        assert!((values[j][cell * 2 + q] - mu).abs() < 1e-12);
                    }
                }
                if q == 0 {
                    for (g, w) in gains.iter().zip(&want_gains) {
                        assert!((g - w).abs() < 1e-12 * (1.0 + w.abs()), "{g} vs {w}");
                    }
                }
            }
        }

        /// A response the cells represent exactly, without noise, keeps
        /// the undamped refit.
        #[test]
        fn the_final_refit_keeps_no_damping_for_an_exact_fit() {
            let n = 40;
            let cells_of: Vec<u8> = (0..n).map(|i| u8::from(i >= n / 2)).collect();
            let y: Vec<f64> = cells_of
                .iter()
                .map(|&c| if c == 1 { 1.0 } else { -1.0 })
                .collect();
            let mut table = CellTable::<u8>::new(1, n);
            table.set(0, &cells_of);
            let mut ts = vec![tessellation(vec![0.25, 0.75], vec![0], 2)];
            let counts = vec![cell_counts(&cells_of, 2)];
            let mut resid = y.clone();
            let mut rng = crate::rng::chain_rng(1);
            let dof = final_refit(&mut ts, &mut resid, &table, &counts, 0.5, 1e-3, &mut rng);
            let shrunk = 20.0 / (20.0 + 1e-3);
            assert!((ts[0].mus[1] - shrunk).abs() < 1e-9, "{:?}", ts[0].mus);
            assert!((ts[0].mus[0] + shrunk).abs() < 1e-9, "{:?}", ts[0].mus);
            assert!(dof > 0.0 && dof < n as f64);
        }

        #[test]
        fn screening_favours_the_informative_column() {
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
            let sc = screening_scores(&x, &y);
            assert!(sc[0] > 10.0 * sc[1] && sc[0] > 10.0 * sc[2]);
            assert!(
                sc.iter().all(|&v| v > 0.0),
                "the floor keeps every column drawable"
            );
        }

        #[test]
        fn usage_weights_count_tessellations_per_column() {
            let t = |dims: Vec<usize>| Tessellation {
                centres: vec![0.0; dims.len()],
                dims,
                mus: vec![0.0],
                betas: Vec::new(),
                tau: None,
            };
            let w = usage_weights(&[t(vec![0, 2]), t(vec![2]), t(vec![2, 3])], 5);
            assert_eq!(w, vec![2.0, 1.0, 4.0, 2.0, 1.0]);
        }

        #[test]
        fn fresh_draws_distinct_columns() {
            use crate::rng::chain_rng;
            let x = Data::from_rows(&[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]).unwrap();
            for seed in 0..50 {
                let (dims, c) = fresh(&x, 3, &[1.0, 5.0, 0.5, 2.0], 4, &mut chain_rng(seed));
                let mut sorted = dims.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), 3);
                assert_eq!(c.len(), 12);
            }
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
            let cells: [u8; 3] = [0, 0, 1];
            let gain = refit(&resid, &mut t, &cells, &cell_counts(&cells, 2), 0.5, 1.0);
            assert!((t.mus[0] - 0.5 * 4.0 / 3.0).abs() < 1e-15);
            assert!((t.mus[1] - 0.5 * -2.0 / 2.0).abs() < 1e-15);
            assert!((gain - (16.0 / 3.0 + 4.0 / 2.0)).abs() < 1e-12);
        }

        #[test]
        fn empty_cells_are_dropped() {
            let x = Data::from_rows(&[[0.0], [0.1], [0.9]]).unwrap();
            let geometry = crate::geometry::Geometry::euclidean(1);
            // Centre 0.5 is nearest to no row once 0.05 and 0.95 are present.
            let mut ws = Workspace::<u8>::default();
            let t = nonempty(&x, &geometry, vec![0.05, 0.5, 0.95], vec![0], &mut ws);
            assert_eq!(t.centres, vec![0.05, 0.95]);
            assert_eq!(ws.cells, vec![0, 0, 1]);
        }
    }
}
