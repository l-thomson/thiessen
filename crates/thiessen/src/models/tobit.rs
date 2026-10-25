//! The tobit model for a censored response,
//! [`Outcome::Tobit`](crate::Outcome::Tobit), experimental
//! (`docs/experimental.md`):
//!
//! ```text
//! y*_i = f(x_i) + e_i,   e_i ~ N(0, sigma^2),   f(x) = sum_{j=1}^m g(x; T_j, M_j),
//! y_i  = lower   if y*_i <= lower,
//!        upper   if y*_i >= upper,
//!        y*_i    otherwise,
//! ```
//!
//! the type-I tobit model (Tobin 1958): the limits are known constants of
//! the design and a response value equal to a limit is read as censored
//! on that side. At least one limit is declared; a value beyond a limit
//! is rejected at fit. Models with unknown censoring points are out of
//! scope.
//!
//! # Sampler
//!
//! The data augmentation of Chib (1992): the latent y*_i of each censored
//! row is refreshed before each sweep from its full conditional
//!
//! ```text
//! y*_i | f, sigma^2, y_i = lower  ~  N(f(x_i), sigma^2) truncated to (-inf, lower],
//! y*_i | f, sigma^2, y_i = upper  ~  N(f(x_i), sigma^2) truncated to [upper, inf),
//! ```
//!
//! (truncated normal by the Robert 1995 exponential rejection), an
//! observed row's latent being its response, and the completed response
//! then runs the Gaussian model's sweep unchanged: sigma^2 from
//! Inv-Gamma((nu + n) / 2, (nu lambda + sum (y* - f)^2) / 2) and the cell
//! means from their Normal conditionals. Each update conditions on the
//! current values of the other blocks, so the scan
//! y* | sigma^2, f, y then sigma^2 | y*, f then f | y*, sigma^2 is a
//! valid Gibbs sampler; the latent refresh runs first so that sigma^2
//! and the ensemble only ever condition on latents drawn from their
//! conditional, which makes a response replacement mid-chain
//! self-repairing. No structural move gains an acceptance-ratio term:
//! the latent refresh is a Gibbs step against a fixed ensemble.
//! With a variance ensemble attached (`variance_params.tessellations`
//! above 0) the truncated draw's variance is the ensemble's product
//! s^2(x_i), the same per-observation precision the backfit uses.
//!
//! # Priors
//!
//! The Gaussian model's exactly: cell means mu ~ N(0, sigma_mu^2) with
//! sigma_mu = 0.5 / (k sqrt m) on the response min-max scaled to
//! [-0.5, 0.5]; sigma^2 ~ nu lambda / chi^2_nu with lambda set so that
//! Pr(sigma < sigma_hat) = q. The limits are mapped by the same frozen
//! affine map as the response; sigma_hat is the least-squares residual
//! standard deviation of the observed response with censored rows at
//! their limits, a calibration heuristic the way the Gaussian model's
//! least-squares fit is. Imputed latents may fall outside the training
//! range, which the frozen map permits.
//!
//! # Fixed rather than estimated
//!
//! Nothing beyond the limits themselves, which are data, not parameters.
//! sigma^2 is sampled, so the scale mode is `Sampled` and a variance
//! ensemble may attach.
//!
//! # Correspondence
//!
//! With MCMCpack `MCMCtobit` (the Chib 1992 sampler for the linear
//! model): lower = `below`, upper = `above` (its defaults are 0 and
//! infinity; here each limit is optional and at least one is required).
//! With the crate's Gaussian model: every structural parameter is
//! shared, and nu and q sit on the tobit parameters. Uncensored data (no
//! row at a limit) reproduces the Gaussian model draw for draw at the
//! same seed: the refresh touches only censored rows and consumes no
//! randomness when there are none.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of the uncensored f(x), the latent
//! mean a tobit analysis targets, and `predict_draws` its per-draw
//! values; `predict_variance` is sigma_d^2 per draw (s_d^2(x) with a
//! variance ensemble); `prediction_interval` is the censored
//! predictive's central interval, the ends of the uncensored interval
//! clamped to the limits; `log_likelihood` is the type-I tobit
//! likelihood, ln Phi((lower - f_d) / s_d) at a row censored below,
//! ln Phi((f_d - upper) / s_d) at a row censored above and the Normal
//! log density otherwise; `sigma` is sigma_d on the caller's scale;
//! `in_sample_rmse` is against the observed response, censored rows at
//! their limits.
//!
//! # Input
//!
//! A continuous response with every value inside the declared limits;
//! a value beyond a limit is
//! [`Error::ResponseBeyondLimit`](crate::Error::ResponseBeyondLimit).

use crate::maths;
use crate::models::censoring::{self, Bound};
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::Rng;

/// The tobit outcome behind the [`OutcomeModel`] contract: known limits
/// on the scaled response and the Chib (1992) latent refresh of the
/// censored rows each sweep, with the observed response kept for the
/// in-sample RMSE. sigma^2 is sampled, so the scale mode is `Sampled`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TobitOutcome {
    lower: Option<f64>,
    upper: Option<f64>,
    observed: Vec<f64>,
    bounds: Vec<Bound>,
}

impl TobitOutcome {
    /// The half-width of the cell-mean prior: the Gaussian model's, the
    /// response being scaled to [-0.5, 0.5].
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 0.5;

    /// An outcome with the limits already on the scaled response scale.
    pub(crate) fn new(lower: Option<f64>, upper: Option<f64>) -> Self {
        Self {
            lower,
            upper,
            observed: Vec::new(),
            bounds: Vec::new(),
        }
    }

    /// The observed scaled response the fit summary is measured against;
    /// censored rows hold their limit.
    pub(crate) fn observed(&self) -> &[f64] {
        &self.observed
    }

    /// Number of censored training rows.
    #[cfg(test)]
    pub(crate) fn n_censored(&self) -> usize {
        self.bounds
            .iter()
            .filter(|&&b| b != Bound::Observed)
            .count()
    }
}

impl OutcomeModel for TobitOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Continuous
    }

    /// Store the observed response and read each row's truncation bound
    /// from equality with a limit; the affine response map preserves
    /// exact equality, so the scaled comparison is the caller-scale one.
    fn init(&mut self, y: &[f64]) {
        self.observed = y.to_vec();
        self.bounds = y
            .iter()
            .map(|&v| {
                if self.lower == Some(v) {
                    Bound::Below(v)
                } else if self.upper == Some(v) {
                    Bound::Above(v)
                } else {
                    Bound::Observed
                }
            })
            .collect();
    }

    fn draw_extra(&mut self, _y: &[f64], _total: &[f64], _precision: &[f64], _rng: &mut Rng) {}

    /// The shared censored refresh over the per-row bounds; no randomness
    /// is consumed for a response with no censored rows.
    fn working_response(&mut self, total: &[f64], precision: &[f64], y: &mut [f64], rng: &mut Rng) {
        censoring::refresh(&self.bounds, total, precision, y, rng);
    }

    fn weights(&self) -> Option<&[f64]> {
        None
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Sampled
    }

    /// Quantile of the censored predictive, scaled space: censoring is
    /// monotone, so it is the Normal quantile clamped to the limits.
    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64> {
        let q = mean + sd * maths::normal_quantile(p);
        Some(q.clamp(
            self.lower.unwrap_or(f64::NEG_INFINITY),
            self.upper.unwrap_or(f64::INFINITY),
        ))
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::*;
    use crate::rng::chain_rng;
    use rand_core::RngCore;

    #[test]
    fn the_tobit_outcome_answers_the_contract() {
        let mut outcome = TobitOutcome::new(Some(-0.3), Some(0.4));
        let y = [-0.3, 0.1, 0.4, 0.0];
        outcome.init(&y);
        assert_eq!(outcome.observed(), &y);
        assert_eq!(outcome.n_censored(), 2);
        assert_eq!(outcome.required_data(), RequiredData::Continuous);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(outcome.weights(), None);

        let mut rng = chain_rng(11);
        let mut latent = y;
        let total = [0.0; 4];
        let precision = [4.0; 4];
        outcome.working_response(&total, &precision, &mut latent, &mut rng);
        assert!(latent[0] <= -0.3);
        assert_eq!(latent[1], 0.1);
        assert!(latent[2] >= 0.4);
        assert_eq!(latent[3], 0.0);

        let median = outcome.predictive_quantile(0.0, 1.0, 0.5).unwrap();
        assert!(median.abs() < 1e-12);
        assert_eq!(outcome.predictive_quantile(0.0, 1.0, 0.001), Some(-0.3));
        assert_eq!(outcome.predictive_quantile(0.0, 1.0, 0.999), Some(0.4));
    }

    /// The refresh of an uncensored response consumes no randomness, the
    /// invariant behind draw-for-draw agreement with the Gaussian model.
    #[test]
    fn an_uncensored_response_consumes_no_randomness() {
        let mut outcome = TobitOutcome::new(Some(-1.0), None);
        let y = [-0.4, 0.1, 0.3];
        outcome.init(&y);
        assert_eq!(outcome.n_censored(), 0);
        let mut rng = chain_rng(3);
        let mut untouched = chain_rng(3);
        let mut latent = y;
        outcome.working_response(&[0.0; 3], &[1.0; 3], &mut latent, &mut rng);
        assert_eq!(latent, y);
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Config, Outcome};
    use crate::data::Data;
    use crate::sampler::Sampler;
    use crate::tessellation::Tessellation;

    /// The rows of one cell as the quadrature sees them: the observed
    /// values, and the counts censored at each limit.
    struct CellData {
        observed: Vec<f64>,
        n_below: f64,
        n_above: f64,
    }

    fn normal_cdf(z: f64) -> f64 {
        0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
    }

    /// Posterior means of sigma^2 and of every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2 with an inner
    /// quadrature over each cell mean. Given sigma^2 the cells are
    /// independent, so cell k contributes its marginal
    ///
    /// ```text
    /// m_k(sigma^2) = int N(mu; 0, sigma_mu^2) prod_obs N(y_i; mu, sigma^2)
    ///                Phi((L - mu) / sigma)^{n_below} Phi((mu - U) / sigma)^{n_above} dmu,
    /// ```
    ///
    /// the model's own censored likelihood integrated numerically, and
    /// E[mu_k | y] averages the per-sigma conditional means under
    /// p(sigma^2 | y) proportional to the Inv-Gamma(nu / 2,
    /// nu lambda / 2) prior times prod_k m_k. Independent of the engine.
    fn quadrature_reference(
        cells: &[CellData],
        nu: f64,
        lambda: f64,
        sigma_mu_sq: f64,
        lower: f64,
        upper: f64,
    ) -> (f64, Vec<f64>) {
        let (a, scale) = (0.5 * nu, 0.5 * nu * lambda);
        let outer = 400;
        let (t_lo, t_hi) = (lambda.ln() - 8.0, lambda.ln() + 8.0);
        let inner = 3000;
        let (mu_lo, mu_hi) = (-1.5_f64, 1.5_f64);
        let mut log_weights = Vec::with_capacity(outer + 1);
        let mut sigmas = Vec::with_capacity(outer + 1);
        let mut cond_means: Vec<Vec<f64>> = vec![Vec::with_capacity(outer + 1); cells.len()];
        for i in 0..=outer {
            let t = t_lo + (t_hi - t_lo) * i as f64 / outer as f64;
            let sigma_sq = t.exp();
            let sigma = sigma_sq.sqrt();
            let mut lp = -(a + 1.0) * t - scale / sigma_sq + t;
            for (k, cell) in cells.iter().enumerate() {
                let mut best = f64::NEG_INFINITY;
                let mut terms = Vec::with_capacity(inner + 1);
                for j in 0..=inner {
                    let mu = mu_lo + (mu_hi - mu_lo) * j as f64 / inner as f64;
                    let mut term = -0.5 * mu * mu / sigma_mu_sq;
                    for &y in &cell.observed {
                        term += -0.5 * (y - mu) * (y - mu) / sigma_sq - 0.5 * t;
                    }
                    if cell.n_below > 0.0 {
                        term += cell.n_below * normal_cdf((lower - mu) / sigma).ln();
                    }
                    if cell.n_above > 0.0 {
                        term += cell.n_above * normal_cdf((mu - upper) / sigma).ln();
                    }
                    best = best.max(term);
                    terms.push((mu, term));
                }
                // A sigma^2 gridpoint the cell's censored likelihood
                // rules out entirely carries zero weight.
                if best == f64::NEG_INFINITY {
                    cond_means[k].push(0.0);
                    lp = f64::NEG_INFINITY;
                    continue;
                }
                let mut total = 0.0;
                let mut mean = 0.0;
                for (mu, term) in terms {
                    let w = (term - best).exp();
                    total += w;
                    mean += w * mu;
                }
                cond_means[k].push(mean / total);
                lp += best + total.ln();
            }
            sigmas.push(sigma_sq);
            log_weights.push(lp);
        }
        let max = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|lp| (lp - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mean_sigma_sq = weights.iter().zip(&sigmas).map(|(w, s)| w * s).sum::<f64>() / total;
        let mean_mus = cond_means
            .iter()
            .map(|means| weights.iter().zip(means).map(|(w, m)| w * m).sum::<f64>() / total)
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

    /// On a fixed tessellation the chain is the Chib (1992) Gibbs sampler;
    /// its means of sigma^2 and of every mu_k match the numerical
    /// integration of the censored likelihood within 4 batch-means MCSE.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let (lower, upper) = (-0.30, 0.32);
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (((i * 7) % 13) as f64 / 13.0 - 0.5).clamp(lower, upper))
            .collect();
        assert!(y.iter().filter(|&&v| v == lower).count() >= 2);
        assert!(y.iter().filter(|&&v| v == upper).count() >= 1);
        let x = Data::new(xs, n, 1).unwrap();
        let lambda = 0.04;
        let config = Config::new()
            .with_outcome(Outcome::tobit(Some(lower), Some(upper)))
            .with_m(1);
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 41_u64), (vec![0.0], 42)] {
            let b = centres.len();
            let mut sampler = Sampler::pinned_prior(&config, &x, &y, lambda, seed).unwrap();
            sampler.fix_mean_tessellation(
                0,
                Tessellation {
                    centres,
                    dims: vec![0],
                    mus: vec![0.0; b],
                    betas: Vec::new(),
                    tau: None,
                },
            );
            let assignments = sampler.mean_cells(0).to_vec();
            let mut cells: Vec<CellData> = (0..b)
                .map(|_| CellData {
                    observed: Vec::new(),
                    n_below: 0.0,
                    n_above: 0.0,
                })
                .collect();
            for (&cell, &v) in assignments.iter().zip(&y) {
                if v == lower {
                    cells[cell as usize].n_below += 1.0;
                } else if v == upper {
                    cells[cell as usize].n_above += 1.0;
                } else {
                    cells[cell as usize].observed.push(v);
                }
            }
            let sigma_mu_sq = sampler.mean_sigma_mu_sq();
            let (ref_sigma_sq, ref_mus) = quadrature_reference(
                &cells,
                sampler.config().sigma2_prior().0,
                lambda,
                sigma_mu_sq,
                lower,
                upper,
            );

            for _ in 0..500 {
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
