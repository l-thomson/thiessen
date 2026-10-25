//! The lognormal accelerated failure time model for a right-censored
//! time-to-event response, [`Outcome::Aft`](crate::Outcome::Aft),
//! experimental (`docs/experimental.md`):
//!
//! ```text
//! ln T_i = f(x_i) + e_i,   e_i ~ N(0, sigma^2),   f(x) = sum_{j=1}^m g(x; T_j, M_j),
//! observed: (t_i, delta_i),  delta_i = 1: T_i = t_i (event),
//!                            delta_i = 0: T_i > t_i (censored at t_i),
//! ```
//!
//! the AFT formulation of survival analysis (Wei 1992) with a lognormal
//! error, the model of the BART package's `abart`. The event times t_i
//! are positive; the event indicator is data, one flag per row.
//! Right-censoring only; interval censoring is a separate model.
//!
//! # Sampler
//!
//! The censored-data augmentation on the log scale: the latent log time
//! of each censored row is refreshed before each sweep from its full
//! conditional
//!
//! ```text
//! ln T_i | f, sigma^2, delta_i = 0  ~  N(f(x_i), sigma^2) truncated to [ln t_i, inf),
//! ```
//!
//! (truncated normal by the Robert 1995 exponential rejection, the
//! censored refresh shared with the tobit model), an event row's latent
//! being ln t_i, and the completed log-time response then runs the
//! Gaussian model's sweep unchanged. The scan is the tobit model's:
//! ln T | sigma^2, f first, then sigma^2 | ln T, then f | ln T, sigma^2,
//! so sigma^2 and the ensemble only ever condition on latents drawn
//! from their conditional. No structural move gains an acceptance-ratio
//! term. With a variance ensemble attached the truncated draw's
//! variance is the ensemble's product s^2(x_i).
//!
//! # Priors
//!
//! The Gaussian model's exactly, on ln t min-max scaled to [-0.5, 0.5];
//! each censored row's truncation point is its own scaled ln t_i, so no
//! separate limit crosses the map. sigma_hat calibrates from the
//! least-squares fit of the observed log times, censored rows at their
//! censoring values, a heuristic the way the Gaussian model's fit is.
//! Imputed latents may fall outside the training range, which the
//! frozen map permits.
//!
//! # Fixed rather than estimated
//!
//! The log transform is fixed: the model is lognormal AFT, not a
//! general transformation model. sigma^2 (the log scale's) is sampled,
//! so the scale mode is `Sampled` and a variance ensemble may attach.
//!
//! # Correspondence
//!
//! With BART `abart` (Sparapani, Spanbauer and McCulloch 2021):
//! times = `times`, events = `delta`, m = `ntree`, k = `k`, nu =
//! `sigdf`, q = `sigquant`, burn_in = `nskip`, draws = `ndpost`,
//! thinning = `keepevery`. `abart` defaults to k = 2, `ntree = 200`,
//! `sigdf = 3` and `sigquant = 0.90`; the crate keeps its own defaults
//! (k = 3, m = 200, nu = 6, q = 0.85). `abart` centres the response
//! with an offset; here the min-max response map carries the centring.
//! The comparison against `abart` on a fixed dataset is informational
//! (`benchmarks/upstream/aft_abart.R`): trees and tessellations are
//! different priors, so the posteriors are close but not equal.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of f(x) on the log-time scale
//! (`abart`'s `yhat`), and `predict_draws` its per-draw values;
//! `predict_variance` is sigma_d^2 per draw on the log scale (s_d^2(x)
//! with a variance ensemble); `prediction_interval` is the predictive
//! interval of a new log time; `log_likelihood` is
//! [`Error::NotApplicable`](crate::Error::NotApplicable), the pointwise
//! likelihood needing the event indicator, and
//! [`log_likelihood_survival`](crate::Fitted::log_likelihood_survival)
//! takes it: ln N(ln t_i; f_d, s_d^2) at an event,
//! ln Phi((f_d - ln t_i) / s_d) at a censored row; `sigma` is sigma_d
//! on the log scale times the training range of ln t; `in_sample_rmse`
//! is on the log-time scale against the observed log times, censored
//! rows at their censoring values.
//!
//! # Input
//!
//! Times finite and positive
//! ([`Error::InvalidSurvivalTime`](crate::Error::InvalidSurvivalTime)),
//! one event flag per row
//! ([`Error::EventCountMismatch`](crate::Error::EventCountMismatch)).
//! The model is fitted through [`fit_aft`](crate::fit_aft) or
//! [`Sampler::aft`](crate::Sampler::aft); [`fit`](crate::fit) has no
//! event channel and rejects the outcome. All-event data reproduces the
//! Gaussian model on ln t draw for draw at the same seed.

use crate::maths;
use crate::models::censoring::{self, Bound};
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::Rng;

/// The AFT outcome behind the [`OutcomeModel`] contract: an event flag
/// per row and the censored refresh of each censored row's latent log
/// time above its own censoring value, with the observed log times kept
/// for the in-sample RMSE. sigma^2 is sampled, so the scale mode is
/// `Sampled`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AftOutcome {
    events: Vec<bool>,
    observed: Vec<f64>,
    bounds: Vec<Bound>,
}

impl AftOutcome {
    /// The half-width of the cell-mean prior: the Gaussian model's, the
    /// log-time response being scaled to [-0.5, 0.5].
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 0.5;

    /// An outcome over the given event flags; the response arrives at
    /// [`init`](OutcomeModel::init) as scaled log times.
    pub(crate) fn new(events: Vec<bool>) -> Self {
        Self {
            events,
            observed: Vec::new(),
            bounds: Vec::new(),
        }
    }

    /// The observed scaled log times the fit summary is measured
    /// against; censored rows hold their censoring values.
    pub(crate) fn observed(&self) -> &[f64] {
        &self.observed
    }

    /// Replace the event flags; the caller has validated the length and
    /// calls [`init`](OutcomeModel::init) with the matching response.
    pub(crate) fn set_events(&mut self, events: &[bool]) {
        self.events = events.to_vec();
    }

    /// Number of censored training rows.
    #[cfg(test)]
    pub(crate) fn n_censored(&self) -> usize {
        self.events.iter().filter(|&&event| !event).count()
    }
}

impl OutcomeModel for AftOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Continuous
    }

    /// Store the observed scaled log times; a censored row's truncation
    /// point is its own value, so the bounds need no separate map.
    fn init(&mut self, y: &[f64]) {
        self.observed = y.to_vec();
        self.bounds = y
            .iter()
            .zip(&self.events)
            .map(|(&v, &event)| {
                if event {
                    Bound::Observed
                } else {
                    Bound::Above(v)
                }
            })
            .collect();
    }

    fn draw_extra(&mut self, _y: &[f64], _total: &[f64], _precision: &[f64], _rng: &mut Rng) {}

    /// The shared censored refresh over the per-row bounds; no randomness
    /// is consumed for all-event data.
    fn working_response(&mut self, total: &[f64], precision: &[f64], y: &mut [f64], rng: &mut Rng) {
        censoring::refresh(&self.bounds, total, precision, y, rng);
    }

    fn weights(&self) -> Option<&[f64]> {
        None
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Sampled
    }

    /// Quantile of the predictive log time, scaled space.
    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64> {
        Some(mean + sd * maths::normal_quantile(p))
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::*;
    use crate::rng::chain_rng;
    use rand_core::RngCore;

    #[test]
    fn the_aft_outcome_answers_the_contract() {
        let mut outcome = AftOutcome::new(vec![true, false, true, false]);
        let y = [-0.2, 0.1, 0.3, -0.4];
        outcome.init(&y);
        assert_eq!(outcome.observed(), &y);
        assert_eq!(outcome.n_censored(), 2);
        assert_eq!(outcome.required_data(), RequiredData::Continuous);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(outcome.weights(), None);

        let mut rng = chain_rng(23);
        let mut latent = y;
        outcome.working_response(&[0.0; 4], &[4.0; 4], &mut latent, &mut rng);
        assert_eq!(latent[0], -0.2);
        assert!(latent[1] >= 0.1);
        assert_eq!(latent[2], 0.3);
        assert!(latent[3] >= -0.4);

        let median = outcome.predictive_quantile(0.2, 1.0, 0.5).unwrap();
        assert!((median - 0.2).abs() < 1e-12);
    }

    /// All-event data consumes no randomness in the refresh, the
    /// invariant behind draw-for-draw agreement with the Gaussian model
    /// on log times.
    #[test]
    fn all_event_data_consumes_no_randomness() {
        let mut outcome = AftOutcome::new(vec![true; 3]);
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

    /// The rows of one cell as the quadrature sees them: log times of
    /// the events, and the censoring log times of the censored rows.
    struct CellData {
        events: Vec<f64>,
        censored: Vec<f64>,
    }

    fn normal_cdf(z: f64) -> f64 {
        0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
    }

    /// Posterior means of sigma^2 and of every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2 with an
    /// inner quadrature over each cell mean. Given sigma^2 the cells are
    /// independent; cell k's marginal integrates the model's own
    /// censored likelihood, an event row contributing
    /// N(ln t_i; mu, sigma^2) and a censored row
    /// Phi((mu - ln t_i) / sigma). Independent of the engine.
    fn quadrature_reference(
        cells: &[CellData],
        nu: f64,
        lambda: f64,
        sigma_mu_sq: f64,
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
                    for &v in &cell.events {
                        term += -0.5 * (v - mu) * (v - mu) / sigma_sq - 0.5 * t;
                    }
                    for &bound in &cell.censored {
                        term += normal_cdf((mu - bound) / sigma).ln();
                    }
                    best = best.max(term);
                    terms.push((mu, term));
                }
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

    /// On a fixed tessellation the chain is the censored-data Gibbs
    /// sampler on log times; its means of sigma^2 and of every mu_k
    /// match the numerical integration of the model's own likelihood
    /// within 4 batch-means MCSE.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let log_times: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 13.0 - 0.5).collect();
        let events: Vec<bool> = (0..n).map(|i| (i * 3) % 4 != 0).collect();
        assert!(events.iter().filter(|&&event| !event).count() >= 3);
        let times: Vec<f64> = log_times.iter().map(|v| v.exp()).collect();
        let x = Data::new(xs, n, 1).unwrap();
        let lambda = 0.04;
        let config = Config::new().with_outcome(Outcome::aft()).with_m(1);
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 51_u64), (vec![0.0], 52)] {
            let b = centres.len();
            let mut sampler =
                Sampler::pinned_prior_aft(&config, &x, &times, &events, lambda, seed).unwrap();
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
            // The engine's response is ln(exp(v)), so the reference reads
            // the same values back rather than reusing the grid.
            let engine_y: Vec<f64> = times.iter().map(|t| t.ln()).collect();
            let mut cells: Vec<CellData> = (0..b)
                .map(|_| CellData {
                    events: Vec::new(),
                    censored: Vec::new(),
                })
                .collect();
            for ((&cell, &v), &event) in assignments.iter().zip(&engine_y).zip(&events) {
                if event {
                    cells[cell as usize].events.push(v);
                } else {
                    cells[cell as usize].censored.push(v);
                }
            }
            let sigma_mu_sq = sampler.mean_sigma_mu_sq();
            let (ref_sigma_sq, ref_mus) = quadrature_reference(
                &cells,
                sampler.config().sigma2_prior().0,
                lambda,
                sigma_mu_sq,
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
