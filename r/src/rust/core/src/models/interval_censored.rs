//! The interval-censored model for a response known only to lie between
//! two row-specific bounds,
//! [`Outcome::IntervalCensored`](crate::Outcome::IntervalCensored),
//! experimental (`docs/experimental.md`):
//!
//! ```text
//! y*_i = f(x_i) + e_i,   e_i ~ N(0, sigma^2),   f(x) = sum_{j=1}^m g(x; T_j, M_j),
//! observed [l_i, u_i]: l_i = u_i  an exact value (y*_i = l_i),
//!                      l_i < u_i  y*_i in [l_i, u_i],
//! ```
//!
//! the interval-censoring observation scheme (Sun 2006): tested negative
//! at one inspection and positive at a later one. An infinite endpoint
//! is one-sided censoring. The bounds are data, one pair per row; the
//! inspection scheme is taken as independent of the response
//! (non-informative censoring), so the bounds enter the likelihood only
//! through the interval probability.
//!
//! # Sampler
//!
//! The censored-data augmentation shared with the tobit model, extended
//! to a two-sided draw: the latent of each censored row is refreshed
//! before each sweep from its full conditional
//!
//! ```text
//! y*_i | f, sigma^2  ~  N(f(x_i), sigma^2) truncated to [l_i, u_i],
//! ```
//!
//! (Robert 1995: the one-sided draw by exponential rejection, the
//! two-sided draw by his section 2 rules), an exact row's latent being
//! its value, and the completed response then runs the Gaussian model's
//! sweep unchanged with the tobit model's scan order: y* | sigma^2, f
//! first, then sigma^2 | y*, then f | y*, sigma^2, so sigma^2 and the
//! ensemble only ever condition on latents drawn from their
//! conditional. No structural move gains an acceptance-ratio term. With
//! a variance ensemble attached the truncated draw's variance is the
//! ensemble's product s^2(x_i).
//!
//! # Priors
//!
//! The Gaussian model's exactly, on the working response min-max scaled
//! to [-0.5, 0.5]. The working response completes each interval with
//! one value: the exact value where l_i = u_i, the midpoint where both
//! bounds are finite and the finite endpoint of a one-sided interval;
//! sigma_hat calibrates from its least-squares fit, the Gaussian
//! model's heuristic, and the bounds cross to the scaled space by the
//! same frozen affine map. Imputed latents may fall outside the
//! training range, which the frozen map permits.
//!
//! # Fixed rather than estimated
//!
//! Nothing beyond the bounds themselves, which are data, not
//! parameters. sigma^2 is sampled, so the scale mode is `Sampled` and a
//! variance ensemble may attach.
//!
//! # Correspondence
//!
//! None in the BART family; the BART package has no interval-censored
//! model. survival's `survreg(dist = "gaussian")` with
//! `Surv(l, u, type = "interval2")` fits the linear-model analogue by
//! maximum likelihood. Exact data (every pair l_i = u_i) reproduces the
//! Gaussian model draw for draw at the same seed: the refresh touches
//! no row and consumes no randomness.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of the latent f(x) and
//! `predict_draws` its per-draw values; `predict_variance` is
//! sigma_d^2 per draw (s_d^2(x) with a variance ensemble);
//! `prediction_interval` is the predictive interval of a new latent
//! value, uncensored, the bounds being an observation scheme rather
//! than part of the response law; `log_likelihood` is
//! [`Error::NotApplicable`](crate::Error::NotApplicable), the pointwise
//! likelihood needing the bounds, and
//! [`log_likelihood_interval_censored`](crate::Fitted::log_likelihood_interval_censored)
//! takes it: ln N(l_i; f_d, s_d^2) at an exact row,
//! ln(Phi((u_i - f_d) / s_d) - Phi((l_i - f_d) / s_d)) at a censored
//! one with an infinite endpoint dropping its term; `sigma` is sigma_d
//! on the caller's scale; `in_sample_rmse` is against the working
//! response.
//!
//! # Input
//!
//! One pair of bounds per row, each pair with no NaN endpoint,
//! l_i <= u_i, at least one finite endpoint and a finite value where
//! l_i = u_i ([`Error::InvalidInterval`](crate::Error::InvalidInterval);
//! length disagreement is
//! [`Error::BoundCountMismatch`](crate::Error::BoundCountMismatch)).
//! The model is fitted through
//! [`fit_interval_censored`](crate::fit_interval_censored) or
//! [`Sampler::interval_censored`](crate::Sampler::interval_censored);
//! [`fit`](crate::fit) has no bound channel and rejects the outcome.

use crate::maths;
use crate::models::censoring::{self, Bound};
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::Rng;

/// The interval-censored outcome behind the [`OutcomeModel`] contract:
/// a pair of bounds per row on the scaled response and the two-sided
/// refresh of each censored row's latent, with the working response
/// kept for the in-sample RMSE. sigma^2 is sampled, so the scale mode
/// is `Sampled`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IntervalCensoredOutcome {
    lower: Vec<f64>,
    upper: Vec<f64>,
    observed: Vec<f64>,
    bounds: Vec<Bound>,
}

impl IntervalCensoredOutcome {
    /// The half-width of the cell-mean prior: the Gaussian model's, the
    /// working response being scaled to [-0.5, 0.5].
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 0.5;

    /// An outcome with the bounds already on the scaled response scale;
    /// the response arrives at [`init`](OutcomeModel::init) as the
    /// scaled working values.
    pub(crate) fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        Self {
            lower,
            upper,
            observed: Vec::new(),
            bounds: Vec::new(),
        }
    }

    /// The scaled working response the fit summary is measured against;
    /// censored rows hold their completion value.
    pub(crate) fn observed(&self) -> &[f64] {
        &self.observed
    }

    /// Replace the bounds; the caller has validated them and calls
    /// [`init`](OutcomeModel::init) with the matching working response.
    pub(crate) fn set_bounds(&mut self, lower: Vec<f64>, upper: Vec<f64>) {
        self.lower = lower;
        self.upper = upper;
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

impl OutcomeModel for IntervalCensoredOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Continuous
    }

    /// Store the working response and read each row's truncation from
    /// its pair of bounds: an equal pair is exact, an infinite endpoint
    /// one-sided.
    fn init(&mut self, y: &[f64]) {
        self.observed = y.to_vec();
        self.bounds = self
            .lower
            .iter()
            .zip(&self.upper)
            .map(|(&lo, &hi)| {
                if lo == hi {
                    Bound::Observed
                } else if lo == f64::NEG_INFINITY {
                    Bound::Below(hi)
                } else if hi == f64::INFINITY {
                    Bound::Above(lo)
                } else {
                    Bound::Between(lo, hi)
                }
            })
            .collect();
    }

    fn draw_extra(&mut self, _y: &[f64], _total: &[f64], _precision: &[f64], _rng: &mut Rng) {}

    /// The shared censored refresh over the per-row bounds; no randomness
    /// is consumed for exact data.
    fn working_response(&mut self, total: &[f64], precision: &[f64], y: &mut [f64], rng: &mut Rng) {
        censoring::refresh(&self.bounds, total, precision, y, rng);
    }

    fn weights(&self) -> Option<&[f64]> {
        None
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Sampled
    }

    /// Quantile of the predictive latent, scaled space: the bounds are
    /// an observation scheme, so a new value is uncensored.
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
    fn the_interval_censored_outcome_answers_the_contract() {
        let lower = vec![-0.2, f64::NEG_INFINITY, 0.1, -0.1, 0.3];
        let upper = vec![-0.2, 0.0, f64::INFINITY, 0.2, 0.3];
        let mut outcome = IntervalCensoredOutcome::new(lower, upper);
        let y = [-0.2, 0.0, 0.1, 0.05, 0.3];
        outcome.init(&y);
        assert_eq!(outcome.observed(), &y);
        assert_eq!(outcome.n_censored(), 3);
        assert_eq!(outcome.required_data(), RequiredData::Continuous);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(outcome.weights(), None);

        let mut rng = chain_rng(29);
        let mut latent = y;
        outcome.working_response(&[0.0; 5], &[4.0; 5], &mut latent, &mut rng);
        assert_eq!(latent[0], -0.2);
        assert!(latent[1] <= 0.0);
        assert!(latent[2] >= 0.1);
        assert!((-0.1..=0.2).contains(&latent[3]));
        assert_eq!(latent[4], 0.3);

        let median = outcome.predictive_quantile(0.2, 1.0, 0.5).unwrap();
        assert!((median - 0.2).abs() < 1e-12);
    }

    /// Exact data consumes no randomness in the refresh, the invariant
    /// behind draw-for-draw agreement with the Gaussian model.
    #[test]
    fn exact_data_consumes_no_randomness() {
        let values = vec![-0.4, 0.1, 0.3];
        let mut outcome = IntervalCensoredOutcome::new(values.clone(), values.clone());
        outcome.init(&values);
        assert_eq!(outcome.n_censored(), 0);
        let mut rng = chain_rng(3);
        let mut untouched = chain_rng(3);
        let mut latent = [-0.4, 0.1, 0.3];
        outcome.working_response(&[0.0; 3], &[1.0; 3], &mut latent, &mut rng);
        assert_eq!(latent, [-0.4, 0.1, 0.3]);
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Config, Outcome};
    use crate::data::Data;
    use crate::sampler::Sampler;
    use crate::tessellation::Tessellation;

    /// The rows of one cell as the quadrature sees them: the exact
    /// values, and the bound pair of each censored row.
    struct CellData {
        exact: Vec<f64>,
        censored: Vec<(f64, f64)>,
    }

    fn normal_cdf(z: f64) -> f64 {
        if z == f64::NEG_INFINITY {
            return 0.0;
        }
        if z == f64::INFINITY {
            return 1.0;
        }
        0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
    }

    /// Posterior means of sigma^2 and of every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2 with an
    /// inner quadrature over each cell mean. Given sigma^2 the cells are
    /// independent; cell k's marginal integrates the model's own
    /// interval likelihood, an exact row contributing
    /// N(l_i; mu, sigma^2) and a censored row
    /// Phi((u_i - mu) / sigma) - Phi((l_i - mu) / sigma). Independent of
    /// the engine.
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
                    for &v in &cell.exact {
                        term += -0.5 * (v - mu) * (v - mu) / sigma_sq - 0.5 * t;
                    }
                    for &(lo, hi) in &cell.censored {
                        let mass = normal_cdf((hi - mu) / sigma) - normal_cdf((lo - mu) / sigma);
                        term += mass.ln();
                    }
                    best = best.max(term);
                    terms.push((mu, term));
                }
                // A sigma^2 gridpoint the cell's interval likelihood
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

    /// On a fixed tessellation the chain is the interval-censored Gibbs
    /// sampler; its means of sigma^2 and of every mu_k match the
    /// numerical integration of the model's own likelihood within 4
    /// batch-means MCSE.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let values: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 13.0 - 0.5).collect();
        // Rows cycle through exact, two-sided, censored below and
        // censored above.
        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);
        for (i, &v) in values.iter().enumerate() {
            match i % 4 {
                0 => {
                    lower.push(v);
                    upper.push(v);
                }
                1 => {
                    lower.push(v - 0.15);
                    upper.push(v + 0.1);
                }
                2 => {
                    lower.push(f64::NEG_INFINITY);
                    upper.push(v);
                }
                _ => {
                    lower.push(v);
                    upper.push(f64::INFINITY);
                }
            }
        }
        let x = Data::new(xs, n, 1).unwrap();
        let lambda = 0.04;
        let config = Config::new()
            .with_outcome(Outcome::interval_censored())
            .with_m(1);
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 61_u64), (vec![0.0], 62)] {
            let b = centres.len();
            let mut sampler =
                Sampler::pinned_prior_interval_censored(&config, &x, &lower, &upper, lambda, seed)
                    .unwrap();
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
                    exact: Vec::new(),
                    censored: Vec::new(),
                })
                .collect();
            for (i, (&cell, &v)) in assignments.iter().zip(&values).enumerate() {
                match i % 4 {
                    0 => cells[cell as usize].exact.push(v),
                    1 => cells[cell as usize].censored.push((v - 0.15, v + 0.1)),
                    2 => cells[cell as usize].censored.push((f64::NEG_INFINITY, v)),
                    _ => cells[cell as usize].censored.push((v, f64::INFINITY)),
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
