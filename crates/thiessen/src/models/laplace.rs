//! The Laplace model for a continuous response with outliers,
//! [`Outcome::Laplace`](crate::Outcome::Laplace), experimental
//! (`docs/experimental.md`):
//!
//! ```text
//! y_i = f(x_i) + e_i,   e_i ~ Laplace(0, sigma),   f(x) = sum_{j=1}^m g(x; T_j, M_j),
//! ```
//!
//! density (2 sigma)^-1 exp(-|e| / sigma): errors with exponential
//! tails, heavier than the Gaussian's and lighter than any low-df
//! Student-t's, so a wild observation is discounted at rate 1 / |r|
//! against the t model's 1 / r^2. sigma is the Laplace scale; the error
//! standard deviation is sigma sqrt 2.
//!
//! # Sampler
//!
//! The scale mixture of normals with exponential mixing (Andrews and
//! Mallows 1974; the hierarchy of Park and Casella 2008):
//!
//! ```text
//! y_i | w_i ~ N(f(x_i), sigma^2 / w_i),   1 / w_i ~ Exp(rate 1 / 2),
//! ```
//!
//! whose marginal over w_i is exactly Laplace(f(x_i), sigma). Each sweep
//! the weights are redrawn from their conditional
//!
//! ```text
//! w_i | r_i, sigma^2 ~ Inverse-Gaussian(mean sigma / |r_i|, shape 1),
//! ```
//!
//! r_i = y_i - f(x_i), the conditional of Park and Casella (2008, s. 3)
//! for the mixing precision, through the shared scale-mixture refresh,
//! which recovers 1 / sigma^2 from the standing precisions and reduces
//! the draw to the prior under prior-only sampling. A residual so small
//! that the mean would overflow the draw takes the r = 0 limit,
//! 1 / w_i ~ chi^2_1. The kernel then draws sigma^2 from
//! Inv-Gamma((nu + n) / 2, (nu lambda + sum w_i r_i^2) / 2) and the cell
//! means from their Normal conditionals against the precisions
//! w_i / sigma^2; the scan w | f, sigma^2 then sigma^2 | w, f then
//! f | w, sigma^2 is a valid Gibbs sampler. No structural move gains an
//! acceptance-ratio term.
//!
//! # Priors
//!
//! The Gaussian model's for the cells and sigma^2, on the response
//! min-max scaled to [-0.5, 0.5]; sigma^2 ~ nu lambda / chi^2_nu with
//! lambda set so that Pr(sigma < sigma_hat) = q, sigma_hat the
//! least-squares residual standard deviation, which under Laplace errors
//! estimates sigma sqrt 2 rather than sigma; the prior's spread absorbs
//! the overstatement. The weights carry the exponential mixing prior;
//! the model has no parameters of its own.
//!
//! # Fixed rather than estimated
//!
//! Nothing: the model has no parameters beyond the cells and sigma^2.
//! sigma^2 is sampled, so the scale mode is `Sampled`; a variance
//! ensemble is nonetheless rejected at validation, as for every
//! scale-mixture outcome, because per-observation weights and a
//! per-observation variance product both model dispersion and their
//! joint identification awaits its argument.
//!
//! # Correspondence
//!
//! With Park and Casella (2008): the mixing hierarchy and the
//! inverse-Gaussian conditional are theirs, applied to the errors rather
//! than to regression coefficients, with the ensemble in place of the
//! linear predictor. No maintained BART-family package ships a Laplace
//! error model. With the crate's Student-t model: the same refresh with
//! the Gamma conditional replaced by the inverse-Gaussian one.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of f(x) and `predict_draws` its
//! per-draw values; `predict_variance` is the error variance 2 sigma_d^2
//! per draw; `prediction_interval` is the central interval of the
//! equal-weight mixture over draws of Laplace(f_d(x), sigma_d), by
//! bisection on the mixture CDF; `log_likelihood` is the Laplace log
//! density per draw; `sigma` is sigma_d on the caller's scale, the
//! Laplace scale rather than the error standard deviation;
//! `in_sample_rmse` is against the observed response.
//!
//! # Input
//!
//! A continuous response, min-max scaled as the Gaussian model's.

use crate::maths;
use crate::models::scale_mixture;
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::{self, Rng};

/// Below this likelihood curvature r^2 / sigma^2 the inverse-Gaussian
/// mean sigma / |r| exceeds 1e100 and its sampler would overflow; the
/// r = 0 limit is drawn instead.
const LIMIT_CURVATURE: f64 = 1e-200;

/// The Laplace outcome behind the [`OutcomeModel`] contract: the
/// response is observed, the per-observation inverse-Gaussian weights
/// answer through the precisions, and sigma^2 is sampled by the kernel
/// from its weighted inverse-gamma conditional.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct LaplaceOutcome {
    weights: Vec<f64>,
}

impl LaplaceOutcome {
    /// The half-width of the cell-mean prior: the Gaussian model's, the
    /// response being scaled to [-0.5, 0.5].
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 0.5;
}

impl OutcomeModel for LaplaceOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Continuous
    }

    /// Start every weight at 1, the Gaussian state. The weights are
    /// sampled state, not data: a response replacement leaves them
    /// standing, because the precisions were written with them.
    fn init(&mut self, y: &[f64]) {
        if self.weights.is_empty() {
            self.weights = vec![1.0; y.len()];
        }
    }

    /// The weight refresh: one inverse-Gaussian draw per observation
    /// (one exponential under prior-only sampling; one chi-squared at
    /// the r = 0 limit).
    fn draw_extra(&mut self, y: &[f64], total: &[f64], precision: &[f64], rng: &mut Rng) {
        scale_mixture::refresh_weights(
            &mut self.weights,
            y,
            total,
            precision,
            rng,
            |residual, scale_precision, rng| {
                if scale_precision == 0.0 {
                    return 1.0 / rng::gamma(1.0, 2.0, rng);
                }
                let curvature = scale_precision * residual * residual;
                if curvature < LIMIT_CURVATURE {
                    return 1.0 / rng::gamma(0.5, 2.0, rng);
                }
                rng::inverse_gaussian(1.0 / curvature.sqrt(), 1.0, rng)
            },
        );
    }

    /// The identity: the response is observed.
    fn working_response(
        &mut self,
        _total: &[f64],
        _precision: &[f64],
        _y: &mut [f64],
        _rng: &mut Rng,
    ) {
    }

    fn weights(&self) -> Option<&[f64]> {
        Some(&self.weights)
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Sampled
    }

    /// Quantile of the predictive Laplace(f, sigma); `sd` is the scale
    /// sigma, not the error standard deviation.
    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64> {
        Some(mean + sd * maths::laplace_quantile(p))
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::*;
    use crate::rng::chain_rng;

    #[test]
    fn the_laplace_outcome_answers_the_contract() {
        let mut outcome = LaplaceOutcome::default();
        let y = [0.1, -0.2, 0.3];
        outcome.init(&y);
        assert_eq!(outcome.required_data(), RequiredData::Continuous);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(outcome.weights(), Some(&[1.0, 1.0, 1.0][..]));

        let mut rng = chain_rng(21);
        let mut latent = y;
        outcome.working_response(&[0.0; 3], &[1.0; 3], &mut latent, &mut rng);
        assert_eq!(latent, y);

        let median = outcome.predictive_quantile(0.3, 1.0, 0.5).unwrap();
        assert!((median - 0.3).abs() < 1e-12);
        let q = outcome.predictive_quantile(0.0, 2.0, 0.975).unwrap();
        assert!((q - 2.0 * -(2.0 * 0.025_f64).ln()).abs() < 1e-12);
        assert!((outcome.predictive_quantile(0.0, 2.0, 0.025).unwrap() + q).abs() < 1e-12);
    }

    /// A second `init`, the response-replacement path, keeps the
    /// standing weights the precisions were written with.
    #[test]
    fn a_response_replacement_keeps_the_weights() {
        let mut outcome = LaplaceOutcome::default();
        outcome.init(&[0.1, -0.2]);
        let mut rng = chain_rng(2);
        outcome.draw_extra(&[0.1, -0.2], &[0.0, 0.0], &[4.0, 4.0], &mut rng);
        let drawn = outcome.weights().unwrap().to_vec();
        assert_ne!(drawn, vec![1.0, 1.0]);
        outcome.init(&[0.3, 0.4]);
        assert_eq!(outcome.weights(), Some(&drawn[..]));
    }

    /// E[w | r] = sigma / |r|, the inverse-Gaussian mean: the weight
    /// decays at rate 1 / |r|.
    #[test]
    fn the_weight_conditional_has_the_park_casella_mean() {
        let sigma_sq = 0.25;
        let y = [0.5, 2.0];
        let total = [0.0, 0.0];
        let mut outcome = LaplaceOutcome::default();
        outcome.init(&y);
        let precision = [1.0 / sigma_sq; 2];
        let mut rng = chain_rng(5);
        let n = 40_000;
        let mut means = [0.0; 2];
        for _ in 0..n {
            outcome.weights.copy_from_slice(&[1.0, 1.0]);
            outcome.draw_extra(&y, &total, &precision, &mut rng);
            for (mean, w) in means.iter_mut().zip(outcome.weights().unwrap()) {
                *mean += w;
            }
        }
        for (mean, &value) in means.iter_mut().zip(&y) {
            *mean /= n as f64;
            let expected = sigma_sq.sqrt() / value.abs();
            assert!(
                (*mean - expected).abs() < 0.02 * expected,
                "{mean} vs {expected}"
            );
        }
    }

    /// Zero precisions reduce the draw to the prior 1 / w ~ Exp(1 / 2),
    /// mean 2; a zero residual takes the limit 1 / w ~ chi^2_1, mean 1.
    #[test]
    fn prior_only_and_zero_residual_draws_come_from_their_limits() {
        let mut outcome = LaplaceOutcome::default();
        outcome.init(&[0.4, 0.0]);
        let mut rng = chain_rng(9);
        let n = 40_000;
        let mut sums = [0.0; 2];
        for _ in 0..n {
            outcome.weights.copy_from_slice(&[1.0, 1.0]);
            outcome.draw_extra(&[0.4, 0.0], &[0.0, 0.0], &[0.0, 4.0], &mut rng);
            for (sum, w) in sums.iter_mut().zip(outcome.weights().unwrap()) {
                *sum += 1.0 / w;
            }
        }
        assert!(
            (sums[0] / n as f64 - 2.0).abs() < 0.05,
            "{}",
            sums[0] / n as f64
        );
        assert!(
            (sums[1] / n as f64 - 1.0).abs() < 0.03,
            "{}",
            sums[1] / n as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Config, Outcome};
    use crate::data::Data;
    use crate::sampler::Sampler;
    use crate::tessellation::Tessellation;

    /// Posterior means of sigma^2 and of every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2 with an
    /// inner quadrature over each cell mean. Given sigma^2 the weights
    /// marginalise analytically, so cell k contributes
    ///
    /// ```text
    /// m_k(sigma^2) = int N(mu; 0, sigma_mu^2) prod_obs (2 sigma)^-1 exp(-|y_i - mu| / sigma) dmu,
    /// ```
    ///
    /// the model's own marginal likelihood integrated numerically,
    /// independent of the engine and of the augmentation.
    fn quadrature_reference(
        cells: &[Vec<f64>],
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
                    for &y in cell {
                        term += -(2.0 * sigma).ln() - (y - mu).abs() / sigma;
                    }
                    best = best.max(term);
                    terms.push((mu, term));
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

    /// On a fixed tessellation the chain is the Park and Casella (2008)
    /// Gibbs sampler; its means of sigma^2 and of every mu_k match the
    /// numerical integration of the marginal Laplace likelihood within
    /// 4 batch-means MCSE.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let mut y: Vec<f64> = (0..n)
            .map(|i| (((i * 5) % 11) as f64 / 11.0 - 0.5) * 0.4)
            .collect();
        y[3] = 0.48;
        y[8] = -0.45;
        let x = Data::new(xs, n, 1).unwrap();
        let lambda = 0.04;
        let config = Config::new().with_outcome(Outcome::laplace()).with_m(1);
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 61_u64), (vec![0.0], 62)] {
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
            let mut cells: Vec<Vec<f64>> = vec![Vec::new(); b];
            for (&cell, &v) in assignments.iter().zip(&y) {
                cells[cell as usize].push(v);
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
