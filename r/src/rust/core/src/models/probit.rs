//! The probit model for a binary response,
//! [`Outcome::Probit`](crate::Outcome::Probit):
//!
//! ```text
//! y_i in {0, 1},   P(y_i = 1 | x_i) = Phi(c + f(x_i)),   f(x) = sum_{j=1}^m g(x; T_j, M_j),
//! ```
//!
//! Phi the standard normal distribution function and c a fixed offset.
//! The sampler is the data augmentation of Albert and Chib (1993): a latent
//! z_i ~ N(c + f(x_i), 1), truncated to z_i > 0 when y_i = 1 and to
//! z_i < 0 when y_i = 0, is refreshed before each sweep (Robert 1995
//! exponential rejection for the truncated draw), and the mean ensemble is
//! then updated as in the Gaussian model with z - c as the response and
//! unit variance. This is the construction of Chipman, George and McCulloch
//! (2010), s. 4, with tessellations in place of trees.
//!
//! # Priors
//!
//! Cell means mu ~ N(0, sigma_mu^2) with sigma_mu = 3 / (k sqrt m) on the
//! latent scale, so that the prior on f(x) puts high probability on
//! [-3, 3] (Chipman, George and McCulloch 2010, s. 4; BART `pbart`). The
//! response is not scaled; the covariates are scaled as in the Gaussian
//! model. The structural priors (cells, active covariates, centres) are
//! those of the Gaussian model.
//!
//! # Fixed rather than estimated
//!
//! The latent variance is 1: it is not identified in a probit model and
//! no sigma^2 is drawn or reported. The offset c is fixed at
//! [`Config::offset`](crate::Config::offset), which defaults to
//! Phi^-1(ybar) (BART `binaryOffset`); the initial state is f = 0, the
//! offset carrying the mean. Chipman, George and McCulloch (2010) centre
//! at c = 0; the authors' Binary AddiVortes script has no offset and
//! initialises the latent fit at ybar.
//!
//! # Correspondence
//!
//! With BART `pbart`: m = `ntree`, k = `k`, c = `binaryOffset`, burn_in =
//! `nskip`, draws = `ndpost`, thinning = `keepevery`. The crate's defaults
//! are k = 3 and m = 200; `pbart` defaults to k = 2 and `ntree = 50`. With
//! the authors' script (`AddiVortes_Algorithm`): m = `m`, k = `k`,
//! sigma_c = `var`, omega = `Omega`, lambda_c = `lambda_rate`, burn_in =
//! `burn_in`, and draws is `max_iter - burn_in`.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of P(y = 1 | x) and `predict_draws` its
//! per-draw values; `predict_latent` is c + f(x) per draw;
//! `credible_interval` is on the probability scale; `log_likelihood` is the
//! Bernoulli log-likelihood; `prediction_interval` and `predict_variance`
//! return [`Error::NotApplicable`](crate::Error::NotApplicable);
//! `sigma` is empty; `in_sample_rmse` is the root Brier score of the
//! posterior-mean probabilities against the labels.
//!
//! # Input
//!
//! The response holds values in {0, 1} with both present; any other value
//! is [`Error::InvalidLabel`](crate::Error::InvalidLabel), a constant
//! response [`Error::DegenerateResponse`](crate::Error::DegenerateResponse).
//!

use crate::maths;
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::{self, Rng};

/// The probit outcome behind the [`OutcomeModel`] contract: labels in
/// {0, 1}, a fixed offset c, and the Albert and Chib (1993) latent
/// response refreshed each sweep by the Robert (1995) truncated draw. The
/// latent variance is fixed at 1, so the scale mode is `Fixed(1.0)` and
/// no variance ensemble may attach.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProbitOutcome {
    labels: Vec<f64>,
    offset: f64,
}

impl ProbitOutcome {
    /// The half-width of the cell-mean prior on the latent scale:
    /// sigma_mu = 3 / (k sqrt m), so the prior on f puts high probability
    /// on [-3, 3] (Chipman, George and McCulloch 2010, s. 4).
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 3.0;

    /// An outcome with the offset resolved: the configured value, or
    /// Phi^-1(ybar) (the BART package's `binaryOffset`).
    pub(crate) fn new(offset: Option<f64>, mean_y: f64) -> Self {
        Self {
            labels: Vec::new(),
            offset: offset.unwrap_or_else(|| maths::normal_quantile(mean_y)),
        }
    }

    /// The offset c of P(y = 1 | x) = Phi(c + f(x)).
    pub(crate) fn offset(&self) -> f64 {
        self.offset
    }

    /// The labels the latent response conditions on.
    pub(crate) fn labels(&self) -> &[f64] {
        &self.labels
    }

    /// Replace the labels; the caller validates them.
    pub(crate) fn set_labels(&mut self, y: &[f64]) {
        self.labels.copy_from_slice(y);
    }
}

impl OutcomeModel for ProbitOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Binary
    }

    fn init(&mut self, y: &[f64]) {
        self.labels = y.to_vec();
    }

    fn draw_extra(&mut self, _y: &[f64], _total: &[f64], _precision: &[f64], _rng: &mut Rng) {}

    /// z_i ~ N(c + f(x_i), 1) truncated to z_i > 0 when y_i = 1 and
    /// z_i < 0 when y_i = 0; the working response is z_i - c. The
    /// precisions are not read: the latent variance is fixed at 1.
    fn working_response(
        &mut self,
        total: &[f64],
        _precision: &[f64],
        y: &mut [f64],
        rng: &mut Rng,
    ) {
        for ((slot, &label), &f) in y.iter_mut().zip(self.labels.iter()).zip(total) {
            let mean = f + self.offset;
            let z = if label == 1.0 {
                mean + rng::truncated_standard_normal_above(-mean, rng)
            } else {
                mean - rng::truncated_standard_normal_above(mean, rng)
            };
            *slot = z - self.offset;
        }
    }

    fn weights(&self) -> Option<&[f64]> {
        None
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Fixed(1.0)
    }

    fn predictive_quantile(&self, _mean: f64, _sd: f64, _p: f64) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::*;
    use crate::rng::chain_rng;

    #[test]
    fn the_probit_outcome_answers_the_contract() {
        let labels = [1.0, 0.0, 1.0];
        let mut outcome = ProbitOutcome::new(None, 2.0 / 3.0);
        assert!((maths::normal_cdf(outcome.offset()) - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(ProbitOutcome::new(Some(0.25), 0.5).offset(), 0.25);
        outcome.init(&labels);
        assert_eq!(outcome.labels(), &labels);
        assert_eq!(outcome.required_data(), RequiredData::Binary);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Fixed(1.0));
        assert_eq!(outcome.weights(), None);
        assert_eq!(outcome.predictive_quantile(0.0, 1.0, 0.5), None);
        let mut rng = chain_rng(11);
        let mut y = vec![0.0; 3];
        outcome.working_response(&[0.0, 0.0, 0.0], &[1.0; 3], &mut y, &mut rng);
        for (z, &label) in y.iter().zip(&labels) {
            assert_eq!((z + outcome.offset() > 0.0), label == 1.0);
        }
        outcome.set_labels(&[0.0, 1.0, 0.0]);
        assert_eq!(outcome.labels(), &[0.0, 1.0, 0.0]);
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Config;
    use crate::data::Data;
    use crate::sampler::Sampler;
    use crate::tessellation::Tessellation;

    /// Posterior mean of mu for a cell with n1 ones and n0 zeros under
    /// p(mu | y) proportional to N(mu; 0, sigma_mu^2) Phi(mu)^n1
    /// (1 - Phi(mu))^n0, by the trapezium rule on [-8, 8]. Independent of
    /// the engine: the normal distribution function through `erfc`.
    fn quadrature_mean(n1: f64, n0: f64, sigma_mu_sq: f64) -> f64 {
        let phi = |m: f64| 0.5 * libm::erfc(-m * std::f64::consts::FRAC_1_SQRT_2);
        let steps = 40_000;
        let (lo, hi) = (-8.0_f64, 8.0_f64);
        let mut log_weights = Vec::with_capacity(steps + 1);
        let mut grid = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let m = lo + (hi - lo) * i as f64 / steps as f64;
            let lp = -0.5 * m * m / sigma_mu_sq + n1 * phi(m).ln() + n0 * (1.0 - phi(m)).ln();
            grid.push(m);
            log_weights.push(lp);
        }
        let max = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|lp| (lp - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter().zip(&grid).map(|(w, m)| w * m).sum::<f64>() / total
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

    /// On a fixed tessellation with c = 0 the chain is the Albert-Chib Gibbs
    /// sampler for independent cells; its mean of mu_k matches quadrature
    /// within 4 batch-means MCSE. With k = 3 and m = 1, sigma_mu^2 = 1.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let labels: Vec<f64> = (0..n)
            .map(|i| if (i * 5) % 7 < 3 { 1.0 } else { 0.0 })
            .collect();
        let x = Data::new(xs, n, 1).unwrap();
        let config = Config::new()
            .with_outcome(crate::config::Outcome::probit())
            .with_offset(0.0)
            .with_m(1);
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 31_u64), (vec![0.0], 32)] {
            let b = centres.len();
            let mut sampler = Sampler::pinned_prior(&config, &x, &labels, 1.0, seed).unwrap();
            assert_eq!(sampler.mean_sigma_mu_sq(), 1.0);
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
            let cells = sampler.mean_cells(0).to_vec();
            let mut n1 = vec![0.0; b];
            let mut n0 = vec![0.0; b];
            for (&cell, &label) in cells.iter().zip(&labels) {
                if label == 1.0 {
                    n1[cell as usize] += 1.0;
                } else {
                    n0[cell as usize] += 1.0;
                }
            }
            assert!(n1.iter().zip(&n0).all(|(a, z)| a + z > 0.0));

            for _ in 0..500 {
                sampler.conjugate_sweep();
            }
            let kept = 60_000;
            let mut mus: Vec<Vec<f64>> = vec![Vec::with_capacity(kept); b];
            for _ in 0..kept {
                sampler.conjugate_sweep();
                for (k, series) in mus.iter_mut().enumerate() {
                    series.push(sampler.tessellations()[0].mus()[k]);
                }
            }
            for k in 0..b {
                let reference = quadrature_mean(n1[k], n0[k], 1.0);
                let (mean, mcse) = batch_means_mcse(&mus[k]);
                assert!(
                    (mean - reference).abs() < 4.0 * mcse,
                    "mu_{k} {mean} vs {reference} +- {mcse}"
                );
            }
        }
    }
}
