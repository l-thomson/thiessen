//! The ordinal probit model for ordered categories,
//! [`Outcome::Ordinal`](crate::Outcome::Ordinal), experimental
//! (`docs/experimental.md`):
//!
//! ```text
//! y_i in {0, ..., K - 1},   z_i = c + f(x_i) + e_i,   e_i ~ N(0, 1),
//! y_i = k  iff  gamma_k < z_i <= gamma_{k+1},
//! -inf = gamma_0 < gamma_1 = 0 < gamma_2 < ... < gamma_{K-1} < gamma_K = inf,
//! ```
//!
//! the model of Albert and Chib (1993, s. 5): ordered categories with no
//! scale behind them, observed as integer codes. The latent variance is
//! fixed at 1 and the first cutpoint at 0, the standard identification;
//! the offset c is fixed at Phi^-1(share of y >= 1) by default, the
//! probit rule at K = 2, and the K - 2 interior cutpoints are sampled.
//!
//! # Sampler
//!
//! Three blocks per sweep: gamma | f, y with the latents integrated
//! out, then z | gamma, f, y, then f | z as the Gaussian sweep with
//! unit variance. The first two compose into one joint draw from
//! p(gamma, z | f, y), so the scan needs no further correction.
//!
//! The cutpoint move follows Cowles (1996): one-at-a-time Gibbs
//! cutpoint updates mix impractically slowly as n grows, so all
//! interior cutpoints move jointly against the collapsed likelihood
//!
//! ```text
//! prod_i [Phi(gamma_{y_i + 1} - c - f_i) - Phi(gamma_{y_i} - c - f_i)],
//! ```
//!
//! by a Gaussian random walk on the log-gap transformation
//! delta_k = ln(gamma_k - gamma_{k-1}) of Albert and Chib (2001), which
//! is unconstrained, so the proposal is symmetric and the acceptance
//! ratio is the collapsed likelihood ratio times the prior ratio in
//! delta. The walk scale is 2.38 / sqrt(n (K - 2)), the optimal-scaling
//! rate of Roberts, Gelman and Gilks (1997) against a per-observation
//! information of order one; it is a constant of the fit, so no
//! adaptation disturbs detailed balance. The latent refresh reuses the
//! censoring draws: category 0 below 0, category K - 1 above
//! gamma_{K-1}, an interior category between its cutpoints (Robert
//! 1995).
//!
//! # Priors
//!
//! Cell means as the probit model's, sigma_mu = 3 / (k sqrt m) on the
//! latent scale; log-gaps delta_k ~ N(0, cutpoint_sd^2) independent,
//! `cutpoint_sd` default 1. The response is not scaled. Initial
//! cutpoints come from the marginal category shares,
//! gamma_k = c + Phi^-1(share of y < k).
//!
//! # Fixed rather than estimated
//!
//! The latent variance (1, not identified; no sigma^2 is drawn and a
//! variance ensemble is rejected, derived from the scale mode as for
//! the probit model), gamma_1 = 0 and the offset c.
//!
//! # Correspondence
//!
//! MCMCpack `MCMCoprobit` fits the linear-model analogue with the
//! Cowles (1996) and Albert-Chib (2001) cutpoint updates; the BART
//! family has no ordinal model. Two-category data reproduces the probit
//! model draw for draw at the same seed: with K = 2 there is no
//! interior cutpoint, the MH step consumes no randomness and the
//! refresh is the probit draw.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of the expected category
//! E[y | x] = sum_{k >= 1} Phi(c + f(x) - gamma_k) and `predict_draws`
//! its per-draw values; `predict_latent` is c + f(x) per draw;
//! `predict_category_probabilities` is the posterior mean of
//! P(y = k | x) per row; `log_likelihood` is the ordinal likelihood;
//! `cutpoint_draws` holds the interior cutpoints per kept draw;
//! `prediction_interval` and `predict_variance` are
//! [`Error::NotApplicable`](crate::Error::NotApplicable); `sigma` is
//! empty; `in_sample_rmse` is against the observed codes.
//!
//! # Input
//!
//! Integer codes 0 to K - 1
//! ([`Error::InvalidOrdinalLabel`](crate::Error::InvalidOrdinalLabel)),
//! a constant response rejected as
//! [`Error::DegenerateResponse`](crate::Error::DegenerateResponse). An
//! empty category is permitted; its cutpoint gap follows the prior. A
//! replacement response through
//! [`set_response`](crate::Sampler::set_response) keeps the offset and
//! the cutpoints, which are a constant of the fit and sampled state.

use crate::maths;
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::{self, Rng};

/// The ordinal outcome behind the [`OutcomeModel`] contract: codes in
/// 0..K, a fixed offset c, the sampled interior cutpoints with their
/// blocked collapsed MH move, and the truncated refresh of every row's
/// latent. The latent variance is fixed at 1, so the scale mode is
/// `Fixed(1.0)` and no variance ensemble may attach.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OrdinalOutcome {
    categories: usize,
    cutpoint_sd: f64,
    offset: f64,
    configured_offset: Option<f64>,
    labels: Vec<f64>,
    /// gamma_1 = 0 through gamma_{K-1}, increasing; length K - 1.
    gamma: Vec<f64>,
    proposal_sd: f64,
    #[cfg(test)]
    pub(crate) drop_cutpoint_prior: bool,
}

impl OrdinalOutcome {
    /// The half-width of the cell-mean prior on the latent scale: the
    /// probit model's 3.
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 3.0;

    /// An outcome over K categories; the offset and the cutpoints
    /// resolve from the labels at [`init`](OutcomeModel::init).
    pub(crate) fn new(categories: usize, offset: Option<f64>, cutpoint_sd: f64) -> Self {
        Self {
            categories,
            cutpoint_sd,
            offset: offset.unwrap_or(0.0),
            configured_offset: offset,
            labels: Vec::new(),
            gamma: Vec::new(),
            proposal_sd: 0.0,
            #[cfg(test)]
            drop_cutpoint_prior: false,
        }
    }

    /// The offset c.
    pub(crate) fn offset(&self) -> f64 {
        self.offset
    }

    /// The codes the latent response conditions on.
    pub(crate) fn labels(&self) -> &[f64] {
        &self.labels
    }

    /// The interior cutpoints gamma_2 through gamma_{K-1}; empty at
    /// K = 2.
    pub(crate) fn free_cutpoints(&self) -> &[f64] {
        &self.gamma[1..]
    }

    /// Replace the interior cutpoints, which must number K - 2 and,
    /// after gamma_1 = 0, increase strictly; returns whether they did.
    pub(crate) fn set_free_cutpoints(&mut self, free: &[f64]) -> bool {
        let valid = free.len() + 2 == self.categories
            && std::iter::once(0.0)
                .chain(free.iter().copied())
                .collect::<Vec<f64>>()
                .windows(2)
                .all(|w| w[0] < w[1] && w[1].is_finite());
        if valid {
            self.gamma[1..].copy_from_slice(free);
        }
        valid
    }

    /// Replace the codes; the caller validates them. The offset and the
    /// cutpoints stand: they are a constant of the fit and sampled
    /// state, and the next sweep's refresh redraws every latent from
    /// the new codes.
    pub(crate) fn set_labels(&mut self, y: &[f64]) {
        self.labels.copy_from_slice(y);
    }

    /// The log collapsed likelihood of the codes under cutpoints
    /// `gamma` (interior values, gamma_1 = 0 implicit) given the means
    /// c + f_i.
    fn collapsed_log_likelihood(&self, gamma: &[f64], total: &[f64]) -> f64 {
        let mut ll = 0.0;
        for (&label, &f) in self.labels.iter().zip(total) {
            let k = label as usize;
            let mean = self.offset + f;
            let above = if k + 1 > self.categories - 1 {
                1.0
            } else {
                maths::normal_cdf(gamma[k] - mean)
            };
            let below = if k == 0 {
                0.0
            } else {
                maths::normal_cdf(gamma[k - 1] - mean)
            };
            ll += maths::ln(above - below);
        }
        ll
    }

    /// The log prior of the interior cutpoints in the log-gap
    /// parameterisation: delta_k ~ N(0, cutpoint_sd^2) independent with
    /// delta_k = ln(gamma_k - gamma_{k-1}).
    fn log_prior(&self, gamma: &[f64]) -> f64 {
        let mut lp = 0.0;
        let mut previous = 0.0;
        for &g in &gamma[1..] {
            let delta = maths::ln(g - previous);
            lp += -0.5 * delta * delta / (self.cutpoint_sd * self.cutpoint_sd);
            previous = g;
        }
        lp
    }
}

impl OutcomeModel for OrdinalOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Ordinal
    }

    /// Store the codes and resolve the offset and the initial
    /// cutpoints from the marginal category shares:
    /// c = Phi^-1(share of y >= 1) unless configured, and
    /// gamma_k = c + Phi^-1(share of y < k), which puts gamma_1 at 0
    /// exactly when the offset is the default. An empty category is
    /// permitted: the shares are clamped away from 0 and 1, a floor the
    /// clamp leaves untouched wherever the share is interior, and the
    /// empty category's cutpoint gap then follows the prior. The walk
    /// scale is 2.38 / sqrt(n (K - 2)).
    fn init(&mut self, y: &[f64]) {
        self.labels = y.to_vec();
        let n = y.len() as f64;
        let mut counts = vec![0.0; self.categories];
        for &label in y {
            counts[label as usize] += 1.0;
        }
        let interior = |share: f64| share.clamp(0.5 / n, 1.0 - 0.5 / n);
        // (n - n_0) / n equals the probit rule's label mean exactly for
        // labels in {0, 1}, the bit-for-bit correspondence at K = 2.
        self.offset = self
            .configured_offset
            .unwrap_or_else(|| maths::normal_quantile(interior((n - counts[0]) / n)));
        let mut cumulative = 0.0;
        self.gamma = vec![0.0];
        for &count in counts.iter().take(self.categories - 1).skip(1) {
            cumulative += count;
            let previous = *self.gamma.last().expect("gamma_1 present");
            let next = self.offset + maths::normal_quantile(interior((counts[0] + cumulative) / n));
            // The marginal quantile can tie or invert under a
            // configured offset or an empty category; a positive floor
            // keeps the log-gap parameterisation defined.
            self.gamma.push(next.max(previous + 1e-3));
        }
        let free = (self.categories - 2) as f64;
        self.proposal_sd = if free > 0.0 {
            2.38 / (n * free).sqrt()
        } else {
            0.0
        };
    }

    /// The blocked collapsed MH move of the interior cutpoints: a
    /// symmetric Gaussian walk on the log-gaps, accepted on the
    /// collapsed likelihood ratio times the prior ratio. No randomness
    /// is consumed at K = 2, where no interior cutpoint exists.
    fn draw_extra(&mut self, _y: &[f64], total: &[f64], _precision: &[f64], rng: &mut Rng) {
        if self.categories < 3 {
            return;
        }
        let mut proposed = Vec::with_capacity(self.gamma.len());
        proposed.push(0.0);
        let mut previous_current = 0.0;
        let mut previous_proposed = 0.0;
        for &g in &self.gamma[1..] {
            let delta = maths::ln(g - previous_current);
            let step = delta + self.proposal_sd * rng::standard_normal(rng);
            previous_proposed += maths::exp(step);
            proposed.push(previous_proposed);
            previous_current = g;
        }
        let prior_ratio = self.log_prior(&proposed) - self.log_prior(&self.gamma);
        #[cfg(test)]
        let prior_ratio = if self.drop_cutpoint_prior {
            0.0
        } else {
            prior_ratio
        };
        let log_alpha = self.collapsed_log_likelihood(&proposed, total)
            - self.collapsed_log_likelihood(&self.gamma, total)
            + prior_ratio;
        if maths::ln(rng::uniform(rng)) < log_alpha {
            self.gamma = proposed;
        }
    }

    /// z_i ~ N(c + f_i, 1) truncated to the code's cutpoint interval;
    /// the working response is z_i - c. The precisions are not read:
    /// the latent variance is fixed at 1. Category 0 and category K - 1
    /// take the one-sided draws, an interior category the two-sided
    /// draw.
    fn working_response(
        &mut self,
        total: &[f64],
        _precision: &[f64],
        y: &mut [f64],
        rng: &mut Rng,
    ) {
        let last = self.categories - 1;
        for ((slot, &label), &f) in y.iter_mut().zip(self.labels.iter()).zip(total) {
            let k = label as usize;
            let mean = f + self.offset;
            let z = if k == 0 {
                mean - rng::truncated_standard_normal_above(mean - self.gamma[0], rng)
            } else if k == last {
                mean + rng::truncated_standard_normal_above(self.gamma[k - 1] - mean, rng)
            } else {
                mean + rng::truncated_standard_normal_between(
                    self.gamma[k - 1] - mean,
                    self.gamma[k] - mean,
                    rng,
                )
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
    use rand_core::RngCore;

    #[test]
    fn the_ordinal_outcome_answers_the_contract() {
        let mut outcome = OrdinalOutcome::new(3, None, 1.0);
        let y = [0.0, 1.0, 2.0, 1.0, 0.0, 2.0];
        outcome.init(&y);
        assert_eq!(outcome.labels(), &y);
        assert_eq!(outcome.required_data(), RequiredData::Ordinal);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Fixed(1.0));
        assert_eq!(outcome.weights(), None);
        assert_eq!(outcome.predictive_quantile(0.0, 1.0, 0.5), None);
        // c = Phi^-1(2/3 above zero): shares 1/3, 1/3, 1/3.
        assert!((maths::normal_cdf(outcome.offset()) - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(outcome.free_cutpoints().len(), 1);
        assert!(outcome.free_cutpoints()[0] > 0.0);

        let total = [0.0; 6];
        let mut rng = chain_rng(41);
        let mut latent = [0.0; 6];
        outcome.working_response(&total, &[1.0; 6], &mut latent, &mut rng);
        let gamma_2 = outcome.free_cutpoints()[0];
        for (z, &label) in latent.iter().zip(&y) {
            let z = z + outcome.offset();
            match label as usize {
                0 => assert!(z <= 0.0),
                1 => assert!((0.0..=gamma_2).contains(&z)),
                _ => assert!(z >= gamma_2),
            }
        }
    }

    /// The cutpoint move keeps gamma increasing and consumes a fixed
    /// amount of randomness per sweep.
    #[test]
    fn the_cutpoint_move_preserves_the_ordering() {
        let mut outcome = OrdinalOutcome::new(4, None, 1.0);
        let y = [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        outcome.init(&y);
        let total = [0.0; 8];
        let mut rng = chain_rng(43);
        for _ in 0..200 {
            outcome.draw_extra(&[], &total, &[], &mut rng);
            let free = outcome.free_cutpoints();
            assert!(free[0] > 0.0 && free[1] > free[0]);
        }
    }

    /// K = 2 has no interior cutpoint: the move consumes no randomness,
    /// the invariant behind draw-for-draw agreement with the probit
    /// model.
    #[test]
    fn two_categories_consume_no_randomness_in_the_move() {
        let mut outcome = OrdinalOutcome::new(2, None, 1.0);
        let y = [0.0, 1.0, 1.0];
        outcome.init(&y);
        assert!(outcome.free_cutpoints().is_empty());
        let mut rng = chain_rng(3);
        let mut untouched = chain_rng(3);
        outcome.draw_extra(&[], &[0.0; 3], &[], &mut rng);
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Config, Outcome};
    use crate::data::Data;
    use crate::sampler::Sampler;
    use crate::tessellation::Tessellation;

    fn normal_cdf(z: f64) -> f64 {
        0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
    }

    /// Posterior means of gamma_2 and of every cell mean for the fixed
    /// tessellation model with K = 3 and c = 0 by quadrature: outer grid
    /// over delta = ln gamma_2 against the N(0, s^2) prior, inner grid
    /// over each cell mean, the cells independent given gamma_2. Each
    /// row contributes its ordinal probability. Independent of the
    /// engine.
    fn quadrature_reference(
        cells: &[Vec<usize>],
        sigma_mu_sq: f64,
        cutpoint_sd: f64,
    ) -> (f64, Vec<f64>) {
        let outer = 400;
        let (d_lo, d_hi) = (-6.0_f64, 3.0_f64);
        let inner = 2000;
        let (mu_lo, mu_hi) = (-5.0_f64, 5.0_f64);
        let mut log_weights = Vec::with_capacity(outer + 1);
        let mut gammas = Vec::with_capacity(outer + 1);
        let mut cond_means: Vec<Vec<f64>> = vec![Vec::with_capacity(outer + 1); cells.len()];
        for i in 0..=outer {
            let delta = d_lo + (d_hi - d_lo) * i as f64 / outer as f64;
            let gamma_2 = delta.exp();
            let mut lp = -0.5 * delta * delta / (cutpoint_sd * cutpoint_sd);
            for (k, labels) in cells.iter().enumerate() {
                let mut best = f64::NEG_INFINITY;
                let mut terms = Vec::with_capacity(inner + 1);
                for j in 0..=inner {
                    let mu = mu_lo + (mu_hi - mu_lo) * j as f64 / inner as f64;
                    let mut term = -0.5 * mu * mu / sigma_mu_sq;
                    for &label in labels {
                        let p = match label {
                            0 => normal_cdf(-mu),
                            1 => normal_cdf(gamma_2 - mu) - normal_cdf(-mu),
                            _ => 1.0 - normal_cdf(gamma_2 - mu),
                        };
                        term += p.ln();
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
            gammas.push(gamma_2);
            log_weights.push(lp);
        }
        let max = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|lp| (lp - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mean_gamma = weights.iter().zip(&gammas).map(|(w, g)| w * g).sum::<f64>() / total;
        let mean_mus = cond_means
            .iter()
            .map(|means| weights.iter().zip(means).map(|(w, m)| w * m).sum::<f64>() / total)
            .collect();
        (mean_gamma, mean_mus)
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

    fn fixture() -> (Config, Data, Vec<f64>) {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let labels: Vec<f64> = (0..n).map(|i| ((i * 5) % 3) as f64).collect();
        let config = Config::new()
            .with_outcome(Outcome::ordinal(3))
            .with_offset(0.0)
            .with_m(1);
        (config, Data::new(xs, n, 1).unwrap(), labels)
    }

    /// (mean, MCSE) of gamma_2, the same per cell mean, and each
    /// cell's labels.
    type ChainSummary = ((f64, f64), Vec<(f64, f64)>, Vec<Vec<usize>>);

    fn chain_summary(sampler: &mut Sampler, centres: Vec<f64>, labels: &[f64]) -> ChainSummary {
        let b = centres.len();
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
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); b];
        for (&cell, &label) in assignments.iter().zip(labels) {
            cells[cell].push(label as usize);
        }
        for _ in 0..2000 {
            sampler.conjugate_sweep();
        }
        let kept = 200_000;
        let mut gamma = Vec::with_capacity(kept);
        let mut mus: Vec<Vec<f64>> = vec![Vec::with_capacity(kept); b];
        for _ in 0..kept {
            sampler.conjugate_sweep();
            gamma.push(sampler.cutpoints()[0]);
            for (k, series) in mus.iter_mut().enumerate() {
                series.push(sampler.tessellations()[0].mus[k]);
            }
        }
        (
            batch_means_mcse(&gamma),
            mus.iter().map(|series| batch_means_mcse(series)).collect(),
            cells,
        )
    }

    /// On a fixed tessellation the chain is the collapsed-cutpoint Gibbs
    /// sampler; its means of gamma_2 and of every mu_k match the
    /// numerical integration of the ordinal likelihood within 4
    /// batch-means MCSE.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let (config, x, labels) = fixture();
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 71_u64), (vec![0.0], 72)] {
            let mut sampler = Sampler::pinned_prior(&config, &x, &labels, 1.0, seed).unwrap();
            let sigma_mu_sq = sampler.mean_sigma_mu_sq();
            let ((g_mean, g_mcse), mu_summaries, cells) =
                chain_summary(&mut sampler, centres, &labels);
            let (ref_gamma, ref_mus) = quadrature_reference(&cells, sigma_mu_sq, 1.0);
            assert!(
                (g_mean - ref_gamma).abs() < 4.0 * g_mcse,
                "gamma_2 {g_mean} vs {ref_gamma} +- {g_mcse}"
            );
            for (k, (mean, mcse)) in mu_summaries.iter().enumerate() {
                assert!(
                    (mean - ref_mus[k]).abs() < 4.0 * mcse,
                    "mu_{k} {mean} vs {} +- {mcse}",
                    ref_mus[k]
                );
            }
        }
    }

    /// Dropping the prior ratio from the cutpoint acceptance changes the
    /// invariant target: the chain's gamma_2 mean leaves the quadrature
    /// gate the intact sampler passes.
    #[test]
    fn dropped_cutpoint_prior_is_rejected_by_the_known_answer_gate() {
        let (config, x, labels) = fixture();
        let config = config.with_cutpoint_sd(0.25);
        let mut sampler = Sampler::pinned_prior(&config, &x, &labels, 1.0, 73).unwrap();
        sampler.breakage = crate::broken::Breakage::DroppedCutpointPrior;
        let sigma_mu_sq = sampler.mean_sigma_mu_sq();
        let ((g_mean, g_mcse), _, cells) = chain_summary(&mut sampler, vec![0.0], &labels);
        let (ref_gamma, _) = quadrature_reference(&cells, sigma_mu_sq, 0.25);
        assert!(
            (g_mean - ref_gamma).abs() > 8.0 * g_mcse,
            "the broken chain stayed at the reference: {g_mean} vs {ref_gamma} +- {g_mcse}"
        );
    }
}
