//! The Student-t model for a continuous response with outliers,
//! [`Outcome::StudentT`](crate::Outcome::StudentT), experimental
//! (`docs/experimental.md`):
//!
//! ```text
//! y_i = f(x_i) + e_i,   e_i ~ sigma t_df,   f(x) = sum_{j=1}^m g(x; T_j, M_j),
//! ```
//!
//! the independent Student-t model of Geweke (1993) with the ensemble in
//! place of the linear predictor: errors with polynomial tails, so a wild
//! observation is discounted rather than dragging the fit. sigma is the
//! scale of the t, not the error standard deviation, which is
//! sigma sqrt(df / (df - 2)) for df > 2.
//!
//! # Sampler
//!
//! The scale mixture of normals (Andrews and Mallows 1974; Geweke 1993):
//!
//! ```text
//! y_i | w_i ~ N(f(x_i), sigma^2 / w_i),   w_i ~ Gamma(df / 2, rate df / 2),
//! ```
//!
//! whose marginal over w_i is exactly sigma t_df. Each sweep the weights
//! are redrawn from their conditional
//!
//! ```text
//! w_i | r_i, sigma^2 ~ Gamma((df + 1) / 2, rate (df + r_i^2 / sigma^2) / 2),
//! ```
//!
//! r_i = y_i - f(x_i), through the shared scale-mixture refresh, which
//! recovers 1 / sigma^2 from the standing precisions and reduces the
//! draw to the prior under prior-only sampling. The kernel then draws
//! sigma^2 from Inv-Gamma((nu + n) / 2, (nu lambda + sum w_i r_i^2) / 2),
//! the weighted form of the Gaussian conditional, and the cell means
//! from their Normal conditionals against the precisions w_i / sigma^2.
//! Each update conditions on the current values of the other blocks, so
//! the scan w | f, sigma^2 then sigma^2 | w, f then f | w, sigma^2 is a
//! valid Gibbs sampler. No structural move gains an acceptance-ratio
//! term: the weight draw is a Gibbs step against a fixed ensemble.
//!
//! With `df` a grid, the degrees of freedom are drawn each sweep, before
//! the weight refresh, from the exact discrete conditional with the
//! weights integrated out,
//!
//! ```text
//! P(df = g | r, sigma^2) proportional to prod_i t_g(r_i; 0, sigma),
//! ```
//!
//! the location-scale t density of the residuals, uniform over the grid
//! a priori; the weights then follow from their conditional under the
//! new df. The pair is one draw of (df, w) from
//! p(df | r, sigma^2) p(w | df, r, sigma^2), the joint conditional, so the
//! scan stays a Gibbs sampler with no Metropolis step. The conditional
//! given the weights alone, prod_i Gamma(w_i; g / 2, rate g / 2), is
//! also exact but cannot move: weights drawn under one grid value favour
//! that value again, and the chain sits where it starts. Under prior-only
//! sampling the residual carries no likelihood and df comes from its
//! uniform prior.
//!
//! # Priors
//!
//! The Gaussian model's for the cells and sigma^2: cell means
//! mu ~ N(0, sigma_mu^2) with sigma_mu = 0.5 / (k sqrt m) on the response
//! min-max scaled to [-0.5, 0.5]; sigma^2 ~ nu lambda / chi^2_nu with
//! lambda set so that Pr(sigma < sigma_hat) = q, sigma_hat the
//! least-squares residual standard deviation. Under t errors that
//! residual standard deviation estimates sigma sqrt(df / (df - 2)), not
//! sigma, so the calibration heuristic overstates the scale at small df;
//! the prior's spread absorbs the overstatement the way it absorbs the
//! least-squares heuristic itself. The weights carry the
//! Gamma(df / 2, rate df / 2) prior with mean 1; a grid-valued df is
//! uniform over its grid.
//!
//! # Fixed rather than estimated
//!
//! The degrees of freedom, fixed at 4 unless a grid is declared: df is
//! weakly identified, continuous random-walk samplers over it mix
//! poorly, and the grid conditional is exact, so no continuous-df
//! sampler exists. sigma^2 is sampled, so the scale mode is `Sampled`; a
//! variance ensemble is nonetheless rejected at validation, because
//! per-observation weights and a per-observation variance product both
//! model dispersion and their joint identification awaits its argument.
//!
//! # Correspondence
//!
//! With Geweke (1993, Journal of Applied Econometrics): the Gibbs scan
//! over the weights, the scale and the mean is his sampler with the
//! ensemble in place of the linear predictor and the exact grid
//! conditional in place of his continuous-df step. No maintained
//! BART-family package ships a Student-t error model. With the crate's
//! Gaussian model: as df grows the weights concentrate at 1 and the
//! posterior converges to the Gaussian model's; the reproduction is in
//! distribution, not draw for draw, because the weight draws consume
//! randomness.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of f(x) and `predict_draws` its
//! per-draw values; `predict_variance` is the error variance
//! sigma_d^2 df_d / (df_d - 2) per draw, and `NotApplicable` where the
//! configuration admits df <= 2, whose t has no variance;
//! `prediction_interval` is the central interval of the equal-weight
//! mixture over draws of f_d(x) + sigma_d t_{df_d}, by bisection on the
//! mixture CDF; `log_likelihood` is the location-scale t log density per
//! draw; `sigma` is sigma_d on the caller's scale, the scale of the t
//! rather than the error standard deviation; `in_sample_rmse` is against
//! the observed response. Under a grid the fit stores one df per kept
//! draw, read back through `Posterior::dfs`.
//!
//! # Input
//!
//! A continuous response, min-max scaled as the Gaussian model's.

use crate::maths;
use crate::models::scale_mixture;
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::{self, Rng};

/// The Student-t outcome behind the [`OutcomeModel`] contract: the
/// response is observed, the per-observation Gamma weights answer
/// through the precisions, and sigma^2 is sampled by the kernel from its
/// weighted inverse-gamma conditional.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StudentTOutcome {
    /// The current degrees of freedom: the fixed value, or the grid
    /// state.
    df: f64,
    /// The df grid, uniform a priori; empty for a fixed df.
    grid: Vec<f64>,
    weights: Vec<f64>,
}

impl StudentTOutcome {
    /// The half-width of the cell-mean prior: the Gaussian model's, the
    /// response being scaled to [-0.5, 0.5].
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 0.5;

    /// An outcome with the initial degrees of freedom and the grid
    /// (empty for a fixed df).
    pub(crate) fn new(df: f64, grid: Vec<f64>) -> Self {
        Self {
            df,
            grid,
            weights: Vec::new(),
        }
    }

    /// The current degrees of freedom.
    pub(crate) fn df(&self) -> f64 {
        self.df
    }

    /// Whether the degrees of freedom are drawn over a grid, so the fit
    /// stores one value per kept draw.
    pub(crate) fn grid_sampled(&self) -> bool {
        !self.grid.is_empty()
    }
}

impl OutcomeModel for StudentTOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Continuous
    }

    /// Start every weight at its prior mean 1, the Gaussian state. The
    /// weights are sampled state, not data: a response replacement
    /// leaves them standing, because the precisions were written with
    /// them and the next sweep's refresh conditions on the new response.
    fn init(&mut self, y: &[f64]) {
        if self.weights.is_empty() {
            self.weights = vec![1.0; y.len()];
        }
    }

    /// The df grid draw (one uniform, the residuals against the
    /// recovered 1 / sigma^2), then the weight refresh (one gamma per
    /// observation, fresh df).
    fn draw_extra(&mut self, y: &[f64], total: &[f64], precision: &[f64], rng: &mut Rng) {
        if !self.grid.is_empty() {
            // precision_i = w_i / sigma^2 on every row, so the first row
            // gives 1 / sigma^2; zero under prior-only sampling.
            let scale_precision = precision
                .first()
                .zip(self.weights.first())
                .map_or(0.0, |(&p, &w)| p / w);
            let n = y.len() as f64;
            let log_conditional: Vec<f64> = self
                .grid
                .iter()
                .map(|&g| {
                    if scale_precision <= 0.0 {
                        return 0.0;
                    }
                    let constant = maths::lgamma(0.5 * (g + 1.0))
                        - maths::lgamma(0.5 * g)
                        - 0.5 * maths::ln(g);
                    let kernel: f64 = y
                        .iter()
                        .zip(total)
                        .map(|(&value, &f)| {
                            let residual = value - f;
                            maths::ln(1.0 + scale_precision * residual * residual / g)
                        })
                        .sum();
                    n * constant - 0.5 * (g + 1.0) * kernel
                })
                .collect();
            self.df = self.grid[rng::draw_discrete(&log_conditional, rng)];
        }
        let df = self.df;
        scale_mixture::refresh_weights(
            &mut self.weights,
            y,
            total,
            precision,
            rng,
            |residual, scale_precision, rng| {
                let shape = if scale_precision > 0.0 {
                    0.5 * (df + 1.0)
                } else {
                    0.5 * df
                };
                let rate = 0.5 * (df + scale_precision * residual * residual);
                rng::gamma(shape, 1.0 / rate, rng)
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

    /// Quantile of the predictive f + sigma t_df with the current
    /// degrees of freedom; `sd` is the scale sigma, not the error
    /// standard deviation.
    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64> {
        Some(mean + sd * maths::student_t_quantile(p, self.df))
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::*;
    use crate::rng::chain_rng;

    #[test]
    fn the_student_t_outcome_answers_the_contract() {
        let mut outcome = StudentTOutcome::new(4.0, Vec::new());
        let y = [0.1, -0.2, 0.3];
        outcome.init(&y);
        assert_eq!(outcome.required_data(), RequiredData::Continuous);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(outcome.weights(), Some(&[1.0, 1.0, 1.0][..]));
        assert!(!outcome.grid_sampled());
        assert_eq!(outcome.df(), 4.0);

        let mut rng = chain_rng(21);
        let mut latent = y;
        outcome.working_response(&[0.0; 3], &[1.0; 3], &mut latent, &mut rng);
        assert_eq!(latent, y);

        let median = outcome.predictive_quantile(0.3, 1.0, 0.5).unwrap();
        assert!((median - 0.3).abs() < 1e-9);
        let q = outcome.predictive_quantile(0.0, 2.0, 0.975).unwrap();
        assert!((q - 2.0 * maths::student_t_quantile(0.975, 4.0)).abs() < 1e-12);
    }

    /// A second `init`, the response-replacement path, keeps the
    /// standing weights the precisions were written with.
    #[test]
    fn a_response_replacement_keeps_the_weights() {
        let mut outcome = StudentTOutcome::new(4.0, Vec::new());
        outcome.init(&[0.1, -0.2]);
        let mut rng = chain_rng(2);
        outcome.draw_extra(&[0.1, -0.2], &[0.0, 0.0], &[4.0, 4.0], &mut rng);
        let drawn = outcome.weights().unwrap().to_vec();
        assert_ne!(drawn, vec![1.0, 1.0]);
        outcome.init(&[0.3, 0.4]);
        assert_eq!(outcome.weights(), Some(&drawn[..]));
    }

    /// E[w | r] = (df + 1) / (df + r^2 / sigma^2): a wild residual is
    /// discounted, a zero residual upweighted.
    #[test]
    fn the_weight_conditional_has_the_geweke_mean() {
        let df = 4.0;
        let sigma_sq = 0.25;
        let y = [0.0, 2.0];
        let total = [0.0, 0.0];
        let mut outcome = StudentTOutcome::new(df, Vec::new());
        outcome.init(&y);
        // precision_i = w_i / sigma^2 with unit weights.
        let precision = [1.0 / sigma_sq; 2];
        let mut rng = chain_rng(5);
        let n = 40_000;
        let mut means = [0.0; 2];
        for _ in 0..n {
            // Unit standing weights keep the recovered scale at
            // sigma^2 across repeats.
            outcome.weights.copy_from_slice(&[1.0, 1.0]);
            outcome.draw_extra(&y, &total, &precision, &mut rng);
            for (mean, w) in means.iter_mut().zip(outcome.weights().unwrap()) {
                *mean += w;
            }
        }
        for (mean, &value) in means.iter_mut().zip(&y) {
            *mean /= n as f64;
            let expected = (df + 1.0) / (df + value * value / sigma_sq);
            assert!((*mean - expected).abs() < 0.02, "{mean} vs {expected}");
        }
    }

    /// Zero precisions reduce the draw to the Gamma(df / 2, rate df / 2)
    /// prior, mean 1 and variance 2 / df.
    #[test]
    fn prior_only_draws_come_from_the_weight_prior() {
        let df = 6.0;
        let mut outcome = StudentTOutcome::new(df, Vec::new());
        outcome.init(&[0.4]);
        let mut rng = chain_rng(9);
        let n = 40_000;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..n {
            outcome.draw_extra(&[0.4], &[0.0], &[0.0], &mut rng);
            let w = outcome.weights().unwrap()[0];
            sum += w;
            sum_sq += w * w;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        assert!((mean - 1.0).abs() < 0.02, "{mean}");
        assert!((variance - 2.0 / df).abs() < 0.02, "{variance}");
    }

    /// Residuals drawn as sigma t_3 pick 3 from the grid whatever the
    /// standing weights, here drawn under the grid's top value, the
    /// state the conditional given the weights alone could not leave.
    #[test]
    fn the_grid_conditional_concentrates_on_the_generating_df() {
        let generating = 3.0;
        let standing = 24.0;
        let grid = vec![3.0, 6.0, 12.0, 24.0];
        let n = 400;
        let sigma_sq: f64 = 0.25;
        let mut rng = chain_rng(31);
        let residuals: Vec<f64> = (0..n)
            .map(|_| {
                let w = rng::gamma(0.5 * generating, 2.0 / generating, &mut rng);
                sigma_sq.sqrt() * rng::standard_normal(&mut rng) / w.sqrt()
            })
            .collect();
        let weights: Vec<f64> = (0..n)
            .map(|_| rng::gamma(0.5 * standing, 2.0 / standing, &mut rng))
            .collect();
        let precision: Vec<f64> = weights.iter().map(|w| w / sigma_sq).collect();
        let zeros = vec![0.0; n];
        let mut outcome = StudentTOutcome::new(standing, grid);
        outcome.init(&zeros);
        let mut hits = 0;
        let draws = 200;
        for _ in 0..draws {
            outcome.weights.copy_from_slice(&weights);
            outcome.draw_extra(&residuals, &zeros, &precision, &mut rng);
            if outcome.df() == generating {
                hits += 1;
            }
        }
        assert!(hits > draws * 9 / 10, "{hits} of {draws}");
    }

    /// Zero precisions draw df from its uniform prior over the grid.
    #[test]
    fn prior_only_draws_df_from_the_grid_prior() {
        let grid = vec![3.0, 6.0, 12.0, 24.0];
        let mut outcome = StudentTOutcome::new(3.0, grid.clone());
        outcome.init(&[0.4, -0.2]);
        let mut rng = chain_rng(17);
        let draws = 8_000;
        let mut counts = vec![0; grid.len()];
        for _ in 0..draws {
            outcome.draw_extra(&[0.4, -0.2], &[0.0, 0.0], &[0.0, 0.0], &mut rng);
            counts[grid.iter().position(|&g| g == outcome.df()).unwrap()] += 1;
        }
        for count in counts {
            let share = count as f64 / draws as f64;
            assert!((share - 0.25).abs() < 0.03, "{share}");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Config, Outcome};
    use crate::data::Data;
    use crate::maths;
    use crate::sampler::Sampler;
    use crate::tessellation::Tessellation;

    /// Posterior means of sigma^2, df and every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2 with an
    /// inner quadrature over each cell mean and a sum over the df grid
    /// (a single-entry `dfs` is the fixed-df model). Given sigma^2 and
    /// df the weights marginalise analytically, so cell k contributes
    ///
    /// ```text
    /// m_k(sigma^2, df) = int N(mu; 0, sigma_mu^2) prod_obs t_df(y_i; mu, sigma) dmu,
    /// ```
    ///
    /// t_df(y; mu, sigma) the location-scale t density: the model's own
    /// marginal likelihood integrated numerically, independent of the
    /// engine and of the augmentation.
    fn quadrature_reference(
        cells: &[Vec<f64>],
        dfs: &[f64],
        nu: f64,
        lambda: f64,
        sigma_mu_sq: f64,
    ) -> (f64, f64, Vec<f64>) {
        let (a, scale) = (0.5 * nu, 0.5 * nu * lambda);
        let outer = 400;
        let (t_lo, t_hi) = (lambda.ln() - 8.0, lambda.ln() + 8.0);
        let inner = 3000;
        let (mu_lo, mu_hi) = (-1.5_f64, 1.5_f64);
        let mut log_weights = Vec::new();
        let mut sigmas = Vec::new();
        let mut df_values = Vec::new();
        let mut cond_means: Vec<Vec<f64>> = vec![Vec::new(); cells.len()];
        for &df in dfs {
            let ln_c = maths::lgamma(0.5 * (df + 1.0))
                - maths::lgamma(0.5 * df)
                - 0.5 * (df * std::f64::consts::PI).ln();
            for i in 0..=outer {
                let t = t_lo + (t_hi - t_lo) * i as f64 / outer as f64;
                let sigma_sq = t.exp();
                let mut lp = -(a + 1.0) * t - scale / sigma_sq + t;
                for (k, cell) in cells.iter().enumerate() {
                    let mut best = f64::NEG_INFINITY;
                    let mut terms = Vec::with_capacity(inner + 1);
                    for j in 0..=inner {
                        let mu = mu_lo + (mu_hi - mu_lo) * j as f64 / inner as f64;
                        let mut term = -0.5 * mu * mu / sigma_mu_sq;
                        for &y in cell {
                            let z_sq = (y - mu) * (y - mu) / (df * sigma_sq);
                            term += ln_c - 0.5 * t - 0.5 * (df + 1.0) * (1.0 + z_sq).ln();
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
                df_values.push(df);
                log_weights.push(lp);
            }
        }
        let max = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|lp| (lp - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mean_sigma_sq = weights.iter().zip(&sigmas).map(|(w, s)| w * s).sum::<f64>() / total;
        let mean_df = weights
            .iter()
            .zip(&df_values)
            .map(|(w, d)| w * d)
            .sum::<f64>()
            / total;
        let mean_mus = cond_means
            .iter()
            .map(|means| weights.iter().zip(means).map(|(w, m)| w * m).sum::<f64>() / total)
            .collect();
        (mean_sigma_sq, mean_df, mean_mus)
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

    /// A response in [-0.5, 0.5] with two wild values, so the weights
    /// have work to do.
    fn training_data() -> (Data, Vec<f64>) {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let mut y: Vec<f64> = (0..n)
            .map(|i| (((i * 5) % 11) as f64 / 11.0 - 0.5) * 0.4)
            .collect();
        y[3] = 0.48;
        y[8] = -0.45;
        (Data::new(xs, n, 1).unwrap(), y)
    }

    fn run_fixed_tessellation(
        config: &Config,
        x: &Data,
        y: &[f64],
        lambda: f64,
        centres: Vec<f64>,
        seed: u64,
    ) -> (Sampler, Vec<Vec<f64>>, f64) {
        let b = centres.len();
        let mut sampler = Sampler::pinned_prior(config, x, y, lambda, seed).unwrap();
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
        for (&cell, &v) in assignments.iter().zip(y) {
            cells[cell].push(v);
        }
        let sigma_mu_sq = sampler.mean_sigma_mu_sq();
        (sampler, cells, sigma_mu_sq)
    }

    /// On a fixed tessellation the chain is the Geweke (1993) Gibbs
    /// sampler; its means of sigma^2 and of every mu_k match the
    /// numerical integration of the marginal t likelihood within 4
    /// batch-means MCSE.
    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let (x, y) = training_data();
        let lambda = 0.04;
        let df = 4.0;
        let config = Config::new().with_outcome(Outcome::student_t(df)).with_m(1);
        for (centres, seed) in [(vec![-0.35, 0.0, 0.3], 51_u64), (vec![0.0], 52)] {
            let (mut sampler, cells, sigma_mu_sq) =
                run_fixed_tessellation(&config, &x, &y, lambda, centres, seed);
            let (ref_sigma_sq, _, ref_mus) = quadrature_reference(
                &cells,
                &[df],
                sampler.config().sigma2_prior().0,
                lambda,
                sigma_mu_sq,
            );
            let b = cells.len();

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

    /// The grid form: the same quadrature with an outer sum over the
    /// grid checks the exact discrete df conditional end to end, E[df]
    /// included.
    #[test]
    fn fixed_tessellation_grid_posterior_matches_quadrature() {
        let (x, y) = training_data();
        let lambda = 0.04;
        let grid = vec![3.0, 6.0, 12.0, 24.0];
        let config = Config::new()
            .with_outcome(Outcome::student_t_grid(grid.clone()))
            .with_m(1);
        let (mut sampler, cells, sigma_mu_sq) =
            run_fixed_tessellation(&config, &x, &y, lambda, vec![-0.35, 0.0, 0.3], 53);
        let (ref_sigma_sq, ref_df, ref_mus) = quadrature_reference(
            &cells,
            &grid,
            sampler.config().sigma2_prior().0,
            lambda,
            sigma_mu_sq,
        );
        let b = cells.len();

        for _ in 0..500 {
            sampler.conjugate_sweep();
        }
        let kept = 40_000;
        let mut sigma_sq = Vec::with_capacity(kept);
        let mut dfs = Vec::with_capacity(kept);
        let mut mus: Vec<Vec<f64>> = vec![Vec::with_capacity(kept); b];
        for _ in 0..kept {
            sampler.conjugate_sweep();
            sigma_sq.push(sampler.noise_variances()[0]);
            dfs.push(sampler.student_df().unwrap());
            for (k, series) in mus.iter_mut().enumerate() {
                series.push(sampler.tessellations()[0].mus[k]);
            }
        }
        let (mean, mcse) = batch_means_mcse(&sigma_sq);
        assert!(
            (mean - ref_sigma_sq).abs() < 4.0 * mcse,
            "sigma^2 {mean} vs {ref_sigma_sq} +- {mcse}"
        );
        let (mean, mcse) = batch_means_mcse(&dfs);
        assert!(
            (mean - ref_df).abs() < 4.0 * mcse,
            "df {mean} vs {ref_df} +- {mcse}"
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
