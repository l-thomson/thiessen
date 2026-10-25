//! Conjugate cell families: the sufficient statistics of a tessellation's
//! cells, the integrated likelihood of its cell structure, and the
//! posterior draw of its cell values. An [`Ensemble`](crate::ensemble)
//! is generic over the family.
//!
//! # Gaussian cell means
//!
//! Observation i carries precision w_i = 1 / sigma_i^2 (the global 1 / sigma^2
//! for the Gaussian model), so the same code serves a per-observation
//! variance. With partial residuals r_i and prior mu_k ~ N(0, sigma_mu^2),
//! cell k with W_k = sum w_i and S_k = sum w_i r_i over its observations has
//!
//! ```text
//! ln p(r | T) = sum_k [ -ln(1 + W_k sigma_mu^2) / 2 + sigma_mu^2 S_k^2 / (2 (1 + W_k sigma_mu^2)) ]
//!               + terms that do not depend on T,
//! mu_k | r ~ N( sigma_mu^2 S_k / (1 + W_k sigma_mu^2), sigma_mu^2 / (1 + W_k sigma_mu^2) ).
//! ```
//!
//! Chipman, George and McCulloch (2010), s. 3.1, with the cell in place of
//! the leaf.
//!
//! # Inverse-gamma cell variances
//!
//! Observation i carries a residual e_i and the product s_i of the other
//! variance tessellations at x_i, so e_i ~ N(0, v_k s_i) in cell k. With
//! prior v_k ~ Inv-Gamma(nu' / 2, nu' lambda' / 2), cell k with n_k
//! observations and E_k = sum e_i^2 / s_i has
//!
//! ```text
//! ln p(e | T) = sum_k [ lgamma((nu' + n_k) / 2) - lgamma(nu' / 2)
//!                      + (nu' / 2) ln(nu' lambda' / 2)
//!                      - ((nu' + n_k) / 2) ln((nu' lambda' + E_k) / 2) ]
//!               + terms that do not depend on T,
//! v_k | e ~ Inv-Gamma((nu' + n_k) / 2, (nu' lambda' + E_k) / 2).
//! ```
//!
//! The per-cell normaliser (nu' / 2) ln(nu' lambda' / 2) - lgamma(nu' / 2)
//! enters once per cell, so it appears in the acceptance ratio of every
//! move that changes the cell count. Pratola, Chipman, George and
//! McCulloch (2020), s. 3.2, with the cell in place of the leaf.

use crate::data::Data;
use crate::maths;
use crate::rng::{self, standard_normal, Rng};
use crate::tessellation::{CellIndex, Tessellation};

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::GaussianCells {}
    impl Sealed for super::InverseGammaCells {}
}

/// Per-cell sufficient statistics, accumulated into a buffer that keeps
/// its capacity across proposals.
pub(crate) trait Stats: Clone + std::fmt::Debug + Default {
    /// Whether every cell holds at least one observation. Counted, not
    /// weighed: under prior-only sampling every weight is zero and the
    /// occupancy rule must not change.
    fn all_occupied(&self) -> bool;
}

/// What an observation's contribution reads beyond its cell, input,
/// weight and partial: the design and the tessellation for the offsets
/// of the linear basis, and under soft membership the row-major n x b
/// membership weights.
#[derive(Clone, Copy)]
pub(crate) struct Context<'a> {
    pub x: &'a Data,
    pub tessellation: &'a Tessellation,
    pub soft: Option<&'a [f64]>,
    /// When every observation carries the same weight, entry c is c
    /// copies of it added in turn from zero, for c up to n: under hard
    /// membership a cell's total weight is then the entry at its count,
    /// the same bits as adding its observations' weights in order. `None`
    /// when the weights differ.
    pub uniform: Option<&'a [f64]>,
}

/// A conjugate family of cell values with its combination rule across an
/// ensemble. Sealed: the two implementations are the crate's.
///
/// An observation's partial is the one number, beyond the caller's input
/// and weight, that its contribution to tessellation j's statistics and
/// the ensemble total after j's update depend on: the partial residual
/// under Gaussian cells, the product of the other tessellations under
/// inverse-gamma cells.
pub(crate) trait CellFamily: sealed::Sealed {
    type Stats: Stats;

    /// The observation's partial from the caller's input, the ensemble
    /// total and the tessellation's own current value at the observation.
    fn partial(&self, input: f64, total: f64, own: f64) -> f64;

    /// The ensemble total after tessellation j's value at the observation
    /// becomes `own`.
    fn total(&self, input: f64, partial: f64, own: f64) -> f64;

    /// Start `out` as the statistics of `b` empty cells, replacing
    /// whatever it held.
    fn begin(&self, b: usize, context: &Context, out: &mut Self::Stats);

    /// Add observation `i`, in cell `cell`, to `out`. Observations enter
    /// in ascending order; under soft membership occupancy still counts
    /// the hard assignment.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        out: &mut Self::Stats,
        i: usize,
        cell: usize,
        input: f64,
        weight: f64,
        partial: f64,
        context: &Context,
    );

    /// Statistics of the tessellation's `b` cells from the assignment,
    /// the caller's inputs and weights and the partials: [`begin`] then
    /// [`add`] over every observation.
    ///
    /// [`begin`]: Self::begin
    /// [`add`]: Self::add
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        cells: &[CellIndex],
        input: &[f64],
        weights: &[f64],
        partials: &[f64],
        b: usize,
        context: &Context,
        out: &mut Self::Stats,
    ) {
        accumulate_by_add(self, cells, input, weights, partials, b, context, out);
    }

    /// Under hard membership and the constant basis, each observation's
    /// partial against its cell value `mus[cells[i] as usize]` into `partials`,
    /// and its [`add`] to `out`, which [`begin`] has started.
    ///
    /// [`add`]: Self::add
    /// [`begin`]: Self::begin
    #[allow(clippy::too_many_arguments)]
    fn hard_partials(
        &self,
        cells: &[CellIndex],
        mus: &[f64],
        input: &[f64],
        weights: &[f64],
        total: &[f64],
        partials: &mut [f64],
        context: &Context,
        out: &mut Self::Stats,
    ) {
        hard_partials_by_add(
            self, cells, mus, input, weights, total, partials, context, out,
        );
    }

    /// Under hard membership and the constant basis for this tessellation
    /// and the next, per observation: the total after this tessellation's
    /// update from `partials[i]` and `mus[cells[i] as usize]` into `total[i]`, then
    /// [`hard_partials`] of the next tessellation against that total. The
    /// same operations in the same order as the update followed by the
    /// next tessellation's own pass.
    ///
    /// [`hard_partials`]: Self::hard_partials
    #[allow(clippy::too_many_arguments)]
    fn hard_totals_then_partials(
        &self,
        cells: &[CellIndex],
        mus: &[f64],
        next_cells: &[CellIndex],
        next_mus: &[f64],
        input: &[f64],
        weights: &[f64],
        total: &mut [f64],
        partials: &mut [f64],
        context: &Context,
        out: &mut Self::Stats,
    ) {
        hard_totals_then_partials_by_add(
            self, cells, mus, next_cells, next_mus, input, weights, total, partials, context, out,
        );
    }

    /// The T-dependent part of the integrated log-likelihood.
    fn log_marginal(
        &self,
        stats: &Self::Stats,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) -> f64;

    /// Posterior draw of every cell, ascending cell index, replacing
    /// `values`, and the row-major slopes of the linear basis replacing
    /// `slopes` (emptied otherwise).
    fn draw(
        &self,
        stats: &Self::Stats,
        rng: &mut Rng,
        values: &mut Vec<f64>,
        slopes: &mut Vec<f64>,
    );

    /// [`draw`](Self::draw) into fresh vectors.
    #[cfg(test)]
    fn draw_owned(&self, stats: &Self::Stats, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let (mut values, mut slopes) = (Vec::new(), Vec::new());
        self.draw(stats, rng, &mut values, &mut slopes);
        (values, slopes)
    }

    /// The per-cell constant of the integrated likelihood, the term a
    /// fixture drops to mis-price the cell-count moves; zero for a family
    /// without one.
    #[cfg(test)]
    fn cell_normaliser(&self) -> f64;
}

/// [`CellFamily::accumulate`] through [`CellFamily::add`].
#[allow(clippy::too_many_arguments)]
fn accumulate_by_add<F: CellFamily + ?Sized>(
    family: &F,
    cells: &[CellIndex],
    input: &[f64],
    weights: &[f64],
    partials: &[f64],
    b: usize,
    context: &Context,
    out: &mut F::Stats,
) {
    family.begin(b, context, out);
    let n = cells.len();
    let (input, weights, partials) = (&input[..n], &weights[..n], &partials[..n]);
    for i in 0..n {
        family.add(
            out,
            i,
            cells[i] as usize,
            input[i],
            weights[i],
            partials[i],
            context,
        );
    }
}

/// [`CellFamily::hard_partials`] through [`CellFamily::add`].
#[allow(clippy::too_many_arguments)]
fn hard_partials_by_add<F: CellFamily + ?Sized>(
    family: &F,
    cells: &[CellIndex],
    mus: &[f64],
    input: &[f64],
    weights: &[f64],
    total: &[f64],
    partials: &mut [f64],
    context: &Context,
    out: &mut F::Stats,
) {
    let n = cells.len();
    let (input, weights, total) = (&input[..n], &weights[..n], &total[..n]);
    let partials = &mut partials[..n];
    for i in 0..n {
        let partial = family.partial(input[i], total[i], mus[cells[i] as usize]);
        partials[i] = partial;
        family.add(
            out,
            i,
            cells[i] as usize,
            input[i],
            weights[i],
            partial,
            context,
        );
    }
}

/// [`CellFamily::hard_totals_then_partials`] through [`CellFamily::add`].
#[allow(clippy::too_many_arguments)]
fn hard_totals_then_partials_by_add<F: CellFamily + ?Sized>(
    family: &F,
    cells: &[CellIndex],
    mus: &[f64],
    next_cells: &[CellIndex],
    next_mus: &[f64],
    input: &[f64],
    weights: &[f64],
    total: &mut [f64],
    partials: &mut [f64],
    context: &Context,
    out: &mut F::Stats,
) {
    let n = cells.len();
    let (next_cells, input, weights) = (&next_cells[..n], &input[..n], &weights[..n]);
    let (total, partials) = (&mut total[..n], &mut partials[..n]);
    for i in 0..n {
        total[i] = family.total(input[i], partials[i], mus[cells[i] as usize]);
        let partial = family.partial(input[i], total[i], next_mus[next_cells[i] as usize]);
        partials[i] = partial;
        family.add(
            out,
            i,
            next_cells[i] as usize,
            input[i],
            weights[i],
            partial,
            context,
        );
    }
}

/// Gaussian cell means under an additive ensemble; with `linear` the
/// cells carry (mu, beta) jointly, u = [1, x_A - c] the within-cell
/// design and N(0, sigma_mu^2 I) the prior on the whole coefficient
/// vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GaussianCells {
    pub sigma_mu_sq: f64,
    /// The linear cell basis ([`Basis::Linear`](crate::Basis)); reachable
    /// only through the `experimental` configuration surface.
    pub linear: bool,
}

/// Per-cell (n_k, W_k, S_k) accumulated in ascending observation order;
/// under the linear basis also the normal-equation blocks, per cell the
/// row-major q x q matrix A_k = sum w u u' and the q-vector
/// b_k = sum w r u with q = d + 1. Under soft membership one block for
/// the whole tessellation, q = b, A = W' D W and b = W' D r with W the
/// n x b membership weights and D the observation precisions; `count`
/// still holds the hard assignment for the occupancy rule.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GaussianStats {
    pub cells: Vec<GaussianCell>,
    soft: bool,
    q: usize,
    a: Vec<f64>,
    bvec: Vec<f64>,
    /// The within-cell design row of the observation being added.
    u: Vec<f64>,
}

/// One cell's (n_k, W_k, S_k).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct GaussianCell {
    pub count: usize,
    pub weight: f64,
    pub sum: f64,
}

impl GaussianCell {
    /// One observation of weight `weight` and partial `partial`; the
    /// weight itself is added only with `weigh`, the total weight otherwise
    /// left to [`weigh_by_count`].
    #[inline]
    fn add(&mut self, weight: f64, partial: f64, weigh: bool) {
        self.count += 1;
        if weigh {
            self.weight += weight;
        }
        self.sum += weight * partial;
    }
}

/// Each cell's total weight from its count and the running sums of the
/// weight every observation shares; see [`Context::uniform`].
fn weigh_by_count(cells: &mut [GaussianCell], running: &[f64]) {
    for c in cells {
        c.weight = running[c.count];
    }
}

impl Stats for GaussianStats {
    fn all_occupied(&self) -> bool {
        self.cells.iter().all(|c| c.count > 0)
    }
}

impl CellFamily for GaussianCells {
    type Stats = GaussianStats;

    fn partial(&self, input: f64, total: f64, own: f64) -> f64 {
        input - total + own
    }

    fn total(&self, input: f64, partial: f64, own: f64) -> f64 {
        input - partial + own
    }

    fn begin(&self, b: usize, context: &Context, out: &mut GaussianStats) {
        out.cells.clear();
        out.cells.resize(b, GaussianCell::default());
        out.a.clear();
        out.bvec.clear();
        out.u.clear();
        if context.soft.is_some() {
            out.soft = true;
            out.q = b;
            out.a.resize(b * b, 0.0);
            out.bvec.resize(b, 0.0);
            return;
        }
        let q = if self.linear {
            context.tessellation.n_dims() + 1
        } else {
            0
        };
        out.soft = false;
        out.q = q;
        out.a.resize(b * q * q, 0.0);
        out.bvec.resize(b * q, 0.0);
        out.u.resize(q, 0.0);
    }

    #[inline]
    fn add(
        &self,
        out: &mut GaussianStats,
        i: usize,
        cell: usize,
        _input: f64,
        weight: f64,
        partial: f64,
        context: &Context,
    ) {
        let c = &mut out.cells[cell];
        c.count += 1;
        if let Some(w) = context.soft {
            add_soft(out, i, weight, partial, w);
            return;
        }
        c.weight += weight;
        c.sum += weight * partial;
        if self.linear {
            add_slopes(out, i, cell, weight, partial, context);
        }
    }

    /// Under the constant basis, without the per-observation membership
    /// and basis tests of [`add`](CellFamily::add).
    fn hard_partials(
        &self,
        cells: &[CellIndex],
        mus: &[f64],
        input: &[f64],
        weights: &[f64],
        total: &[f64],
        partials: &mut [f64],
        context: &Context,
        out: &mut GaussianStats,
    ) {
        if self.linear {
            return hard_partials_by_add(
                self, cells, mus, input, weights, total, partials, context, out,
            );
        }
        let n = cells.len();
        let (input, weights, total) = (&input[..n], &weights[..n], &total[..n]);
        let partials = &mut partials[..n];
        let stats = &mut out.cells[..mus.len()];
        let weigh = context.uniform.is_none();
        for i in 0..n {
            let partial = self.partial(input[i], total[i], mus[cells[i] as usize]);
            partials[i] = partial;
            stats[cells[i] as usize].add(weights[i], partial, weigh);
        }
        if let Some(running) = context.uniform {
            weigh_by_count(stats, running);
        }
    }

    /// Under the constant basis, without the per-observation membership
    /// and basis tests of [`add`](CellFamily::add).
    fn hard_totals_then_partials(
        &self,
        cells: &[CellIndex],
        mus: &[f64],
        next_cells: &[CellIndex],
        next_mus: &[f64],
        input: &[f64],
        weights: &[f64],
        total: &mut [f64],
        partials: &mut [f64],
        context: &Context,
        out: &mut GaussianStats,
    ) {
        if self.linear {
            return hard_totals_then_partials_by_add(
                self, cells, mus, next_cells, next_mus, input, weights, total, partials, context,
                out,
            );
        }
        let n = cells.len();
        let (next_cells, input, weights) = (&next_cells[..n], &input[..n], &weights[..n]);
        let (total, partials) = (&mut total[..n], &mut partials[..n]);
        let stats = &mut out.cells[..next_mus.len()];
        let weigh = context.uniform.is_none();
        for i in 0..n {
            total[i] = self.total(input[i], partials[i], mus[cells[i] as usize]);
            let partial = self.partial(input[i], total[i], next_mus[next_cells[i] as usize]);
            partials[i] = partial;
            stats[next_cells[i] as usize].add(weights[i], partial, weigh);
        }
        if let Some(running) = context.uniform {
            weigh_by_count(stats, running);
        }
    }

    /// Under hard membership and the constant basis, without the
    /// per-observation membership and basis tests of [`add`].
    ///
    /// [`add`]: CellFamily::add
    fn accumulate(
        &self,
        cells: &[CellIndex],
        input: &[f64],
        weights: &[f64],
        partials: &[f64],
        b: usize,
        context: &Context,
        out: &mut GaussianStats,
    ) {
        if context.soft.is_some() || self.linear {
            return accumulate_by_add(self, cells, input, weights, partials, b, context, out);
        }
        self.begin(b, context, out);
        let n = cells.len();
        let (weights, partials) = (&weights[..n], &partials[..n]);
        let stats = &mut out.cells[..b];
        let weigh = context.uniform.is_none();
        for i in 0..n {
            stats[cells[i] as usize].add(weights[i], partials[i], weigh);
        }
        if let Some(running) = context.uniform {
            weigh_by_count(stats, running);
        }
    }

    fn log_marginal(
        &self,
        stats: &GaussianStats,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) -> f64 {
        if stats.soft {
            // -ln det(I + sigma_mu^2 A) / 2 + b' (Sigma0^-1 + A)^-1 b / 2
            // with one b x b block for the whole tessellation.
            let q = stats.q;
            let mut p_mat = stats.a.clone();
            for r in 0..q {
                p_mat[r * q + r] += 1.0 / self.sigma_mu_sq;
            }
            let log_det_p = cholesky_in_place(&mut p_mat, q);
            let mut m = stats.bvec.clone();
            cholesky_solve(&p_mat, q, &mut m);
            let fit: f64 = stats.bvec.iter().zip(&m).map(|(&b, &m)| b * m).sum();
            let det_term = q as f64 * maths::ln(self.sigma_mu_sq) + log_det_p;
            #[cfg(test)]
            let det_term = if breakage == crate::broken::Breakage::DroppedSoftDeterminant {
                0.0
            } else {
                det_term
            };
            return -0.5 * det_term + 0.5 * fit;
        }
        if !self.linear {
            let mut total = 0.0;
            for c in &stats.cells {
                let den = 1.0 + c.weight * self.sigma_mu_sq;
                total += -0.5 * maths::ln(den) + self.sigma_mu_sq * c.sum * c.sum / (2.0 * den);
            }
            return total;
        }
        // Per cell -ln det(I + sigma_mu^2 A) / 2 + b' (Sigma0^-1 + A)^-1 b / 2,
        // the (d + 1)-dimensional form of the constant-basis expression.
        let q = stats.q;
        let mut total = 0.0;
        for cell in 0..stats.cells.len() {
            let mut p_mat = stats.a[cell * q * q..(cell + 1) * q * q].to_vec();
            for r in 0..q {
                p_mat[r * q + r] += 1.0 / self.sigma_mu_sq;
            }
            let log_det_p = cholesky_in_place(&mut p_mat, q);
            let mut m = stats.bvec[cell * q..(cell + 1) * q].to_vec();
            cholesky_solve(&p_mat, q, &mut m);
            let fit: f64 = stats.bvec[cell * q..(cell + 1) * q]
                .iter()
                .zip(&m)
                .map(|(&b, &m)| b * m)
                .sum();
            let det_term = q as f64 * maths::ln(self.sigma_mu_sq) + log_det_p;
            #[cfg(test)]
            let det_term = if breakage == crate::broken::Breakage::DroppedTiltDeterminant {
                0.0
            } else {
                det_term
            };
            total += -0.5 * det_term + 0.5 * fit;
        }
        total
    }

    /// One standard normal per cell; under the linear basis q = d + 1
    /// standard normals per cell, coefficient order. Under soft
    /// membership one joint b-dimensional draw, b standard normals in
    /// cell order.
    fn draw(
        &self,
        stats: &GaussianStats,
        rng: &mut Rng,
        values: &mut Vec<f64>,
        slopes: &mut Vec<f64>,
    ) {
        values.clear();
        slopes.clear();
        if stats.soft {
            let q = stats.q;
            let mut p_mat = stats.a.clone();
            for r in 0..q {
                p_mat[r * q + r] += 1.0 / self.sigma_mu_sq;
            }
            cholesky_in_place(&mut p_mat, q);
            let mut theta = stats.bvec.clone();
            cholesky_solve(&p_mat, q, &mut theta);
            // theta = m + L^-T z: the upper-triangular solve of L' y = z.
            let mut z: Vec<f64> = (0..q).map(|_| standard_normal(rng)).collect();
            for r in (0..q).rev() {
                for c in r + 1..q {
                    z[r] -= p_mat[c * q + r] * z[c];
                }
                z[r] /= p_mat[r * q + r];
            }
            values.extend(theta.iter().zip(&z).map(|(&m, &e)| m + e));
            return;
        }
        if !self.linear {
            values.extend(stats.cells.iter().map(|c| {
                let den = 1.0 + c.weight * self.sigma_mu_sq;
                let mean = self.sigma_mu_sq * c.sum / den;
                let var = self.sigma_mu_sq / den;
                mean + var.sqrt() * standard_normal(rng)
            }));
            return;
        }
        let q = stats.q;
        let b = stats.cells.len();
        for cell in 0..b {
            let mut p_mat = stats.a[cell * q * q..(cell + 1) * q * q].to_vec();
            for r in 0..q {
                p_mat[r * q + r] += 1.0 / self.sigma_mu_sq;
            }
            cholesky_in_place(&mut p_mat, q);
            let mut theta = stats.bvec[cell * q..(cell + 1) * q].to_vec();
            cholesky_solve(&p_mat, q, &mut theta);
            // theta = m + L^-T z: the upper-triangular solve of L' y = z.
            let mut z: Vec<f64> = (0..q).map(|_| standard_normal(rng)).collect();
            for r in (0..q).rev() {
                for c in r + 1..q {
                    z[r] -= p_mat[c * q + r] * z[c];
                }
                z[r] /= p_mat[r * q + r];
            }
            values.push(theta[0] + z[0]);
            for r in 1..q {
                slopes.push(theta[r] + z[r]);
            }
        }
    }

    #[cfg(test)]
    fn cell_normaliser(&self) -> f64 {
        0.0
    }
}

/// The lower Cholesky factor of the row-major q x q symmetric positive
/// definite matrix, in place (the upper triangle is left stale); returns
/// ln det of the matrix.
fn cholesky_in_place(matrix: &mut [f64], q: usize) -> f64 {
    let mut log_det = 0.0;
    for j in 0..q {
        let mut d = matrix[j * q + j];
        for k in 0..j {
            d -= matrix[j * q + k] * matrix[j * q + k];
        }
        let root = d.sqrt();
        log_det += 2.0 * maths::ln(root);
        matrix[j * q + j] = root;
        for i in j + 1..q {
            let mut v = matrix[i * q + j];
            for k in 0..j {
                v -= matrix[i * q + k] * matrix[j * q + k];
            }
            matrix[i * q + j] = v / root;
        }
    }
    log_det
}

/// Solve (L L') x = rhs in place given the lower factor `l`.
fn cholesky_solve(l: &[f64], q: usize, rhs: &mut [f64]) {
    for r in 0..q {
        for c in 0..r {
            rhs[r] -= l[r * q + c] * rhs[c];
        }
        rhs[r] /= l[r * q + r];
    }
    for r in (0..q).rev() {
        for c in r + 1..q {
            rhs[r] -= l[c * q + r] * rhs[c];
        }
        rhs[r] /= l[r * q + r];
    }
}

/// Observation `i`'s soft-membership contribution: its row of the
/// membership weights `w` into the tessellation's one block.
fn add_soft(out: &mut GaussianStats, i: usize, weight: f64, partial: f64, w: &[f64]) {
    let b = out.q;
    let row = &w[i * b..(i + 1) * b];
    for r in 0..b {
        out.bvec[r] += weight * partial * row[r];
        for c in 0..b {
            out.a[r * b + c] += weight * row[r] * row[c];
        }
    }
}

/// Observation `i`'s contribution to the normal-equation blocks of cell
/// `cell` under the linear basis.
fn add_slopes(
    out: &mut GaussianStats,
    i: usize,
    cell: usize,
    weight: f64,
    partial: f64,
    context: &Context,
) {
    let q = out.q;
    out.u[0] = 1.0;
    let row = context.x.row(i);
    let centre = context.tessellation.centre(cell);
    for (j, &dim) in context.tessellation.dims().iter().enumerate() {
        out.u[j + 1] = row[dim] - centre[j];
    }
    for r in 0..q {
        out.bvec[cell * q + r] += weight * partial * out.u[r];
        for c in 0..q {
            out.a[cell * q * q + r * q + c] += weight * out.u[r] * out.u[c];
        }
    }
}

/// Inverse-gamma cell variances under a multiplicative ensemble. Under
/// `prior_only` the likelihood is removed: every cell counts as empty in
/// the integrated likelihood and the draw, while occupancy still uses the
/// true counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InverseGammaCells {
    pub nu: f64,
    pub lambda: f64,
    pub prior_only: bool,
}

/// Per-cell (n_k, E_k) accumulated in ascending observation order.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct InverseGammaStats {
    pub cells: Vec<InverseGammaCell>,
}

/// One cell's (n_k, E_k).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct InverseGammaCell {
    pub count: usize,
    pub sum: f64,
}

impl Stats for InverseGammaStats {
    fn all_occupied(&self) -> bool {
        self.cells.iter().all(|c| c.count > 0)
    }
}

impl InverseGammaCells {
    /// Shape and rate of cell k's posterior.
    fn posterior(&self, count: usize, sum: f64) -> (f64, f64) {
        if self.prior_only {
            (0.5 * self.nu, 0.5 * self.nu * self.lambda)
        } else {
            (
                0.5 * (self.nu + count as f64),
                0.5 * (self.nu * self.lambda + sum),
            )
        }
    }

    fn normaliser(&self) -> f64 {
        0.5 * self.nu * maths::ln(0.5 * self.nu * self.lambda) - maths::lgamma(0.5 * self.nu)
    }
}

impl CellFamily for InverseGammaCells {
    type Stats = InverseGammaStats;

    fn partial(&self, _input: f64, total: f64, own: f64) -> f64 {
        total / own
    }

    fn total(&self, _input: f64, partial: f64, own: f64) -> f64 {
        partial * own
    }

    fn begin(&self, b: usize, _context: &Context, out: &mut InverseGammaStats) {
        out.cells.clear();
        out.cells.resize(b, InverseGammaCell::default());
    }

    #[inline]
    fn add(
        &self,
        out: &mut InverseGammaStats,
        _i: usize,
        cell: usize,
        input: f64,
        _weight: f64,
        partial: f64,
        _context: &Context,
    ) {
        let c = &mut out.cells[cell];
        c.count += 1;
        c.sum += 1.0 / partial * input * input;
    }

    fn log_marginal(
        &self,
        stats: &InverseGammaStats,
        #[cfg(test)] _breakage: crate::broken::Breakage,
    ) -> f64 {
        let normaliser = self.normaliser();
        let mut total = 0.0;
        for c in &stats.cells {
            let (shape, rate) = self.posterior(c.count, c.sum);
            total += maths::lgamma(shape) - shape * maths::ln(rate) + normaliser;
        }
        total
    }

    /// One gamma draw per cell.
    fn draw(
        &self,
        stats: &InverseGammaStats,
        rng: &mut Rng,
        values: &mut Vec<f64>,
        slopes: &mut Vec<f64>,
    ) {
        values.clear();
        slopes.clear();
        values.extend(stats.cells.iter().map(|c| {
            let (shape, rate) = self.posterior(c.count, c.sum);
            1.0 / rng::gamma(shape, 1.0 / rate, rng)
        }));
    }

    #[cfg(test)]
    fn cell_normaliser(&self) -> f64 {
        self.normaliser()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{chain_rng, uniform_index};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    /// A design and tessellation for the constant-basis calls, which
    /// ignore both.
    fn any_ctx(n: usize) -> (Data, Tessellation) {
        let rows: Vec<[f64; 1]> = (0..n.max(2)).map(|i| [i as f64]).collect();
        let t = Tessellation {
            centres: vec![0.0],
            dims: vec![0],
            mus: vec![0.0],
            betas: Vec::new(),
            tau: None,
        };
        (Data::from_rows(&rows).unwrap(), t)
    }

    /// The per-row inputs of one accumulation.
    struct Rows {
        input: Vec<f64>,
        weights: Vec<f64>,
        partials: Vec<f64>,
    }

    /// [`CellFamily::accumulate`] into a fresh buffer.
    fn accumulated<F: CellFamily>(
        family: &F,
        x: &Data,
        tessellation: &Tessellation,
        cells: &[CellIndex],
        rows: &Rows,
        b: usize,
        soft: Option<&[f64]>,
    ) -> F::Stats {
        let mut out = F::Stats::default();
        let context = Context {
            x,
            tessellation,
            soft,
            uniform: None,
        };
        family.accumulate(
            cells,
            &rows.input,
            &rows.weights,
            &rows.partials,
            b,
            &context,
            &mut out,
        );
        out
    }

    /// Gaussian rows: the residuals are the partials.
    fn gaussian_partials(residuals: &[f64], precision: &[f64]) -> Rows {
        Rows {
            input: residuals.to_vec(),
            weights: precision.to_vec(),
            partials: residuals.to_vec(),
        }
    }

    /// Inverse-gamma rows: the residuals are the input and the products
    /// of the other tessellations the partials.
    fn inverse_gamma_partials(residuals: &[f64], rest: &[f64]) -> Rows {
        Rows {
            input: residuals.to_vec(),
            weights: vec![1.0; residuals.len()],
            partials: rest.to_vec(),
        }
    }

    #[test]
    fn gaussian_accumulate_and_occupancy() {
        let ctx = any_ctx(8);
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let stats = accumulated(
            &family,
            &ctx.0,
            &ctx.1,
            &[0, 1, 1],
            &gaussian_partials(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]),
            3,
            None,
        );
        let cell = |count, weight, sum| GaussianCell { count, weight, sum };
        assert_eq!(
            stats.cells,
            vec![cell(1, 2.0, 2.0), cell(2, 4.0, 10.0), cell(0, 0.0, 0.0)]
        );
        assert!(!stats.all_occupied());
        assert!(accumulated(
            &family,
            &ctx.0,
            &ctx.1,
            &[0, 1],
            &gaussian_partials(&[1.0, 1.0], &[1.0, 1.0]),
            2,
            None
        )
        .all_occupied());
    }

    #[test]
    fn gaussian_partial_and_total_use_the_backfitting_order() {
        // r = y - F + mu and F' = y - r + mu', in that order of operations.
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let p = family.partial(0.7, 0.2, 0.05);
        assert_eq!(p, 0.7 - 0.2 + 0.05);
        assert_eq!(family.total(0.7, p, 0.1), 0.7 - p + 0.1);
    }

    #[test]
    fn gaussian_log_marginal_matches_the_unweighted_form() {
        let ctx = any_ctx(8);
        // With w_i = 1 / sigma^2: n_k = 4, S = 2, sigma^2 = 0.5, sigma_mu^2 = 0.25
        // gives 0.5 ln(sigma^2 / (n sigma_mu^2 + sigma^2)) + sigma_mu^2 S^2 /
        // (2 sigma^2 (n sigma_mu^2 + sigma^2)).
        let sigma_sq = 0.5;
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let stats = accumulated(
            &family,
            &ctx.0,
            &ctx.1,
            &[0; 4],
            &gaussian_partials(&[0.5; 4], &[1.0 / sigma_sq; 4]),
            1,
            None,
        );
        let expected = 0.5 * (sigma_sq / (4.0 * 0.25 + sigma_sq)).ln()
            + 0.25 * 4.0 / (2.0 * sigma_sq * (4.0 * 0.25 + sigma_sq));
        close(
            family.log_marginal(&stats, crate::broken::Breakage::None),
            expected,
            1e-14,
        );
    }

    /// ln N(r; 0, D^-1 + sigma_mu^2 Z Z^T) with D = diag(w) and Z the cell
    /// indicator matrix, by dense Cholesky. Shares no code with
    /// `log_marginal`.
    fn dense_log_density(r: &[f64], w: &[f64], cells: &[CellIndex], sigma_mu_sq: f64) -> f64 {
        let n = r.len();
        let mut sigma = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut v = if cells[i] == cells[j] {
                    sigma_mu_sq
                } else {
                    0.0
                };
                if i == j {
                    v += 1.0 / w[i];
                }
                sigma[i * n + j] = v;
            }
        }
        let mut l = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = sigma[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    l[i * n + i] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        let mut u = r.to_vec();
        for i in 0..n {
            for k in 0..i {
                u[i] -= l[i * n + k] * u[k];
            }
            u[i] /= l[i * n + i];
        }
        let log_det: f64 = (0..n).map(|i| 2.0 * l[i * n + i].ln()).sum();
        let quad: f64 = u.iter().map(|v| v * v).sum();
        -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad)
    }

    #[test]
    fn gaussian_log_marginal_difference_matches_a_dense_recomputation() {
        let ctx = any_ctx(8);
        // log_marginal drops terms that do not depend on the assignment,
        // so differences between assignments equal the dense evaluation.
        let mut rng = chain_rng(17);
        for case in 0..60 {
            let n = 4 + uniform_index(9, &mut rng);
            let b = 1 + uniform_index(4, &mut rng);
            let family = GaussianCells {
                sigma_mu_sq: 0.05 + 0.4 * (case as f64 / 60.0),
                linear: false,
            };
            let r: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let w: Vec<f64> = (0..n)
                .map(|_| (0.6 * standard_normal(&mut rng)).exp())
                .collect();
            let cells_a: Vec<CellIndex> = (0..n)
                .map(|_| uniform_index(b, &mut rng) as CellIndex)
                .collect();
            let cells_b: Vec<CellIndex> = (0..n)
                .map(|_| uniform_index(b, &mut rng) as CellIndex)
                .collect();
            let partials = gaussian_partials(&r, &w);
            let delta = family.log_marginal(
                &accumulated(&family, &ctx.0, &ctx.1, &cells_a, &partials, b, None),
                crate::broken::Breakage::None,
            ) - family.log_marginal(
                &accumulated(&family, &ctx.0, &ctx.1, &cells_b, &partials, b, None),
                crate::broken::Breakage::None,
            );
            let dense = dense_log_density(&r, &w, &cells_a, family.sigma_mu_sq)
                - dense_log_density(&r, &w, &cells_b, family.sigma_mu_sq);
            close(delta, dense, 1e-9);
        }
    }

    #[test]
    fn gaussian_posterior_mean_and_variance() {
        let ctx = any_ctx(8);
        // n = 4, S = 2, sigma^2 = 0.5, sigma_mu^2 = 0.25: mean 1/3, var 1/12.
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let stats = accumulated(
            &family,
            &ctx.0,
            &ctx.1,
            &[0; 4],
            &gaussian_partials(&[0.5; 4], &[2.0; 4]),
            1,
            None,
        );
        let mut rng = chain_rng(1);
        let n = 40_000;
        let draws: Vec<f64> = (0..n)
            .map(|_| family.draw_owned(&stats, &mut rng).0[0])
            .collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        close(mean, 1.0 / 3.0, 0.01);
        close(var, 1.0 / 12.0, 0.005);
    }

    fn ig_family() -> InverseGammaCells {
        InverseGammaCells {
            nu: 3.0,
            lambda: 0.4,
            prior_only: false,
        }
    }

    #[test]
    fn inverse_gamma_partial_and_total_divide_and_multiply() {
        let family = ig_family();
        let p = family.partial(0.3, 6.0, 2.0);
        close(p, 3.0, 1e-15);
        close(family.total(0.3, p, 0.5), 1.5, 1e-15);
    }

    #[test]
    fn inverse_gamma_accumulate_and_occupancy() {
        let ctx = any_ctx(8);
        let family = ig_family();
        let partials = inverse_gamma_partials(&[1.0, 2.0, 3.0], &[0.5, 2.0, 1.0]);
        let stats = accumulated(&family, &ctx.0, &ctx.1, &[0, 1, 1], &partials, 3, None);
        let cell = |count, sum| InverseGammaCell { count, sum };
        assert_eq!(
            stats.cells,
            vec![cell(1, 2.0), cell(2, 2.0 + 9.0), cell(0, 0.0)]
        );
        assert!(!stats.all_occupied());
    }

    /// ln p(e | T) = sum_k ln integral prod_{i in k} N(e_i; 0, v s_i)
    /// Inv-Gamma(v; nu / 2, nu lambda / 2) dv, each cell integral by the
    /// trapezium rule over t = ln v. Shares no code with `log_marginal`.
    fn quadrature_log_density(
        e: &[f64],
        s: &[f64],
        cells: &[CellIndex],
        b: usize,
        nu: f64,
        lambda: f64,
    ) -> f64 {
        let (a, rate) = (0.5 * nu, 0.5 * nu * lambda);
        let log_prior_const = a * rate.ln() - libm::lgamma(a);
        let steps = 20_000;
        let (lo, hi) = (lambda.ln() - 14.0, lambda.ln() + 14.0);
        let h = (hi - lo) / steps as f64;
        let mut total = 0.0;
        for k in 0..b {
            let members: Vec<usize> = (0..e.len()).filter(|&i| cells[i] as usize == k).collect();
            let mut log_terms = Vec::with_capacity(steps + 1);
            for step in 0..=steps {
                let t = lo + h * step as f64;
                let v = t.exp();
                // ln prior density in v times the Jacobian v, plus the likelihood.
                let mut lp = log_prior_const - (a + 1.0) * t - rate / v + t;
                for &i in &members {
                    let var = v * s[i];
                    lp += -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + e[i] * e[i] / var);
                }
                log_terms.push(lp);
            }
            let max = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = log_terms.iter().map(|lp| (lp - max).exp()).sum();
            total += max + (sum * h).ln();
        }
        total
    }

    #[test]
    fn inverse_gamma_log_marginal_difference_matches_quadrature() {
        let ctx = any_ctx(8);
        let mut rng = chain_rng(23);
        for case in 0..40 {
            let n = 3 + uniform_index(8, &mut rng);
            let b = 1 + uniform_index(3, &mut rng);
            let family = InverseGammaCells {
                nu: 2.5 + 0.1 * case as f64,
                lambda: 0.2 + 0.02 * case as f64,
                prior_only: false,
            };
            let e: Vec<f64> = (0..n).map(|_| 0.5 * standard_normal(&mut rng)).collect();
            let s: Vec<f64> = (0..n)
                .map(|_| (0.5 * standard_normal(&mut rng)).exp())
                .collect();
            let partials = inverse_gamma_partials(&e, &s);
            let cells_a: Vec<CellIndex> = (0..n)
                .map(|_| uniform_index(b, &mut rng) as CellIndex)
                .collect();
            let cells_b: Vec<CellIndex> = (0..n)
                .map(|_| uniform_index(b, &mut rng) as CellIndex)
                .collect();
            let delta = family.log_marginal(
                &accumulated(&family, &ctx.0, &ctx.1, &cells_a, &partials, b, None),
                crate::broken::Breakage::None,
            ) - family.log_marginal(
                &accumulated(&family, &ctx.0, &ctx.1, &cells_b, &partials, b, None),
                crate::broken::Breakage::None,
            );
            let reference = quadrature_log_density(&e, &s, &cells_a, b, family.nu, family.lambda)
                - quadrature_log_density(&e, &s, &cells_b, b, family.nu, family.lambda);
            close(delta, reference, 1e-6);
        }
    }

    #[test]
    fn inverse_gamma_log_marginal_closed_form() {
        let ctx = any_ctx(8);
        // nu = 3, lambda = 0.4; cell 0 holds one observation with e^2 / s = 2,
        // cell 1 holds two with e^2 / s = 0.5 each.
        let family = ig_family();
        let partials = inverse_gamma_partials(&[2.0, 1.0, 1.0], &[2.0; 3]);
        let stats = accumulated(&family, &ctx.0, &ctx.1, &[0, 1, 1], &partials, 2, None);
        assert_eq!(stats.cells[0].sum, 2.0);
        assert_eq!(stats.cells[1].sum, 1.0);
        let normaliser = 1.5 * 0.6_f64.ln() - libm::lgamma(1.5);
        let cell = |n: f64, e: f64| {
            libm::lgamma(0.5 * (3.0 + n)) - 0.5 * (3.0 + n) * (0.5 * (1.2 + e)).ln() + normaliser
        };
        close(family.cell_normaliser(), normaliser, 1e-14);
        close(
            family.log_marginal(&stats, crate::broken::Breakage::None),
            cell(1.0, 2.0) + cell(2.0, 1.0),
            1e-12,
        );
        // Under prior_only every cell contributes zero.
        let prior = InverseGammaCells {
            prior_only: true,
            ..family
        };
        close(
            prior.log_marginal(&stats, crate::broken::Breakage::None),
            0.0,
            1e-12,
        );
    }

    #[test]
    fn inverse_gamma_posterior_mean() {
        let ctx = any_ctx(8);
        // nu = 3, lambda = 0.4, n = 4, E = 2: Inv-Gamma(3.5, 1.6), mean 1.6 / 2.5.
        let family = ig_family();
        let partials = inverse_gamma_partials(&[1.0; 4], &[2.0; 4]);
        let stats = accumulated(&family, &ctx.0, &ctx.1, &[0; 4], &partials, 1, None);
        assert_eq!(stats.cells, vec![InverseGammaCell { count: 4, sum: 2.0 }]);
        let mut rng = chain_rng(2);
        let n = 100_000;
        let mean = (0..n)
            .map(|_| family.draw_owned(&stats, &mut rng).0[0])
            .sum::<f64>()
            / n as f64;
        close(mean, 1.6 / 2.5, 0.01);
        let prior = InverseGammaCells {
            prior_only: true,
            ..family
        };
        // Inv-Gamma(1.5, 0.6) has no finite variance; its reciprocal is
        // Gamma(1.5, rate 0.6) with mean 2.5.
        let mean = (0..n)
            .map(|_| 1.0 / prior.draw_owned(&stats, &mut rng).0[0])
            .sum::<f64>()
            / n as f64;
        close(mean, 2.5, 0.05);
    }

    /// The hard-membership passes against their definition through `add`,
    /// bit for bit: `{:?}` prints every f64 exactly, signed zero included.
    mod hard {
        use super::*;
        use proptest::prelude::*;

        fn value() -> impl Strategy<Value = f64> {
            prop::num::f64::NORMAL | prop::num::f64::SUBNORMAL | prop::num::f64::ZERO
        }

        /// Rows over `b` cells and `next_b` next cells, and the values of
        /// both tessellations.
        #[derive(Debug, Clone)]
        struct Case {
            cells: Vec<CellIndex>,
            next_cells: Vec<CellIndex>,
            mus: Vec<f64>,
            next_mus: Vec<f64>,
            input: Vec<f64>,
            weights: Vec<f64>,
            total: Vec<f64>,
            partials: Vec<f64>,
        }

        fn case() -> impl Strategy<Value = Case> {
            (0usize..48, 1usize..40, 1usize..40).prop_flat_map(|(n, b, next_b)| {
                let values = || prop::collection::vec(value(), n);
                (
                    prop::collection::vec(0..b as CellIndex, n),
                    prop::collection::vec(0..next_b as CellIndex, n),
                    prop::collection::vec(value(), b),
                    prop::collection::vec(value(), next_b),
                    values(),
                    weights(n),
                    values(),
                    values(),
                )
                    .prop_map(
                        |(cells, next_cells, mus, next_mus, input, weights, total, partials)| {
                            Case {
                                cells,
                                next_cells,
                                mus,
                                next_mus,
                                input,
                                weights,
                                total,
                                partials,
                            }
                        },
                    )
            })
        }

        /// Non-negative weights, all with the same bits half the time.
        fn weights(n: usize) -> impl Strategy<Value = Vec<f64>> {
            let weight = || value().prop_map(f64::abs);
            prop_oneof![
                prop::collection::vec(weight(), n),
                weight().prop_map(move |w| vec![w; n]),
            ]
        }

        /// [`Context::uniform`] for `weights`, from its definition; empty
        /// when the weights differ.
        fn running(weights: &[f64]) -> Vec<f64> {
            let equal = weights.windows(2).all(|w| w[0].to_bits() == w[1].to_bits());
            if weights.is_empty() || !equal {
                return Vec::new();
            }
            (0..=weights.len())
                .map(|c| weights[..c].iter().fold(0.0, |s, w| s + w))
                .collect()
        }

        /// The context of the pass under test, with the running sums when
        /// the weights allow, and of its definition, without.
        fn contexts<'a>(
            x: &'a Data,
            t: &'a Tessellation,
            running: &'a [f64],
        ) -> (Context<'a>, Context<'a>) {
            let definition = Context {
                x,
                tessellation: t,
                soft: None,
                uniform: None,
            };
            let fast = Context {
                uniform: (!running.is_empty()).then_some(running),
                ..definition
            };
            (fast, definition)
        }

        fn families() -> (GaussianCells, InverseGammaCells) {
            (
                GaussianCells {
                    sigma_mu_sq: 0.25,
                    linear: false,
                },
                InverseGammaCells {
                    nu: 3.0,
                    lambda: 0.9,
                    prior_only: false,
                },
            )
        }

        /// `hard_partials` and its definition, as printed state.
        fn partials_pair<F: CellFamily>(family: &F, c: &Case) -> (String, String) {
            let (x, t) = any_ctx(c.cells.len());
            let running = running(&c.weights);
            let (fast, definition) = contexts(&x, &t, &running);
            let run = |by_add: bool| {
                let mut out = F::Stats::default();
                let mut partials = vec![0.0; c.cells.len()];
                let (cells, mus, input, weights) = (&c.cells, &c.mus, &c.input, &c.weights);
                if by_add {
                    family.begin(c.mus.len(), &definition, &mut out);
                    hard_partials_by_add(
                        family,
                        cells,
                        mus,
                        input,
                        weights,
                        &c.total,
                        &mut partials,
                        &definition,
                        &mut out,
                    );
                } else {
                    family.begin(c.mus.len(), &fast, &mut out);
                    family.hard_partials(
                        cells,
                        mus,
                        input,
                        weights,
                        &c.total,
                        &mut partials,
                        &fast,
                        &mut out,
                    );
                }
                format!("{partials:?} {out:?}")
            };
            (run(false), run(true))
        }

        /// `accumulate` and its definition, as printed state.
        fn accumulate_pair<F: CellFamily>(family: &F, c: &Case) -> (String, String) {
            let (x, t) = any_ctx(c.cells.len());
            let running = running(&c.weights);
            let (fast, definition) = contexts(&x, &t, &running);
            let (cells, input, weights, partials) = (&c.cells, &c.input, &c.weights, &c.partials);
            let b = c.mus.len();
            let mut got = F::Stats::default();
            family.accumulate(cells, input, weights, partials, b, &fast, &mut got);
            let mut want = F::Stats::default();
            accumulate_by_add(
                family,
                cells,
                input,
                weights,
                partials,
                b,
                &definition,
                &mut want,
            );
            (format!("{got:?}"), format!("{want:?}"))
        }

        /// The fused pass, and the running-total update followed by the
        /// next tessellation's `hard_partials`, as printed state.
        fn fused_pair<F: CellFamily>(family: &F, c: &Case) -> (String, String) {
            let (x, t) = any_ctx(c.cells.len());
            let running = running(&c.weights);
            let (fast, definition) = contexts(&x, &t, &running);
            let n = c.cells.len();
            let mut fused = F::Stats::default();
            let (mut total, mut partials) = (c.total.clone(), c.partials.clone());
            family.begin(c.next_mus.len(), &fast, &mut fused);
            family.hard_totals_then_partials(
                &c.cells,
                &c.mus,
                &c.next_cells,
                &c.next_mus,
                &c.input,
                &c.weights,
                &mut total,
                &mut partials,
                &fast,
                &mut fused,
            );
            let got = format!("{total:?} {partials:?} {fused:?}");

            let total: Vec<f64> = (0..n)
                .map(|i| family.total(c.input[i], c.partials[i], c.mus[c.cells[i] as usize]))
                .collect();
            let mut partials = vec![0.0; n];
            let mut separate = F::Stats::default();
            family.begin(c.next_mus.len(), &definition, &mut separate);
            hard_partials_by_add(
                family,
                &c.next_cells,
                &c.next_mus,
                &c.input,
                &c.weights,
                &total,
                &mut partials,
                &definition,
                &mut separate,
            );
            (got, format!("{total:?} {partials:?} {separate:?}"))
        }

        proptest! {
            #[test]
            fn accumulate_equals_its_definition(c in case()) {
                let (gaussian, inverse_gamma) = families();
                let (got, want) = accumulate_pair(&gaussian, &c);
                prop_assert_eq!(got, want);
                let (got, want) = accumulate_pair(&inverse_gamma, &c);
                prop_assert_eq!(got, want);
            }

            #[test]
            fn hard_partials_equal_their_definition(c in case()) {
                let (gaussian, inverse_gamma) = families();
                let (got, want) = partials_pair(&gaussian, &c);
                prop_assert_eq!(got, want);
                let (got, want) = partials_pair(&inverse_gamma, &c);
                prop_assert_eq!(got, want);
            }

            #[test]
            fn the_fused_pass_equals_the_update_then_the_next_pass(c in case()) {
                let (gaussian, inverse_gamma) = families();
                let (got, want) = fused_pair(&gaussian, &c);
                prop_assert_eq!(got, want);
                let (got, want) = fused_pair(&inverse_gamma, &c);
                prop_assert_eq!(got, want);
            }
        }
    }

    mod soft {
        use super::*;

        fn family() -> GaussianCells {
            GaussianCells {
                sigma_mu_sq: 0.25,
                linear: false,
            }
        }

        /// The hard indicator as a weight matrix.
        fn indicator(cells: &[CellIndex], b: usize) -> Vec<f64> {
            let mut w = vec![0.0; cells.len() * b];
            for (i, &cell) in cells.iter().enumerate() {
                w[i * b + cell as usize] = 1.0;
            }
            w
        }

        /// A normalised random weight row per observation.
        fn random_weights(n: usize, b: usize, rng: &mut Rng) -> Vec<f64> {
            let mut w: Vec<f64> = (0..n * b)
                .map(|_| maths::exp(standard_normal(rng)))
                .collect();
            for row in w.chunks_exact_mut(b) {
                let total: f64 = row.iter().sum();
                row.iter_mut().for_each(|v| *v /= total);
            }
            w
        }

        #[test]
        fn indicator_weights_match_the_constant_marginal() {
            let ctx = any_ctx(8);
            let mut rng = chain_rng(29);
            for _ in 0..40 {
                let n = 4 + uniform_index(9, &mut rng);
                let b = 1 + uniform_index(4, &mut rng);
                let family = family();
                let r: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
                let w: Vec<f64> = (0..n)
                    .map(|_| maths::exp(0.6 * standard_normal(&mut rng)))
                    .collect();
                let cells: Vec<CellIndex> = (0..n)
                    .map(|_| uniform_index(b, &mut rng) as CellIndex)
                    .collect();
                let partials = gaussian_partials(&r, &w);
                let soft = accumulated(
                    &family,
                    &ctx.0,
                    &ctx.1,
                    &cells,
                    &partials,
                    b,
                    Some(&indicator(&cells, b)),
                );
                let hard = accumulated(&family, &ctx.0, &ctx.1, &cells, &partials, b, None);
                assert_eq!(soft.all_occupied(), hard.all_occupied());
                close(
                    family.log_marginal(&soft, crate::broken::Breakage::None),
                    family.log_marginal(&hard, crate::broken::Breakage::None),
                    1e-10,
                );
            }
        }

        /// ln N(r; 0, D^-1 + sigma_mu^2 W W^T) by dense Cholesky. Shares
        /// no code with `log_marginal`.
        fn dense_log_density(
            r: &[f64],
            prec: &[f64],
            w: &[f64],
            b: usize,
            sigma_mu_sq: f64,
        ) -> f64 {
            let n = r.len();
            let mut sigma = vec![0.0_f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut v =
                        sigma_mu_sq * (0..b).map(|k| w[i * b + k] * w[j * b + k]).sum::<f64>();
                    if i == j {
                        v += 1.0 / prec[i];
                    }
                    sigma[i * n + j] = v;
                }
            }
            let mut l = vec![0.0_f64; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = sigma[i * n + j];
                    for k in 0..j {
                        sum -= l[i * n + k] * l[j * n + k];
                    }
                    if i == j {
                        l[i * n + i] = sum.sqrt();
                    } else {
                        l[i * n + j] = sum / l[j * n + j];
                    }
                }
            }
            let mut u = r.to_vec();
            for i in 0..n {
                for k in 0..i {
                    u[i] -= l[i * n + k] * u[k];
                }
                u[i] /= l[i * n + i];
            }
            let log_det: f64 = (0..n).map(|i| 2.0 * l[i * n + i].ln()).sum();
            let quad: f64 = u.iter().map(|v| v * v).sum();
            -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad)
        }

        #[test]
        fn the_soft_marginal_difference_matches_a_dense_recomputation() {
            let ctx = any_ctx(8);
            // log_marginal drops terms that do not depend on the weights,
            // so differences between weight matrices equal the dense
            // evaluation.
            let mut rng = chain_rng(31);
            for case in 0..60 {
                let n = 4 + uniform_index(9, &mut rng);
                let b = 1 + uniform_index(4, &mut rng);
                let family = GaussianCells {
                    sigma_mu_sq: 0.05 + 0.4 * (case as f64 / 60.0),
                    linear: false,
                };
                let r: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
                let prec: Vec<f64> = (0..n)
                    .map(|_| maths::exp(0.6 * standard_normal(&mut rng)))
                    .collect();
                let cells: Vec<CellIndex> = (0..n)
                    .map(|_| uniform_index(b, &mut rng) as CellIndex)
                    .collect();
                let partials = gaussian_partials(&r, &prec);
                let w_a = random_weights(n, b, &mut rng);
                let w_b = random_weights(n, b, &mut rng);
                let delta = family.log_marginal(
                    &accumulated(&family, &ctx.0, &ctx.1, &cells, &partials, b, Some(&w_a)),
                    crate::broken::Breakage::None,
                ) - family.log_marginal(
                    &accumulated(&family, &ctx.0, &ctx.1, &cells, &partials, b, Some(&w_b)),
                    crate::broken::Breakage::None,
                );
                let dense = dense_log_density(&r, &prec, &w_a, b, family.sigma_mu_sq)
                    - dense_log_density(&r, &prec, &w_b, b, family.sigma_mu_sq);
                close(delta, dense, 1e-9);
            }
        }

        #[test]
        fn the_joint_draw_matches_the_conjugate_posterior() {
            let ctx = any_ctx(4);
            // Two cells, four observations, hand-picked weights: the
            // posterior is N(P^-1 W'Dr, P^-1) with P = W'DW + I / sigma_mu^2.
            let family = family();
            let values = [1.0, 0.2, 0.4, 0.9];
            let prec = [1.0, 2.0, 1.0, 1.0];
            let w = [0.9, 0.1, 0.3, 0.7, 0.5, 0.5, 0.8, 0.2];
            let cells = [0, 1, 0, 0];
            let partials = gaussian_partials(&values, &prec);
            let stats = accumulated(&family, &ctx.0, &ctx.1, &cells, &partials, 2, Some(&w));
            let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for i in 0..4 {
                a11 += prec[i] * w[i * 2] * w[i * 2];
                a12 += prec[i] * w[i * 2] * w[i * 2 + 1];
                a22 += prec[i] * w[i * 2 + 1] * w[i * 2 + 1];
                b1 += prec[i] * values[i] * w[i * 2];
                b2 += prec[i] * values[i] * w[i * 2 + 1];
            }
            let (p11, p12, p22) = (a11 + 4.0, a12, a22 + 4.0);
            let det = p11 * p22 - p12 * p12;
            let mean = [(p22 * b1 - p12 * b2) / det, (p11 * b2 - p12 * b1) / det];
            let cov = [p22 / det, -p12 / det, p11 / det];

            let mut rng = rng::chain_rng(13);
            let n = 200_000;
            let (mut m0, mut m1) = (0.0, 0.0);
            let (mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0);
            for _ in 0..n {
                let (values, slopes) = family.draw_owned(&stats, &mut rng);
                assert!(slopes.is_empty());
                m0 += values[0];
                m1 += values[1];
                v0 += values[0] * values[0];
                v1 += values[1] * values[1];
                c01 += values[0] * values[1];
            }
            let n = n as f64;
            let (m0, m1) = (m0 / n, m1 / n);
            close(m0, mean[0], 5e-3);
            close(m1, mean[1], 5e-3);
            close(v0 / n - m0 * m0, cov[0], 5e-3);
            close(c01 / n - m0 * m1, cov[1], 5e-3);
            close(v1 / n - m1 * m1, cov[2], 5e-3);
        }

        #[test]
        fn the_dropped_determinant_shifts_the_marginal() {
            let ctx = any_ctx(4);
            let family = family();
            let cells = [0, 1, 0, 0];
            let partials = gaussian_partials(&[1.0, 0.2, 0.4, 0.9], &[1.0; 4]);
            let w = indicator(&cells, 2);
            let stats = accumulated(&family, &ctx.0, &ctx.1, &cells, &partials, 2, Some(&w));
            let whole = family.log_marginal(&stats, crate::broken::Breakage::None);
            let broken =
                family.log_marginal(&stats, crate::broken::Breakage::DroppedSoftDeterminant);
            assert!(broken > whole, "{broken} vs {whole}");
        }
    }

    mod linear {
        use super::*;

        fn linear_family() -> GaussianCells {
            GaussianCells {
                sigma_mu_sq: 0.25,
                linear: true,
            }
        }

        /// One cell on one dimension centred at 0.4.
        fn one_cell() -> Tessellation {
            Tessellation {
                centres: vec![0.4],
                dims: vec![0],
                mus: vec![0.0],
                betas: vec![0.0],
                tau: None,
            }
        }

        #[test]
        fn rows_at_the_centre_match_the_constant_marginal() {
            // Every offset zero: the slope block is prior-only and the
            // integrated likelihood reduces to the constant expression.
            let x = Data::from_rows(&[[0.4], [0.4], [0.4]]).unwrap();
            let t = one_cell();
            let partials = gaussian_partials(&[0.6, -0.2, 0.1], &[2.0, 1.0, 3.0]);
            let cells = [0, 0, 0];
            let linear = linear_family();
            let constant = GaussianCells {
                sigma_mu_sq: 0.25,
                linear: false,
            };
            let a = linear.log_marginal(
                &accumulated(&linear, &x, &t, &cells, &partials, 1, None),
                crate::broken::Breakage::None,
            );
            let b = constant.log_marginal(
                &accumulated(&constant, &x, &t, &cells, &partials, 1, None),
                crate::broken::Breakage::None,
            );
            close(a, b, 1e-12);
        }

        #[test]
        fn the_marginal_matches_the_closed_form() {
            // Two observations, u = (1, x - 0.4), unit weights.
            let x = Data::from_rows(&[[0.9], [-0.1]]).unwrap();
            let t = one_cell();
            let partials = gaussian_partials(&[1.0, 0.2], &[1.0, 1.0]);
            let cells = [0, 0];
            let family = linear_family();
            let stats = accumulated(&family, &x, &t, &cells, &partials, 1, None);
            // A = [[2, 0], [0, 0.5]], b = (1.2, 0.4).
            let sigma = 0.25_f64;
            let p11 = 2.0 + 1.0 / sigma;
            let p22 = 0.5 + 1.0 / sigma;
            let expected = -0.5 * ((sigma * p11).ln() + (sigma * p22).ln())
                + 0.5 * (1.2 * 1.2 / p11 + 0.4 * 0.4 / p22);
            close(
                family.log_marginal(&stats, crate::broken::Breakage::None),
                expected,
                1e-12,
            );
        }

        #[test]
        fn the_joint_draw_matches_the_conjugate_posterior() {
            let x = Data::from_rows(&[[0.9], [-0.1], [0.15], [0.65]]).unwrap();
            let t = one_cell();
            let partials = gaussian_partials(&[1.0, 0.2, 0.4, 0.9], &[1.0, 2.0, 1.0, 1.0]);
            let cells = [0, 0, 0, 0];
            let family = linear_family();
            let stats = accumulated(&family, &x, &t, &cells, &partials, 1, None);
            // The closed-form posterior mean, from the normal equations.
            let offsets = [0.5, -0.5, -0.25, 0.25];
            let values = [1.0, 0.2, 0.4, 0.9];
            let weights = [1.0, 2.0, 1.0, 1.0];
            let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for i in 0..4 {
                a11 += weights[i];
                a12 += weights[i] * offsets[i];
                a22 += weights[i] * offsets[i] * offsets[i];
                b1 += weights[i] * values[i];
                b2 += weights[i] * values[i] * offsets[i];
            }
            let (p11, p12, p22) = (a11 + 4.0, a12, a22 + 4.0);
            let det = p11 * p22 - p12 * p12;
            let mean = [(p22 * b1 - p12 * b2) / det, (p11 * b2 - p12 * b1) / det];
            let cov = [p22 / det, -p12 / det, p11 / det];

            let mut rng = rng::chain_rng(11);
            let n = 200_000;
            let (mut m0, mut m1) = (0.0, 0.0);
            let (mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0);
            for _ in 0..n {
                let (values, slopes) = family.draw_owned(&stats, &mut rng);
                m0 += values[0];
                m1 += slopes[0];
                v0 += values[0] * values[0];
                v1 += slopes[0] * slopes[0];
                c01 += values[0] * slopes[0];
            }
            let n = n as f64;
            let (m0, m1) = (m0 / n, m1 / n);
            close(m0, mean[0], 5e-3);
            close(m1, mean[1], 5e-3);
            close(v0 / n - m0 * m0, cov[0], 5e-3);
            close(c01 / n - m0 * m1, cov[1], 5e-3);
            close(v1 / n - m1 * m1, cov[2], 5e-3);
        }
    }
}
