//! An ensemble of tessellations over one conjugate cell family: the
//! tessellations, their assignment caches, the running total, and the
//! backfitting sweep of Stone and Gosling (2025, Algorithm 1) with the
//! Metropolis-Hastings structural moves. The Gaussian mean function is an
//! additive ensemble of [`GaussianCells`](crate::cells::GaussianCells); the
//! heteroscedastic variance function is a multiplicative ensemble of
//! [`InverseGammaCells`](crate::cells::InverseGammaCells).

use crate::cells::{CellFamily, Context, Stats};
use crate::data::Data;
use crate::geometry::Geometry;
use crate::maths;
use crate::moves::{self, Prior};
use crate::rng::{self, Rng};
use crate::tessellation::{Assignment, Tessellation};

/// The standard deviation of the soft-membership bandwidth proposal on
/// ln tau.
const BANDWIDTH_STEP: f64 = 0.5;

/// The tessellations of one cell family and their running total at the
/// training rows.
#[derive(Debug, Clone)]
pub(crate) struct Ensemble<F: CellFamily> {
    family: F,
    prior: Prior,
    /// Rate of the exponential prior on the soft-membership bandwidth;
    /// `None` under hard membership.
    bandwidth_rate: Option<f64>,
    tessellations: Vec<Tessellation>,
    assignments: Vec<Assignment>,
    /// The combined value of the ensemble at each training row.
    total: Vec<f64>,
    /// For the sweep in progress, [`Context::uniform`]'s running sums;
    /// empty when the weights differ.
    running: Vec<f64>,
    scratch: Scratch<F::Stats>,
}

/// The running sums of `weights` into `out` when every weight has the
/// same bits: entry c is c copies added in turn from zero. Empty
/// otherwise.
fn running_sums(weights: &[f64], out: &mut Vec<f64>) {
    out.clear();
    let Some(&w) = weights.first() else {
        return;
    };
    if weights.iter().any(|v| v.to_bits() != w.to_bits()) {
        return;
    }
    let mut sum = 0.0;
    out.push(sum);
    for _ in weights {
        sum += w;
        out.push(sum);
    }
}

/// [`Context::uniform`] from [`running_sums`].
fn uniform_of(running: &[f64]) -> Option<&[f64]> {
    (!running.is_empty()).then_some(running)
}

/// The buffers of one backfitting step, kept across steps with their
/// capacity. A proposal is built here and swapped into the ensemble on
/// acceptance, so the hard-membership path allocates nothing per step.
#[derive(Debug, Clone)]
struct Scratch<S> {
    partials: Vec<f64>,
    /// Scratch of the streaming assignment paths.
    keys: Vec<f64>,
    tessellation: Tessellation,
    assignment: Assignment,
    /// Statistics of the proposal; after an accepting swap, of the
    /// tessellation the proposal replaced.
    proposed: S,
    /// Statistics of the tessellation in the ensemble.
    current: S,
    values: Vec<f64>,
    slopes: Vec<f64>,
    /// Whether `partials` and `current` already hold the next
    /// tessellation's, formed in the previous tessellation's running-total
    /// pass.
    primed: bool,
}

impl<S: Default> Default for Scratch<S> {
    fn default() -> Self {
        Self {
            partials: Vec::new(),
            keys: Vec::new(),
            tessellation: Tessellation::empty(),
            assignment: Assignment::default(),
            proposed: S::default(),
            current: S::default(),
            values: Vec::new(),
            slopes: Vec::new(),
            primed: false,
        }
    }
}

impl<F: CellFamily> Ensemble<F> {
    /// `m` single-cell tessellations on one covariate each, every cell
    /// holding `cell_value`, the total set to `total` at every row. Draw
    /// order per tessellation: the covariate, then the centre coordinate,
    /// then, under soft membership, the bandwidth from its prior.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        family: F,
        prior: Prior,
        bandwidth_rate: Option<f64>,
        x: &Data,
        m: usize,
        cell_value: f64,
        total: f64,
        rng: &mut Rng,
    ) -> Self {
        let tessellations: Vec<Tessellation> = (0..m)
            .map(|_| {
                let dim = prior.initial_dim(rng);
                let centre = prior.coordinate(dim, rng);
                let tau = bandwidth_rate.map(|rate| -maths::ln(1.0 - rng::uniform(rng)) / rate);
                Tessellation {
                    centres: vec![centre],
                    dims: vec![dim],
                    mus: vec![cell_value],
                    betas: Vec::new(),
                    tau,
                }
            })
            .collect();
        let assignments = tessellations
            .iter()
            .map(|t| Assignment::full(x, t, &prior.geometry))
            .collect();
        Self {
            family,
            prior,
            bandwidth_rate,
            tessellations,
            assignments,
            total: vec![total; x.n_rows()],
            running: Vec::new(),
            scratch: Scratch::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn family(&self) -> &F {
        &self.family
    }

    pub(crate) fn tessellations(&self) -> &[Tessellation] {
        &self.tessellations
    }

    pub(crate) fn geometry(&self) -> &Geometry {
        &self.prior.geometry
    }

    pub(crate) fn total(&self) -> &[f64] {
        &self.total
    }

    /// One sweep: every tessellation in turn through [`backfit`](Self::backfit).
    /// Replace the inclusion weights of the structural prior; the DART
    /// update writes the sampled vector here each sweep.
    #[cfg(feature = "experimental")]
    pub(crate) fn set_inclusion_weights(&mut self, weights: crate::moves::InclusionWeights) {
        self.prior.weights = Some(weights);
    }

    /// The cell of every training row under each tessellation.
    pub(crate) fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }

    /// Replace every tessellation by `tessellations`, of the same count,
    /// rebuilding the assignment caches and the total. `identity` is the
    /// family's neutral value, 0 for an additive ensemble and 1 for a
    /// multiplicative one: each row's total folds the tessellations'
    /// values into it through the family's own update. Returns the index
    /// of the first tessellation with a cell no training row falls in.
    #[cfg(feature = "experimental")]
    pub(crate) fn restore(
        &mut self,
        x: &Data,
        tessellations: Vec<Tessellation>,
        identity: f64,
    ) -> std::result::Result<(), usize> {
        debug_assert_eq!(tessellations.len(), self.tessellations.len());
        let geometry = &self.prior.geometry;
        let assignments: Vec<Assignment> = tessellations
            .iter()
            .map(|t| Assignment::full(x, t, geometry))
            .collect();
        for (j, (t, a)) in tessellations.iter().zip(&assignments).enumerate() {
            let mut occupied = vec![false; t.n_cells()];
            a.cells.iter().for_each(|&k| occupied[k] = true);
            if occupied.iter().any(|&o| !o) {
                return Err(j);
            }
        }
        for (i, slot) in self.total.iter_mut().enumerate() {
            let row = x.row(i);
            let mut total = identity;
            for t in &tessellations {
                let partial = self.family.partial(0.0, total, identity);
                total = self.family.total(0.0, partial, t.value_at(row, geometry));
            }
            *slot = total;
        }
        self.tessellations = tessellations;
        self.assignments = assignments;
        Ok(())
    }

    pub(crate) fn sweep(
        &mut self,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.primed = false;
        running_sums(weights, &mut self.running);
        for j in 0..self.tessellations.len() {
            self.backfit(
                &mut scratch,
                j,
                x,
                input,
                weights,
                rng,
                #[cfg(test)]
                breakage,
            );
        }
        self.scratch = scratch;
    }

    /// The partials of tessellation `j` at every training row into
    /// `scratch.partials` and its statistics into `scratch.current`, in
    /// one pass.
    fn partials(
        &self,
        scratch: &mut Scratch<F::Stats>,
        j: usize,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        soft: Option<&[f64]>,
    ) {
        let current = &self.tessellations[j];
        let cells = &self.assignments[j].cells;
        let b = current.n_cells();
        let context = Context {
            x,
            tessellation: current,
            soft,
            uniform: uniform_of(&self.running),
        };
        self.family.begin(b, &context, &mut scratch.current);
        let n = input.len();
        let (weights, total, cells) = (&weights[..n], &self.total[..n], &cells[..n]);
        scratch.partials.resize(n, 0.0);
        let partials = &mut scratch.partials[..n];
        if soft.is_none() && current.betas.is_empty() {
            self.family.hard_partials(
                cells,
                &current.mus,
                input,
                weights,
                total,
                partials,
                &context,
                &mut scratch.current,
            );
            return;
        }
        for i in 0..n {
            let cell = cells[i];
            let own = match soft {
                Some(w) => soft_value(&current.mus, &w[i * b..(i + 1) * b]),
                None => current.training_value(cell, x, i),
            };
            let partial = self.family.partial(input[i], total[i], own);
            partials[i] = partial;
            self.family.add(
                &mut scratch.current,
                i,
                cell,
                input[i],
                weights[i],
                partial,
                &context,
            );
        }
    }

    /// The backfitting update of tessellation `j`: the partials against the
    /// rest of the ensemble, one structural move with the empty-cell guard,
    /// under soft membership one bandwidth move, the cell values, and the
    /// running total.
    #[allow(clippy::too_many_arguments)]
    fn backfit(
        &mut self,
        scratch: &mut Scratch<F::Stats>,
        j: usize,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        let tau = self.tessellations[j].tau;
        let mut soft_weights = tau.map(|tau| self.assignments[j].soft_weights(tau));
        if !std::mem::take(&mut scratch.primed) {
            self.partials(scratch, j, x, input, weights, soft_weights.as_deref());
        }
        let current = &self.tessellations[j];

        let m = moves::select(current, &self.prior, rng);
        let proposal = moves::propose(m, current, &self.prior, rng, &mut scratch.tessellation);
        let known_empty = self.assignments[j].updated_into(
            x,
            &scratch.tessellation,
            proposal.delta,
            &self.prior.geometry,
            &mut scratch.keys,
            &mut scratch.assignment,
        );
        let proposed_weights = tau.map(|tau| scratch.assignment.soft_weights(tau));
        if !known_empty {
            self.family.accumulate(
                &scratch.assignment.cells,
                input,
                weights,
                &scratch.partials,
                scratch.tessellation.n_cells(),
                &Context {
                    x,
                    tessellation: &scratch.tessellation,
                    soft: proposed_weights.as_deref(),
                    uniform: uniform_of(&self.running),
                },
                &mut scratch.proposed,
            );
        }
        // A proposal leaving a cell empty is rejected before the acceptance
        // draw, so no uniform is consumed.
        if !known_empty && scratch.proposed.all_occupied() {
            #[allow(unused_mut)]
            let mut log_alpha = self.family.log_marginal(
                &scratch.proposed,
                #[cfg(test)]
                breakage,
            ) - self.family.log_marginal(
                &scratch.current,
                #[cfg(test)]
                breakage,
            ) + proposal.log_structure_ratio
                + moves::log_selection_ratio(m, current, &scratch.tessellation, &self.prior);
            #[cfg(test)]
            {
                log_alpha += crate::broken::log_alpha_shift(
                    breakage,
                    m,
                    current,
                    &scratch.tessellation,
                    &self.prior,
                    self.family.cell_normaliser(),
                );
            }
            debug_assert!(!log_alpha.is_nan());
            let u = rng::uniform(rng);
            if maths::ln(u) < log_alpha {
                std::mem::swap(&mut self.tessellations[j], &mut scratch.tessellation);
                std::mem::swap(&mut self.assignments[j], &mut scratch.assignment);
                std::mem::swap(&mut scratch.current, &mut scratch.proposed);
                soft_weights = proposed_weights;
            }
        }
        if tau.is_some() {
            self.update_bandwidth(
                scratch,
                j,
                x,
                input,
                weights,
                &mut soft_weights,
                rng,
                #[cfg(test)]
                breakage,
            );
        }
        self.redraw(
            scratch,
            j,
            x,
            input,
            soft_weights.as_deref(),
            Some(weights),
            rng,
        );
    }

    /// The bandwidth move of soft tessellation `j`: a random-walk
    /// Metropolis step on ln tau with the cell values integrated out,
    /// prior tau ~ Exponential(rate). Draw order: the proposal normal,
    /// then the acceptance uniform.
    #[allow(clippy::too_many_arguments)]
    fn update_bandwidth(
        &mut self,
        scratch: &mut Scratch<F::Stats>,
        j: usize,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        soft_weights: &mut Option<Vec<f64>>,
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        let rate = self
            .bandwidth_rate
            .expect("a soft ensemble carries a bandwidth prior");
        let tau = self.tessellations[j].tau.expect("a soft tessellation");
        let proposed_tau = tau * maths::exp(BANDWIDTH_STEP * rng::standard_normal(rng));
        let proposed_weights = self.assignments[j].soft_weights(proposed_tau);
        self.family.accumulate(
            &self.assignments[j].cells,
            input,
            weights,
            &scratch.partials,
            self.tessellations[j].n_cells(),
            &Context {
                x,
                tessellation: &self.tessellations[j],
                soft: Some(&proposed_weights),
                uniform: None,
            },
            &mut scratch.proposed,
        );
        // The exponential prior ratio and the Jacobian of the log-scale
        // walk, tau' / tau.
        let log_alpha = self.family.log_marginal(
            &scratch.proposed,
            #[cfg(test)]
            breakage,
        ) - self.family.log_marginal(
            &scratch.current,
            #[cfg(test)]
            breakage,
        ) - rate * (proposed_tau - tau)
            + maths::ln(proposed_tau)
            - maths::ln(tau);
        debug_assert!(!log_alpha.is_nan());
        if maths::ln(rng::uniform(rng)) < log_alpha {
            self.tessellations[j].tau = Some(proposed_tau);
            *soft_weights = Some(proposed_weights);
            std::mem::swap(&mut scratch.current, &mut scratch.proposed);
        }
    }

    /// The cell values of tessellation `j` from the statistics in
    /// `scratch.current`, then the running total. Given the sweep's
    /// `weights`, when tessellation `j` and the next are both hard with the
    /// constant basis, the next one's partials and statistics are formed
    /// in the same pass and `scratch.primed` is set.
    #[allow(clippy::too_many_arguments)]
    fn redraw(
        &mut self,
        scratch: &mut Scratch<F::Stats>,
        j: usize,
        x: &Data,
        input: &[f64],
        soft: Option<&[f64]>,
        weights: Option<&[f64]>,
        rng: &mut Rng,
    ) {
        let tessellation = &mut self.tessellations[j];
        let cells = &self.assignments[j].cells;
        self.family.draw(
            &scratch.current,
            rng,
            &mut scratch.values,
            &mut scratch.slopes,
        );
        std::mem::swap(&mut tessellation.mus, &mut scratch.values);
        std::mem::swap(&mut tessellation.betas, &mut scratch.slopes);
        let b = tessellation.n_cells();
        let n = input.len();
        let hard = |t: &Tessellation| t.tau.is_none() && t.betas.is_empty();
        if soft.is_none() && tessellation.betas.is_empty() {
            let tessellation = &self.tessellations[j];
            if let (Some(weights), Some(next)) = (weights, self.tessellations.get(j + 1)) {
                if hard(next) {
                    let context = Context {
                        x,
                        tessellation: next,
                        soft: None,
                        uniform: uniform_of(&self.running),
                    };
                    self.family
                        .begin(next.n_cells(), &context, &mut scratch.current);
                    self.family.hard_totals_then_partials(
                        &cells[..n],
                        &tessellation.mus,
                        &self.assignments[j + 1].cells[..n],
                        &next.mus,
                        input,
                        &weights[..n],
                        &mut self.total[..n],
                        &mut scratch.partials[..n],
                        &context,
                        &mut scratch.current,
                    );
                    scratch.primed = true;
                    return;
                }
            }
            let (total, partials, cells) =
                (&mut self.total[..n], &scratch.partials[..n], &cells[..n]);
            let mus = &tessellation.mus;
            for i in 0..n {
                total[i] = self.family.total(input[i], partials[i], mus[cells[i]]);
            }
            return;
        }
        let (total, partials, cells) = (&mut self.total[..n], &scratch.partials[..n], &cells[..n]);
        for i in 0..n {
            let own = match soft {
                Some(w) => soft_value(&tessellation.mus, &w[i * b..(i + 1) * b]),
                None => tessellation.training_value(cells[i], x, i),
            };
            total[i] = self.family.total(input[i], partials[i], own);
        }
    }
}

/// The kernel-weighted sum of the cell values at one observation.
fn soft_value(mus: &[f64], weights: &[f64]) -> f64 {
    mus.iter().zip(weights).map(|(&mu, &w)| mu * w).sum()
}

#[cfg(test)]
impl<F: CellFamily> Ensemble<F> {
    /// One sweep with the structural moves disabled: every tessellation's
    /// cell values given its current structure. On fixed tessellations the
    /// chain is the conjugate Gibbs sampler of the known-answer tests.
    pub(crate) fn conjugate_sweep(
        &mut self,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
    ) {
        let mut scratch = std::mem::take(&mut self.scratch);
        running_sums(weights, &mut self.running);
        for j in 0..self.tessellations.len() {
            let tau = self.tessellations[j].tau;
            let soft = tau.map(|tau| self.assignments[j].soft_weights(tau));
            self.partials(&mut scratch, j, x, input, weights, soft.as_deref());
            self.redraw(&mut scratch, j, x, input, soft.as_deref(), None, rng);
        }
        self.scratch = scratch;
    }

    /// Replace tessellation `j` and rebuild its cache; the caller resets
    /// the total.
    pub(crate) fn set_tessellation(&mut self, j: usize, x: &Data, t: Tessellation, total: f64) {
        self.assignments[j] = Assignment::full(x, &t, &self.prior.geometry);
        self.tessellations[j] = t;
        self.total.iter_mut().for_each(|v| *v = total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn weight() -> impl Strategy<Value = f64> {
        prop::num::f64::ANY
    }

    proptest! {
        /// Entry c is c copies added in turn from zero, and nothing is
        /// formed unless every weight has the same bits.
        #[test]
        fn running_sums_are_the_in_order_sums_of_equal_weights(
            w in weight(),
            n in 0usize..64,
            other in prop::option::of((0usize..64, weight())),
        ) {
            let mut weights = vec![w; n];
            if let Some((i, v)) = other {
                if i < n {
                    weights[i] = v;
                }
            }
            let mut out = vec![1.0];
            running_sums(&weights, &mut out);
            let equal = weights.iter().all(|v| v.to_bits() == w.to_bits());
            if n == 0 || !equal {
                prop_assert!(out.is_empty());
            } else {
                let want: Vec<u64> = (0..=n)
                    .map(|c| weights[..c].iter().fold(0.0, |s, v| s + v).to_bits())
                    .collect();
                let got: Vec<u64> = out.iter().map(|v| v.to_bits()).collect();
                prop_assert_eq!(got, want);
            }
        }
    }
}
