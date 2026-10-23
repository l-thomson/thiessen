//! One chain's state at one kept draw, and where the chains of a fit
//! start.
//!
//! A chain can start from a kept draw of a previous fit of the same
//! model to the same training data instead of the cold start. Only the
//! initial distribution changes: the transition kernel and its
//! stationary distribution are those of the cold-started chain, so a
//! warm start needs no new validity argument (Roberts and Rosenthal,
//! 2004). The precedents are stochtree's `previous_model_json` warm
//! start of BART, CmdStanR's `init = <fit>` and the many short chains
//! from distinct retained draws of He and Hahn (2023, section 5).
//!
//! Experimental (`docs/experimental.md`): the types exist in every build,
//! and using a state needs the `experimental` feature.
//!
//! # References
//!
//! - He, J. and Hahn, P. R. (2023). Stochastic tree ensembles for
//!   regularized nonlinear regression. Journal of the American
//!   Statistical Association 118(541), 551-570.
//! - Roberts, G. O. and Rosenthal, J. S. (2004). General state space
//!   Markov chains and MCMC algorithms. Probability Surveys 1, 20-71.

use serde::{Deserialize, Serialize};

use crate::fitted::Fitted;
use crate::tessellation::Tessellation;

/// The sampled state of one chain at one draw, on the sampler's scaled
/// space: what [`Posterior`](crate::Posterior) stores per kept draw.
///
/// The latent responses of the augmented outcomes and the Student-t and
/// Laplace scale-mixture weights are not part of it: the first sweep
/// after a restore redraws them from their full conditionals before
/// anything conditions on them, a valid Gibbs step. Neither is the
/// generator's state, which comes from the seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsembleState {
    tessellations: Vec<Tessellation>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    variance_tessellations: Vec<Tessellation>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    sigma_sq: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    cutpoints: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    df: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    inclusion: Option<(Vec<f64>, f64)>,
}

impl EnsembleState {
    pub(crate) fn new(
        tessellations: Vec<Tessellation>,
        variance_tessellations: Vec<Tessellation>,
        sigma_sq: Option<f64>,
        cutpoints: Vec<f64>,
        df: Option<f64>,
        inclusion: Option<(Vec<f64>, f64)>,
    ) -> Self {
        Self {
            tessellations,
            variance_tessellations,
            sigma_sq,
            cutpoints,
            df,
            inclusion,
        }
    }

    /// The mean tessellations.
    pub fn tessellations(&self) -> &[Tessellation] {
        &self.tessellations
    }

    /// The variance tessellations; empty outside the heteroscedastic
    /// model.
    pub fn variance_tessellations(&self) -> &[Tessellation] {
        &self.variance_tessellations
    }

    /// The global sigma^2, where the model draws one.
    pub fn sigma_sq(&self) -> Option<f64> {
        self.sigma_sq
    }

    /// The free cutpoints of the ordinal model; empty otherwise.
    pub fn cutpoints(&self) -> &[f64] {
        &self.cutpoints
    }

    /// The Student-t degrees of freedom, where they are drawn on a grid.
    pub fn df(&self) -> Option<f64> {
        self.df
    }

    /// The DART inclusion weights and concentration, where the structure
    /// prior samples them.
    pub fn inclusion(&self) -> Option<(&[f64], f64)> {
        self.inclusion
            .as_ref()
            .map(|(s, theta)| (s.as_slice(), *theta))
    }
}

/// Where each chain of a multi-chain fit starts.
#[derive(Debug, Clone, Copy, Default)]
pub enum Init<'a> {
    /// The cold start: one single-cell tessellation per slot, every cell
    /// at the mean response.
    #[default]
    Cold,
    /// Kept draws of a previous fit of the same model to the same data.
    /// Chain k (from 0) starts from draw `draw - k`, `draw` defaulting
    /// to the last; when the fit holds fewer than `n_chains` draws up to
    /// `draw`, every chain starts from `draw` and the fit records
    /// [`Warning::WarmStartDrawsReused`](crate::Warning::WarmStartDrawsReused).
    /// This is stochtree's rule. Experimental: `RequiresFeature` without
    /// the feature.
    From {
        /// The previous fit.
        fitted: &'a Fitted,
        /// The 0-based pooled draw chain 0 starts from; `None` for the
        /// last.
        draw: Option<usize>,
    },
    /// Chain k starts from kept draw `draws[k]` (0-based, pooled order)
    /// of a previous fit of the same model to the same data; `draws` has
    /// one entry per chain. A binding continuing every chain of a fit
    /// from its own last draw names them here. Experimental:
    /// `RequiresFeature` without the feature.
    Draws {
        /// The previous fit.
        fitted: &'a Fitted,
        /// The starting draw of each chain.
        draws: &'a [usize],
    },
}

/// The draw each of `n_chains` chains starts from under [`Init::From`],
/// and whether every chain shares one draw, for a fit of `n_draws`
/// draws.
#[cfg_attr(not(feature = "experimental"), allow(dead_code))]
pub(crate) fn starting_draws(
    n_draws: usize,
    n_chains: usize,
    draw: Option<usize>,
) -> crate::Result<(Vec<usize>, bool)> {
    if n_draws == 0 {
        return Err(crate::error::invalid(
            "init",
            "the previous fit has no draws",
        ));
    }
    let first = draw.unwrap_or(n_draws - 1);
    if first >= n_draws {
        return Err(crate::error::invalid(
            "init_draw",
            format!("draw {first} of a fit with {n_draws} draws"),
        ));
    }
    if n_chains > first + 1 {
        return Ok((vec![first; n_chains], true));
    }
    Ok(((0..n_chains).map(|k| first - k).collect(), false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_count_back_from_the_starting_draw() {
        assert_eq!(
            starting_draws(10, 4, None).unwrap(),
            (vec![9, 8, 7, 6], false)
        );
        assert_eq!(
            starting_draws(10, 4, Some(5)).unwrap(),
            (vec![5, 4, 3, 2], false)
        );
        assert_eq!(
            starting_draws(10, 4, Some(3)).unwrap(),
            (vec![3, 2, 1, 0], false)
        );
    }

    #[test]
    fn too_few_draws_share_the_starting_draw() {
        assert_eq!(starting_draws(10, 4, Some(2)).unwrap(), (vec![2; 4], true));
        assert_eq!(starting_draws(1, 3, None).unwrap(), (vec![0; 3], true));
    }

    #[test]
    fn out_of_range_draws_are_refused() {
        assert!(starting_draws(0, 1, None).is_err());
        assert!(starting_draws(5, 1, Some(5)).is_err());
    }
}
