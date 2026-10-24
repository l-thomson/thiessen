//! The observation models, one module each: the model statement, its
//! priors, what is fixed rather than estimated, the correspondence of its
//! parameters with the paper and with the BART-family reference
//! implementation, and the prediction semantics of the fitted model. The
//! model is selected by [`Config::outcome`]; every model runs through
//! [`fit`] and the [`Sampler`].

use crate::chain_seed;
use crate::config::Config;
use crate::data::Data;
use crate::error::Result;
use crate::fitted::Fitted;
use crate::sampler::Sampler;
use crate::state::Init;

#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod aft;
#[cfg(feature = "experimental")]
pub(crate) mod censoring;
pub mod gaussian;
pub mod heteroscedastic;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod interval_censored;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod laplace;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod ordinal;
pub mod probit;
#[cfg(feature = "experimental")]
pub(crate) mod scale_mixture;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod student_t;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod tobit;

/// Fit the model the configuration names: validate, run `burn_in` sweeps,
/// then keep every `thinning`-th of the next `draws * thinning` sweeps.
///
/// # Arguments
///
/// `x` is n by p, `y` has n rows, both on the caller's scale (labels in
/// {0, 1} under the probit model); `seed` keys the chain RNG.
///
/// # Errors
///
/// [`Sampler::new`].
pub fn fit(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Fitted> {
    fit_with_progress(config, x, y, seed, |_, _| {})
}

/// Fit as [`fit`], calling `progress(completed, total)` after every sweep.
///
/// # Arguments
///
/// As [`fit`]. `total` is `burn_in + draws * thinning`, `completed` counts
/// sweeps from one, and the draws do not depend on `progress`.
///
/// # Errors
///
/// [`Sampler::new`].
pub fn fit_with_progress(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    mut progress: impl FnMut(usize, usize),
) -> Result<Fitted> {
    run(config, x, y, seed, &mut progress)
}

/// Fit `n_chains` chains of the model the configuration names, seeded by
/// [`chain_seed`]`(seed, k)` for k below `n_chains`,
/// and pool their draws in chain order.
///
/// # Arguments
///
/// As [`fit`]. The second value is the posterior mean at the rows of
/// `x`, the fitted values, from the one prediction pass that also gives
/// the pooled in-sample RMSE ([`Fitted::pool_samplers`]).
///
/// # Errors
///
/// [`Sampler::new`]; `InvalidHyperparameter` for `n_chains` of zero.
pub fn fit_chains(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    n_chains: usize,
) -> Result<(Fitted, Vec<f64>)> {
    fit_chains_with_threads(config, x, y, seed, n_chains, 1)
}

/// Fit as [`fit_chains`], the chains spread over at most `n_threads`
/// threads, and at most the parallelism available to the process
/// ([`Sampler::advance_all`]). The draws do not depend on
/// `n_threads`: each chain runs on one thread with its own generator. The
/// fitted model's [`threads`](Fitted::threads) is `n_threads`, so its
/// predictions use the same count.
///
/// # Errors
///
/// [`fit_chains`].
pub fn fit_chains_with_threads(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
) -> Result<(Fitted, Vec<f64>)> {
    fit_chains_with_init(config, x, y, seed, n_chains, n_threads, Init::Cold)
}

/// Fit as [`fit_chains_with_threads`], each chain starting where `init`
/// says: the cold start, or a kept draw of a previous fit of the same
/// model to the same data ([`Init::From`], [`Sampler::start_from`]).
/// Under the cold start the draws are those of
/// [`fit_chains_with_threads`] bit for bit. A warm start is experimental
/// (`docs/experimental.md`).
///
/// # Errors
///
/// [`fit_chains`]; under [`Init::From`], `RequiresFeature` in a build
/// without the feature, then [`Sampler::start_from`].
pub fn fit_chains_with_init(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
    init: Init<'_>,
) -> Result<(Fitted, Vec<f64>)> {
    let mut samplers = chain_samplers(n_chains, |k| {
        Sampler::new(config, x, y, chain_seed(seed, k))
    })?;
    start_chains(&mut samplers, init)?;
    fit_samplers_with_threads(samplers, config, x, y, n_threads)
}

/// Fit the AFT model: as [`fit`], with the times and the event
/// indicator in place of a plain response (`Outcome::Aft`).
/// Experimental (`docs/experimental.md`): present in every build, served
/// only with the `experimental` feature.
///
/// # Arguments
///
/// `x` is n by p; `times` has n positive event or censoring times and
/// `events` one flag per row (true is an event, false right-censoring);
/// `seed` keys the chain RNG.
///
/// # Errors
///
/// `RequiresFeature` in a build without the feature; otherwise
/// [`Sampler::aft`].
pub fn fit_aft(
    config: &Config,
    x: &Data,
    times: &[f64],
    events: &[bool],
    seed: u64,
) -> Result<Fitted> {
    fit_aft_chains_with_threads(config, x, times, events, seed, 1, 1).map(|(fitted, _)| fitted)
}

/// Fit the AFT model as [`fit_chains_with_threads`]: `n_chains` chains
/// seeded by [`chain_seed`]`(seed, k)`, spread over at most `n_threads`
/// threads, their draws pooled in chain order; the second value is the
/// posterior mean at the rows of `x`. Experimental
/// (`docs/experimental.md`): present in every build, served only with
/// the `experimental` feature.
///
/// # Errors
///
/// `RequiresFeature` in a build without the feature; otherwise
/// [`Sampler::aft`] and [`fit_chains`].
#[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
pub fn fit_aft_chains_with_threads(
    config: &Config,
    x: &Data,
    times: &[f64],
    events: &[bool],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
) -> Result<(Fitted, Vec<f64>)> {
    fit_aft_chains_with_init(
        config,
        x,
        times,
        events,
        seed,
        n_chains,
        n_threads,
        Init::Cold,
    )
}

/// Fit the AFT model as [`fit_aft_chains_with_threads`], each chain
/// starting where `init` says ([`fit_chains_with_init`]).
/// Experimental (`docs/experimental.md`).
///
/// # Errors
///
/// [`fit_aft_chains_with_threads`]; under [`Init::From`],
/// [`Sampler::start_from`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
pub fn fit_aft_chains_with_init(
    config: &Config,
    x: &Data,
    times: &[f64],
    events: &[bool],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
    init: Init<'_>,
) -> Result<(Fitted, Vec<f64>)> {
    #[cfg(not(feature = "experimental"))]
    return Err(crate::config::Gated::AFT.requires_feature());
    #[cfg(feature = "experimental")]
    {
        let mut samplers = chain_samplers(n_chains, |k| {
            Sampler::aft(config, x, times, events, chain_seed(seed, k))
        })?;
        start_chains(&mut samplers, init)?;
        let y = samplers[0].training_response();
        fit_samplers_with_threads(samplers, config, x, &y, n_threads)
    }
}

/// Fit the interval-censored model: as [`fit`], with one pair of bounds
/// per row in place of a plain response (`Outcome::IntervalCensored`).
/// Experimental (`docs/experimental.md`): present in every build, served
/// only with the `experimental` feature.
///
/// # Arguments
///
/// `x` is n by p; `lower` and `upper` hold n bounds each (an equal pair
/// is an exact value, an infinite endpoint one-sided censoring); `seed`
/// keys the chain RNG.
///
/// # Errors
///
/// `RequiresFeature` in a build without the feature; otherwise
/// [`Sampler::interval_censored`].
pub fn fit_interval_censored(
    config: &Config,
    x: &Data,
    lower: &[f64],
    upper: &[f64],
    seed: u64,
) -> Result<Fitted> {
    fit_interval_censored_chains_with_threads(config, x, lower, upper, seed, 1, 1)
        .map(|(fitted, _)| fitted)
}

/// Fit the interval-censored model as [`fit_chains_with_threads`]:
/// `n_chains` chains seeded by [`chain_seed`]`(seed, k)`, spread over at
/// most `n_threads` threads, their draws pooled in chain order; the
/// second value is the posterior mean at the rows of `x`. Experimental
/// (`docs/experimental.md`): present in every build, served only with
/// the `experimental` feature.
///
/// # Errors
///
/// `RequiresFeature` in a build without the feature; otherwise
/// [`Sampler::interval_censored`] and [`fit_chains`].
#[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
pub fn fit_interval_censored_chains_with_threads(
    config: &Config,
    x: &Data,
    lower: &[f64],
    upper: &[f64],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
) -> Result<(Fitted, Vec<f64>)> {
    fit_interval_censored_chains_with_init(
        config,
        x,
        lower,
        upper,
        seed,
        n_chains,
        n_threads,
        Init::Cold,
    )
}

/// Fit the interval-censored model as
/// [`fit_interval_censored_chains_with_threads`], each chain starting
/// where `init` says ([`fit_chains_with_init`]). Experimental
/// (`docs/experimental.md`).
///
/// # Errors
///
/// [`fit_interval_censored_chains_with_threads`]; under [`Init::From`],
/// [`Sampler::start_from`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "experimental"), allow(unused_variables))]
pub fn fit_interval_censored_chains_with_init(
    config: &Config,
    x: &Data,
    lower: &[f64],
    upper: &[f64],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
    init: Init<'_>,
) -> Result<(Fitted, Vec<f64>)> {
    #[cfg(not(feature = "experimental"))]
    return Err(crate::config::Gated::INTERVAL_CENSORED.requires_feature());
    #[cfg(feature = "experimental")]
    {
        let mut samplers = chain_samplers(n_chains, |k| {
            Sampler::interval_censored(config, x, lower, upper, chain_seed(seed, k))
        })?;
        start_chains(&mut samplers, init)?;
        let y = samplers[0].training_response();
        fit_samplers_with_threads(samplers, config, x, &y, n_threads)
    }
}

/// Start each sampler where `init` says ([`Sampler::start_all`]).
fn start_chains(samplers: &mut [Sampler], init: Init<'_>) -> Result<()> {
    let mut chains: Vec<&mut Sampler> = samplers.iter_mut().collect();
    Sampler::start_all(&mut chains, init)
}

/// One sampler per chain, chain k on [`chain_seed`]`(seed, k)`.
fn chain_samplers(
    n_chains: usize,
    sampler: impl Fn(usize) -> Result<Sampler>,
) -> Result<Vec<Sampler>> {
    if n_chains == 0 {
        return Err(crate::error::invalid(
            "n_chains",
            "a fit needs at least one chain",
        ));
    }
    (0..n_chains).map(sampler).collect()
}

/// Run the configured schedule over `samplers`, one chain each, over at
/// most `n_threads` threads, and pool their draws; the in-sample RMSE is
/// measured against `y`.
fn fit_samplers_with_threads(
    mut samplers: Vec<Sampler>,
    config: &Config,
    x: &Data,
    y: &[f64],
    n_threads: usize,
) -> Result<(Fitted, Vec<f64>)> {
    let schedule = &config.general_params;
    let mut chains: Vec<&mut Sampler> = samplers.iter_mut().collect();
    Sampler::advance_all(
        &mut chains,
        schedule.burn_in,
        schedule.draws,
        schedule.thinning,
        n_threads,
    );
    let (mut fitted, values) = Fitted::pool_samplers(samplers, x, y)?;
    fitted.set_threads(n_threads);
    Ok((fitted, values))
}

/// The sweep schedule shared by every model.
pub(crate) fn run(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    run_schedule(Sampler::new(config, x, y, seed)?, config, progress)
}

/// Run a constructed sampler through the configured schedule and finish
/// it.
fn run_schedule(
    mut sampler: Sampler,
    config: &Config,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    advance(&mut sampler, config, progress);
    sampler.finish()
}

/// Advance a constructed sampler through the configured schedule.
fn advance(sampler: &mut Sampler, config: &Config, progress: &mut dyn FnMut(usize, usize)) {
    let schedule = config.general_params.clone();
    let total = schedule.burn_in + schedule.draws * schedule.thinning;
    let mut completed = 0;
    for _ in 0..schedule.burn_in {
        sampler.step();
        completed += 1;
        progress(completed, total);
    }
    for _ in 0..schedule.draws {
        for _ in 0..schedule.thinning {
            sampler.step();
            completed += 1;
            progress(completed, total);
        }
        sampler.keep();
    }
}
