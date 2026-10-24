//! extendr bindings over the `thiessen` core: the configuration as JSON,
//! the design as an R matrix, and the fitted model as a live handle behind
//! an external pointer.
//!
//! The configuration crosses the boundary as JSON so that the core's serde
//! representation stays the single definition of the field names, their
//! defaults and their validation. The fitted model stays on the Rust side:
//! methods take the handle, and the state is encoded to bytes only for
//! persistence, as MessagePack with field names, so the serde
//! representation stays the single definition there too. A double survives
//! the byte encoding bit for bit, which text cannot promise.

use extendr_api::prelude::*;
use extendr_api::Result;
use std::cell::{Ref, RefCell, RefMut};

/// Leads the message of a feature error. The extendr error channel
/// carries a string, so the condition class travels as this prefix and
/// `core_call()` in R takes the class from it.
const REQUIRES_FEATURE: &str = "thiessen_requires_feature: ";

fn core_error(error: thiessen::Error) -> Error {
    match error {
        thiessen::Error::RequiresFeature { .. } => {
            Error::Other(format!("{REQUIRES_FEATURE}{error}"))
        }
        _ => Error::Other(error.to_string()),
    }
}

fn json_error(error: serde_json::Error) -> Error {
    Error::Other(error.to_string())
}

fn config(json: &str) -> Result<thiessen::Config> {
    serde_json::from_str(json).map_err(json_error)
}

/// The fitted model, alive behind an external pointer. R releases it
/// through the pointer's finalizer.
struct FittedHandle {
    inner: thiessen::Fitted,
}

/// The design in row-major order; `RMatrix::data` is column-major.
fn design(x: &RMatrix<f64>) -> Result<thiessen::Data> {
    let (n_rows, n_cols) = (x.nrows(), x.ncols());
    let column_major = x.data();
    let mut values = Vec::with_capacity(n_rows * n_cols);
    for row in 0..n_rows {
        for col in 0..n_cols {
            values.push(column_major[col * n_rows + row]);
        }
    }
    thiessen::Data::new(values, n_rows, n_cols).map_err(core_error)
}

/// Draw-major rows as a matrix of one row per draw, as `dbarts` returns.
fn draw_matrix(draws: &[Vec<f64>]) -> RMatrix<f64> {
    let n_cols = draws.first().map_or(0, Vec::len);
    RMatrix::new_matrix(draws.len(), n_cols, |row, col| draws[row][col])
}

/// A whole non-negative double as the core's `u64` seed.
fn seed(seed: f64) -> Result<u64> {
    if !seed.is_finite() || seed < 0.0 || seed.fract() != 0.0 || seed > 9_007_199_254_740_992.0 {
        return Err(Error::Other(format!(
            "seed must be a whole number in [0, 2^53]; got {seed}"
        )));
    }
    Ok(seed as u64)
}

/// Whether the core was built with the `experimental` feature.
#[extendr]
fn core_experimental() -> bool {
    cfg!(feature = "experimental")
}

/// The prefix a feature error's message carries.
#[extendr]
fn core_requires_feature_prefix() -> &'static str {
    REQUIRES_FEATURE
}

/// Version of the vendored core crate.
#[extendr]
fn core_version() -> &'static str {
    thiessen::VERSION
}

/// The core's default configuration as JSON.
#[extendr]
fn core_defaults() -> Result<String> {
    serde_json::to_string(&thiessen::Config::new()).map_err(json_error)
}

/// The default parameters of each outcome family the build carries, as
/// JSON.
#[extendr]
fn core_outcome_defaults() -> Result<String> {
    serde_json::to_string(&thiessen::Outcome::catalogue()).map_err(json_error)
}

/// Validate a configuration without data.
#[extendr]
fn core_validate(config_json: &str) -> Result<()> {
    config(config_json)?.validate().map_err(core_error)
}

/// Whether `state` is a fitted-model pointer with a live address. A
/// pointer read back by `readRDS` deserialises with a null address, so a
/// caller checks here and restores from the payload before using it.
#[extendr]
fn core_state_is_live(state: Robj) -> bool {
    <&ExternalPtr<FittedHandle>>::try_from(&state).is_ok()
}

/// The fitted state as bytes, for persistence.
#[extendr]
fn core_state_payload(state: ExternalPtr<FittedHandle>) -> Result<Raw> {
    let bytes = rmp_serde::encode::to_vec_named(&state.inner)
        .map_err(|error| Error::Other(error.to_string()))?;
    Ok(Raw::from_bytes(&bytes))
}

/// A live fitted-model pointer from the bytes of `core_state_payload`,
/// predicting on `threads` threads.
#[extendr]
fn core_state_restore(payload: &[u8], threads: i32) -> Result<ExternalPtr<FittedHandle>> {
    let mut inner = thiessen::Fitted::load(&mut rmp_serde::Deserializer::from_read_ref(payload))
        .map_err(core_error)?;
    inner.set_threads(threads.max(1) as usize);
    Ok(ExternalPtr::new(FittedHandle { inner }))
}

/// Set the number of threads the predictions of `state` run on.
#[extendr]
fn core_state_set_threads(mut state: ExternalPtr<FittedHandle>, threads: i32) {
    state.inner.set_threads(threads.max(1) as usize);
}

/// The posterior mean at each row of `x`.
#[extendr]
fn core_predict(state: ExternalPtr<FittedHandle>, x: RMatrix<f64>) -> Result<Vec<f64>> {
    state.inner.predict(&design(&x)?).map_err(core_error)
}

/// Per-draw predictions: `kind` selects the quantity of `predict`, the mean
/// function, or the variance of y given f.
#[extendr]
fn core_predict_draws(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    kind: &str,
) -> Result<RMatrix<f64>> {
    let fitted = &state.inner;
    let data = design(&x)?;
    let draws = match kind {
        "draws" => fitted.predict_draws(&data),
        "latent" => fitted.predict_latent(&data),
        "variance" => fitted.predict_variance(&data),
        other => return Err(Error::Other(format!("unknown draw kind {other}"))),
    }
    .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// Central credible or posterior predictive interval, `n_rows` by 2.
#[extendr]
fn core_interval(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    kind: &str,
    level: f64,
) -> Result<RMatrix<f64>> {
    let fitted = &state.inner;
    let data = design(&x)?;
    let intervals = match kind {
        "credible" => fitted.credible_interval(&data, level),
        "prediction" => fitted.prediction_interval(&data, level),
        other => return Err(Error::Other(format!("unknown interval kind {other}"))),
    }
    .map_err(core_error)?;
    Ok(RMatrix::new_matrix(intervals.len(), 2, |row, col| {
        if col == 0 {
            intervals[row].lower
        } else {
            intervals[row].upper
        }
    }))
}

/// Pointwise log-likelihood, one row per draw.
#[extendr]
fn core_log_lik(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    y: &[f64],
) -> Result<RMatrix<f64>> {
    let draws = state
        .inner
        .log_likelihood(&design(&x)?, y)
        .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// Pointwise survival log-likelihood under the AFT model, one row per
/// draw.
#[extendr]
fn core_log_lik_survival(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    times: &[f64],
    events: Robj,
) -> Result<RMatrix<f64>> {
    let events = Vec::<bool>::try_from(events)?;
    let draws = state
        .inner
        .log_likelihood_survival(&design(&x)?, times, &events)
        .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// Pointwise interval log-likelihood under the interval-censored model,
/// one row per draw.
#[extendr]
fn core_log_lik_interval_censored(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    lower: &[f64],
    upper: &[f64],
) -> Result<RMatrix<f64>> {
    let draws = state
        .inner
        .log_likelihood_interval_censored(&design(&x)?, lower, upper)
        .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// Posterior-mean category probabilities under the ordinal model, one
/// row per row of `x` and one column per category.
#[extendr]
fn core_predict_probs(state: ExternalPtr<FittedHandle>, x: RMatrix<f64>) -> Result<RMatrix<f64>> {
    let rows = state
        .inner
        .predict_category_probabilities(&design(&x)?)
        .map_err(core_error)?;
    Ok(draw_matrix(&rows))
}

/// The posterior mean beside a central interval, `n_rows` by 3 (fit,
/// lower, upper), from one traversal of the draws.
#[extendr]
fn core_predict_interval(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    kind: &str,
    level: f64,
) -> Result<RMatrix<f64>> {
    let data = design(&x)?;
    let kind = match kind {
        "credible" => thiessen::IntervalKind::Credible,
        "prediction" => thiessen::IntervalKind::Prediction,
        other => return Err(Error::Other(format!("unknown interval kind {other}"))),
    };
    let (fit, intervals) = state
        .inner
        .predict_with_interval(&data, kind, level)
        .map_err(core_error)?;
    Ok(RMatrix::new_matrix(fit.len(), 3, |row, col| match col {
        0 => fit[row],
        1 => intervals[row].lower,
        _ => intervals[row].upper,
    }))
}

/// sigma per kept draw; empty outside the Gaussian model.
#[extendr]
fn core_sigma(state: ExternalPtr<FittedHandle>) -> Result<Vec<f64>> {
    Ok(state.inner.sigma())
}

/// The per-draw quantities the experimental models sample: the error
/// degrees of freedom, the interior cutpoints, the soft-membership
/// bandwidths, the DART inclusion weights and their concentration. Each
/// is empty under a model that does not sample it.
#[extendr]
fn core_posterior_draws(state: ExternalPtr<FittedHandle>) -> Result<List> {
    let fitted = &state.inner;
    Ok(list!(
        df = fitted.posterior().dfs().to_vec(),
        cutpoint = draw_matrix(fitted.cutpoint_draws()),
        bandwidth = draw_matrix(&fitted.bandwidth_draws()),
        inclusion_weight = draw_matrix(fitted.inclusion_weight_draws()),
        concentration = fitted.concentration_draws().to_vec()
    ))
}

/// The per-draw ensemble summaries and the covariate inclusion shares.
#[extendr]
fn core_diagnostics(state: ExternalPtr<FittedHandle>) -> Result<List> {
    let fitted = &state.inner;
    Ok(list!(
        cell_count = fitted.cell_counts(),
        dimension_count = fitted.dimension_counts(),
        inclusion = fitted.variable_inclusion_proportions()
    ))
}

/// A live sampler over the core's Gibbs loop, held behind an external
/// pointer. `finish` consumes the state; every later call errors.
struct SamplerHandle {
    inner: RefCell<Option<thiessen::Sampler>>,
    data: thiessen::Data,
}

fn finished() -> Error {
    Error::Other("the sampler is finished".to_string())
}

impl SamplerHandle {
    /// The live sampler.
    fn live(&self) -> Result<Ref<'_, thiessen::Sampler>> {
        Ref::filter_map(self.inner.borrow(), Option::as_ref).map_err(|_| finished())
    }

    /// The live sampler, mutably.
    fn live_mut(&self) -> Result<RefMut<'_, thiessen::Sampler>> {
        RefMut::filter_map(self.inner.borrow_mut(), Option::as_mut).map_err(|_| finished())
    }

    /// The sampler itself, leaving the handle finished.
    fn take(&self) -> Result<thiessen::Sampler> {
        self.inner.borrow_mut().take().ok_or_else(finished)
    }

    /// A handle over a constructed sampler.
    fn new(inner: thiessen::Sampler, data: thiessen::Data) -> ExternalPtr<Self> {
        ExternalPtr::new(SamplerHandle {
            inner: RefCell::new(Some(inner)),
            data,
        })
    }
}

/// The seed the core derives for chain `chain` of `seed_value`, so
/// driving the configured schedule by hand reproduces that chain of a
/// fit bit for bit.
fn chain_seed(seed_value: f64, chain: i32) -> Result<u64> {
    Ok(thiessen::chain_seed(
        seed(seed_value)?,
        chain.max(0) as usize,
    ))
}

/// Construct a sampler over a plain response on the seed of chain
/// `chain`.
#[extendr]
fn core_sampler_new(
    config_json: &str,
    x: RMatrix<f64>,
    y: &[f64],
    seed_value: f64,
    chain: i32,
) -> Result<ExternalPtr<SamplerHandle>> {
    let config = config(config_json)?;
    let data = design(&x)?;
    let inner = thiessen::Sampler::new(&config, &data, y, chain_seed(seed_value, chain)?)
        .map_err(core_error)?;
    Ok(SamplerHandle::new(inner, data))
}

/// Construct a sampler for the AFT model over `times` and `events` on
/// the seed of chain `chain`.
#[extendr]
fn core_sampler_new_aft(
    config_json: &str,
    x: RMatrix<f64>,
    times: &[f64],
    events: Robj,
    seed_value: f64,
    chain: i32,
) -> Result<ExternalPtr<SamplerHandle>> {
    let config = config(config_json)?;
    let data = design(&x)?;
    let events = Vec::<bool>::try_from(events)?;
    let inner = thiessen::Sampler::aft(
        &config,
        &data,
        times,
        &events,
        chain_seed(seed_value, chain)?,
    )
    .map_err(core_error)?;
    Ok(SamplerHandle::new(inner, data))
}

/// Construct a sampler for the interval-censored model over the bound
/// pairs on the seed of chain `chain`.
#[extendr]
fn core_sampler_new_interval_censored(
    config_json: &str,
    x: RMatrix<f64>,
    lower: &[f64],
    upper: &[f64],
    seed_value: f64,
    chain: i32,
) -> Result<ExternalPtr<SamplerHandle>> {
    let config = config(config_json)?;
    let data = design(&x)?;
    let inner = thiessen::Sampler::interval_censored(
        &config,
        &data,
        lower,
        upper,
        chain_seed(seed_value, chain)?,
    )
    .map_err(core_error)?;
    Ok(SamplerHandle::new(inner, data))
}

/// Run `n` sweeps of the Gibbs loop.
#[extendr]
fn core_sampler_step(sampler: ExternalPtr<SamplerHandle>, n: i32) -> Result<()> {
    let mut live = sampler.live_mut()?;
    for _ in 0..n.max(0) {
        live.step();
    }
    Ok(())
}

/// Advance every sampler through `burn_in` sweeps and then `draws` kept
/// draws of `thinning` sweeps each, the samplers spread over at most
/// `threads` threads.
#[extendr]
fn core_samplers_advance(
    samplers: List,
    burn_in: i32,
    draws: i32,
    thinning: i32,
    threads: i32,
) -> Result<()> {
    let handles = samplers
        .values()
        .map(ExternalPtr::<SamplerHandle>::try_from)
        .collect::<Result<Vec<_>>>()?;
    let mut guards: Vec<_> = handles
        .iter()
        .map(|handle| handle.inner.borrow_mut())
        .collect();
    let mut live = guards
        .iter_mut()
        .map(|guard| guard.as_mut().ok_or_else(finished))
        .collect::<Result<Vec<_>>>()?;
    thiessen::Sampler::advance_all(
        &mut live,
        burn_in.max(0) as usize,
        draws.max(0) as usize,
        thinning.max(0) as usize,
        threads.max(1) as usize,
    );
    Ok(())
}

/// Start every sampler of a fit from draws of `previous` by the core's
/// rule ([`thiessen::Init::From`]), counting back from the 0-based
/// `draw` when it is not `NULL`.
#[extendr]
fn core_samplers_start(
    samplers: List,
    previous: ExternalPtr<FittedHandle>,
    draw: Robj,
) -> Result<()> {
    let handles = samplers
        .values()
        .map(ExternalPtr::<SamplerHandle>::try_from)
        .collect::<Result<Vec<_>>>()?;
    let mut guards: Vec<_> = handles
        .iter()
        .map(|handle| handle.inner.borrow_mut())
        .collect();
    let mut live = guards
        .iter_mut()
        .map(|guard| guard.as_mut().ok_or_else(finished))
        .collect::<Result<Vec<_>>>()?;
    let draw = if draw.is_null() {
        None
    } else {
        Some(f64::try_from(draw)?.max(0.0) as usize)
    };
    let init = thiessen::Init::From {
        fitted: &previous.inner,
        draw,
    };
    thiessen::Sampler::start_all(&mut live, init).map_err(core_error)
}

/// Record the current state as a posterior draw.
#[extendr]
fn core_sampler_keep(sampler: ExternalPtr<SamplerHandle>) -> Result<()> {
    sampler.live_mut()?.keep();
    Ok(())
}

/// Number of draws kept so far.
#[extendr]
fn core_sampler_n_kept(sampler: ExternalPtr<SamplerHandle>) -> Result<i32> {
    Ok(sampler.live()?.n_kept() as i32)
}

/// Replace the response on the caller's scale.
#[extendr]
fn core_sampler_set_response(sampler: ExternalPtr<SamplerHandle>, y: &[f64]) -> Result<()> {
    sampler.live_mut()?.set_response(y).map_err(core_error)
}

/// Replace the AFT model's times and event indicator.
#[extendr]
fn core_sampler_set_aft_response(
    sampler: ExternalPtr<SamplerHandle>,
    times: &[f64],
    events: Robj,
) -> Result<()> {
    let events = Vec::<bool>::try_from(events)?;
    sampler
        .live_mut()?
        .set_aft_response(times, &events)
        .map_err(core_error)
}

/// Replace the interval-censored model's bound pairs.
#[extendr]
fn core_sampler_set_interval_censored_response(
    sampler: ExternalPtr<SamplerHandle>,
    lower: &[f64],
    upper: &[f64],
) -> Result<()> {
    sampler
        .live_mut()?
        .set_interval_censored_response(lower, upper)
        .map_err(core_error)
}

/// The current mean function at the training rows, caller scale.
#[extendr]
fn core_sampler_fitted_values(sampler: ExternalPtr<SamplerHandle>) -> Result<Vec<f64>> {
    Ok(sampler.live()?.fitted_values())
}

/// The current variance of y given f at each training row, caller scale.
#[extendr]
fn core_sampler_noise_variances(sampler: ExternalPtr<SamplerHandle>) -> Result<Vec<f64>> {
    Ok(sampler.live()?.noise_variances())
}

/// The fitted model from the kept draws of every sampler, their chains
/// pooled, predicting on `threads` threads, with the response the
/// in-sample fit is measured against. Consumes the samplers.
#[extendr]
fn core_finish(samplers: List, threads: i32) -> Result<List> {
    let handles = samplers
        .values()
        .map(ExternalPtr::<SamplerHandle>::try_from)
        .collect::<Result<Vec<_>>>()?;
    let first = handles
        .first()
        .ok_or_else(|| Error::Other("a fit needs at least one chain".to_string()))?;
    // The in-sample fit is measured against the response the core reports,
    // the observed values under the models whose working response is
    // replaced by latents.
    let y = first.live()?.training_response();
    let mut samplers = Vec::with_capacity(handles.len());
    for handle in &handles {
        samplers.push(handle.take()?);
    }
    let n_chains = samplers.len() as i32;
    let (mut fitted, fitted_values) =
        thiessen::Fitted::pool_samplers(samplers, &first.data, &y).map_err(core_error)?;
    fitted.set_threads(threads.max(1) as usize);
    let warnings: Vec<String> = fitted.warnings().iter().map(ToString::to_string).collect();
    Ok(list!(
        config = serde_json::to_string(fitted.config()).map_err(json_error)?,
        model = fitted.model_name().to_string(),
        n_chains = n_chains,
        n_draws = fitted.n_draws() as i32,
        in_sample_rmse = fitted.in_sample_rmse(),
        warnings = warnings,
        fitted_values = fitted_values,
        y = y,
        state = ExternalPtr::new(FittedHandle { inner: fitted })
    ))
}

/// The XAddiVortes fast fit on a live sampler under the settings
/// `params_json` (the core's `XAddiVortesParams` as JSON): replaces the
/// chain's state, so it can be kept as a point fit or advanced as a warm
/// start. Returns the core's `FastFit` as JSON.
#[extendr]
fn core_sampler_xaddivortes(
    sampler: ExternalPtr<SamplerHandle>,
    params_json: &str,
) -> Result<String> {
    let params: thiessen::XAddiVortesParams =
        serde_json::from_str(params_json).map_err(json_error)?;
    let fast = sampler
        .live_mut()?
        .xaddivortes(&params)
        .map_err(core_error)?;
    serde_json::to_string(&fast).map_err(json_error)
}

/// The XAddiVortes fast fit on every sampler, over at most `threads`
/// threads ([`thiessen::Sampler::xaddivortes_all`]).
#[extendr]
fn core_samplers_xaddivortes(samplers: List, params_json: &str, threads: i32) -> Result<()> {
    let params: thiessen::XAddiVortesParams =
        serde_json::from_str(params_json).map_err(json_error)?;
    let handles = samplers
        .values()
        .map(ExternalPtr::<SamplerHandle>::try_from)
        .collect::<Result<Vec<_>>>()?;
    let mut guards: Vec<_> = handles
        .iter()
        .map(|handle| handle.inner.borrow_mut())
        .collect();
    let mut live = guards
        .iter_mut()
        .map(|guard| guard.as_mut().ok_or_else(finished))
        .collect::<Result<Vec<_>>>()?;
    thiessen::Sampler::xaddivortes_all(&mut live, &params, threads.max(1) as usize)
        .map_err(core_error)?;
    Ok(())
}

/// The default XAddiVortes settings as JSON.
#[extendr]
fn core_xaddivortes_defaults() -> Result<String> {
    serde_json::to_string(&thiessen::XAddiVortesParams::default()).map_err(json_error)
}

extendr_module! {
    mod thiessen;
    fn core_version;
    fn core_experimental;
    fn core_requires_feature_prefix;
    fn core_defaults;
    fn core_outcome_defaults;
    fn core_validate;
    fn core_state_is_live;
    fn core_state_payload;
    fn core_state_restore;
    fn core_state_set_threads;
    fn core_predict;
    fn core_predict_draws;
    fn core_interval;
    fn core_predict_interval;
    fn core_sigma;
    fn core_log_lik;
    fn core_log_lik_survival;
    fn core_log_lik_interval_censored;
    fn core_predict_probs;
    fn core_posterior_draws;
    fn core_diagnostics;
    fn core_sampler_new;
    fn core_sampler_new_aft;
    fn core_sampler_new_interval_censored;
    fn core_sampler_step;
    fn core_samplers_advance;
    fn core_samplers_start;
    fn core_sampler_xaddivortes;
    fn core_samplers_xaddivortes;
    fn core_xaddivortes_defaults;
    fn core_sampler_keep;
    fn core_sampler_n_kept;
    fn core_sampler_set_response;
    fn core_sampler_set_aft_response;
    fn core_sampler_set_interval_censored_response;
    fn core_sampler_fitted_values;
    fn core_sampler_noise_variances;
    fn core_finish;
}
