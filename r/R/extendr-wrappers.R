# Wrappers for the functions exported by src/rust/src/lib.rs, one per
# `#[extendr]` function, by the registered symbol name.

#' Version of the core crate
#'
#' @return A character string: the semantic version of the Rust core the
#'   package was built with, equal to the `Config/thiessen/core-version`
#'   field of the package `DESCRIPTION`.
#' @examples
#' core_version()
#' @export
core_version <- function() .Call(wrap__core_version)

#' Whether the core was built with its experimental feature
#'
#' The outcome families and options the core crate keeps behind its
#' `experimental` Cargo feature are absent from a build without it, so a
#' configuration or a saved fit naming one is rejected with the condition
#' class `thiessen_requires_feature`. Every released build has the feature
#' off; a build with it on is installed from source with
#' `THIESSEN_EXPERIMENTAL=1` in the environment, is outside semantic
#' versioning, and reads a fit the other build cannot. Report this with
#' [core_version()] in a bug report, since a fit rejected for naming a
#' gated option looks the same either way.
#'
#' @return A logical of length one: whether the core in use was built with
#'   the `experimental` feature.
#' @examples
#' core_experimental()
#' @export
core_experimental <- function() .Call(wrap__core_experimental)

core_requires_feature_prefix <- function() {
  .Call(wrap__core_requires_feature_prefix)
}

core_defaults <- function() .Call(wrap__core_defaults)

core_outcome_defaults <- function() .Call(wrap__core_outcome_defaults)

core_validate <- function(config_json) .Call(wrap__core_validate, config_json)

core_state_is_live <- function(state) .Call(wrap__core_state_is_live, state)

core_state_payload <- function(state) .Call(wrap__core_state_payload, state)

core_state_restore <- function(payload, threads) {
  .Call(wrap__core_state_restore, payload, threads)
}

core_state_set_threads <- function(state, threads) invisible(.Call(wrap__core_state_set_threads, state, threads))

core_predict <- function(state, x) .Call(wrap__core_predict, state, x)

core_predict_draws <- function(state, x, kind) {
  .Call(wrap__core_predict_draws, state, x, kind)
}

core_interval <- function(state, x, kind, level) {
  .Call(wrap__core_interval, state, x, kind, level)
}

core_predict_interval <- function(state, x, kind, level) {
  .Call(wrap__core_predict_interval, state, x, kind, level)
}

core_sigma <- function(state) .Call(wrap__core_sigma, state)

core_log_lik_survival <- function(state, x, times, events) {
  .Call(wrap__core_log_lik_survival, state, x, times, events)
}

core_log_lik_interval_censored <- function(state, x, lower, upper) {
  .Call(wrap__core_log_lik_interval_censored, state, x, lower, upper)
}

core_predict_probs <- function(state, x) .Call(wrap__core_predict_probs, state, x)

core_posterior_draws <- function(state) .Call(wrap__core_posterior_draws, state)

core_log_lik <- function(state, x, y) {
  .Call(wrap__core_log_lik, state, x, y)
}

core_diagnostics <- function(state) {
  .Call(wrap__core_diagnostics, state)
}

core_sampler_new <- function(config_json, x, y, seed_value, chain) {
  .Call(wrap__core_sampler_new, config_json, x, y, seed_value, chain)
}

core_sampler_new_aft <- function(config_json, x, times, events, seed_value, chain) {
  .Call(wrap__core_sampler_new_aft, config_json, x, times, events, seed_value, chain)
}

core_sampler_new_interval_censored <- function(config_json, x, lower, upper, seed_value, chain) {
  .Call(wrap__core_sampler_new_interval_censored, config_json, x, lower, upper, seed_value, chain)
}

core_sampler_step <- function(sampler, n) .Call(wrap__core_sampler_step, sampler, n)

core_samplers_advance <- function(samplers, burn_in, draws, thinning, threads) {
  .Call(wrap__core_samplers_advance, samplers, burn_in, draws, thinning, threads)
}

core_samplers_start <- function(samplers, previous, draw) {
  .Call(wrap__core_samplers_start, samplers, previous, draw)
}

core_sampler_keep <- function(sampler) .Call(wrap__core_sampler_keep, sampler)

core_sampler_xaddivortes <- function(sampler, params_json) {
  .Call(wrap__core_sampler_xaddivortes, sampler, params_json)
}

core_samplers_xaddivortes <- function(samplers, params_json, threads) {
  .Call(wrap__core_samplers_xaddivortes, samplers, params_json, threads)
}

core_xaddivortes_defaults <- function() .Call(wrap__core_xaddivortes_defaults)

core_sampler_n_kept <- function(sampler) .Call(wrap__core_sampler_n_kept, sampler)

core_sampler_set_response <- function(sampler, y) {
  .Call(wrap__core_sampler_set_response, sampler, y)
}

core_sampler_set_aft_response <- function(sampler, times, events) {
  .Call(wrap__core_sampler_set_aft_response, sampler, times, events)
}

core_sampler_set_interval_censored_response <- function(sampler, lower, upper) {
  .Call(wrap__core_sampler_set_interval_censored_response, sampler, lower, upper)
}

core_sampler_fitted_values <- function(sampler) {
  .Call(wrap__core_sampler_fitted_values, sampler)
}

core_sampler_noise_variances <- function(sampler) {
  .Call(wrap__core_sampler_noise_variances, sampler)
}

core_finish <- function(samplers, threads) {
  .Call(wrap__core_finish, samplers, threads)
}
