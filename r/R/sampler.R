# The sampler API: the core's Gibbs loop as an object the caller drives.

#' Drive the sampler one call at a time
#'
#' `r lifecycle::badge("experimental")`
#'
#' An outcome family, a censoring scheme or an imputation scheme the
#' package does not ship is written in R against this loop, with no Rust
#' and no recompilation. A model is reachable exactly when it is a Gaussian
#' regression on a response the caller can rewrite each sweep, which covers
#' every latent-Gaussian data augmentation: probit, tobit, accelerated
#' failure time, interval-censored and ordinal. An augmentation needing
#' per-observation weights, such as logistic through Polya-Gamma, is not
#' reachable, because nothing sets the noise variances. Neither is the
#' geometry, tessellation membership, cell internals, the inclusion prior
#' or the proposals.
#'
#' It follows the updatable sampler object of dbarts and the low-level
#' interface of stochtree: construct with the configuration, the data and a
#' seed, then drive the Gibbs loop yourself. Burn-in and thinning are the
#' caller's loop. Parameters sampled in the caller's loop, cutpoints for
#' instance, are not in `$finish()`'s draws, so the caller keeps and
#' diagnoses those. `vignette("sampler-api")` reimplements the probit
#' family in R and checks it against the built-in one.
#'
#' The response is on the caller's scale through an affine map frozen at
#' construction, so a response outside the training range is legitimate.
#' The sampler owns its RNG, seeded at construction with the chain-0 seed
#' of [thiessen()]; driving the configured schedule by hand reproduces a
#' one-chain fit bit for bit. The loop cannot rewire tessellation
#' membership or cell internals. The `burn_in`, `draws` and `thinning`
#' settings of the control play no part here.
#'
#' The returned object holds the loop's verbs:
#' \describe{
#'   \item{`$step(n = 1)`}{Run `n` sweeps of the Gibbs loop.}
#'   \item{`$keep()`}{Record the current state as a posterior draw.}
#'   \item{`$n_kept()`}{The number of draws kept so far.}
#'   \item{`$set_response(y)`}{Replace the response, keeping the
#'     tessellations, the cell values and sigma^2; the next sweep
#'     conditions on it. Takes the shapes [thiessen()] takes, checked
#'     against the sampler's family: labels in \{0, 1\} or a two-level
#'     factor under the probit family, an ordered factor under the
#'     ordinal family, a `Surv` under the AFT and interval-censored
#'     families. Under the censored families the latents reset to the
#'     new observed values.}
#'   \item{`$fitted_values()`}{The current mean function at the training
#'     rows: f(x_i), or c + f(x_i) under the probit family.}
#'   \item{`$noise_variances()`}{The current variance of y given f at each
#'     training row: sigma^2 under the Gaussian model, 1 under the probit
#'     family (the latent scale), s^2(x_i) under the heteroscedastic
#'     model.}
#'   \item{`$finish()`}{The fit of the kept draws, as [thiessen()] returns
#'     one. Consumes the sampler: every later call on it errors.}
#' }
#'
#' @param x A numeric matrix of covariates, one row per observation. A
#'   numeric vector is taken as one column.
#' @param y The response, one value per row of `x`, in the shapes
#'   [thiessen()] takes: a numeric vector (labels in \{0, 1\} under the
#'   probit family), a two-level or ordered factor, or a
#'   [survival::Surv()]. It selects the family where `control` names
#'   none.
#' @param control An object of class `"thiessen_control"`, from
#'   [thiessen_control()].
#' @param seed The seed. `NULL`, the default, draws one from R's stream,
#'   so [set.seed()] governs; a whole number in `[0, 2^53]` gives the
#'   chain that `thiessen()` would run first.
#'
#' @return An object of class `"thiessen_sampler"`: an environment holding
#'   the verbs above.
#'
#' @seealso [thiessen()], whose loop this is.
#'
#' @examples
#' fixture <- matrix(seq(0, 1, length.out = 40), ncol = 1)
#' response <- 3 * fixture[, 1]^2 - fixture[, 1]
#'
#' sampler <- thiessen_sampler(fixture, response,
#'                             thiessen_control(tessellations = 10),
#'                             seed = 1)
#' sampler$step(20)
#' for (draw in seq_len(30)) {
#'   sampler$step(1)
#'   sampler$keep()
#' }
#' fit <- sampler$finish()
#' fit$n_draws
#' @export
thiessen_sampler <- function(x, y, control = thiessen_control(), seed = NULL) {
  lifecycle::signal_stage("experimental", "thiessen_sampler()")
  if (!inherits(control, "thiessen_control")) {
    thiessen_abort("`control` must come from `thiessen_control()`.")
  }
  design <- as_design(x)
  response <- as_response(y)
  if (response$n != nrow(design)) {
    thiessen_abort(sprintf(
      "`y` must have one value per row of `x`, %d values.", nrow(design)
    ))
  }
  control <- resolve_outcome(control, response)
  kind <- attr(control$outcome, "kind")
  resolved <- resolve_seed(seed)
  handle <- core_call(
    new_sampler_handle(config_json(control), design, response, resolved, 0L)
  )
  current <- response

  self <- new.env(parent = emptyenv())
  self$step <- function(n = 1) {
    check_whole_number(n, min = 0)
    core_call(core_sampler_step(handle, as.integer(n)))
    invisible(NULL)
  }
  self$keep <- function() {
    core_call(core_sampler_keep(handle))
    invisible(NULL)
  }
  self$n_kept <- function() {
    core_call(core_sampler_n_kept(handle))
  }
  self$set_response <- function(y) {
    response <- as_response(y)
    check_outcome_response(kind, response)
    core_call(set_sampler_response(handle, response))
    current <<- response
    invisible(NULL)
  }
  self$fitted_values <- function() {
    core_call(core_sampler_fitted_values(handle))
  }
  self$noise_variances <- function() {
    core_call(core_sampler_noise_variances(handle))
  }
  self$finish <- function() {
    fit <- core_call(core_finish(list(handle), 1L))
    assemble_fit(fit, design = design, response = current, seed = resolved,
                 call = sys.call(-1L))
  }
  class(self) <- "thiessen_sampler"
  self
}

#' Print a sampler
#'
#' Prints the number of draws kept so far, or that the sampler is
#' finished.
#'
#' @param x An object of class `"thiessen_sampler"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' design <- matrix(seq(0, 1, length.out = 40), ncol = 1)
#' print(thiessen_sampler(design, design[, 1]^2,
#'                        thiessen_control(tessellations = 10), seed = 1))
#' @export
print.thiessen_sampler <- function(x, ...) {
  kept <- tryCatch(x$n_kept(), error = function(condition) NULL)
  if (is.null(kept)) {
    cat("<thiessen_sampler> finished\n")
  } else {
    cat(sprintf("<thiessen_sampler> %d draw(s) kept\n", kept))
  }
  invisible(x)
}
