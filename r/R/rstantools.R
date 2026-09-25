# Methods on the rstantools generics, re-exported so a caller needs no
# library(rstantools), as rstanarm and brms do.

#' @importFrom rstantools posterior_predict
#' @export
rstantools::posterior_predict

#' @importFrom rstantools posterior_epred
#' @export
rstantools::posterior_epred

#' @importFrom rstantools log_lik
#' @export
rstantools::log_lik

#' @importFrom rstantools predictive_interval
#' @export
rstantools::predictive_interval

#' Draw from the posterior predictive distribution
#'
#' One replicate per kept draw from the fitted family's observation model:
#' the mean function of that draw plus a Gaussian residual at the scale of
#' that draw under the Gaussian and heteroscedastic models; Bernoulli labels
#' under the probit model; category codes, 0 to K - 1, from the latent value
#' against the cutpoints of that draw under the ordinal model; a time, the
#' exponential of the log-time draw, under the AFT model; the value clipped
#' to the censoring limits under the tobit model; the working value under
#' the interval-censored model; and a Student-t or Laplace error at the
#' drawn scale under those models. The residuals are drawn from R's stream,
#' so [set.seed()] governs them; they are not part of the chain the core
#' draws.
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param ... Ignored.
#'
#' @return A double matrix, one row per kept draw and one column per row of
#'   `newdata`.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' dim(posterior_predict(fit))
#' @exportS3Method rstantools::posterior_predict
posterior_predict.thiessen <- function(object, newdata = NULL, ...) {
  design <- predict_design(object, newdata)
  replicate_draws(object$control$outcome, fit_state(object), design)
}

#' Replicates of the response under a family's observation model
#'
#' @param outcome The fit's outcome family, an object of class
#'   `"thiessen_outcome"`.
#' @param state An external pointer to the fitted state.
#' @param design The numeric design.
#' @return A double matrix, one row per kept draw and one column per row of
#'   `design`.
#' @noRd
replicate_draws <- function(outcome, state, design) {
  UseMethod("replicate_draws")
}

#' @export
replicate_draws.default <- function(outcome, state, design) {
  thiessen_abort(sprintf(
    "`posterior_predict()` is not defined under the %s model.",
    attr(outcome, "kind")
  ))
}

#' @export
replicate_draws.thiessen_gaussian <- function(outcome, state, design) {
  draws <- core_call(core_predict_draws(state, design, "draws"))
  draws + noise_at(state, design, draws)
}

#' @export
replicate_draws.thiessen_probit <- function(outcome, state, design) {
  draws <- core_call(core_predict_draws(state, design, "draws"))
  labels <- stats::rbinom(length(draws), 1L, pmin(pmax(draws, 0), 1))
  matrix(as.double(labels), nrow = nrow(draws))
}

#' @export
replicate_draws.thiessen_ordinal <- function(outcome, state, design) {
  latent <- core_call(core_predict_draws(state, design, "latent"))
  z <- latent + matrix(stats::rnorm(length(latent)), nrow = nrow(latent))
  # The first cutpoint is fixed at zero; a category is the count of
  # cutpoints below the latent value.
  cutpoints <- cbind(0, core_call(core_posterior_draws(state))$cutpoint)
  codes <- matrix(0, nrow(z), ncol(z))
  for (k in seq_len(ncol(cutpoints))) {
    codes <- codes + (z > cutpoints[, k])
  }
  codes
}

#' @export
replicate_draws.thiessen_aft <- function(outcome, state, design) {
  latent <- core_call(core_predict_draws(state, design, "latent"))
  exp(latent + noise_at(state, design, latent))
}

#' @export
replicate_draws.thiessen_tobit <- function(outcome, state, design) {
  latent <- core_call(core_predict_draws(state, design, "latent"))
  values <- latent + noise_at(state, design, latent)
  lower <- if (is.null(outcome$lower)) -Inf else outcome$lower
  upper <- if (is.null(outcome$upper)) Inf else outcome$upper
  pmin(pmax(values, lower), upper)
}

#' @export
replicate_draws.thiessen_interval_censored <- function(outcome, state,
                                                       design) {
  latent <- core_call(core_predict_draws(state, design, "latent"))
  latent + noise_at(state, design, latent)
}

#' @export
replicate_draws.thiessen_student_t <- function(outcome, state, design) {
  latent <- core_call(core_predict_draws(state, design, "latent"))
  df <- core_call(core_posterior_draws(state))$df
  if (length(df) == 0L) {
    df <- outcome$df
  }
  latent + sigma_draws(state) * matrix(
    stats::rt(length(latent), df = df), nrow = nrow(latent)
  )
}

#' @export
replicate_draws.thiessen_laplace <- function(outcome, state, design) {
  latent <- core_call(core_predict_draws(state, design, "latent"))
  # Laplace(0, 1) by inversion of the distribution function.
  u <- stats::runif(length(latent)) - 0.5
  errors <- -sign(u) * log1p(-2 * abs(u))
  latent + sigma_draws(state) * matrix(errors, nrow = nrow(latent))
}

#' The scale sigma of the error per kept draw, on the caller's scale
#'
#' Under the Student-t and Laplace models the scale of the t or Laplace
#' error, not the error standard deviation `predict_variance` squares.
#'
#' @param state An external pointer to the fitted state.
#' @return A double vector, one value per kept draw, recycled down the
#'   columns of a draws by rows matrix.
#' @noRd
sigma_draws <- function(state) {
  core_call(core_sigma(state))
}

#' The residual scale of y given f at each row, per draw
#'
#' @param state An external pointer to the fitted state.
#' @param design The numeric design.
#' @return A double matrix, one row per kept draw.
#' @noRd
scale_at <- function(state, design) {
  sqrt(core_call(core_predict_draws(state, design, "variance")))
}

#' Gaussian residuals at the residual scale, shaped as `draws`
#'
#' @param state An external pointer to the fitted state.
#' @param design The numeric design.
#' @param draws A double matrix, one row per kept draw.
#' @return A double matrix of the shape of `draws`.
#' @noRd
noise_at <- function(state, design, draws) {
  scale <- scale_at(state, design)
  matrix(
    stats::rnorm(length(draws), 0, as.vector(scale)), nrow = nrow(draws)
  )
}

#' Draw from the posterior distribution of the expected response
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param ... Ignored.
#'
#' @return A double matrix, one row per kept draw and one column per row of
#'   `newdata`: the mean of the response, the probability under the probit
#'   model.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' dim(posterior_epred(fit))
#' @exportS3Method rstantools::posterior_epred
posterior_epred.thiessen <- function(object, newdata = NULL, ...) {
  design <- predict_design(object, newdata)
  core_call(core_predict_draws(fit_state(object), design, "draws"))
}

#' Pointwise log-likelihood of a fitted model
#'
#' The matrix `loo::loo()` takes.
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param y The response at `newdata`, in the shape [thiessen()] took:
#'   a `Surv` under the AFT and interval-censored families. Taken from
#'   `newdata` where the fit came from a formula and `newdata` carries the
#'   response column.
#' @param ... Ignored.
#'
#' @return A double matrix, one row per kept draw and one column per
#'   observation.
#'
#' @examples
#' if (requireNamespace("loo", quietly = TRUE)) {
#'   n <- 60
#'   x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#'   y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#'   control <- thiessen_control(
#'     tessellations = 10,
#'     general_params = general_params(burn_in = 20, draws = 40)
#'   )
#'   fit <- thiessen(x, y, control, seed = 1)
#'   loo::loo(log_lik(fit))
#' }
#' @exportS3Method rstantools::log_lik
log_lik.thiessen <- function(object, newdata = NULL, y = NULL, ...) {
  if (is.null(newdata)) {
    return(core_call(
      log_lik_of(fit_state(object), object$x, object$response)
    ))
  }
  design <- predict_design(object, newdata)
  response <- log_lik_response(object, newdata, y)
  if (response$n != nrow(design)) {
    thiessen_abort(
      "`y` must have one value per row of `newdata`."
    )
  }
  check_outcome_response(attr(object$control$outcome, "kind"), response)
  core_call(log_lik_of(fit_state(object), design, response))
}

#' Central posterior predictive interval
#'
#' @param object An object of class `"thiessen"`.
#' @param prob The mass of the interval. Default 0.9.
#' @param newdata New covariates, as [predict.thiessen()] takes them.
#'   `NULL`, the default, is the training rows.
#' @param ... Ignored.
#'
#' @return A double matrix of one row per row of `newdata`, with the lower
#'   and upper bounds as columns named by their percentiles.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' head(predictive_interval(fit))
#' @exportS3Method rstantools::predictive_interval
predictive_interval.thiessen <- function(object, prob = 0.9, newdata = NULL,
                                         ...) {
  check_probability(prob)
  design <- predict_design(object, newdata)
  bounds <- core_call(
    core_interval(fit_state(object), design, "prediction", prob)
  )
  tail <- (1 - prob) / 2
  colnames(bounds) <- paste0(
    format(100 * c(tail, 1 - tail), trim = TRUE), "%"
  )
  bounds
}

#' The design a prediction method should use
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata `NULL` for the training rows, otherwise new covariates.
#' @param call The calling environment to report.
#' @return A double matrix.
#' @noRd
predict_design <- function(object, newdata, call = rlang::caller_env()) {
  if (is.null(newdata)) {
    return(object$x)
  }
  if (is.null(object$blueprint)) {
    return(as_design(newdata, "newdata", call = call))
  }
  forge_design(object, newdata, call = call)
}

#' The response at `newdata` for the log-likelihood
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata New covariates.
#' @param y The response as the caller gave it, or `NULL`.
#' @param call The calling environment to report.
#' @return An object of class `"thiessen_response"`.
#' @noRd
log_lik_response <- function(object, newdata, y, call = rlang::caller_env()) {
  if (!is.null(y)) {
    return(as_response(y, call = call))
  }
  if (is.null(object$blueprint)) {
    thiessen_abort(
      "`y` is required: the fit came from a matrix, so `newdata` carries no response.",
      call = call
    )
  }
  forged <- core_call(
    hardhat::forge(newdata, object$blueprint, outcomes = TRUE),
    call = call
  )
  encode_response(forged$outcomes, call = call)
}
