# Methods on the posterior package's generics, registered on load without
# attaching posterior.

#' The posterior draws of a fit as a matrix of named variables
#'
#' `mu[i]` is the mean function at training row i, `sigma` the residual
#' standard deviation where the model has one, `cell_count` and
#' `dimension_count` the mean cells and mean active covariates per mean
#' tessellation, and, where the model samples them, `df`, `cutpoint[k]`,
#' `bandwidth[j]`, `inclusion_weight[j]` and `concentration`.
#'
#' @param object An object of class `"thiessen"`.
#' @return A double matrix, one row per kept draw.
#' @noRd
draws_matrix_of <- function(object) {
  state <- fit_state(object)
  latent <- core_call(core_predict_draws(state, object$x, "latent"))
  colnames(latent) <- paste0("mu[", seq_len(ncol(latent)), "]")
  diagnostics <- core_call(core_diagnostics(state))
  sigma <- core_call(core_sigma(state))
  blocks <- list(latent)
  if (length(sigma) > 0L) {
    blocks$sigma <- matrix(sigma, ncol = 1L, dimnames = list(NULL, "sigma"))
  }
  blocks$counts <- matrix(
    c(diagnostics$cell_count, diagnostics$dimension_count),
    ncol = 2L, dimnames = list(NULL, c("cell_count", "dimension_count"))
  )
  extras <- core_call(core_posterior_draws(state))
  if (length(extras$df) > 0L) {
    blocks$df <- matrix(extras$df, ncol = 1L, dimnames = list(NULL, "df"))
  }
  for (name in c("cutpoint", "bandwidth", "inclusion_weight")) {
    draws <- extras[[name]]
    if (ncol(draws) > 0L) {
      colnames(draws) <- paste0(name, "[", seq_len(ncol(draws)), "]")
      blocks[[name]] <- draws
    }
  }
  if (length(extras$concentration) > 0L) {
    blocks$concentration <- matrix(
      extras$concentration, ncol = 1L, dimnames = list(NULL, "concentration")
    )
  }
  do.call(cbind, blocks)
}

#' Pooled draws as an iteration by chain by variable array
#'
#' The pooled draws are in chain order, so chain k holds the rows from
#' `(k - 1) * iterations + 1`.
#'
#' @param draws A double matrix, one row per pooled draw.
#' @param chains The number of chains the rows hold.
#' @return A double array of `iterations`, `chains` and variables.
#' @noRd
chain_array <- function(draws, chains) {
  iterations <- nrow(draws) / chains
  array(
    as.vector(draws),
    dim = c(iterations, chains, ncol(draws)),
    dimnames = list(NULL, NULL, colnames(draws))
  )
}

#' Posterior draws of a fitted model
#'
#' The variables are `mu[i]`, the mean function at training row i; `sigma`,
#' where the model has a global residual scale; `cell_count` and
#' `dimension_count`, the mean cells and mean active covariates per mean
#' tessellation; and the quantities the experimental models sample where
#' the fit has them: `df`, the error degrees of freedom under a Student-t
#' grid; `cutpoint[k]`, the interior cutpoints of the ordinal model;
#' `bandwidth[j]`, the soft-membership bandwidth of tessellation j;
#' `inclusion_weight[j]` and `concentration`, the DART inclusion weight of
#' covariate j and the Dirichlet concentration.
#'
#' The chain dimension holds the chains of the fit. A fit of one chain has
#' one chain, and `posterior::summarise_draws()` then reports R-hat as `NA`;
#' effective sample sizes are reported as usual.
#'
#' @param x An object of class `"thiessen"`.
#' @param ... Passed to the posterior method.
#'
#' @return A `draws_df`, from [posterior::as_draws_df()].
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1, chains = 2)
#' posterior::summarise_draws(posterior::as_draws_df(fit), "mean", "sd")
#' @exportS3Method posterior::as_draws_df
as_draws_df.thiessen <- function(x, ...) {
  posterior::as_draws_df(as_draws_array.thiessen(x), ...)
}

#' @rdname as_draws_df.thiessen
#' @return For `as_draws_array()`, a `draws_array`.
#' @exportS3Method posterior::as_draws_array
as_draws_array.thiessen <- function(x, ...) {
  posterior::as_draws_array(chain_array(draws_matrix_of(x), x$n_chains), ...)
}

#' Number of draws and chains of a fitted model
#'
#' The `posterior` generics `ndraws()` and `nchains()` for a fit.
#'
#' @param x An object of class `"thiessen"`.
#'
#' @return The number of kept draws over every chain, and the number of
#'   chains the fit ran.
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
#' posterior::ndraws(fit)
#' @exportS3Method posterior::ndraws
ndraws.thiessen <- function(x) {
  x$n_draws
}

#' @rdname ndraws.thiessen
#' @exportS3Method posterior::nchains
nchains.thiessen <- function(x) {
  x$n_chains
}
