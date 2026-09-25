# Convergence diagnostics of a multi-chain fit: rank-normalised split R-hat
# and bulk and tail effective sample sizes (Vehtari, Gelman, Simpson,
# Carpenter and Buerkner, 2021), computed by posterior at the thresholds
# posterior documents.

# Above this R-hat the chains have not mixed.
RHAT_THRESHOLD <- 1.01

# Below this effective sample size the summaries are not reliable.
ESS_THRESHOLD <- 400

# Training rows the mean function is monitored at.
CONVERGENCE_POINTS <- 20L

#' The convergence diagnostics of a fit, or `NULL` for one chain
#'
#' R-hat and the bulk and tail effective sample sizes of sigma, where the
#' model has one, and of the mean function at a subsample of the training
#' rows, reduced to their worst value over those variables.
#'
#' @param object An object of class `"thiessen"`.
#' @param points The number of training rows to monitor.
#' @return A list of the chain count, the number of variables monitored, the
#'   largest R-hat and the smallest bulk and tail effective sample sizes, or
#'   `NULL` where fewer than two chains ran.
#' @noRd
convergence_of <- function(object, points = CONVERGENCE_POINTS) {
  if (object$n_chains < 2L) {
    return(NULL)
  }
  rows <- monitored_rows(nrow(object$x), points)
  state <- fit_state(object)
  latent <- core_call(
    core_predict_draws(state, object$x[rows, , drop = FALSE], "latent")
  )
  colnames(latent) <- paste0("mu[", rows, "]")
  sigma <- core_call(core_sigma(state))
  if (length(sigma) > 0L) {
    latent <- cbind(sigma = sigma, latent)
  }
  # The measures are passed as functions: summarise_draws() resolves a bare
  # name through the calling environment, where a user's own rhat() would
  # shadow posterior's.
  summary <- posterior::summarise_draws(
    posterior::as_draws_array(chain_array(latent, object$n_chains)),
    rhat = posterior::rhat, ess_bulk = posterior::ess_bulk,
    ess_tail = posterior::ess_tail
  )
  list(
    n_chains = object$n_chains,
    n_variables = nrow(summary),
    rhat = max(summary$rhat, na.rm = TRUE),
    ess_bulk = min(summary$ess_bulk, na.rm = TRUE),
    ess_tail = min(summary$ess_tail, na.rm = TRUE)
  )
}

#' Rows of the design the mean function is monitored at
#'
#' @param n The number of training rows.
#' @param points The number of rows to monitor.
#' @return An integer vector of row indices.
#' @noRd
monitored_rows <- function(n, points) {
  if (n <= points) {
    return(seq_len(n))
  }
  unique(as.integer(round(seq(1, n, length.out = points))))
}

#' The message a fit that has not converged carries
#'
#' @param convergence The diagnostics, or `NULL`.
#' @return A character string, or `NULL` where the thresholds are met.
#' @noRd
convergence_message <- function(convergence) {
  if (is.null(convergence)) {
    return(NULL)
  }
  ess <- min(convergence$ess_bulk, convergence$ess_tail)
  if (!isTRUE(convergence$rhat > RHAT_THRESHOLD) &&
        !isTRUE(ess < ESS_THRESHOLD)) {
    return(NULL)
  }
  sprintf(
    paste0(
      "The chains may not have converged: largest R-hat %.3f (threshold ",
      "%.2f), smallest effective sample size %.0f (threshold %d). ",
      "Run more draws or more chains."
    ),
    convergence$rhat, RHAT_THRESHOLD, ess, ESS_THRESHOLD
  )
}

#' Warn where a fit has not met the convergence thresholds
#'
#' @param object An object of class `"thiessen"`.
#' @param call The calling environment to report.
#' @return `object`, invisibly.
#' @noRd
warn_convergence <- function(object, call = rlang::caller_env()) {
  message <- convergence_message(object$convergence)
  if (!is.null(message)) {
    rlang::warn(message, class = "thiessen_warning", call = call)
  }
  invisible(object)
}

#' The convergence line the print and summary methods report
#'
#' @param object An object of class `"thiessen"`, or a summary of one.
#' @return A character string.
#' @noRd
convergence_line <- function(object) {
  if (is.null(object$convergence)) {
    return(paste(
      "1 chain; R-hat and effective sample sizes need two or more chains"
    ))
  }
  sprintf(
    "%d chains, largest R-hat %.3f, smallest effective sample size %.0f",
    object$convergence$n_chains, object$convergence$rhat,
    min(object$convergence$ess_bulk, object$convergence$ess_tail)
  )
}
