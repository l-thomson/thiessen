#' XAddiVortes settings
#'
#' The settings of [xaddivortes()], and of the fast start of
#' `thiessen(start = "fast")`. The defaults are the core's, which
#' [xaddivortes_defaults()] returns.
#'
#' @param rounds Replacement rounds after the build, a whole number from 0.
#'   The fit time grows linearly with it; with 0 the build goes straight to
#'   the final refit.
#' @param max_columns The most columns of a tessellation, a whole number
#'   from 1; capped at the number of columns.
#' @param centres The most centres of a tessellation, drawn at random
#'   training rows before empty cells are dropped; fresh tessellations take
#'   it or one of its halvings down to 4. A whole number from 1 to 65,535.
#' @param value_scale The share of the shrunk cell mean used as the cell
#'   value in the build and the rounds, in (0, 1]; the final refit chooses
#'   from it up to 1. Smaller values fit more slowly and overfit less.
#' @param replace_share The share of tessellations replaced each round, in
#'   \[0, 1\]; 0 makes the rounds pure refits.
#'
#' @return An object of class `"xaddivortes_params"`.
#' @seealso [xaddivortes()], [xaddivortes_defaults()]
#' @examples
#' xaddivortes_params(rounds = 20, max_columns = 2)
#' @export
xaddivortes_params <- function(rounds = 40, max_columns = 3, centres = 64,
                               value_scale = 0.5, replace_share = 0.3) {
  check_whole_number(rounds, min = 0)
  check_whole_number(max_columns, min = 1)
  check_whole_number(centres, min = 1, max = 65535)
  check_positive(value_scale, max = 1)
  check_number(replace_share, min = 0, max = 1)
  structure(
    list(
      rounds = rounds, max_columns = max_columns, centres = centres,
      value_scale = value_scale, replace_share = replace_share
    ),
    class = "xaddivortes_params"
  )
}

#' @export
format.xaddivortes_params <- function(x, ...) {
  constructor_call("xaddivortes_params", compact(unclass(x)))
}

#' @rdname print.params
#' @export
print.xaddivortes_params <- print.term_params

#' The core's default XAddiVortes settings
#'
#' @return A named list: every setting of [xaddivortes_params()] and its
#'   default, as the core defines them.
#' @examples
#' xaddivortes_defaults()
#' @export
xaddivortes_defaults <- function() {
  jsonlite::fromJSON(core_xaddivortes_defaults(), simplifyVector = TRUE)
}

#' The settings as the core's JSON
#'
#' Doubles are written with 17 significant digits, which round-trip
#' exactly, so the R and Python bindings send the core the same values.
#'
#' @param params An object of class `"xaddivortes_params"`, or `NULL` for
#'   the defaults.
#' @param arg The argument name to report.
#' @param call The calling environment to report.
#' @return A JSON string.
#' @noRd
fast_json <- function(params, arg = "params", call = rlang::caller_env()) {
  check_group(params, arg, "xaddivortes_params", null_ok = TRUE, call = call)
  if (is.null(params)) {
    params <- xaddivortes_params()
  }
  jsonlite::toJSON(unclass(params), auto_unbox = TRUE, digits = I(17))
}

#' Fit XAddiVortes, the fast fit of AddiVortes
#'
#' XAddiVortes is a fast point fit of the additive ensemble of Voronoi
#' tessellations of AddiVortes (Stone and Gosling, 2025), in place of
#' Markov chain Monte Carlo. The columns are screened by binned mutual
#' information with the response, and the tessellations are built in turn
#' on the running residual with damped, shrunk cell values. Each round then
#' replaces the share with the smallest log Bayes factor under the model's
#' prior by fresh tessellations and refits every cell value, and a final
#' refit chooses the damping by generalised cross-validation. The result is
#' a single point estimate, not a posterior sample. Starting the sampler
#' from the fit, `thiessen(start = "fast")`, gives posterior draws after
#' the burn-in.
#'
#' The noise estimate `sigma(fit)` is the root of the training residual sum
#' of squares over the residual degrees of freedom of the final refit.
#' Those degrees of freedom do not count the choice of the cells, so the
#' intervals of `predict(fit, interval = "prediction")` can cover less than
#' their level; `vignette("fast-fits", package = "thiessen")` measures the
#' coverage on held-out data. Split-conformal intervals around the fit (Lei
#' and others, 2018) are calibrated without that assumption: fit on one
#' part of the data and widen the prediction by the `level` quantile of the
#' absolute residuals on the other part.
#'
#' The fit holds one draw, so it has no posterior to compare models by:
#' `log_lik()`, and with it `loo::loo()` and `loo::waic()`, refuse it with
#' an error of class `thiessen_error`.
#'
#' The design and response are handled as by [thiessen()]: a matrix, a data
#' frame or a formula, factors encoded as treatment-contrast indicators.
#' The model is the Gaussian one, so the response must be numeric; a
#' response or a `control` that selects another family is an error.
#'
#' XAddiVortes is experimental: it needs a build with the core's
#' `experimental` feature ([core_experimental()]) and errors with the
#' condition class `thiessen_requires_feature` otherwise.
#'
#' @param x A numeric matrix or data frame of covariates.
#' @param y The numeric response.
#' @param formula A formula, `response ~ covariates`.
#' @param data A data frame holding the variables of `formula`.
#' @param params The settings, from [xaddivortes_params()]; `NULL` takes the
#'   defaults.
#' @param control The model configuration, from [thiessen_control()]; its
#'   `mean_params` set the number of tessellations and the covariate space.
#'   The sweep schedule is not used.
#' @param seed The seed, as for [thiessen()].
#' @param threads Threads for prediction.
#' @param ... Not used; must be empty.
#'
#' @return An object of class `c("xaddivortes", "thiessen")`: a
#'   [thiessen()] fit with one draw, the fast fit, whose noise draw is the
#'   fit's sigma, so every method of the fit uses that value. Its `fast`
#'   element is a list of `sigma`, the noise estimate on the response's
#'   scale; `column_scores`, the screening scores named by column;
#'   `train_score`, the training residual root mean square after the build,
#'   after each round and after the final refit; and `seconds`, the elapsed
#'   fit time.
#' @references
#' Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
#' (2018). Distribution-free predictive inference for regression. *Journal
#' of the American Statistical Association* 113(523), 1094-1111.
#' \doi{10.1080/01621459.2017.1307116}
#'
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. *Journal of Computational and Graphical
#' Statistics* 34(3), 859-871. \doi{10.1080/10618600.2024.2414104}
#' @seealso [xaddivortes_params()], [thiessen()], [predict.xaddivortes()],
#'   [summary.xaddivortes()]
#' @examplesIf core_experimental()
#' set.seed(1)
#' x <- matrix(runif(400), ncol = 2)
#' y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
#' control <- thiessen_control(tessellations = 20)
#' fit <- xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
#'                    control = control, seed = 1)
#' fit
#'
#' frame <- data.frame(y = y, a = x[, 1], b = x[, 2])
#' xaddivortes(y ~ a + b, frame, control = control, seed = 1)
#' @export
xaddivortes <- function(x, ...) {
  UseMethod("xaddivortes")
}

#' @rdname xaddivortes
#' @export
xaddivortes.default <- function(x, y, params = NULL,
                                control = thiessen_control(), seed = NULL,
                                threads = getOption("mc.cores", 1L), ...) {
  rlang::check_dots_empty()
  new_fast_fit(as_design(x), as_response(y), params, control, seed, threads,
               generic_call(match.call(), "xaddivortes"))
}

#' @rdname xaddivortes
#' @export
xaddivortes.data.frame <- function(x, y, params = NULL,
                                   control = thiessen_control(), seed = NULL,
                                   threads = getOption("mc.cores", 1L),
                                   ...) {
  rlang::check_dots_empty()
  molded <- mold_design(~ ., x, control)
  new_fast_fit(molded$design, as_response(y), params, control, seed,
               threads, generic_call(match.call(), "xaddivortes"),
               blueprint = molded$blueprint)
}

#' @rdname xaddivortes
#' @export
xaddivortes.formula <- function(formula, data, params = NULL,
                                control = thiessen_control(), seed = NULL,
                                threads = getOption("mc.cores", 1L), ...) {
  rlang::check_dots_empty()
  molded <- mold_design(formula, data, control)
  new_fast_fit(molded$design, encode_response(molded$outcomes), params,
               control, seed, threads,
               generic_call(match.call(), "xaddivortes"),
               blueprint = molded$blueprint)
}

#' Run the fast fit and assemble the object the methods return
#'
#' One sampler on chain 0 of the seed, the fast fit, kept as the single
#' draw, as the core's `fit_xaddivortes` does.
#'
#' @noRd
new_fast_fit <- function(design, response, params, control, seed, threads,
                         call, blueprint = NULL,
                         call_env = rlang::caller_env()) {
  check_design_rows(design, response, call = call_env)
  check_group(control, "control", "thiessen_control", call = call_env)
  params_json <- fast_json(params, call = call_env)
  control <- resolve_outcome(control, response, call = call_env)
  check_fast_family(control, "`xaddivortes()`", call = call_env)
  threads <- resolve_threads(threads, call = call_env)
  resolved <- resolve_seed(seed, call = call_env)
  config <- config_json(control, call = call_env)
  run <- core_call({
    sampler <- new_sampler_handle(config, design, response, resolved, 0L)
    seconds <- system.time(
      fast <- core_sampler_xaddivortes(sampler, params_json)
    )[["elapsed"]]
    core_sampler_keep(sampler)
    list(fit = core_finish(list(sampler), threads), fast = fast,
         seconds = seconds)
  }, call = call_env)
  fit <- assemble_fit(run$fit, design, response, resolved, call,
                      threads = threads, blueprint = blueprint,
                      call_env = call_env)
  fast <- jsonlite::fromJSON(run$fast, simplifyVector = TRUE)
  names(fast$column_scores) <- if (is.null(colnames(design))) {
    paste0("x", seq_len(ncol(design)))
  } else {
    colnames(design)
  }
  fast$seconds <- run$seconds
  fit$fast <- fast
  fit$fast_params <- if (is.null(params)) xaddivortes_params() else params
  class(fit) <- c("xaddivortes", class(fit))
  fit
}

#' Refuse a family other than the Gaussian one for the fast fit
#'
#' @param control An object of class `"thiessen_control"` with its family
#'   resolved.
#' @param what The function or argument to name in the error.
#' @param call The calling environment to report.
#' @noRd
check_fast_family <- function(control, what, call = rlang::caller_env()) {
  kind <- attr(control$outcome, "kind")
  if (identical(kind, "gaussian")) {
    return(invisible(NULL))
  }
  thiessen_abort(
    sprintf("%s fits the Gaussian family only, not the %s family.", what,
            kind),
    call = call
  )
}

#' Predictions of an XAddiVortes fit
#'
#' The prediction is the sum of the fitted cell values. The prediction
#' interval is the prediction plus and minus the normal quantile times
#' `sigma(object)`, the fit's noise estimate: a rough interval that takes
#' the fit as known, not a posterior interval (see [xaddivortes()] for
#' calibrated intervals).
#'
#' @param object An object of class `"xaddivortes"`.
#' @param newdata New covariates, as for [predict.thiessen()]; `NULL`
#'   predicts at the training rows.
#' @param interval `"none"` or `"prediction"`.
#' @param level The interval level, in (0, 1).
#' @param threads Threads for this prediction; `NULL` takes the fit's.
#' @param ... Not used; must be empty.
#'
#' @return A numeric vector, or with an interval a matrix with columns
#'   `fit`, `lower` and `upper`.
#' @examplesIf core_experimental()
#' set.seed(1)
#' x <- matrix(runif(400), ncol = 2)
#' y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
#' fit <- xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
#'                    control = thiessen_control(tessellations = 20),
#'                    seed = 1)
#' head(predict(fit, x))
#' head(predict(fit, x, interval = "prediction"))
#' @export
predict.xaddivortes <- function(object, newdata = NULL,
                                interval = c("none", "prediction"),
                                level = 0.95, threads = NULL, ...) {
  rlang::check_dots_empty()
  interval <- resignal(rlang::arg_match0(interval, c("none", "prediction")))
  predict.thiessen(object, newdata = newdata, type = "mean",
                   interval = interval, level = level, threads = threads)
}

#' Print an XAddiVortes fit
#'
#' @param x An object of class `"xaddivortes"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examplesIf core_experimental()
#' set.seed(1)
#' x <- matrix(runif(400), ncol = 2)
#' y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
#' print(xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
#'                   control = thiessen_control(tessellations = 20),
#'                   seed = 1))
#' @export
print.xaddivortes <- function(x, ...) {
  cat("XAddiVortes fast fit (experimental)\n")
  cat("Call: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat(sprintf(
    "%s model, %d observations, %d covariates\n",
    x$model, nobs(x), x$n_features
  ))
  cat(fast_line(x$control$mean_params$tessellations, x$fast))
  cat(sprintf("In-sample RMSE %.4g, seed %.0f\n", x$in_sample_rmse, x$seed))
  invisible(x)
}

#' The size, the rounds and the time of a fast fit
#'
#' @param tessellations The number of tessellations.
#' @param fast The `fast` element of an XAddiVortes fit.
#' @return One line of text, ending in a newline.
#' @noRd
fast_line <- function(tessellations, fast) {
  sprintf("%d tessellations, %d rounds (%.2f s)\n", tessellations,
          length(fast$train_score) - 2L, fast$seconds)
}

#' Summarise an XAddiVortes fit
#'
#' @param object An object of class `"xaddivortes"`.
#' @param ... Ignored.
#'
#' @return An object of class `"summary.xaddivortes"`: a list of the call,
#'   the model, the dimensions of the fit, the number of tessellations, the
#'   `fast` element of the fit, the in-sample root mean squared error and
#'   the column scores, highest first.
#'
#' @examplesIf core_experimental()
#' set.seed(1)
#' x <- matrix(runif(400), ncol = 2)
#' y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
#' summary(xaddivortes(x, y, params = xaddivortes_params(rounds = 10),
#'                     control = thiessen_control(tessellations = 20),
#'                     seed = 1))
#' @export
summary.xaddivortes <- function(object, ...) {
  structure(
    list(
      call = object$call,
      model = object$model,
      nobs = nobs(object),
      n_features = object$n_features,
      tessellations = object$control$mean_params$tessellations,
      fast = object$fast,
      in_sample_rmse = object$in_sample_rmse,
      column_scores = sort(object$fast$column_scores, decreasing = TRUE)
    ),
    class = "summary.xaddivortes"
  )
}

#' @param x An object of class `"summary.xaddivortes"`.
#' @return `x`, invisibly.
#' @rdname summary.xaddivortes
#' @export
print.summary.xaddivortes <- function(x, ...) {
  cat("XAddiVortes fast fit (experimental)\n")
  cat("Call: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat(sprintf(
    "%s model, %d observations, %d covariates\n",
    x$model, x$nobs, x$n_features
  ))
  cat(fast_line(x$tessellations, x$fast))
  cat(sprintf("\nIn-sample RMSE %.4g\n", x$in_sample_rmse))
  cat("\nColumn scores, mutual information with the response:\n")
  shown <- seq_len(min(length(x$column_scores), 10L))
  print(round(x$column_scores[shown], 4))
  if (length(x$column_scores) > 10L) {
    cat(sprintf("... and %d more\n", length(x$column_scores) - 10L))
  }
  invisible(x)
}

#' @exportS3Method rstantools::log_lik
log_lik.xaddivortes <- function(object, ...) {
  thiessen_abort(paste(
    "An XAddiVortes fit holds one point fit, not posterior draws, so it",
    "has no pointwise log-likelihood; fit with `thiessen()` for one."
  ))
}
