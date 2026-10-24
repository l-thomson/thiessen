#' XAddiVortes settings
#'
#' The settings of [xaddivortes()], and of the fast start of
#' `thiessen(start = "fast")`. The defaults are the core's, which
#' [xaddivortes_defaults()] returns; `budget` and `max_columns` are the
#' settings most users change.
#'
#' @param budget The compute budget, in approximate sweeps of the exact
#'   sampler on the same data; positive and finite. More budget runs more
#'   replacement rounds. Measured fit times are typically below the time
#'   of that many exact sweeps. Scoring the columns costs roughly
#'   `115 / m` sweeps at the default `pair_screen`, for m tessellations, so
#'   a small budget or ensemble leaves tessellations at zero; the fit then
#'   warns.
#' @param max_columns Columns per tessellation for the tessellations that
#'   are not main effects; at least 1.
#' @param centres Centres drawn per tessellation at random training rows,
#'   before empty cells are dropped; from 1 to 65,535.
#' @param value_scale The share of the shrunk cell mean used as a cell
#'   value, in (0, 1].
#' @param main_effect_share The share of tessellations on one column, in
#'   \[0, 1\].
#' @param replace_share The share of tessellations replaced each round, in
#'   \[0, 1\].
#' @param max_rounds The most replacement rounds, at least 0; the budget
#'   usually stops the fit first.
#' @param pair_screen The best-scoring columns whose pairs are scored; at
#'   least 2. The budget is charged for `pair_screen * (pair_screen - 1) / 2`
#'   pairs whatever the column count.
#' @param column_bins Equal-width bins of a column for its score; from 2 to
#'   255.
#' @param response_bins Rank bins of the response for the scores; from 2 to
#'   255.
#' @param score_floor The floor added to every score, as a share of the
#'   mean score; at least 0.
#' @param tries Fresh candidates per replacement, the one of highest gain
#'   kept; at least 1.
#' @param rescore Re-score the columns from the residual after every round.
#' @param refit_passes Refit passes over all tessellations per round; at
#'   least 1.
#' @param update_sigma Re-estimate the noise variance used in the shrinkage
#'   after every round.
#' @param worth How a round ranks the tessellations it may replace:
#'   `"gain"` or `"squared_values"`.
#' @param accept_if_better Keep a replacement only when it lowers the
#'   training error and a refit only when it does not raise it, so the
#'   training error never rises from round to round.
#'
#' @return An object of class `"xaddivortes_params"`.
#' @seealso [xaddivortes()], [xaddivortes_defaults()]
#' @examples
#' xaddivortes_params(budget = 20, max_columns = 3)
#' @export
xaddivortes_params <- function(budget = 50, max_columns = 2, centres = 11,
                               value_scale = 0.5, main_effect_share = 0.2,
                               replace_share = 0.3, max_rounds = 200,
                               pair_screen = 23, column_bins = 8,
                               response_bins = 4, score_floor = 0.02,
                               tries = 4, rescore = FALSE, refit_passes = 2,
                               update_sigma = TRUE,
                               worth = c("gain", "squared_values"),
                               accept_if_better = FALSE) {
  check_positive(budget)
  check_whole_number(max_columns, min = 1)
  check_whole_number(centres, min = 1, max = 65535)
  check_positive(value_scale, max = 1)
  check_number(main_effect_share, min = 0, max = 1)
  check_number(replace_share, min = 0, max = 1)
  check_whole_number(max_rounds, min = 0)
  check_whole_number(pair_screen, min = 2)
  check_whole_number(column_bins, min = 2, max = 255)
  check_whole_number(response_bins, min = 2, max = 255)
  check_number(score_floor, min = 0, allow_infinite = FALSE)
  check_whole_number(tries, min = 1)
  check_flag(rescore)
  check_whole_number(refit_passes, min = 1)
  check_flag(update_sigma)
  worth <- resignal(rlang::arg_match0(worth, c("gain", "squared_values")))
  check_flag(accept_if_better)
  structure(
    list(
      budget = budget, max_columns = max_columns, centres = centres,
      value_scale = value_scale, main_effect_share = main_effect_share,
      replace_share = replace_share, max_rounds = max_rounds,
      pair_screen = pair_screen, column_bins = column_bins,
      response_bins = response_bins, score_floor = score_floor,
      tries = tries, rescore = rescore, refit_passes = refit_passes,
      update_sigma = update_sigma, worth = worth,
      accept_if_better = accept_if_better
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
#' XAddiVortes fits the additive ensemble of Voronoi tessellations of
#' AddiVortes (Stone and Gosling, 2025) under a compute budget instead of
#' by Markov chain Monte Carlo. The columns are scored by binned mutual
#' information with the response, the tessellations are built with cell
#' values fitted to the running residual, and rounds replace the least
#' useful share and refit the rest while the budget lasts. The result is a
#' single point estimate, not a posterior sample. Starting the sampler
#' from the fit, `thiessen(start = "fast")`, gives posterior draws after
#' the burn-in.
#'
#' The noise estimate `sigma(fit)` is the in-sample residual root mean
#' square. It understates the noise when the fit overfits, most on small
#' data, so the intervals of `predict(fit, interval = "prediction")` can
#' cover far less than their level;
#' `vignette("fast-fits", package = "thiessen")` measures the coverage on
#' held-out data. Split-conformal intervals around the fit (Lei and
#' others, 2018) are calibrated without that assumption: fit on one part
#' of the data and widen the prediction by the `level` quantile of the
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
#'   element is a list of `sigma`, the in-sample residual root mean square
#'   on the response's scale; `rounds`; `planned_sweeps`, the budget spent;
#'   `column_scores`, named by column; `zeroed`, the tessellations left at
#'   zero when the budget ran out while building; `train_score`, the
#'   residual root mean square after the build and after each round; and
#'   `seconds`, the elapsed fit time.
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
#' fit <- xaddivortes(x, y, params = xaddivortes_params(budget = 100),
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
#' `sigma(object)`, the in-sample residual root mean square: a rough
#' interval that takes the fit as known, not a posterior interval (see
#' [xaddivortes()] for calibrated intervals).
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
#' fit <- xaddivortes(x, y, params = xaddivortes_params(budget = 100),
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
#' print(xaddivortes(x, y, params = xaddivortes_params(budget = 100),
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

#' The budget, the rounds and the zeroed tessellations of a fast fit
#'
#' @param tessellations The number of tessellations.
#' @param fast The `fast` element of an XAddiVortes fit.
#' @return One or two lines of text, each ending in a newline.
#' @noRd
fast_line <- function(tessellations, fast) {
  line <- sprintf(
    "%d tessellations, %.4g sweeps of budget spent in %d rounds (%.2f s)\n",
    tessellations, fast$planned_sweeps, fast$rounds, fast$seconds
  )
  if (fast$zeroed > 0) {
    line <- paste0(line, sprintf(
      "%d tessellations left at zero: the budget ran out\n", fast$zeroed
    ))
  }
  line
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
#' summary(xaddivortes(x, y, params = xaddivortes_params(budget = 100),
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
