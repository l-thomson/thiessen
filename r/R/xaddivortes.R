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
#'   of that many exact sweeps.
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
#' @param max_rounds The most replacement rounds; the budget usually stops
#'   the fit first.
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
#' @param accept_if_better Keep a replacement or a refit only when it does
#'   not raise the training error, so the training error never rises from
#'   round to round.
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
  for (name in c("budget", "value_scale", "main_effect_share",
                 "replace_share", "score_floor")) {
    check_number(get(name), allow_infinite = FALSE, arg = name)
  }
  minimum <- c(max_columns = 1, centres = 1, max_rounds = 0, pair_screen = 2,
               column_bins = 2, response_bins = 2, tries = 1,
               refit_passes = 1)
  for (name in names(minimum)) {
    check_whole_number(get(name), min = minimum[[name]], arg = name)
  }
  check_flag(rescore)
  check_flag(update_sigma)
  check_flag(accept_if_better)
  worth <- rlang::arg_match(worth)
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
#' from the fit, `thiessen(start = "fast")`, gives the posterior.
#'
#' The noise estimate `sigma(fit)` is the in-sample residual root mean
#' square. It understates the noise when the fit overfits, most on small
#' data, so the intervals of `predict(fit, interval = "prediction")` are a
#' rough guide. Split-conformal intervals around the fit (Lei and others,
#' 2018) are calibrated without that assumption: fit on one part of the
#' data and widen the prediction by the `level` quantile of the absolute
#' residuals on the other part.
#'
#' The design and response are handled as by [thiessen()]: a matrix, a data
#' frame or a formula, factors encoded as treatment-contrast indicators.
#' The response must be numeric.
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
#'   on the response's scale; `rounds`; `planned_sweeps`, the budget used;
#'   `column_scores`, named by column; `zeroed`, the tessellations left at
#'   zero when the budget ran out while building; `train_score`, the
#'   residual root mean square after the build and after each round; and
#'   `seconds`, the elapsed fit time.
#' @references
#' Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
#' (2018). Distribution-free predictive inference for regression. Journal
#' of the American Statistical Association 113(523), 1094-1111.
#'
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. Journal of Computational and Graphical Statistics
#' 34(3), 859-871.
#' @seealso [xaddivortes_params()], [thiessen()], [predict.xaddivortes()]
#' @examples
#' if (core_experimental()) {
#'   set.seed(1)
#'   x <- matrix(runif(400), ncol = 2)
#'   y <- sin(6 * x[, 1]) + x[, 2] + rnorm(200, sd = 0.1)
#'   fit <- xaddivortes(x, y, params = xaddivortes_params(budget = 10),
#'                      seed = 1)
#'   fit
#'   head(predict(fit, x, interval = "prediction"))
#' }
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
               fast_call(match.call()))
}

#' @rdname xaddivortes
#' @export
xaddivortes.data.frame <- function(x, y, params = NULL,
                                   control = thiessen_control(), seed = NULL,
                                   threads = getOption("mc.cores", 1L),
                                   ...) {
  rlang::check_dots_empty()
  molded <- core_call(
    hardhat::mold(~ ., data = x, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric
  )
  new_fast_fit(design, as_response(y), params, control, seed, threads,
               fast_call(match.call()), blueprint = molded$blueprint)
}

#' @rdname xaddivortes
#' @export
xaddivortes.formula <- function(formula, data, params = NULL,
                                control = thiessen_control(), seed = NULL,
                                threads = getOption("mc.cores", 1L), ...) {
  rlang::check_dots_empty()
  molded <- core_call(
    hardhat::mold(formula, data, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric
  )
  new_fast_fit(design, encode_response(molded$outcomes), params, control,
               seed, threads, fast_call(match.call()),
               blueprint = molded$blueprint)
}

#' The method call with the generic's name, so `update()` re-evaluates it
#'
#' @param call The method's `match.call()`.
#' @return A call to `xaddivortes`.
#' @noRd
fast_call <- function(call) {
  call[[1L]] <- as.name("xaddivortes")
  call
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
  if (response$n != nrow(design)) {
    thiessen_abort(
      sprintf(
        "The design has %d rows and the response has %d values; they must agree.",
        nrow(design), response$n
      ),
      call = call_env
    )
  }
  check_group(control, "control", "thiessen_control", call = call_env)
  params_json <- fast_json(params, call = call_env)
  control <- resolve_outcome(control, response, call = call_env)
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
#' @export
predict.xaddivortes <- function(object, newdata = NULL,
                                interval = c("none", "prediction"),
                                level = 0.95, threads = NULL, ...) {
  rlang::check_dots_empty()
  interval <- match.arg(interval)
  predict.thiessen(object, newdata = newdata, type = "mean",
                   interval = interval, level = level, threads = threads)
}

#' @export
print.xaddivortes <- function(x, ...) {
  cat("XAddiVortes fast fit (experimental)\n")
  cat(sprintf("  %d observations, %d design columns, %d tessellations\n",
              length(x$y), x$n_features, tessellation_count(x)))
  cat(sprintf("  budget used: %.2f sweeps in %d rounds (%.2f s)\n",
              x$fast$planned_sweeps, x$fast$rounds, x$fast$seconds))
  if (x$fast$zeroed > 0) {
    cat(sprintf("  %d tessellations left at zero: the budget ran out\n",
                x$fast$zeroed))
  }
  cat(sprintf("  in-sample residual standard deviation: %s\n",
              format(signif(x$fast$sigma, 4))))
  invisible(x)
}

#' The number of tessellations of a fit
#' @noRd
tessellation_count <- function(fit) {
  m <- fit$control$mean_params$tessellations
  if (is.null(m)) 200L else as.integer(m)
}

#' @export
summary.xaddivortes <- function(object, ...) {
  scores <- sort(object$fast$column_scores, decreasing = TRUE)
  structure(
    list(
      n = length(object$y), p = object$n_features,
      tessellations = tessellation_count(object),
      budget_used = object$fast$planned_sweeps, rounds = object$fast$rounds,
      zeroed = object$fast$zeroed, sigma = object$fast$sigma,
      in_sample_rmse = object$in_sample_rmse, column_scores = scores,
      seconds = object$fast$seconds
    ),
    class = "summary.xaddivortes"
  )
}

#' @export
print.summary.xaddivortes <- function(x, ...) {
  cat("XAddiVortes fast fit (experimental)\n")
  cat(sprintf("  n = %d, %d design columns, %d tessellations\n", x$n, x$p,
              x$tessellations))
  cat(sprintf("  budget used %.2f sweeps, %d rounds, %d zeroed, %.2f s\n",
              x$budget_used, x$rounds, x$zeroed, x$seconds))
  cat(sprintf("  in-sample residual standard deviation %s\n",
              format(signif(x$sigma, 4))))
  cat("  column scores (mutual information with the response):\n")
  shown <- utils::head(x$column_scores, 10L)
  print(round(shown, 4))
  if (length(x$column_scores) > 10L) {
    cat(sprintf("  ... and %d more\n", length(x$column_scores) - 10L))
  }
  invisible(x)
}
