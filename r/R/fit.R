#' Fit an AddiVortes model
#'
#' AddiVortes is Bayesian regression on a sum of Voronoi tessellations
#' (Stone and Gosling, 2025): the mean function is a sum of m tessellations,
#' each with a mean per cell, drawn by the Gibbs sampler of the paper. It
#' stands to BART (Chipman, George and McCulloch, 2010) as a tessellation
#' stands to a tree: a cell is a region of the covariate space rather than a
#' box, so a boundary oblique to the axes costs one cell rather than many
#' splits.
#'
#' A factor covariate becomes d - 1 treatment-contrast indicators, the first
#' level as reference, as `model.matrix` and upstream AddiVortes encode it.
#' Where `control` declares a `metric`, one entry per column, factors are
#' passed as integer level codes instead and each factor column must declare
#' `"categorical"`.
#'
#' The response selects the outcome family where `control` names none: a
#' numeric vector the Gaussian family, a two-level factor the probit family
#' (0 and 1 with the first level as the zero, as `glm` treats one), an
#' ordered factor the ordinal family (the `MASS::polr` convention), a
#' [survival::Surv()] of type `"right"` the AFT family and one of type
#' `"interval2"` the interval-censored family (an `NA` bound is one-sided
#' censoring and an equal pair an exact value). A family named in
#' `control` is checked against the response and never coerced: the
#' probit family also takes the numbers 0 and 1, and every other family
#' takes only the shape that selects it. This is `glm`'s rule, the
#' declared family authoritative and the response validated against it,
#' with "not declared" representable. The families beyond the Gaussian
#' and probit ones need a build with the core's `experimental` feature;
#' see [thiessen_control()].
#'
#' Missing (`NA`) and non-finite values in the covariates or the response
#' are rejected with an error; no row is dropped silently.
#'
#' `stats::update()` works on a fit: the call is stored, so
#' `update(fit, seed = 2)` refits with that argument replaced.
#'
#' With `chains` of two or more, the chains are run together with the seeds
#' the core derives from `seed`, each on one of `threads` threads, their
#' draws are pooled, and the fit carries
#' rank-normalised split R-hat and the bulk and tail effective sample sizes
#' of sigma and of the mean function at up to twenty training rows
#' (`posterior::summarise_draws()`). A fit warns, and `print()` and
#' `summary()` repeat the warning, where R-hat exceeds 1.01 or an effective
#' sample size falls below 400 (Vehtari and others, 2021). A fit of one
#' chain says so instead.
#'
#' A default fit runs four chains of 200 burn-in and 1000 draws each, on
#' `getOption("mc.cores", 1L)` threads, so on one thread it costs four
#' chains; `options(mc.cores = 4)` runs the chains on four cores for the
#' same draws, and the fit then takes less than half the one-thread time
#' (about 45 per cent at n = 200 on four cores of a 2025 laptop; pooling
#' the draws and the diagnostics run on one thread), so it costs under
#' two chains rather than four. On Friedman #1 with n = 200 and p = 10 the
#' smallest effective sample size over the monitored points is about 100
#' (threshold 400) and the largest R-hat about 1.05 (threshold 1.01), so
#' the default fit warns. More draws per chain, `general_params(draws = )`,
#' is the answer; `thinning` is not.
#'
#' @section Progress:
#'
#' Progress over the whole fit is signalled with progressr, so a session
#' reports it after `progressr::handlers()` and nothing is printed by
#' default; `progressr::handlers(global = TRUE)` sets one for a whole
#' session. The schedule raises one progression per sweep, to a maximum of
#' a hundred over the sweeps of every chain, then pooling the draws, saving
#' the state and the convergence summary, so the report closes when the
#' fit is complete rather than at the last sweep. The phases after the
#' sweeps scale with the kept draws and run on one thread, while the
#' sweeps scale with the training rows and run on `threads`, so their
#' share of the bar is set from that ratio: the bar is most of the way
#' along when the sweeps of a one-chain fit end, and about a third of the
#' way for four chains on four threads at n = 200. Each phase names itself
#' in a sticky message, which a terminal handler pushes above the bar
#' rather than overwriting, so the phase that is running is named whatever
#' handler is set. The draws do not depend on whether a handler is set.
#'
#' @section Persistence:
#'
#' A fit is a plain R object holding the sampler state, so [saveRDS()]
#' writes one and a later session reads it and predicts the same values,
#' with no refit. A fit written by a build with the core's `experimental`
#' feature and read by a build without it errors with the condition class
#' `thiessen_requires_feature`, naming the item and the feature, at the
#' first call that needs the state.
#'
#' @section Conditions:
#'
#' Errors raised by this package and by the core carry the condition class
#' `thiessen_error`, and its warnings carry `thiessen_warning`, so either
#' can be handled or silenced by class rather than by message. An error
#' naming the core's `experimental` feature carries
#' `thiessen_requires_feature` before it, so a configuration needing an
#' opt-in build is told apart from an invalid one. The
#' convergence warning fires on any short schedule, so silencing it
#' deliberately is a routine need:
#'
#' ```
#' withCallingHandlers(
#'   fit <- thiessen(x, y, control),
#'   thiessen_warning = function(condition) {
#'     invokeRestart("muffleWarning")
#'   }
#' )
#' ```
#'
#' @param x A numeric matrix of covariates, one row per observation, or a
#'   data frame. A numeric vector is taken as one column.
#' @param formula A two-sided formula. The left side names the response and
#'   the right side the covariates, `.` for every remaining column.
#' @param data A data frame holding the columns the formula names.
#' @param y The response, with one value per row of `x`: a numeric
#'   vector, a two-level factor, an ordered factor, or a
#'   [survival::Surv()] of type `"right"` or `"interval2"`. Under the
#'   probit model a numeric vector must hold 0 and 1.
#' @param control An object of class `"thiessen_control"`, from
#'   [thiessen_control()].
#' @param chains The number of chains to run, a whole number. Each chain
#'   has its own seed, derived from `seed` in the core, and the draws of the
#'   chains are pooled. Two or more chains give the convergence
#'   diagnostics; one chain does not. Default 4, the number Vehtari and
#'   others (2021) recommend and the Stan default.
#' @param threads The number of threads, a whole number. The chains are
#'   spread over at most this many threads, each chain on one thread with
#'   its own generator, so the draws do not depend on it; `predict()` on
#'   the fit splits the rows over the same number. The default reads
#'   `getOption("mc.cores", 1L)` when the fit is called, as Stan's
#'   interfaces do, so `options(mc.cores = 4)` is the one-line opt-in and
#'   a session that sets nothing runs on one thread, where a default fit
#'   is four chains long.
#' @param init `NULL`, the default, for the cold start; or a previous
#'   fit of the same model to the same data, from `thiessen()`. Each chain
#'   then starts from one of its kept draws instead of the cold start;
#'   only the starting state changes, not the posterior. Without
#'   `init_draw` and with as many chains as `init`, chain k continues from
#'   the last draw of chain k of `init`. Otherwise chain k (from 1) starts
#'   from pooled draw `init_draw - k + 1`, `init_draw` defaulting to the
#'   last; with fewer draws than chains up to it, every chain starts from
#'   it and the fit warns. This is the warm-start rule of stochtree. A
#'   clean R-hat across
#'   chains started from one previous fit is a check within that run, not
#'   across dispersed starts (Vehtari and others, 2021). Experimental:
#'   needs a build with the `experimental` feature.
#' @param init_draw The pooled draw of `init` the first chain starts
#'   from, a whole number from 1.
#' @param start Where each chain starts: `"cold"`, the sampler's own
#'   start, or `"fast"`, its own XAddiVortes fit ([xaddivortes()]), after
#'   which the same burn-in and draws run. Only the starting state differs,
#'   so the chains target the same posterior either way. `"fast"` needs the
#'   Gaussian family, cannot be combined with `init`, and needs a build
#'   with the `experimental` feature.
#' @param fast The XAddiVortes settings of a fast start, from
#'   [xaddivortes_params()]; `NULL` takes the defaults. Given with
#'   `start = "cold"` it is an error.
#' @param seed The seed of the chain. `NULL`, the default, draws one from
#'   R's stream, so [set.seed()] governs; a whole number in `[0, 2^53]`
#'   passes to the core unchanged, so the same value reproduces the same
#'   draws for a given package version and platform.
#' @param ... Passed to the method.
#'
#' @return An object of class `"thiessen"`: a list with the fitted state,
#'   the resolved configuration, the number of chains, the thread count,
#'   the number of kept draws, the
#'   convergence diagnostics where two or more chains ran, the seed used,
#'   the design, the response the in-sample fit is measured against (the
#'   numeric response, the factor codes, or under a censored family the
#'   observed values on the model's scale, the log times under the AFT
#'   family), the fitted values, the residuals against that response, the
#'   hardhat blueprint where one applies, and the call.
#'
#' @references
#' Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART: Bayesian
#' additive regression trees. *The Annals of Applied Statistics* 4(1),
#' 266-298. \doi{10.1214/09-AOAS285}
#'
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. *Journal of Computational and Graphical
#' Statistics* 34(3), 859-871. \doi{10.1080/10618600.2024.2414104}
#'
#' Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Buerkner, P.-C.
#' (2021). Rank-normalization, folding, and localization: an improved R-hat
#' for assessing convergence of MCMC. *Bayesian Analysis* 16(2), 667-718.
#' \doi{10.1214/20-BA1221}
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
#' fit
#'
#' # The default four chains carry the convergence diagnostics, and warn on
#' # a schedule this short; one chain reports none.
#' thiessen(x, y, control, seed = 1, chains = 1)
#'
#' frame <- data.frame(y = y, a = x[, 1], b = factor(x[, 2] > 0))
#' thiessen(y ~ a + b, frame, control, seed = 1)
#' @export
thiessen <- function(x, ...) {
  UseMethod("thiessen")
}

#' @rdname thiessen
#' @export
thiessen.default <- function(x, y, control = thiessen_control(), seed = NULL,
                             chains = 4,
                             threads = getOption("mc.cores", 1L),
                             init = NULL, init_draw = NULL,
                             start = c("cold", "fast"), fast = NULL, ...) {
  rlang::check_dots_empty()
  start <- rlang::arg_match(start)
  design <- as_design(x)
  response <- as_response(y)
  new_fit(design, response, control, seed, chains, threads,
          generic_call(match.call()), init = init, init_draw = init_draw,
          start = start, fast = fast)
}

#' @rdname thiessen
#' @export
thiessen.data.frame <- function(x, y, control = thiessen_control(),
                                seed = NULL, chains = 4,
                                threads = getOption("mc.cores", 1L),
                                init = NULL, init_draw = NULL,
                                start = c("cold", "fast"), fast = NULL,
                                ...) {
  rlang::check_dots_empty()
  start <- rlang::arg_match(start)
  molded <- core_call(
    hardhat::mold(~ ., data = x, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric
  )
  response <- as_response(y)
  new_fit(design, response, control, seed, chains, threads,
          generic_call(match.call()), blueprint = molded$blueprint,
          init = init, init_draw = init_draw, start = start, fast = fast)
}

#' @rdname thiessen
#' @export
thiessen.formula <- function(formula, data, control = thiessen_control(),
                             seed = NULL, chains = 4,
                             threads = getOption("mc.cores", 1L),
                             init = NULL, init_draw = NULL,
                             start = c("cold", "fast"), fast = NULL, ...) {
  rlang::check_dots_empty()
  start <- rlang::arg_match(start)
  molded <- core_call(
    hardhat::mold(formula, data, blueprint = blueprint_for(control))
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric
  )
  response <- encode_response(molded$outcomes)
  new_fit(design, response, control, seed, chains, threads,
          generic_call(match.call()), blueprint = molded$blueprint,
          init = init, init_draw = init_draw, start = start, fast = fast)
}

#' The method call with the generic's name, so `update()` can re-evaluate it
#'
#' @param call The method's `match.call()`.
#' @return A call to `thiessen`.
#' @noRd
generic_call <- function(call) {
  call[[1L]] <- as.name("thiessen")
  call
}

#' Fit the core and assemble the object the methods return
#'
#' @param design The numeric design.
#' @param response An object of class `"thiessen_response"`.
#' @param control An object of class `"thiessen_control"`.
#' @param seed The seed as the caller gave it.
#' @param chains The number of chains to run.
#' @param threads The number of threads the chains run on.
#' @param call The call to store.
#' @param blueprint The hardhat blueprint, or `NULL` for a matrix fit.
#' @param call_env The calling environment to report.
#' @return An object of class `"thiessen"`.
#' @noRd
new_fit <- function(design, response, control, seed, chains, threads, call,
                    blueprint = NULL, call_env = rlang::caller_env(),
                    init = NULL, init_draw = NULL, start = "cold",
                    fast = NULL) {
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
  control <- resolve_outcome(control, response, call = call_env)
  chains <- resolve_chains(chains, call = call_env)
  threads <- resolve_threads(threads, call = call_env)
  resolved <- resolve_seed(seed, call = call_env)
  if (start == "cold" && !is.null(fast)) {
    thiessen_abort(
      "`fast` sets the fast start; it needs `start = \"fast\"`.",
      call = call_env
    )
  }
  if (start == "fast" && !is.null(init)) {
    thiessen_abort(
      "`start = \"fast\"` and `init` both set the start; give one.",
      call = call_env
    )
  }
  fast_start <- if (start == "fast") {
    fast_json(fast, arg = "fast", call = call_env)
  }
  start <- resolve_init(init, init_draw, chains, call = call_env)
  # The progressor's life must span the whole fit, so the count never ends
  # it: `auto_finish` would close the handler on the last step, and the
  # frame this progressor belongs to exits when the fit is complete.
  phase_steps <- progress_phase_steps(control, nrow(design), chains, threads)
  report <- progressr::progressor(
    steps = progress_updates(control, chains) + sum(phase_steps),
    auto_finish = FALSE
  )
  fit <- core_call(
    run_schedule(control, design, response, resolved, chains, threads,
                 report, phase_steps[["pooling"]], start = start,
                 fast_start = fast_start),
    call = call_env
  )
  assemble_fit(fit, design, response, resolved, call, threads = threads,
               blueprint = blueprint, call_env = call_env, report = report,
               phase_steps = phase_steps,
               warm_start = !is.null(start) || !is.null(fast_start))
}

#' Run the sweep schedule of every chain and pool the draws
#'
#' The schedule runs here rather than in the core because progressr reports
#' by signalling a condition, and extendr evaluates an R callback through
#' `R_tryEval`, which clears R's handler stack: a condition raised inside a
#' callback the core calls reaches no handler established outside the
#' `.Call`. Driving the sampler from R signals where the handlers are live.
#' The chains advance together, a slice of sweeps at a time between
#' reports, each chain on its own thread when `threads` allows; the sweep
#' order within a chain is the core's own, so the draws are unchanged.
#'
#' @param control An object of class `"thiessen_control"`.
#' @param design The numeric design.
#' @param response An object of class `"thiessen_response"`.
#' @param seed The resolved seed.
#' @param chains The number of chains to run.
#' @param threads The number of threads the chains run on.
#' @param report A progressr progressor.
#' @param pooling_steps The steps pooling the chains takes.
#' @param start `NULL` for the cold start, or the list `resolve_init()`
#'   returns.
#' @param fast_start `NULL`, or the XAddiVortes settings as JSON, to
#'   start every chain from its own fast fit.
#' @return The list `core_finish()` returns.
#' @noRd
run_schedule <- function(control, design, response, seed, chains, threads,
                         report, pooling_steps = 1L, start = NULL,
                         fast_start = NULL) {
  schedule <- control$general_params
  config <- config_json(control)
  thinning <- as.integer(schedule$thinning)
  sweeps <- schedule$burn_in + schedule$draws * thinning
  updates <- progress_updates(control, chains)
  emitted <- 0L
  # `updates` reports spread evenly over the sweeps, and the sweeps of every
  # chain to the next of them, so burn-in advances in one call per report.
  gap <- function(completed) {
    ceiling((emitted + 1L) * sweeps / updates) - completed
  }
  tick <- function(completed) {
    while (emitted < updates &&
             (emitted + 1L) * sweeps <= completed * updates) {
      emitted <<- emitted + 1L
      report()
    }
  }
  report(amount = 0, class = "sticky",
         message = sweep_message(chains, threads))
  samplers <- lapply(seq_len(chains) - 1L, function(chain) {
    new_sampler_handle(config, design, response, seed, chain)
  })
  if (!is.null(start)) {
    core_samplers_start(samplers, start$state, start$draw, start$draws)
  }
  if (!is.null(fast_start)) {
    core_samplers_xaddivortes(samplers, fast_start, threads)
  }
  completed <- 0L
  while (completed < schedule$burn_in) {
    run <- max(1L, min(schedule$burn_in - completed, gap(completed)))
    core_samplers_advance(samplers, as.integer(run), 0L, thinning, threads)
    completed <- completed + run
    tick(completed)
  }
  kept <- 0L
  while (kept < schedule$draws) {
    run <- max(1L, min(schedule$draws - kept,
                       ceiling(gap(completed) / thinning)))
    core_samplers_advance(samplers, 0L, as.integer(run), thinning, threads)
    kept <- kept + run
    completed <- completed + run * thinning
    tick(completed)
  }
  report(amount = 0, class = "sticky", message = "pooling the draws")
  fit <- core_finish(samplers, threads)
  report(amount = pooling_steps)
  fit
}

#' Assemble the object the methods return from the core's fit list
#'
#' @param fit The list `core_finish()` returns.
#' @param design The numeric design.
#' @param response An object of class `"thiessen_response"`.
#' @param seed The resolved seed.
#' @param call The call to store.
#' @param threads The number of threads the fit predicts on.
#' @param blueprint The hardhat blueprint, or `NULL` for a matrix fit.
#' @param call_env The calling environment to report.
#' @param report A progressr progressor. The default is a disabled one,
#'   for a caller with no report of its own to advance.
#' @param phase_steps The steps each phase takes, named as
#'   `progress_phase_steps()` names them; `saving` and `summarising` are
#'   read here.
#' @param warm_start Whether the chains started from draws of a previous
#'   fit.
#' @return An object of class `"thiessen"`.
#' @noRd
assemble_fit <- function(fit, design, response, seed, call, threads = 1L,
                         blueprint = NULL, call_env = rlang::caller_env(),
                         report = progressr::progressor(
                           steps = 2L, enable = FALSE
                         ),
                         phase_steps = c(saving = 1L, summarising = 1L),
                         warm_start = FALSE) {
  # The core measures the in-sample fit against the observed values on
  # the model's scale where the working response is replaced by latents;
  # a plain response is kept as given.
  y <- if (response$kind == "plain") response$y else fit$y
  for (warning in fit$warnings) {
    rlang::warn(warning, class = "thiessen_warning")
  }
  # The payload exists before any save because `saveRDS` offers no hook to
  # create it at write time; an external pointer alone saves as a null
  # address.
  report(amount = 0, class = "sticky", message = "saving the state")
  state <- new.env(parent = emptyenv())
  state$handle <- fit$state
  state$payload <- core_call(core_state_payload(fit$state), call = call_env)
  report(amount = phase_steps[["saving"]])
  fit_object <- structure(
    list(
      state = state,
      control = control_from_config(
        jsonlite::fromJSON(fit$config, simplifyVector = FALSE)
      ),
      model = fit$model,
      n_chains = fit$n_chains,
      threads = threads,
      n_draws = fit$n_draws,
      in_sample_rmse = fit$in_sample_rmse,
      warnings = fit$warnings,
      seed = seed,
      warm_start = warm_start,
      n_features = ncol(design),
      blueprint = blueprint,
      response = response,
      x = design,
      y = y,
      fitted.values = fit$fitted_values,
      residuals = y - fit$fitted_values,
      call = call
    ),
    class = "thiessen"
  )
  report(amount = 0, class = "sticky", message = "summarising the draws")
  fit_object$convergence <- convergence_of(fit_object)
  warn_convergence(fit_object, call = call_env)
  report(amount = phase_steps[["summarising"]])
  fit_object
}
