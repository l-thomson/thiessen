#' Configuration of a fit
#'
#' The configuration of Stone and Gosling (2025), s. 2, in the shape the
#' core stores it: an outcome family, one parameter group per ensemble,
#' and the sweep schedule. Each part has its own constructor with its own
#' documentation: [gaussian_outcome()] and [probit_outcome()] for the family,
#' [term_params()] for an ensemble, and [general_params()] for the
#' schedule. An argument left at its default gives the core's default, so
#' `thiessen_control()` is the published configuration.
#'
#' Attaching `variance_params` with a positive tessellation count selects
#' the heteroscedastic model, in which the residual variance varies with
#' x; it needs the Gaussian family with nu > 2, and the paper's count is
#' 40. The two ensembles share one covariate space: `geometry` and
#' `structure` set on `mean_params` apply to `variance_params` as well.
#'
#' One shortcut: `thiessen_control(tessellations = 200)` sets the mean
#' ensemble's size without spelling the group, since that count is the
#' single number most fits tune. Every other setting is named in its
#' group.
#'
#' `outcome` left `NULL` takes the family from the response at fit: a
#' numeric vector the Gaussian family, a two-level factor the probit
#' family, an ordered factor the ordinal family, and a [survival::Surv()]
#' the AFT or the interval-censored family by its type. A named family is
#' checked against the response and a mismatch is an error naming both;
#' see [thiessen()].
#'
#' The models reachable with [gaussian_outcome()] and [probit_outcome()]
#' are the published method and follow semantic versioning. Everything
#' else the core crate adds sits behind its `experimental` Cargo feature,
#' which a released build does not enable, so a configuration or a saved
#' fit naming such an option is rejected with the condition class
#' `thiessen_requires_feature`. The families it gates each have a
#' constructor here ([tobit_outcome()], [aft_outcome()],
#' [interval_censored_outcome()], [ordinal_outcome()],
#' [student_t_outcome()] and [laplace_outcome()]), so naming one is
#' portable, and a build accepting them is installed from source with
#' `THIESSEN_EXPERIMENTAL=1` in the environment; [core_experimental()]
#' reports the setting of the build in use. Each item graduates on its own
#' once calibrated and is then accepted as any other option, with no
#' separate opt-in. The table of experimental items and their status is
#' [`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).
#'
#' The core's calibration suite covers the configurations listed in
#' [`docs/calibrated.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/calibrated.md);
#' component options are verified in isolation, and every other combination
#' of the documented options is valid to run and is not separately
#' verified.
#'
#' @param outcome The outcome family, from [gaussian_outcome()],
#'   [probit_outcome()] or one of the experimental constructors listed
#'   under Details.
#'   `NULL`, the default, takes the family the response selects.
#' @param mean_params The ensemble describing the average, from
#'   [term_params()].
#' @param variance_params The ensemble describing the spread, from
#'   [term_params()]. `NULL`, the default, keeps the spread constant.
#' @param general_params The sweep schedule, from [general_params()].
#'   `NULL`, the default, is `general_params()`.
#' @param tessellations Shortcut for the mean ensemble's size:
#'   `thiessen_control(tessellations = 200)` is
#'   `thiessen_control(mean_params = term_params(tessellations = 200))`.
#'   An error if `mean_params` also sets a count.
#'
#' @return An object of class `"thiessen_control"` holding the four
#'   groups.
#'
#' @references
#' Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
#' Voronoi tessellations. *Journal of Computational and Graphical
#' Statistics* 34(3), 859-871. \doi{10.1080/10618600.2024.2414104}
#'
#' @examples
#' thiessen_control(tessellations = 50)
#'
#' thiessen_control(outcome = probit_outcome())
#'
#' thiessen_control(
#'   outcome = gaussian_outcome(nu = 10),
#'   mean_params = term_params(tessellations = 200, lambda_c = 25),
#'   variance_params = term_params(tessellations = 40),
#'   general_params = general_params(burn_in = 500, draws = 2000)
#' )
#' @export
thiessen_control <- function(outcome = NULL,
                             mean_params = term_params(),
                             variance_params = NULL,
                             general_params = NULL,
                             tessellations = NULL) {
  # The argument shares the constructor's name, so the default is resolved
  # here rather than in the signature, where it would be a recursive
  # promise.
  if (is.null(general_params)) {
    general_params <- thiessen::general_params()
  }
  if (!is.null(outcome) && !inherits(outcome, "thiessen_outcome")) {
    thiessen_abort(paste0(
      "`outcome` must come from an outcome constructor such as ",
      "`gaussian_outcome()`."
    ))
  }
  check_group(mean_params, "mean_params", "term_params")
  check_group(variance_params, "variance_params", "term_params",
              null_ok = TRUE)
  check_group(general_params, "general_params", "general_params")
  if (!is.null(tessellations)) {
    check_whole_number(tessellations, min = 0)
    if (!is.null(mean_params$tessellations)) {
      thiessen_abort(paste0(
        "`tessellations` is a shortcut for `mean_params`; set the count ",
        "in one place."
      ))
    }
    mean_params$tessellations <- tessellations
  }
  control <- structure(
    list(
      outcome = outcome,
      mean_params = mean_params,
      variance_params = variance_params,
      general_params = general_params
    ),
    class = "thiessen_control"
  )
  core_call(core_validate(config_json(control)))
  control
}

#' Print a control object
#'
#' Prints the outcome family and each parameter group, one per line.
#'
#' @param x An object of class `"thiessen_control"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(thiessen_control(tessellations = 50))
#' @export
print.thiessen_control <- function(x, ...) {
  cat("<thiessen_control>\n")
  shown <- list(
    outcome = if (is.null(x$outcome)) {
      "from the response"
    } else {
      format(x$outcome)
    },
    mean_params = format(x$mean_params),
    variance_params = if (is.null(x$variance_params)) {
      "none (constant spread)"
    } else {
      format(x$variance_params)
    },
    general_params = format(x$general_params)
  )
  for (name in names(shown)) {
    cat(sprintf("  %-15s %s\n", name, shown[[name]]))
  }
  invisible(x)
}
