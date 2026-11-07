# The outcome families: constructor functions returning classed objects,
# the `binomial(link = "probit")` idiom. The object carries its own
# parameters and serialises as the configuration's `outcome` group, so the
# constructor arguments and the stored form share one set of names.

#' The Gaussian outcome family
#'
#' The Gaussian observation model of Stone and Gosling (2025): one sigma^2
#' drawn per sweep. Attaching a variance ensemble
#' (`variance_params = term_params(tessellations = ...)` with a positive
#' count) makes the model heteroscedastic, so the residual variance varies
#' with x.
#'
#' @param nu Degrees of freedom nu of the sigma^2 prior,
#'   sigma^2 ~ nu lambda / chi^2_nu. Default 6. A variance ensemble
#'   requires nu > 2.
#' @param q Calibration quantile q of the sigma^2 prior,
#'   Pr(sigma < sigma_hat) = q. Default 0.85.
#'
#' @return An object of class `c("thiessen_gaussian", "thiessen_outcome")`.
#'
#' @seealso [probit_outcome()], [thiessen_control()]
#' @examples
#' gaussian_outcome(nu = 3)
#' @export
gaussian_outcome <- function(nu = 6, q = 0.85) {
  check_number(nu)
  check_number(q)
  new_outcome("gaussian", list(nu = nu, q = q))
}

#' The binary probit outcome family
#'
#' P(y = 1 | x) = Phi(c + f(x)) with the Albert and Chib (1993) latent
#' augmentation. The latent scale is fixed at 1 for identification, so a
#' variance ensemble is not available under this family.
#'
#' @param offset The offset c. `NULL`, the default, resolves to
#'   Phi^-1(ybar) at fit, the BART `binaryOffset` default.
#'
#' @return An object of class `c("thiessen_probit", "thiessen_outcome")`.
#'
#' @seealso [gaussian_outcome()], [thiessen_control()]
#' @examples
#' probit_outcome()
#' @export
probit_outcome <- function(offset = NULL) {
  check_number(offset, allow_null = TRUE)
  new_outcome("probit", list(offset = offset))
}

#' Outcome families behind the core's experimental feature
#'
#' The tobit, AFT, interval-censored, ordinal, Student-t and Laplace
#' families are compiled only into a core built with its `experimental`
#' feature. This page states the policy they share.
#'
#' @section Experimental:
#'
#' This family is compiled only into a core built with its `experimental`
#' Cargo feature. The constructor exists in every build, so a script
#' naming the family is portable, but a fit or a validated configuration
#' is rejected with the condition class `thiessen_requires_feature`
#' unless the package was installed from source with
#' `THIESSEN_EXPERIMENTAL=1` in the environment; [core_experimental()]
#' reports the setting of the build in use. An experimental family sits
#' outside semantic versioning: its configuration and the values it draws
#' may change in any release. The table of experimental items and their
#' status is
#' [`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).
#'
#' @param nu Degrees of freedom nu of the sigma^2 prior,
#'   sigma^2 ~ nu lambda / chi^2_nu. Default 6. A variance ensemble
#'   requires nu > 2.
#' @param q Calibration quantile q of the sigma^2 prior,
#'   Pr(sigma < sigma_hat) = q. Default 0.85.
#'
#' @name experimental_outcomes
#' @keywords internal
NULL

#' The tobit outcome family
#'
#' `r lifecycle::badge("experimental")`
#'
#' The type-I tobit model (Tobin 1958) for a response censored at known
#' limits: a response value equal to a limit is read as censored on that
#' side, and the latent value behind it is drawn by the augmentation of
#' Chib (1992). At least one limit is required, and a response value
#' beyond a limit is rejected at fit.
#'
#' @inheritSection experimental_outcomes Experimental
#' @inheritParams experimental_outcomes
#' @param lower The lower censoring limit. `NULL`, the default, is no
#'   lower limit.
#' @param upper The upper censoring limit. `NULL`, the default, is no
#'   upper limit.
#'
#' @return An object of class `c("thiessen_tobit", "thiessen_outcome")`.
#'
#' @references
#' Tobin, J. (1958). Estimation of relationships for limited dependent
#' variables. *Econometrica* 26(1), 24-36. \doi{10.2307/1907382}
#'
#' @seealso [thiessen_control()], [core_experimental()]
#' @examplesIf core_experimental()
#' tobit_outcome(lower = 0)
#' @export
tobit_outcome <- function(lower = NULL, upper = NULL, nu = 6, q = 0.85) {
  lifecycle::signal_stage("experimental", "tobit_outcome()")
  check_number(lower, allow_null = TRUE)
  check_number(upper, allow_null = TRUE)
  check_number(nu)
  check_number(q)
  new_outcome("tobit", list(lower = lower, upper = upper, nu = nu, q = q))
}

#' The accelerated failure time outcome family
#'
#' `r lifecycle::badge("experimental")`
#'
#' The lognormal accelerated failure time model (Wei 1992) for a
#' right-censored time to event, the model of the BART package's `abart`:
#' ln T = f(x) + e with e ~ N(0, sigma^2), the log time of a censored row
#' drawn from its truncated conditional before each sweep.
#'
#' The times and the event indicator are data, not parameters: the
#' response is a [survival::Surv()] of type `"right"`, which selects this
#' family by itself. [predict()] and the fitted values are f(x) on the
#' log-time scale, the `abart` convention, and [log_lik()] is the
#' censored likelihood of a `Surv` response.
#'
#' @inheritSection experimental_outcomes Experimental
#' @inheritParams experimental_outcomes
#'
#' @return An object of class `c("thiessen_aft", "thiessen_outcome")`.
#'
#' @references
#' Wei, L. J. (1992). The accelerated failure time model: a useful
#' alternative to the Cox regression model in survival analysis.
#' *Statistics in Medicine* 11(14-15), 1871-1879.
#' \doi{10.1002/sim.4780111409}
#'
#' @seealso [thiessen_control()], [core_experimental()]
#' @examplesIf core_experimental()
#' aft_outcome()
#' @export
aft_outcome <- function(nu = 6, q = 0.85) {
  lifecycle::signal_stage("experimental", "aft_outcome()")
  check_number(nu)
  check_number(q)
  new_outcome("aft", list(nu = nu, q = q))
}

#' The interval-censored outcome family
#'
#' `r lifecycle::badge("experimental")`
#'
#' The interval-censoring observation scheme (Sun 2006) for a response
#' known only to lie between two row-specific bounds, an equal pair being
#' an exact value and an infinite endpoint one-sided censoring. The
#' censoring is taken as independent of the response, so the bounds enter
#' the likelihood only through the interval probability.
#'
#' The bounds are data, not parameters: the response is a
#' [survival::Surv()] of type `"interval2"`, `Surv(lower, upper, type =
#' "interval2")`, in which an `NA` bound is one-sided censoring and an
#' equal pair an exact value; it selects this family by itself.
#' [predict()] and the fitted values are the uncensored f(x), and
#' [log_lik()] is the interval likelihood of a `Surv` response.
#'
#' @inheritSection experimental_outcomes Experimental
#' @inheritParams experimental_outcomes
#'
#' @return An object of class
#'   `c("thiessen_interval_censored", "thiessen_outcome")`.
#'
#' @references
#' Sun, J. (2006). *The Statistical Analysis of Interval-censored Failure
#' Time Data*. Springer. \doi{10.1007/0-387-37119-2}
#'
#' @seealso [thiessen_control()], [core_experimental()]
#' @examplesIf core_experimental()
#' interval_censored_outcome()
#' @export
interval_censored_outcome <- function(nu = 6, q = 0.85) {
  lifecycle::signal_stage("experimental", "interval_censored_outcome()")
  check_number(nu)
  check_number(q)
  new_outcome("interval_censored", list(nu = nu, q = q))
}

#' The ordinal outcome family
#'
#' `r lifecycle::badge("experimental")`
#'
#' The ordered probit model of Albert and Chib (1993),
#' P(y <= k | x) = Phi(gamma_(k+1) - c - f(x)), for an ordered factor
#' response (the `MASS::polr` convention), its levels the categories 0 to
#' K - 1 in order. The latent variance is fixed at 1 and the first
#' cutpoint at 0 for identification, and the interior cutpoints are drawn
#' on the log-gap scale of Albert and Chib (2001). At K = 2 the model is
#' [probit_outcome()]. [predict()] is the expected category on the code
#' scale, and `predict(type = "probs")` the category probabilities with
#' the levels as column names.
#'
#' @inheritSection experimental_outcomes Experimental
#' @param categories Number of ordered categories K, at least 2. `NULL`,
#'   the default, is the number of levels of the response; a count that
#'   differs from it is an error.
#' @param offset The offset c. `NULL`, the default, resolves at fit to
#'   Phi^-1 of the share of rows above the first category.
#' @param cutpoint_sd Standard deviation of the N(0, cutpoint_sd^2) prior
#'   on the log-gaps between interior cutpoints. Default 1.
#'
#' @return An object of class `c("thiessen_ordinal", "thiessen_outcome")`.
#'
#' @references
#' Albert, J. H. and Chib, S. (2001). Sequential ordinal modeling with
#' applications to survival data. *Biometrics* 57(3), 829-836.
#' \doi{10.1111/j.0006-341X.2001.00829.x}
#'
#' @seealso [probit_outcome()], [thiessen_control()], [core_experimental()]
#' @examplesIf core_experimental()
#' ordinal_outcome(categories = 4)
#' @export
ordinal_outcome <- function(categories = NULL, offset = NULL,
                            cutpoint_sd = 1) {
  lifecycle::signal_stage("experimental", "ordinal_outcome()")
  check_whole_number(categories, min = 2, allow_null = TRUE)
  check_number(offset, allow_null = TRUE)
  check_number(cutpoint_sd)
  new_outcome(
    "ordinal",
    list(
      categories = categories, offset = offset, cutpoint_sd = cutpoint_sd
    )
  )
}

#' The Student-t outcome family
#'
#' `r lifecycle::badge("experimental")`
#'
#' The independent Student-t model of Geweke (1993) for a continuous
#' response with outliers: y = f(x) + e with e ~ t_df(0, sigma^2), drawn
#' through its scale-mixture representation. The degrees of freedom are
#' fixed at a value, or drawn each sweep over a grid carrying a uniform
#' prior; no continuous sampler over them exists, df being weakly
#' identified.
#'
#' @inheritSection experimental_outcomes Experimental
#' @inheritParams experimental_outcomes
#' @param df The error degrees of freedom: one value, the default being
#'   4, or a grid of at least two strictly increasing values drawn over.
#'
#' @return An object of class
#'   `c("thiessen_student_t", "thiessen_outcome")`.
#'
#' @references
#' Geweke, J. (1993). Bayesian treatment of the independent Student-t
#' linear model. *Journal of Applied Econometrics* 8(S1), S19-S40.
#' \doi{10.1002/jae.3950080504}
#'
#' @seealso [laplace_outcome()], [thiessen_control()],
#'   [core_experimental()]
#' @examplesIf core_experimental()
#' student_t_outcome(df = c(3, 6, 12))
#' @export
student_t_outcome <- function(df = 4, nu = 6, q = 0.85) {
  lifecycle::signal_stage("experimental", "student_t_outcome()")
  if (!is.numeric(df) || length(df) == 0L || anyNA(df)) {
    thiessen_abort(
      "`df` must be a numeric vector of degrees of freedom, without NA."
    )
  }
  check_number(nu)
  check_number(q)
  new_outcome("student_t", list(df = as.double(df), nu = nu, q = q))
}

#' The Laplace outcome family
#'
#' `r lifecycle::badge("experimental")`
#'
#' The Laplace model for a continuous response with outliers:
#' y = f(x) + e with e ~ Laplace(0, sigma), drawn through the
#' normal-exponential mixture of Park and Casella (2008). The errors have
#' exponential tails, so a wild observation is discounted at rate 1/|r|
#' against the Student-t model's 1/r^2.
#'
#' @inheritSection experimental_outcomes Experimental
#' @inheritParams experimental_outcomes
#'
#' @return An object of class `c("thiessen_laplace", "thiessen_outcome")`.
#'
#' @references
#' Park, T. and Casella, G. (2008). The Bayesian lasso. *Journal of the
#' American Statistical Association* 103(482), 681-686.
#' \doi{10.1198/016214508000000337}
#'
#' @seealso [student_t_outcome()], [thiessen_control()],
#'   [core_experimental()]
#' @examplesIf core_experimental()
#' laplace_outcome()
#' @export
laplace_outcome <- function(nu = 6, q = 0.85) {
  lifecycle::signal_stage("experimental", "laplace_outcome()")
  check_number(nu)
  check_number(q)
  new_outcome("laplace", list(nu = nu, q = q))
}

#' Construct a classed outcome family
#'
#' @param kind The core's name for the family.
#' @param fields The family's parameters, `NULL` entries kept.
#' @return A classed list carrying `kind` as an attribute.
#' @noRd
new_outcome <- function(kind, fields) {
  structure(
    fields,
    kind = kind,
    class = c(paste0("thiessen_", kind), "thiessen_outcome")
  )
}

#' The core's tagged form of a family or a component option
#'
#' @param x An object carrying its variant name as the `kind` attribute.
#' @return A one-entry named list.
#' @noRd
tagged_group <- function(x) {
  stats::setNames(list(compact(unclass(x))), attr(x, "kind"))
}

#' @export
format.thiessen_outcome <- function(x, ...) {
  # The constructor's name, not the core's: `kind` is the serialised form.
  constructor_call(
    paste0(attr(x, "kind"), "_outcome"), compact(unclass(x))
  )
}

#' Print an outcome family
#'
#' Prints the family and its parameters on one line.
#'
#' @param x An object of class `"thiessen_outcome"`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(gaussian_outcome())
#' @export
print.thiessen_outcome <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}
