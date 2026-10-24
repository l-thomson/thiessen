# The hardhat layer: molding a formula or a data frame into the numeric
# design the core takes, and forging new data against the stored blueprint so
# columns are matched by name and type.

#' The blueprint a control object calls for
#'
#' A declared `metric` has one entry per column of the data, so factors must
#' not expand; the entries would no longer line up with the design. With no
#' declared metric every column is Euclidean and a factor becomes d - 1
#' treatment-contrast indicators, the encoding of `model.matrix` and of
#' upstream AddiVortes.
#'
#' @param control An object of class `"thiessen_control"`.
#' @return A hardhat blueprint.
#' @noRd
blueprint_for <- function(control) {
  codes <- length(control$mean_params$geometry$metric) > 0L
  # Treatment contrasts are d - 1 columns only against an intercept; without
  # one `model.matrix` returns a column per level. The intercept column is
  # dropped from the design, so only the contrasts remain.
  hardhat::default_formula_blueprint(
    intercept = !codes,
    indicators = if (codes) "none" else "traditional"
  )
}

#' Mold a data frame or a formula into the core's design
#'
#' @param formula The formula to mold by: the caller's, or `~ .` for a
#'   data frame of covariates.
#' @param data A data frame.
#' @param control An object of class `"thiessen_control"`.
#' @param call The calling environment to report.
#' @return A list of the double matrix `design`, the hardhat `blueprint`
#'   and the molded `outcomes`.
#' @noRd
mold_design <- function(formula, data, control, call = rlang::caller_env()) {
  molded <- core_call(
    hardhat::mold(formula, data, blueprint = blueprint_for(control)),
    call = call
  )
  design <- encode_predictors(
    molded$predictors, molded$blueprint$indicators,
    control$mean_params$geometry$metric, call = call
  )
  list(design = design, blueprint = molded$blueprint,
       outcomes = molded$outcomes)
}

#' Encode molded predictors as the core's design
#'
#' @param predictors The `predictors` element of a mold or a forge.
#' @param indicators The blueprint's `indicators` setting.
#' @param metric The declared `metric`, at fit; `NULL` at predict, where the
#'   fit has already checked it.
#' @param call The calling environment to report.
#' @return A double matrix.
#' @noRd
encode_predictors <- function(predictors, indicators, metric = NULL,
                              call = rlang::caller_env()) {
  columns <- as.list(predictors)
  columns <- columns[names(columns) != "(Intercept)"]
  is_factor <- vapply(columns, is.factor, logical(1))
  if (indicators != "none" && any(is_factor)) {
    thiessen_abort(
      paste0(
        "Column ", names(columns)[which(is_factor)[1L]],
        " is not numeric and the blueprint did not encode it."
      ),
      call = call
    )
  }
  for (j in which(is_factor)) {
    if (!is.null(metric) && !identical(metric[[j]], "categorical")) {
      thiessen_abort(
        paste0(
          "Column ", names(columns)[j], " is a factor, so its `metric` entry ",
          "must be \"categorical\"."
        ),
        call = call
      )
    }
    # The forged factor carries the training levels, so the codes a fit and a
    # prediction produce agree.
    columns[[j]] <- as.double(as.integer(columns[[j]]) - 1L)
  }
  design <- as_design(
    matrix(
      unlist(columns, use.names = FALSE),
      nrow = length(columns[[1L]]),
      dimnames = list(NULL, names(columns))
    ),
    call = call
  )
  design
}

#' Parse a molded response
#'
#' hardhat carries a `Surv` object and an ordered factor through the mold
#' as one outcome column, so the column is parsed as `thiessen()` parses
#' `y`.
#'
#' @param outcomes The `outcomes` element of a mold or a forge.
#' @param call The calling environment to report.
#' @return An object of class `"thiessen_response"`.
#' @noRd
encode_response <- function(outcomes, call = rlang::caller_env()) {
  if (ncol(outcomes) != 1L) {
    thiessen_abort("The formula must name one response.", call = call)
  }
  as_response(outcomes[[1L]], call = call)
}

#' Forge new data against a fit's blueprint
#'
#' @param object An object of class `"thiessen"`.
#' @param newdata A data frame.
#' @param call The calling environment to report.
#' @return A double matrix with the columns of the fitted design.
#' @noRd
forge_design <- function(object, newdata, call = rlang::caller_env()) {
  forged <- core_call(
    hardhat::forge(newdata, object$blueprint),
    call = call
  )
  encode_predictors(
    forged$predictors, object$blueprint$indicators, call = call
  )
}
