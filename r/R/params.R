# The parameter groups of a configuration: nested control constructors,
# the `rpart.control()` pattern, so each group documents and validates its
# own arguments. Every name is a name in the core's stored configuration.

#' One ensemble of tessellations
#'
#' The size, priors and covariate space of one ensemble.
#' `thiessen_control()` takes one as `mean_params` and, for the
#' heteroscedastic model, one as `variance_params`.
#'
#' @param tessellations Number of tessellations in the ensemble. `NULL`,
#'   the default, resolves at fit to 200 as `mean_params` and to 0 as
#'   `variance_params`; a positive count as `variance_params` selects the
#'   heteroscedastic model (the paper's count is 40).
#' @param k Cell-value prior spread k: sigma_mu = w / (k sqrt(m)) with the
#'   half-width w the outcome family owns (Chipman, George and McCulloch
#'   2010, s. 4). Default 3. The variance ensemble's inverse-gamma cells
#'   do not use it.
#' @param lambda_c Cell-count prior rate lambda_c:
#'   b - 1 ~ Poisson(lambda_c). Default 5, following AddiVortes 0.6.8 and
#'   later; the paper reports 25.
#' @param geometry The covariate space, from [geometry_params()]. `NULL`,
#'   the default, takes the core's defaults. The ensembles share one
#'   covariate space: set it on `mean_params` and it applies to
#'   `variance_params` as well.
#' @param structure The covariate-inclusion prior, from
#'   [structure_params()]. `NULL` takes the core's defaults. Shared
#'   between the ensembles like `geometry`.
#' @param cell The within-cell response surface, from [cell_params()].
#'   `NULL` takes the core's default, one constant value per cell.
#'
#' @return An object of class `"term_params"`.
#'
#' @seealso [thiessen_control()], [geometry_params()],
#'   [structure_params()], [cell_params()]
#' @examples
#' term_params(tessellations = 200, lambda_c = 25)
#' @export
term_params <- function(tessellations = NULL, k = 3, lambda_c = 5,
                        geometry = NULL, structure = NULL, cell = NULL) {
  check_whole_number(tessellations, min = 0, allow_null = TRUE)
  check_number(k)
  check_number(lambda_c)
  check_group(geometry, "geometry", "geometry_params", null_ok = TRUE)
  check_group(structure, "structure", "structure_params", null_ok = TRUE)
  check_group(cell, "cell", "cell_params", null_ok = TRUE)
  # The `structure` argument shadows `base::structure` in this body.
  group <- list(
    tessellations = tessellations, k = k, lambda_c = lambda_c,
    geometry = geometry, structure = structure, cell = cell
  )
  class(group) <- "term_params"
  group
}

#' The covariate space of the ensembles
#'
#' @section Experimental metrics:
#'
#' The entries beyond `"euclidean"`, `"categorical"` and the sphere are
#' compiled only into a core built with its `experimental` feature (see
#' [experimental_outcomes] for the policy) and are named as the core
#' stores them: `list(minkowski = list(p = 3))` for the Minkowski distance
#' of order p >= 1, `"manhattan"` for its order-1 case, `"cosine"` for the
#' cosine distance, `list(gower = list(kind = "numeric"))` or
#' `list(gower = list(kind = "categorical"))` for one column of the Gower
#' distance, and `"mahalanobis"` for the Mahalanobis distance under
#' `precision`. The Minkowski, Manhattan, cosine and Gower entries take a
#' `group` label (default 0), `list(cosine = list(group = 1))`, so the
#' columns sharing a label form one composite distance.
#'
#' @param metric The metric of each covariate column, in column order: a
#'   list whose entries are `"euclidean"`, `"categorical"`, or
#'   `list(spherical = list(sphere = i))` for one coordinate of the sphere
#'   labelled `i`, its latitudes first and its longitude last, in radians;
#'   or one of the experimental entries below. `NULL`, the default, is
#'   Euclidean on every column. Entries are matched case-sensitively, so
#'   `"Euclidean"` is rejected. Non-Euclidean columns are not scaled.
#' @param sigma_c Prior and proposal standard deviation sigma_c of a
#'   centre coordinate. A Euclidean column is min-max scaled to
#'   \[-0.5, 0.5\] over its training range inside the sampler and `sigma_c`
#'   is on that scale, so 1 is the full range of a column. Default 0.8.
#' @param membership How an observation belongs to a tessellation's
#'   cells: `"hard"`, the published rule, or [soft_membership()]. `NULL`,
#'   the default, is `"hard"`.
#' @param precision The precision matrix of the Mahalanobis metric, a
#'   square numeric matrix over the columns of the encoded design,
#'   required exactly when an entry of `metric` is `"mahalanobis"`; it is
#'   checked at fit to be symmetric and positive definite. `NULL`, the
#'   default, is none. Experimental, as the metric it serves.
#'
#' @return An object of class `"geometry_params"`.
#'
#' @seealso [term_params()], [soft_membership()]
#' @examples
#' geometry_params(metric = list("euclidean", "categorical"))
#' @export
geometry_params <- function(metric = NULL, sigma_c = 0.8, membership = NULL,
                            precision = NULL) {
  check_number(sigma_c)
  if (!is.null(metric)) metric <- as.list(metric)
  check_option(membership, "membership", "hard", "soft_membership")
  if (!is.null(precision) &&
        (!is.matrix(precision) || !is.numeric(precision) ||
           nrow(precision) != ncol(precision))) {
    thiessen_abort("`precision` must be a square numeric matrix.")
  }
  structure(
    list(
      metric = metric, sigma_c = sigma_c, membership = membership,
      precision = precision
    ),
    class = "geometry_params"
  )
}

#' The covariate-inclusion prior of the ensembles
#'
#' @param omega Dimension-count prior parameter omega; omega / p is the
#'   prior probability of including a covariate. `NULL`, the default,
#'   resolves to min(3, p) at fit. Must satisfy 0 < omega <= p.
#' @param inclusion The prior weight of each covariate: `"uniform"`, the
#'   published prior, [weighted_inclusion()] or [dart_inclusion()].
#'   `NULL`, the default, is `"uniform"`.
#'
#' @return An object of class `"structure_params"`.
#'
#' @seealso [term_params()], [weighted_inclusion()], [dart_inclusion()]
#' @examples
#' structure_params(omega = 2)
#' @export
structure_params <- function(omega = NULL, inclusion = NULL) {
  check_number(omega, allow_null = TRUE)
  check_option(
    inclusion, "inclusion", "uniform",
    c("weighted_inclusion", "dart_inclusion")
  )
  structure(list(inclusion = inclusion, omega = omega),
            class = "structure_params")
}

#' The within-cell response surface of the ensembles
#'
#' @param basis The value a cell holds: `"constant"`, one value per cell,
#'   the published basis; or `"linear"`, a value that tilts across the
#'   cell, mu + beta' (x_A - c) over the active covariates centred at the
#'   cell's centre, with the slopes under the cell-value prior. The linear
#'   basis is compiled only into a core built with its `experimental`
#'   feature (see [experimental_outcomes] for the policy), needs every
#'   column min-max scaled, and applies to the mean ensemble only. `NULL`,
#'   the default, is `"constant"`.
#'
#' @return An object of class `"cell_params"`.
#'
#' @seealso [term_params()]
#' @examples
#' cell_params(basis = "constant")
#' @export
cell_params <- function(basis = NULL) {
  if (!is.null(basis)) {
    basis <- resignal(rlang::arg_match0(basis, c("constant", "linear")))
  }
  structure(list(basis = basis), class = "cell_params")
}

#' The sweep schedule of a fit
#'
#' @param burn_in Sweeps discarded before the kept draws. Default 200.
#' @param draws Posterior draws kept. Default 1000.
#' @param thinning Keep every `thinning`-th sweep after burn-in. Default 1.
#' @param prior_only Switch off the likelihood, so the chain draws from
#'   the prior and `predict()` gives prior predictive draws. Default
#'   `FALSE`.
#'
#' @return An object of class `"general_params"`.
#'
#' @seealso [thiessen_control()]
#' @examples
#' general_params(burn_in = 100, draws = 500)
#' @export
general_params <- function(burn_in = 200, draws = 1000, thinning = 1,
                           prior_only = FALSE) {
  check_whole_number(burn_in, min = 0)
  check_whole_number(draws, min = 0)
  check_whole_number(thinning, min = 0)
  check_flag(prior_only)
  structure(
    list(burn_in = burn_in, draws = draws, thinning = thinning,
         prior_only = prior_only),
    class = "general_params"
  )
}

#' Soft membership of observations in cells
#'
#' `r lifecycle::badge("experimental")`
#'
#' Kernel-weighted membership, the softening of the tree split of Linero
#' and Yang (2018) carried to the Voronoi assignment: observation i takes
#' weight proportional to exp(-d^2 / (2 tau^2)) in each cell, normalised
#' over the tessellation's centres, with tau a per-tessellation bandwidth
#' under an exponential prior and updated by a Metropolis step. Constant
#' cell basis and constant spread only. The bandwidth draws are carried
#' by [posterior::as_draws_df()] as `bandwidth[j]`.
#'
#' @inheritSection experimental_outcomes Experimental
#' @param rate Rate of the exponential prior on the bandwidth, on the
#'   scaled covariate space. Default 10, so the prior mean bandwidth is a
#'   tenth of a column's range.
#'
#' @return An object of class
#'   `c("thiessen_soft", "thiessen_membership", "thiessen_option")`, for
#'   the `membership` argument of [geometry_params()].
#'
#' @references
#' Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles
#' that adapt to smoothness and sparsity. *Journal of the Royal
#' Statistical Society: Series B* 80(5), 1087-1110.
#' \doi{10.1111/rssb.12293}
#'
#' @seealso [geometry_params()], [core_experimental()]
#' @examplesIf core_experimental()
#' geometry_params(membership = soft_membership(rate = 10))
#' @export
soft_membership <- function(rate = 10) {
  lifecycle::signal_stage("experimental", "soft_membership()")
  check_number(rate)
  new_option("soft", list(rate = rate), "membership")
}

#' Fixed inclusion weights over the covariates
#'
#' `r lifecycle::badge("experimental")`
#'
#' A fixed prior weight per column, the `cov_prior_vec` of bartMachine
#' (Kapelner and Bleich 2016): the prior on a subset of covariates given
#' its size is proportional to the product of the member weights, a
#' proposal picks the incoming covariate with probability proportional
#' to its weight, and a zero weight excludes the column. Equal weights
#' are the uniform prior and reproduce its draws exactly.
#'
#' @inheritSection experimental_outcomes Experimental
#' @param weights One non-negative finite weight per column of the
#'   encoded design, in column order, at least one positive.
#'
#' @return An object of class
#'   `c("thiessen_weighted", "thiessen_inclusion", "thiessen_option")`,
#'   for the `inclusion` argument of [structure_params()].
#'
#' @references
#' Kapelner, A. and Bleich, J. (2016). bartMachine: machine learning with
#' Bayesian additive regression trees. *Journal of Statistical Software*
#' 70(4), 1-40. \doi{10.18637/jss.v070.i04}
#'
#' @seealso [structure_params()], [dart_inclusion()],
#'   [core_experimental()]
#' @examplesIf core_experimental()
#' structure_params(inclusion = weighted_inclusion(c(2, 1, 1)))
#' @export
weighted_inclusion <- function(weights) {
  lifecycle::signal_stage("experimental", "weighted_inclusion()")
  if (!is.numeric(weights) || length(weights) == 0L || anyNA(weights)) {
    thiessen_abort(
      "`weights` must be a numeric vector of weights, without NA."
    )
  }
  new_option("weighted", list(weights = as.double(weights)), "inclusion")
}

#' The DART sparsity prior over the covariates
#'
#' `r lifecycle::badge("experimental")`
#'
#' The Dirichlet prior of Linero (2018) on the inclusion weights, as the
#' BART package ships it (`sparse = TRUE` with `a`, `b` and `rho`): the
#' weights are a sampled vector s ~ Dirichlet(theta / p) and the
#' concentration theta is drawn on a grid with lambda = theta / (theta +
#' rho) under a Beta(a, b) prior. The sampled weights and concentration
#' are carried by [posterior::as_draws_df()] as `inclusion_weight[j]` and
#' `concentration`.
#'
#' This is the prior of Linero (2018) carried over from trees. It is not the
#' Dirichlet AddiVortes of Stone and Gosling (in preparation), whose subset
#' prior draws covariates sequentially without replacement and whose weights
#' have an exact Gibbs update by latent augmentation.
#'
#' @inheritSection experimental_outcomes Experimental
#' @param a Beta shape a of the concentration prior. Default 0.5.
#' @param b Beta shape b of the concentration prior. Default 1.
#' @param rho The concentration scale rho. `NULL`, the default, resolves
#'   to the number of columns at fit.
#'
#' @return An object of class
#'   `c("thiessen_dart", "thiessen_inclusion", "thiessen_option")`, for
#'   the `inclusion` argument of [structure_params()].
#'
#' @references
#' Linero, A. R. (2018). Bayesian regression trees for high-dimensional
#' prediction and variable selection. *Journal of the American
#' Statistical Association* 113(522), 626-636.
#' \doi{10.1080/01621459.2016.1264957}
#'
#' @seealso [structure_params()], [weighted_inclusion()],
#'   [core_experimental()]
#' @examplesIf core_experimental()
#' structure_params(inclusion = dart_inclusion())
#' @export
dart_inclusion <- function(a = 0.5, b = 1, rho = NULL) {
  lifecycle::signal_stage("experimental", "dart_inclusion()")
  check_number(a)
  check_number(b)
  check_number(rho, allow_null = TRUE)
  new_option("dart", list(a = a, b = b, rho = rho), "inclusion")
}

#' Construct a classed component option
#'
#' The `new_outcome()` idiom for a value of a parameter-group field: the
#' object carries its own parameters and serialises as the field's tagged
#' form, so the constructor arguments and the stored form share one set
#' of names.
#'
#' @param kind The core's name for the variant.
#' @param fields The variant's parameters, `NULL` entries kept.
#' @param slot The field the option sits on.
#' @return A classed list carrying `kind` and `slot` as attributes.
#' @noRd
new_option <- function(kind, fields, slot) {
  structure(
    fields,
    kind = kind,
    slot = slot,
    class = c(
      paste0("thiessen_", kind), paste0("thiessen_", slot), "thiessen_option"
    )
  )
}

#' Reject an option argument that is neither its published value nor a
#' constructed one
#'
#' @param value The value to check.
#' @param name The argument name to report.
#' @param published The published value, a string.
#' @param constructors The constructors of the other values.
#' @param call The calling environment to report.
#' @noRd
check_option <- function(value, name, published, constructors,
                         call = rlang::caller_env()) {
  if (is.null(value) || identical(value, published) ||
        inherits(value, paste0("thiessen_", name))) {
    return(invisible(NULL))
  }
  thiessen_abort(
    paste0(
      "`", name, "` must be \"", published, "\" or come from `",
      paste0(constructors, "()", collapse = "` or `"), "`."
    ),
    call = call
  )
}

#' @export
format.term_params <- function(x, ...) {
  constructor_call("term_params", compact(unclass(x)))
}

#' @export
format.geometry_params <- function(x, ...) {
  constructor_call("geometry_params", compact(unclass(x)))
}

#' @export
format.structure_params <- function(x, ...) {
  constructor_call("structure_params", compact(unclass(x)))
}

#' @export
format.cell_params <- function(x, ...) {
  constructor_call("cell_params", compact(unclass(x)))
}

#' @export
format.general_params <- function(x, ...) {
  constructor_call("general_params", compact(unclass(x)))
}

#' @export
format.thiessen_option <- function(x, ...) {
  constructor_call(
    paste0(attr(x, "kind"), "_", attr(x, "slot")), compact(unclass(x))
  )
}

#' Print a parameter group
#'
#' @param x A parameter group from [term_params()], [geometry_params()],
#'   [structure_params()], [cell_params()] or [general_params()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' print(term_params(tessellations = 40))
#' @name print.params
NULL

#' @rdname print.params
#' @export
print.term_params <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

#' @rdname print.params
#' @export
print.geometry_params <- print.term_params

#' @rdname print.params
#' @export
print.structure_params <- print.term_params

#' @rdname print.params
#' @export
print.cell_params <- print.term_params

#' @rdname print.params
#' @export
print.general_params <- print.term_params

#' Print a component option
#'
#' @param x An option from [soft_membership()], [weighted_inclusion()] or
#'   [dart_inclusion()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examplesIf core_experimental()
#' print(dart_inclusion())
#' @export
print.thiessen_option <- print.term_params

# The classes a constructor call renders as a nested call.
NESTED_CLASSES <- c(
  "term_params", "geometry_params", "structure_params", "cell_params",
  "thiessen_option"
)

#' Drop the `NULL` entries of a list
#'
#' @param fields A named list.
#' @return The list without its `NULL` entries.
#' @noRd
compact <- function(fields) {
  fields[!vapply(fields, is.null, logical(1))]
}

#' Render a constructor call from a group's set fields
#'
#' A nested group or option renders as its own call; every other value is
#' deparsed, so the string parses back to the object.
#'
#' @param name The constructor's name.
#' @param fields The non-`NULL` fields.
#' @return A character string, `name(field = value, ...)`.
#' @noRd
constructor_call <- function(name, fields) {
  shown <- vapply(
    fields,
    function(value) {
      if (inherits(value, NESTED_CLASSES)) {
        format(value)
      } else {
        deparse1(value)
      }
    },
    character(1)
  )
  arguments <- paste(names(fields), "=", shown, collapse = ", ")
  if (length(fields) == 0L) arguments <- ""
  paste0(name, "(", arguments, ")")
}
