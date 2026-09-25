# One fit per row of docs/experimental.md through `thiessen()`: in a build
# with the feature each row fits, predicts and round-trips through
# `serialize()`; in a build without it each reports the feature by its
# condition class. The design has two columns, so a metric names two.

catalogue_rows <- function() {
  fixture <- small_fixture()
  x <- fixture$x
  y <- fixture$y
  n <- nrow(x)
  minkowski <- function(p, group = 0) {
    list(minkowski = list(p = p, group = group))
  }
  gower <- list(gower = list(kind = "numeric"))
  geometry <- function(...) term_params(geometry = geometry_params(...))
  floor <- unname(quantile(y, 0.2))
  list(
    minkowski = list(
      control = function() {
        small_control(
          mean_params = geometry(metric = list(minkowski(3), minkowski(3)))
        )
      },
      y = y
    ),
    manhattan = list(
      control = function() {
        small_control(
          mean_params = geometry(metric = list("manhattan", "manhattan"))
        )
      },
      y = y
    ),
    cosine = list(
      control = function() {
        small_control(
          mean_params = geometry(metric = list("cosine", "cosine"))
        )
      },
      y = y
    ),
    gower = list(
      control = function() {
        small_control(mean_params = geometry(metric = list(gower, gower)))
      },
      y = y
    ),
    mahalanobis = list(
      control = function() {
        small_control(
          mean_params = geometry(
            metric = list("mahalanobis", "mahalanobis"), precision = diag(2)
          )
        )
      },
      y = y
    ),
    composite = list(
      control = function() {
        small_control(
          mean_params = geometry(
            metric = list(minkowski(1.5, group = 0), minkowski(3, group = 1))
          )
        )
      },
      y = y
    ),
    weighted = list(
      control = function() {
        small_control(
          mean_params = term_params(
            structure = structure_params(
              inclusion = weighted_inclusion(c(2, 1))
            )
          )
        )
      },
      y = y
    ),
    dart = list(
      control = function() {
        small_control(
          mean_params = term_params(
            structure = structure_params(inclusion = dart_inclusion())
          )
        )
      },
      y = y
    ),
    linear = list(
      control = function() {
        small_control(
          mean_params = term_params(cell = cell_params(basis = "linear"))
        )
      },
      y = y
    ),
    soft = list(
      control = function() {
        small_control(mean_params = geometry(membership = soft_membership()))
      },
      y = y
    ),
    tobit = list(
      control = function() {
        small_control(outcome = tobit_outcome(lower = floor))
      },
      y = pmax(y, floor)
    ),
    aft = list(
      control = function() small_control(),
      y = right_censored_fixture(y)
    ),
    interval_censored = list(
      control = function() small_control(),
      y = interval_fixture(y)
    ),
    ordinal = list(
      control = function() small_control(),
      y = ordered_fixture(n)
    ),
    student_t = list(
      control = function() {
        small_control(outcome = student_t_outcome(df = 4))
      },
      y = y
    ),
    student_t_grid = list(
      control = function() {
        small_control(outcome = student_t_outcome(df = c(3, 6, 12)))
      },
      y = y
    ),
    laplace = list(
      control = function() small_control(outcome = laplace_outcome()),
      y = y
    )
  )
}

test_that("every row of the catalogue is reachable from thiessen()", {
  skip_if_not_installed("survival")
  x <- small_fixture()$x
  rows <- catalogue_rows()
  for (name in names(rows)) {
    row <- rows[[name]]
    if (!core_experimental()) {
      expect_error(
        thiessen(x, row$y, row$control(), seed = 1, chains = 1),
        class = "thiessen_requires_feature", label = name
      )
      next
    }
    fit <- thiessen(x, row$y, row$control(), seed = 1, chains = 1)
    expect_s3_class(fit, "thiessen")
    expect_length(predict(fit), nrow(x))
    copy <- unserialize(serialize(fit, NULL))
    expect_identical(predict(copy), predict(fit), label = name)
    expect_identical(
      dim(log_lik(fit)), c(fit$n_draws, nrow(x)), label = name
    )
  }
})

test_that("a Surv response of type right reaches the AFT family", {
  skip_if_not(core_experimental())
  skip_if_not_installed("survival")
  fixture <- small_fixture()
  y <- right_censored_fixture(fixture$y)

  fit <- thiessen(fixture$x, y, small_control(), seed = 1, chains = 1)

  expect_identical(attr(fit$control$outcome, "kind"), "aft")
  expect_identical(fit$model, "aft")
  expect_equal(fit$y, log(unclass(y)[, "time"]))
  expect_identical(log_lik(fit, newdata = fixture$x, y = y), log_lik(fit))
  expect_true(is.numeric(sigma(fit)))
})

test_that("a Surv response reaches the AFT family through a formula", {
  skip_if_not(core_experimental())
  skip_if_not_installed("survival")
  fixture <- small_fixture()
  frame <- data.frame(
    time = exp(fixture$y),
    event = rep(c(1, 0), length.out = nrow(fixture$x)),
    a = fixture$x[, 1], b = fixture$x[, 2]
  )

  fit <- thiessen(
    survival::Surv(time, event) ~ a + b, frame, small_control(), seed = 1,
    chains = 1
  )

  expect_identical(fit$model, "aft")
  expect_identical(log_lik(fit, newdata = frame), log_lik(fit))
})

test_that("an interval2 Surv reaches the interval-censored family", {
  skip_if_not(core_experimental())
  skip_if_not_installed("survival")
  fixture <- small_fixture()
  y <- interval_fixture(fixture$y)

  fit <- thiessen(fixture$x, y, small_control(), seed = 1, chains = 1)

  expect_identical(fit$model, "interval_censored")
  expect_identical(fit$response$lower[7], -Inf)
  expect_identical(fit$response$upper[6], Inf)
  expect_identical(log_lik(fit, newdata = fixture$x, y = y), log_lik(fit))
})

test_that("an ordered factor reaches the ordinal family", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  n <- nrow(fixture$x)
  y <- ordered_fixture(n)

  fit <- thiessen(fixture$x, y, small_control(), seed = 1, chains = 1)
  probs <- predict(fit, type = "probs")

  expect_identical(fit$control$outcome$categories, 3L)
  expect_identical(dim(probs), c(n, 3L))
  expect_identical(colnames(probs), levels(y))
  expect_equal(unname(rowSums(probs)), rep(1, n))
  expect_identical(sigma(fit), 1)
  expect_true(
    "cutpoint[1]" %in% posterior::variables(posterior::as_draws_df(fit))
  )
  expect_error(
    thiessen(
      fixture$x, y, small_control(outcome = ordinal_outcome(categories = 4)),
      seed = 1, chains = 1
    ),
    "levels",
    class = "thiessen_error"
  )
})

test_that("the draws carry the quantities the experimental models sample", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  variables <- function(control) {
    posterior::variables(posterior::as_draws_df(
      thiessen(fixture$x, fixture$y, control, seed = 1, chains = 1)
    ))
  }

  expect_true("df" %in% variables(
    small_control(outcome = student_t_outcome(df = c(3, 6, 12)))
  ))
  dart <- variables(small_control(mean_params = term_params(
    structure = structure_params(inclusion = dart_inclusion())
  )))
  expect_true(all(
    c("inclusion_weight[1]", "inclusion_weight[2]", "concentration") %in% dart
  ))
  expect_true("bandwidth[1]" %in% variables(
    small_control(mean_params = term_params(
      geometry = geometry_params(membership = soft_membership())
    ))
  ))
})

test_that("the sampler takes a Surv and reproduces an AFT fit bit for bit", {
  skip_if_not(core_experimental())
  skip_if_not_installed("survival")
  fixture <- core_fixture()
  y <- right_censored_fixture(fixture$y)

  through_fit <- thiessen(fixture$x, y, small_control(), seed = 1, chains = 1)
  sampler <- thiessen_sampler(fixture$x, y, small_control(), seed = 1)
  sampler$step(10)
  for (draw in seq_len(20)) {
    sampler$step(1)
    sampler$keep()
  }
  through_sampler <- sampler$finish()

  expect_identical(
    predict(through_sampler, type = "draws"),
    predict(through_fit, type = "draws")
  )
  expect_identical(
    through_sampler$in_sample_rmse, through_fit$in_sample_rmse
  )
})

test_that("set_response takes the sampler's own response shape", {
  skip_if_not(core_experimental())
  skip_if_not_installed("survival")
  fixture <- small_fixture()
  y <- right_censored_fixture(fixture$y)
  sampler <- thiessen_sampler(fixture$x, y, small_control(), seed = 1)

  expect_no_error(sampler$set_response(y))
  expect_error(
    sampler$set_response(fixture$y), "aft", class = "thiessen_error"
  )
})

test_that("posterior_predict draws on each family's support", {
  skip_if_not(core_experimental())
  skip_if_not_installed("survival")
  fixture <- small_fixture()
  x <- fixture$x
  y <- fixture$y
  n <- nrow(x)
  floor <- 0.2
  ceiling <- 0.55
  replicates <- function(response, ...) {
    fit <- thiessen(x, response, small_control(...), seed = 1)
    set.seed(5)
    draws <- posterior_predict(fit)
    expect_identical(dim(draws), c(fit$n_draws, n))
    draws
  }

  ordinal <- replicates(ordered_fixture(n))
  expect_true(all(ordinal %in% c(0, 1, 2)))
  expect_gt(length(unique(as.vector(ordinal))), 1L)

  expect_true(all(replicates(right_censored_fixture(y)) > 0))

  tobit <- replicates(
    pmin(pmax(y, floor), ceiling),
    outcome = tobit_outcome(lower = floor, upper = ceiling)
  )
  expect_true(all(tobit >= floor & tobit <= ceiling))
  expect_true(any(tobit == floor) || any(tobit == ceiling))

  expect_true(all(is.finite(replicates(interval_fixture(y)))))
  expect_true(all(is.finite(replicates(
    y, outcome = student_t_outcome(df = c(3, 6, 12))
  ))))
  expect_true(all(is.finite(replicates(y, outcome = student_t_outcome()))))
  expect_true(all(is.finite(replicates(y, outcome = laplace_outcome()))))
  expect_true(all(is.finite(replicates(
    y, outcome = student_t_outcome(df = c(2, 4))
  ))))
})

test_that("the Student-t and Laplace replicates are the mean plus errors at sigma", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  x <- fixture$x
  y <- fixture$y

  fit <- thiessen(x, y, small_control(outcome = student_t_outcome(df = 3)), seed = 1)
  state <- fit_state(fit)
  latent <- core_predict_draws(state, x, "latent")
  sigma <- core_sigma(state)
  set.seed(6)
  expected <- latent + sigma * matrix(stats::rt(length(latent), df = 3), nrow = nrow(latent))
  set.seed(6)
  expect_identical(posterior_predict(fit), expected)
  expect_true(all(sigma * sqrt(3) - sqrt(core_predict_draws(state, x, "variance")[, 1]) < 1e-12))

  fit <- thiessen(x, y, small_control(outcome = laplace_outcome()), seed = 1)
  state <- fit_state(fit)
  latent <- core_predict_draws(state, x, "latent")
  sigma <- core_sigma(state)
  set.seed(7)
  u <- stats::runif(length(latent)) - 0.5
  expected <- latent + sigma * matrix(-sign(u) * log1p(-2 * abs(u)), nrow = nrow(latent))
  set.seed(7)
  expect_identical(posterior_predict(fit), expected)
})
