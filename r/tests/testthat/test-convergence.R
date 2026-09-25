test_that("a one-chain fit reports no diagnostics and says so", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)

  expect_null(fit$convergence)
  expect_output(print(fit), "1 chain")
  expect_output(print(summary(fit)), "1 chain")
})

test_that("two chains carry R-hat and the effective sample sizes", {
  fixture <- small_fixture()

  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 2)
  )

  expect_identical(fit$convergence$n_chains, 2L)
  expect_gt(fit$convergence$rhat, 0.9)
  expect_gt(fit$convergence$ess_bulk, 0)
  expect_gt(fit$convergence$ess_tail, 0)
  expect_identical(summary(fit)$convergence, fit$convergence)
})

test_that("a short chain warns and the warning is repeated", {
  fixture <- small_fixture()

  # Twenty draws per chain cannot reach an effective sample size of 400.
  expect_warning(
    fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1,
                    chains = 2),
    class = "thiessen_warning"
  )

  expect_output(print(fit), "may not have converged")
  expect_output(print(summary(fit)), "may not have converged")
})

test_that("the message states both thresholds", {
  convergence <- list(
    n_chains = 2L, n_variables = 3L, rhat = 1.2, ess_bulk = 100,
    ess_tail = 150
  )

  expect_match(convergence_message(convergence), "R-hat 1.200")
  expect_match(convergence_message(convergence), "sample size 100")
  expect_null(convergence_message(NULL))
  expect_null(convergence_message(
    list(n_chains = 2L, n_variables = 3L, rhat = 1.0, ess_bulk = 500,
         ess_tail = 500)
  ))
})

test_that("the monitored rows are a subsample of the design", {
  expect_identical(monitored_rows(5L, 20L), 1:5)
  expect_identical(length(monitored_rows(1000L, 20L)), 20L)
  expect_identical(range(monitored_rows(1000L, 20L)), c(1L, 1000L))
})

test_that("a user function named after a measure does not break the fit", {
  fixture <- small_fixture()
  assign("rhat", function(...) stop("shadowed"), envir = globalenv())
  assign("ess_bulk", function(...) stop("shadowed"), envir = globalenv())
  assign("ess_tail", function(...) stop("shadowed"), envir = globalenv())
  withr::defer(
    rm(list = c("rhat", "ess_bulk", "ess_tail"), envir = globalenv())
  )

  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 2)
  )

  expect_true(is.numeric(fit$convergence$rhat))
  expect_true(is.numeric(fit$convergence$ess_bulk))
  expect_true(is.numeric(fit$convergence$ess_tail))
})
