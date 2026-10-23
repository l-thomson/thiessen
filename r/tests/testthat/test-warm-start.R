# Warm starts from a previous fit: `thiessen(init = , init_draw = )`.

quiet_fit <- function(...) {
  suppressWarnings(thiessen(...))
}

test_that("a warm start needs the feature", {
  skip_if(core_experimental())
  data <- small_fixture()
  previous <- thiessen(data$x, data$y, small_control(), seed = 1, chains = 1)
  expect_error(
    thiessen(data$x, data$y, small_control(), seed = 2, chains = 1,
             init = previous),
    class = "thiessen_requires_feature"
  )
})

test_that("a warm start reproduces and differs from the cold start", {
  skip_if_not(core_experimental())
  data <- small_fixture()
  previous <- thiessen(data$x, data$y, small_control(), seed = 1, chains = 1)
  a <- quiet_fit(data$x, data$y, small_control(), seed = 2, chains = 2,
                 init = previous)
  b <- quiet_fit(data$x, data$y, small_control(), seed = 2, chains = 2,
                 init = previous)
  cold <- quiet_fit(data$x, data$y, small_control(), seed = 2, chains = 2)
  expect_identical(predict(a, data$x, type = "draws"),
                   predict(b, data$x, type = "draws"))
  expect_false(identical(predict(a, data$x, type = "draws"),
                         predict(cold, data$x, type = "draws")))
  expect_true(a$warm_start)
  expect_false(cold$warm_start)
})

test_that("a saved previous fit warm-starts after readRDS", {
  skip_if_not(core_experimental())
  data <- small_fixture()
  previous <- thiessen(data$x, data$y, small_control(), seed = 1, chains = 1)
  path <- withr::local_tempfile(fileext = ".rds")
  saveRDS(previous, path)
  loaded <- readRDS(path)
  a <- thiessen(data$x, data$y, small_control(), seed = 2, chains = 1,
                init = previous)
  b <- thiessen(data$x, data$y, small_control(), seed = 2, chains = 1,
                init = loaded)
  expect_identical(predict(a, data$x, type = "draws"),
                   predict(b, data$x, type = "draws"))
})

test_that("too few draws warn that the chains share a start", {
  skip_if_not(core_experimental())
  data <- small_fixture()
  previous <- thiessen(data$x, data$y, small_control(), seed = 1, chains = 1)
  expect_no_warning(
    thiessen(data$x, data$y, small_control(), seed = 2, chains = 1,
             init = previous, init_draw = 20)
  )
  messages <- capture_warnings(
    thiessen(data$x, data$y, small_control(), seed = 2, chains = 3,
             init = previous, init_draw = 2)
  )
  expect_true(any(grepl("every chain starts from the same draw", messages)))
  expect_true(any(grepl("warm-started", messages)))
})

test_that("the previous fit must be of this model and data", {
  skip_if_not(core_experimental())
  data <- small_fixture()
  previous <- thiessen(data$x, data$y, small_control(), seed = 1, chains = 1)
  expect_error(
    thiessen(data$x, data$y + 1, small_control(), seed = 2, chains = 1,
             init = previous),
    "different data"
  )
})

test_that("the init arguments are checked", {
  data <- small_fixture()
  previous <- thiessen(data$x, data$y, small_control(), seed = 1, chains = 1)
  expect_error(
    thiessen(data$x, data$y, small_control(), chains = 1, init = "previous"),
    "must be a fit"
  )
  expect_error(
    thiessen(data$x, data$y, small_control(), chains = 1, init_draw = 2),
    "needs `init`"
  )
  expect_error(
    thiessen(data$x, data$y, small_control(), chains = 1, init = previous,
             init_draw = 21),
    "holds 20 draws"
  )
})

test_that("each chain continues its counterpart by default", {
  skip_if_not(core_experimental())
  data <- small_fixture()
  previous <- quiet_fit(data$x, data$y, small_control(), seed = 1,
                        chains = 2)
  default <- quiet_fit(data$x, data$y, small_control(), seed = 2, chains = 2,
                       init = previous)
  last <- quiet_fit(data$x, data$y, small_control(), seed = 2, chains = 2,
                    init = previous, init_draw = previous$n_draws)
  first <- quiet_fit(data$x, data$y, small_control(), seed = 2, chains = 2,
                     init = previous, init_draw = previous$n_draws / 2)
  draws <- function(fit) predict(fit, data$x, type = "draws")
  expect_false(identical(draws(default), draws(last)))
  # Chain 1 of the default starts where `first` does.
  expect_identical(draws(default)[1:20, ], draws(first)[1:20, ])
})
