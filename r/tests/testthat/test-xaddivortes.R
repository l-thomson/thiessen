# The XAddiVortes fast fit and the fast start: the feature gate, the
# settings, agreement with the core through the cross-binding fixture
# (core-xaddivortes-fixture.txt, copied from the crate by tools/vendor.sh),
# prediction and the warm start.

fast_control <- function() thiessen_control(tessellations = 15)

test_that("the fast fit needs the experimental feature", {
  skip_if(core_experimental())
  f <- core_fixture()
  expect_error(
    xaddivortes(f$x, f$y, control = fast_control(), seed = CORE_SEED),
    class = "thiessen_requires_feature"
  )
  expect_error(
    thiessen(f$x, f$y, control = small_control(), start = "fast", seed = 1),
    class = "thiessen_requires_feature"
  )
})

test_that("the settings' defaults are the core's", {
  shown <- lapply(formals(xaddivortes_params), eval)
  shown$worth <- shown$worth[[1]]
  core <- xaddivortes_defaults()
  expect_setequal(names(shown), names(core))
  expect_equal(shown[names(core)], core)
})

test_that("settings are checked in R and sent at full precision", {
  expect_s3_class(xaddivortes_params(budget = 5), "xaddivortes_params")
  expect_error(xaddivortes_params(budget = "a"))
  expect_error(xaddivortes_params(budget = Inf))
  expect_error(xaddivortes_params(max_columns = 1.5))
  expect_error(xaddivortes_params(tries = 0))
  expect_error(xaddivortes_params(worth = "best"))
  json <- jsonlite::fromJSON(thiessen:::fast_json(
    xaddivortes_params(value_scale = 1 / 3)
  ))
  expect_identical(json$value_scale, 1 / 3)
  expect_equal(length(json), length(formals(xaddivortes_params)))
})

test_that("a list in place of the settings names the argument", {
  f <- core_fixture()
  expect_error(
    xaddivortes(f$x, f$y, params = list(budget = 5), control = fast_control()),
    "`params`"
  )
  expect_error(
    thiessen(f$x, f$y, control = small_control(), start = "fast",
             fast = list(budget = 5)),
    "`fast`"
  )
})

test_that("the fast fit equals the core's bit for bit", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  lines <- readLines(test_path("core-xaddivortes-fixture.txt"))
  sigma_want <- as.numeric(sub("^sigma ", "", lines[1]))
  want <- as.numeric(lines[-1])
  fit <- xaddivortes(f$x, f$y, control = fast_control(), seed = CORE_SEED)
  expect_identical(fit$fast$sigma, sigma_want)
  expect_identical(sigma(fit), sigma_want)
  expect_identical(unname(predict(fit, f$x)), want)
})

test_that("the fast fit predicts, gives intervals from its sigma and prints", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  fit <- xaddivortes(f$x, f$y, params = xaddivortes_params(budget = 10),
                     control = fast_control(), seed = 3)
  expect_s3_class(fit, c("xaddivortes", "thiessen"))
  p <- predict(fit, f$x)
  expect_length(p, nrow(f$x))
  wide <- predict(fit, f$x, interval = "prediction", level = 0.95)
  expect_equal(colnames(wide), c("fit", "lower", "upper"))
  expect_equal(unname(wide[, "fit"]), p)
  expect_equal(unname(wide[, "upper"] - wide[, "lower"]),
               rep(2 * qnorm(0.975) * sigma(fit), nrow(f$x)),
               tolerance = 1e-10)
  expect_error(predict(fit, f$x, interval = "credible"))
  expect_error(predict(fit, f$x, type = "draws"))
  expect_true(fit$fast$planned_sweeps <= 10)
  expect_named(fit$fast$column_scores)
  expect_length(fit$fast$train_score, fit$fast$rounds + 1)
  expect_output(print(fit), "XAddiVortes fast fit")
  expect_output(print(summary(fit)), "column scores")
  again <- xaddivortes(f$x, f$y, params = xaddivortes_params(budget = 10),
                       control = fast_control(), seed = 3)
  expect_identical(predict(again, f$x), p)
})

test_that("the noise draw is the fit's sigma without the noise update", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  fit <- xaddivortes(
    f$x, f$y, params = xaddivortes_params(budget = 0.5, update_sigma = FALSE),
    control = fast_control(), seed = 3
  )
  expect_identical(sigma(fit), fit$fast$sigma)
})

test_that("a data frame and a formula reach the same fit as a matrix", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  df <- data.frame(a = f$x[, 1], b = f$x[, 2])
  m <- xaddivortes(f$x, f$y, control = fast_control(), seed = 2)
  d <- xaddivortes(df, f$y, control = fast_control(), seed = 2)
  g <- xaddivortes(y ~ a + b, data = cbind(df, y = f$y),
                   control = fast_control(), seed = 2)
  expect_identical(predict(d, df), predict(m, f$x))
  expect_identical(predict(g, df), predict(m, f$x))
})

test_that("chains start from their own fast fits", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  run <- function(start, threads = 1, fast = xaddivortes_params(budget = 5)) {
    suppressWarnings(thiessen(f$x, f$y, control = small_control(), seed = 4,
                              chains = 2, threads = threads, start = start,
                              fast = fast))
  }
  a <- run("fast")
  b <- run("fast", threads = 2)
  cold <- run("cold", fast = NULL)
  expect_identical(predict(a, f$x), predict(b, f$x))
  expect_false(identical(predict(a, f$x), predict(cold, f$x)))
  expect_true(a$warm_start)
})

test_that("the fast start's arguments are checked", {
  f <- core_fixture()
  expect_error(thiessen(f$x, f$y, start = "lukewarm"))
  expect_error(
    thiessen(f$x, f$y, control = small_control(),
             fast = xaddivortes_params(budget = 5)),
    "needs `start = \"fast\"`"
  )
  previous <- thiessen(f$x, f$y, control = small_control(), seed = 1,
                       chains = 1)
  expect_error(
    thiessen(f$x, f$y, control = small_control(), start = "fast",
             init = previous),
    "give one"
  )
})

test_that("guarded rounds never raise the training score", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  guarded <- xaddivortes(
    f$x, f$y, params = xaddivortes_params(budget = 10, accept_if_better = TRUE),
    control = fast_control(), seed = 3
  )
  steps <- diff(guarded$fast$train_score)
  expect_true(all(steps <= 1e-12 * head(guarded$fast$train_score, -1)))
})
