# The XAddiVortes fast fit and the fast start: the feature gate, the
# settings, agreement with the core through the cross-binding fixture
# (core-xaddivortes-fixture.txt, copied from the crate by tools/vendor.sh),
# prediction and the warm start.

fast_control <- function() thiessen_control(tessellations = 15)

# Few rounds, so the fits stay quick.
working <- function(...) xaddivortes_params(rounds = 5, ...)

expect_working_fit <- function(fit, rounds = 5) {
  expect_length(fit$fast$train_score, rounds + 2)
  expect_true(is.finite(fit$fast$sigma) && fit$fast$sigma > 0)
}

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
  core <- xaddivortes_defaults()
  expect_setequal(names(shown), names(core))
  expect_equal(shown[names(core)], core)
})

test_that("settings are checked in R and sent at full precision", {
  expect_s3_class(xaddivortes_params(rounds = 5), "xaddivortes_params")
  expect_error(xaddivortes_params(rounds = "a"),
               "`rounds` must be a whole number", class = "thiessen_error")
  expect_error(xaddivortes_params(rounds = Inf),
               "`rounds` must be a whole number", class = "thiessen_error")
  expect_error(xaddivortes_params(max_columns = 1.5),
               "`max_columns` must be a whole number",
               class = "thiessen_error")
  expect_error(xaddivortes_params(replace_share = NA),
               "`replace_share` must be a number", class = "thiessen_error")
  json <- jsonlite::fromJSON(thiessen:::fast_json(
    xaddivortes_params(value_scale = 1 / 3)
  ))
  expect_identical(json$value_scale, 1 / 3)
  expect_equal(length(json), length(formals(xaddivortes_params)))
})

test_that("an out-of-range setting names the argument and its range", {
  expect_error(xaddivortes_params(rounds = -1),
               "`rounds` must be a whole number larger than or equal to 0",
               class = "thiessen_error")
  expect_error(xaddivortes_params(max_columns = 0),
               "`max_columns` must be a whole number larger than or equal to 1",
               class = "thiessen_error")
  expect_error(xaddivortes_params(value_scale = 0),
               "`value_scale` must be a number larger than 0",
               class = "thiessen_error")
  expect_error(xaddivortes_params(value_scale = 1.5),
               "`value_scale` must be a number between 0 and 1",
               class = "thiessen_error")
  expect_error(xaddivortes_params(replace_share = -0.1),
               "`replace_share` must be a number between 0 and 1",
               class = "thiessen_error")
  expect_error(xaddivortes_params(centres = 70000),
               "`centres` must be a whole number between 1 and 65535",
               class = "thiessen_error")
})

test_that("the settings print as their constructor call", {
  expect_output(print(xaddivortes_params(rounds = 20)),
                "^xaddivortes_params\\(rounds = 20, ")
})

test_that("a response outside the Gaussian family is refused up front", {
  f <- core_fixture()
  binary <- factor(f$y > stats::median(f$y))
  ordered <- factor(cut(f$y, 3), ordered = TRUE)
  only <- "fits the Gaussian family only, not the"
  expect_error(xaddivortes(f$x, binary, control = fast_control()),
               paste("`xaddivortes\\(\\)`", only, "probit"),
               class = "thiessen_error")
  expect_error(
    thiessen(f$x, binary, control = small_control(), start = "fast"),
    paste("`start = \"fast\"`", only, "probit"),
    class = "thiessen_error"
  )
  expect_error(
    thiessen(f$x, ordered, control = small_control(), start = "fast"),
    paste("`start = \"fast\"`", only, "ordinal"),
    class = "thiessen_error"
  )
})

test_that("a list in place of the settings names the argument", {
  f <- core_fixture()
  expect_error(
    xaddivortes(f$x, f$y, params = list(rounds = 5), control = fast_control()),
    "`params`",
    class = "thiessen_error"
  )
  expect_error(
    thiessen(f$x, f$y, control = small_control(), start = "fast",
             fast = list(rounds = 5)),
    "`fast`",
    class = "thiessen_error"
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
  fit <- xaddivortes(f$x, f$y, params = working(), control = fast_control(),
                     seed = 3)
  expect_working_fit(fit)
  expect_s3_class(fit, c("xaddivortes", "thiessen"))
  p <- predict(fit, f$x)
  expect_length(p, nrow(f$x))
  wide <- predict(fit, f$x, interval = "prediction", level = 0.95)
  expect_equal(colnames(wide), c("fit", "lower", "upper"))
  expect_equal(unname(wide[, "fit"]), p)
  expect_equal(unname(wide[, "upper"] - wide[, "lower"]),
               rep(2 * qnorm(0.975) * sigma(fit), nrow(f$x)),
               tolerance = 1e-10)
  expect_error(predict(fit, f$x, interval = "credible"),
               "`interval` must be one of", class = "thiessen_error")
  expect_error(predict(fit, f$x, type = "draws"), "type",
               class = "rlib_error_dots_nonempty")
  expect_named(fit$fast$column_scores)
  expect_length(fit$fast$train_score, 5 + 2)
  expect_output(print(fit), "XAddiVortes fast fit")
  expect_output(print(fit), "Call: xaddivortes")
  expect_output(print(fit), "seed 3")
  expect_output(print(fit), "15 tessellations, 5 rounds")
  expect_output(print(summary(fit)), "Column scores")
  expect_output(print(summary(fit)), "In-sample RMSE")
  again <- xaddivortes(f$x, f$y, params = working(), control = fast_control(),
                       seed = 3)
  expect_identical(predict(again, f$x), p)
})

test_that("the noise draw is the fit's sigma", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  for (rounds in c(0, 5)) {
    fit <- xaddivortes(f$x, f$y, params = xaddivortes_params(rounds = rounds),
                       control = fast_control(), seed = 3)
    expect_working_fit(fit, rounds)
    expect_identical(sigma(fit), fit$fast$sigma)
  }
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
  run <- function(start, threads = 1, fast = working()) {
    suppressWarnings(thiessen(f$x, f$y, control = small_control(), seed = 4,
                              chains = 2, threads = threads, start = start,
                              fast = fast))
  }
  a <- run("fast")
  b <- run("fast", threads = 2)
  cold <- run("cold", fast = NULL)
  expect_identical(predict(a, f$x), predict(b, f$x))
  expect_false(identical(predict(a, f$x), predict(cold, f$x)))
  expect_true(a$fast_start)
  expect_false(a$warm_start)
})

test_that("the fast start's arguments are checked", {
  f <- core_fixture()
  expect_error(thiessen(f$x, f$y, start = "lukewarm"),
               "`start` must be one of", class = "thiessen_error")
  expect_error(
    thiessen(f$x, f$y, control = small_control(),
             fast = xaddivortes_params(rounds = 5)),
    "needs `start = \"fast\"`",
    class = "thiessen_error"
  )
  previous <- thiessen(f$x, f$y, control = small_control(), seed = 1,
                       chains = 1)
  expect_error(
    thiessen(f$x, f$y, control = small_control(), start = "fast",
             init = previous),
    "give one",
    class = "thiessen_error"
  )
})

test_that("the one-draw fit has no log-likelihood", {
  skip_if_not(core_experimental())
  f <- core_fixture()
  fit <- xaddivortes(f$x, f$y, params = working(), control = fast_control(),
                     seed = 3)
  expect_error(log_lik(fit), "one point fit, not posterior draws",
               class = "thiessen_error")
})
