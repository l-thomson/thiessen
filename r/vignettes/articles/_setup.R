# Shared setup for the precomputed experimental articles: the chunk
# options, the schedule, the seeds, the data generators, the paired fits
# and the scoring. Every article sources it in a visible chunk.

library(thiessen)
library(ggplot2)

options(mc.cores = 4)
theme_set(theme_minimal())
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  fig.width = 7, fig.height = 4, dpi = 96, out.width = "100%"
)

schedule <- general_params(burn_in = 300, draws = 300)
seeds <- 1:5

# The Friedman #1 function (Friedman 1991) over the first five columns.
friedman_f <- function(x) {
  10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
    10 * x[, 4] + 5 * x[, 5]
}

# Friedman #1: five informative covariates, p - 5 noise covariates, and
# Gaussian noise of standard deviation `sd`. A share `contaminate` of the
# training rows is moved by three noise standard deviations either way.
# Training rows and a held-out set drawn from the same law, with the
# noise-free truth.
friedman <- function(n, p = 10, sd = 1, seed, n_test = 500, contaminate = 0) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    list(x = x, f = friedman_f(x), y = friedman_f(x) + rnorm(n, sd = sd))
  }
  data <- list(train = draw(n), test = draw(n_test))
  moved <- sample(n, round(contaminate * n))
  data$train$y[moved] <- data$train$y[moved] + 3 * sd * sample(c(-1, 1), length(moved), TRUE)
  data
}

# A surface on the unit square: sin(pi x1) cos(pi x2), or
# 2 x1 - x2 + 0.5 sin(2 pi x1) under `f = "linear"`.
surface_f <- function(x, f = c("sine", "linear")) {
  f <- match.arg(f)
  switch(f,
    sine = sin(pi * x[, "x1"]) * cos(pi * x[, "x2"]),
    linear = 2 * x[, "x1"] - x[, "x2"] + 0.5 * sin(2 * pi * x[, "x1"])
  )
}

# The surface observed with Gaussian noise of standard deviation `sd`.
smooth_surface <- function(n, sd = 0.1, seed, n_test = 500, f = c("sine", "linear")) {
  f <- match.arg(f)
  set.seed(seed)
  draw <- function(n) {
    x <- cbind(x1 = runif(n), x2 = runif(n))
    truth <- surface_f(x, f)
    list(x = x, f = truth, y = truth + rnorm(n, sd = sd))
  }
  list(train = draw(n), test = draw(n_test))
}

# A plane in five covariates with one oblique crease,
# 3 |x1 + x2 - 1| / sqrt(2) + x . (1, -0.5, 1.5, -1, 0.5), which a Voronoi
# boundary can lie along and an axis-aligned split cannot; `f = "plane"`
# drops the crease.
creased_plane_f <- function(x, f = c("crease", "plane")) {
  f <- match.arg(f)
  plane <- drop(x %*% c(1, -0.5, 1.5, -1, 0.5))
  if (f == "plane") plane else 3 * abs(x[, 1] + x[, 2] - 1) / sqrt(2) + plane
}

# The creased plane observed with Gaussian noise of standard deviation
# `sd`.
creased_plane <- function(n, sd = 0.3, seed, n_test = 500, f = c("crease", "plane")) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * 5), n, 5, dimnames = list(NULL, paste0("x", 1:5)))
    truth <- creased_plane_f(x, f)
    list(x = x, f = truth, y = truth + rnorm(n, sd = sd))
  }
  list(train = draw(n), test = draw(n_test))
}

# Lognormal accelerated failure times on Friedman #1 over p columns, log
# time = f / 10 + N(0, sd^2), with independent lognormal censoring times
# scaled so that about `censored` of the training rows are censored. The
# training response is a `survival::Surv()`; the held-out rows carry
# their true log times and are uncensored.
lognormal_friedman <- function(n, p = 5, sd = 0.5, seed, n_test = 500, censored = 1 / 3) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    f <- friedman_f(cbind(x, matrix(0.5, n, max(0, 5 - p)))) / 10
    list(x = x, f = f, log_time = f + rnorm(n, sd = sd))
  }
  train <- draw(n)
  cutoff <- train$log_time + rnorm(n, mean = stats::qnorm(censored, lower.tail = FALSE) * sd, sd = sd)
  train$event <- as.numeric(train$log_time <= cutoff)
  train$time <- exp(pmin(train$log_time, cutoff))
  train$y <- survival::Surv(train$time, train$event)
  test <- draw(n_test)
  test$y <- exp(test$log_time)
  list(train = train, test = test)
}

# Ordered categories from a latent Friedman #1 surface over p columns:
# z = f / 10 + N(0, 1) cut at the latent quartiles into `categories`
# levels. The training response is an ordered factor; the held-out rows
# carry their true category probabilities.
ordinal_friedman <- function(n, p = 5, categories = 4, seed, n_test = 500) {
  set.seed(seed)
  draw <- function(n) {
    x <- matrix(runif(n * p), n, p,
                dimnames = list(NULL, paste0("x", seq_len(p))))
    list(x = x, f = friedman_f(cbind(x, matrix(0.5, n, max(0, 5 - p)))) / 10)
  }
  train <- draw(n)
  test <- draw(n_test)
  cutpoints <- quantile(train$f + rnorm(n), seq_len(categories - 1) / categories)
  categorise <- function(z) factor(findInterval(z, cutpoints), levels = 0:(categories - 1), ordered = TRUE)
  train$y <- categorise(train$f + rnorm(n))
  test$probs <- t(sapply(test$f, function(f) diff(c(0, pnorm(cutpoints - f), 1))))
  test$y <- categorise(test$f + rnorm(n_test))
  list(train = train, test = test, cutpoints = cutpoints)
}

# Evaluates a fit and records its wall-clock as the attribute `seconds`.
timed <- function(expr) {
  start <- Sys.time()
  fit <- expr
  attr(fit, "seconds") <- as.numeric(Sys.time() - start, units = "secs")
  fit
}

# One fit per seed for each named control, on the training rows.
paired_fits <- function(data, controls, seeds) {
  lapply(controls, function(control) {
    lapply(seeds, function(seed) {
      timed(thiessen(data$train$x, data$train$y, control, seed = seed))
    })
  })
}

# SoftBart chains as one fit: one `softbart_regression()` run per seed,
# on as many cores as there are seeds, pooled by the methods below the
# way the chains of a thiessen fit are pooled. A forked worker cannot
# hand its forest back, so each chain returns its draws of the mean at
# the rows of `newdata`, and the fit predicts at those rows only.
softbart_chains <- function(formula, data, newdata, seeds, ...) {
  frame <- as.data.frame(newdata)
  frame[[all.vars(formula)[1]]] <- 0
  timed(structure(
    list(
      newdata = newdata,
      chains = parallel::mclapply(seeds, function(seed) {
        set.seed(seed)
        fit <- SoftBart::softbart_regression(formula, data, frame, ..., verbose = FALSE)
        list(mean = fit$mu_test, sigma = fit$sigma)
      }, mc.cores = length(seeds))
    ),
    class = "softbart_chains"
  ))
}

# The rows of `x` among the rows a fit predicted at.
predicted_rows <- function(fit, x) {
  key <- function(m) apply(m, 1, paste, collapse = ",")
  rows <- match(key(x), key(fit$newdata))
  if (anyNA(rows)) stop("rows of `x` are not among the rows the fit predicted at")
  rows
}

# The predictive distribution of a fit at the rows of `x` as a mixture of
# normals with equal weights: a draws by rows matrix of the mean function
# and one of the residual standard deviation. Every method the articles
# score is reduced to this pair, so one scorer serves them all.
predictive <- function(fit, x) UseMethod("predictive")

predictive.thiessen <- function(fit, x) {
  list(mean = predict(fit, x, type = "draws"),
       sd = sqrt(predict(fit, x, type = "variance")))
}

predictive.softbart_chains <- function(fit, x) {
  parts <- chains(fit, x)
  list(mean = do.call(rbind, parts),
       sd = do.call(rbind, lapply(seq_along(parts), function(k) {
         matrix(fit$chains[[k]]$sigma, nrow(parts[[k]]), ncol(parts[[k]]))
       })))
}

# stochtree BART with a linear regression in every leaf, as one fit: one
# chain per seed in a forked worker, each returning its draws of the mean
# at the rows of `newdata` and of the residual standard deviation, pooled
# as the SoftBart chains are.
stochtree_chains <- function(data, newdata, seeds, num_trees, burn_in, draws) {
  basis <- function(x) cbind(1, x)
  timed(structure(
    list(
      newdata = newdata,
      chains = parallel::mclapply(seeds, function(seed) {
        fit <- stochtree::bart(
          data$train$x, data$train$y, leaf_basis_train = basis(data$train$x),
          X_test = newdata, leaf_basis_test = basis(newdata),
          num_gfr = 0, num_burnin = burn_in, num_mcmc = draws,
          general_params = list(random_seed = seed),
          mean_forest_params = list(num_trees = num_trees)
        )
        list(mean = t(fit$y_hat_test), sigma = sqrt(fit$sigma2_global_samples))
      }, mc.cores = length(seeds))
    ),
    class = "stochtree_chains"
  ))
}

predictive.stochtree_chains <- predictive.softbart_chains

# A fit without draws, as its predictions at the rows of `newdata`: a
# normal predictive (tgp's kriging mean and standard deviation) or a
# point prediction (Cubist, `sd` absent), scored on what it returns.
normal_fit <- function(newdata, mean, sd = NULL) {
  structure(list(newdata = newdata, mean = mean, sd = sd), class = "normal_fit")
}

predictive.normal_fit <- function(fit, x) {
  rows <- predicted_rows(fit, x)
  sd <- if (is.null(fit$sd)) rep(NA_real_, length(rows)) else fit$sd[rows]
  list(mean = matrix(fit$mean[rows], 1), sd = matrix(sd, 1))
}

# tgp's Bayesian treed linear model, predicting at the rows of `newdata`.
# `btlm()` draws from R's generator, so the seed is set here.
tgp_fit <- function(data, newdata, seed, burn_in, draws) {
  set.seed(seed)
  timed({
    fit <- tgp::btlm(data$train$x, data$train$y, XX = newdata,
                     BTE = c(burn_in, burn_in + draws, 1), R = 1, verb = 0)
    normal_fit(newdata, fit$ZZ.mean, sqrt(fit$ZZ.ks2))
  })
}

# Cubist's rules with a linear model in each, a point prediction at the
# rows of `newdata`.
cubist_fit <- function(data, newdata) {
  timed({
    fit <- Cubist::cubist(as.data.frame(data$train$x), data$train$y)
    normal_fit(newdata, predict(fit, as.data.frame(newdata)))
  })
}

# The quantile of each row's mixture, by bisection on the mixture CDF.
mixture_quantile <- function(p, m, s) {
  lower <- apply(m - 8 * s, 1, min)
  upper <- apply(m + 8 * s, 1, max)
  for (step in seq_len(60)) {
    mid <- (lower + upper) / 2
    below <- rowMeans(matrix(pnorm(mid, m, s), nrow(m))) < p
    lower[below] <- mid[below]
    upper[!below] <- mid[!below]
  }
  (lower + upper) / 2
}

# Held-out scores of one predictive distribution: the root mean squared
# error of the posterior mean against the noise-free truth, the coverage
# and mean width of the central predictive interval, and the CRPS and the
# log score of the response, exact for the mixture (scoringRules). The
# mixture is thinned to at most 400 draws, since the CRPS of a mixture
# costs the square of its size. A point prediction scores its error alone.
score_one <- function(p, data, level = 0.95) {
  rmse <- sqrt(mean((colMeans(p$mean) - data$test$f)^2))
  if (anyNA(p$sd)) {
    return(c(rmse = rmse, coverage = NA, width = NA, crps = NA, log_score = NA))
  }
  keep <- unique(round(seq(1, nrow(p$mean), length.out = min(nrow(p$mean), 400))))
  m <- t(p$mean[keep, , drop = FALSE])
  s <- t(p$sd[keep, , drop = FALSE])
  y <- data$test$y
  lower <- mixture_quantile((1 - level) / 2, m, s)
  upper <- mixture_quantile((1 + level) / 2, m, s)
  c(rmse = rmse,
    coverage = mean(y >= lower & y <= upper),
    width = mean(upper - lower),
    crps = mean(scoringRules::crps_mixnorm(y, m, s)),
    log_score = -mean(scoringRules::logs_mixnorm(y, m, s)))
}

# One row per fit and metric for every named list of fits. A method with
# no sampling variation is passed as a list of one fit.
score <- function(fits, data, level = 0.95) {
  rows <- lapply(names(fits), function(model) {
    per_fit <- lapply(seq_along(fits[[model]]), function(i) {
      scores <- score_one(predictive(fits[[model]][[i]], data$test$x), data, level)
      data.frame(model = model, fit = i, metric = names(scores), value = unname(scores))
    })
    do.call(rbind, per_fit)
  })
  scores <- do.call(rbind, rows)
  scores$model <- factor(scores$model, names(fits))
  scores$metric <- factor(scores$metric, c("rmse", "coverage", "width", "crps", "log_score"))
  scores
}

# The mean over fits with its standard error in parentheses, one row per
# model; a model with one fit shows the value alone, a score it lacks an
# empty cell.
score_table <- function(scores) {
  cell <- function(value) {
    if (anyNA(value)) return("")
    if (length(value) == 1) return(sprintf("%.3f", value))
    sprintf("%.3f (%.3f)", mean(value), sd(value) / sqrt(length(value)))
  }
  wide <- tapply(scores$value, scores[c("model", "metric")], cell)
  table <- data.frame(model = rownames(wide), as.data.frame.matrix(wide), row.names = NULL)
  names(table) <- c("model", "RMSE", "coverage", "width", "CRPS", "log score")
  table
}

# Held-out root mean squared error of one fit against the truth.
rmse <- function(fit, data) {
  sqrt(mean((colMeans(predictive(fit, data$test$x)$mean) - data$test$f)^2))
}

# One row per ensemble size and model for a list of `paired_fits()`
# results, one per size: the mean and standard error of the held-out
# RMSE over the fits, the largest R-hat among them, and the mean number
# of cells per tessellation.
size_table <- function(fits_by_size, sizes, data) {
  do.call(rbind, lapply(seq_along(sizes), function(i) {
    do.call(rbind, lapply(names(fits_by_size[[i]]), function(model) {
      fits <- fits_by_size[[i]][[model]]
      errors <- sapply(fits, rmse, data = data)
      data.frame(
        tessellations = sizes[i], model = model,
        RMSE = mean(errors), SE = sd(errors) / sqrt(length(errors)),
        `largest R-hat` = max(sapply(fits, function(fit) fit$convergence$rhat)),
        cells = mean(sapply(fits, function(fit) mean(thiessen_diagnostics(fit)$cell_count))),
        check.names = FALSE
      )
    }))
  }))
}

# The draws of the mean function at the rows of `x`, one draws by rows
# matrix per chain. A thiessen fit keeps its pooled draws in chain order.
chains <- function(fit, x) UseMethod("chains")

chains.thiessen <- function(fit, x) {
  draws <- predict(fit, x, type = "draws")
  iterations <- nrow(draws) / fit$n_chains
  lapply(seq_len(fit$n_chains), function(k) {
    draws[(k - 1) * iterations + seq_len(iterations), , drop = FALSE]
  })
}

chains.softbart_chains <- function(fit, x) {
  rows <- predicted_rows(fit, x)
  lapply(fit$chains, function(chain) chain$mean[, rows, drop = FALSE])
}

chains.stochtree_chains <- chains.softbart_chains

# The convergence of one fit over the mean function at the rows of `x`:
# the largest and median split R-hat and the smallest and median bulk
# effective sample size over those rows (Vehtari and others 2021), the
# wall-clock of the fit, and the smallest effective sample size per
# second.
mixing <- function(fit, x) {
  by_chain <- chains(fit, x)
  draws <- nrow(by_chain[[1]])
  array <- aperm(
    array(unlist(by_chain), dim = c(draws, ncol(by_chain[[1]]), length(by_chain))),
    c(1, 3, 2)
  )
  summary <- posterior::summarise_draws(posterior::as_draws_array(array), "rhat", "ess_bulk")
  seconds <- attr(fit, "seconds")
  data.frame(
    `largest R-hat` = max(summary$rhat), `median R-hat` = median(summary$rhat),
    `smallest ESS` = min(summary$ess_bulk), `median ESS` = median(summary$ess_bulk),
    seconds = seconds, `ESS per second` = min(summary$ess_bulk) / seconds,
    check.names = FALSE
  )
}

# One row of `mixing()` per named fit.
mixing_table <- function(fits, x) {
  rows <- lapply(names(fits), function(name) cbind(method = name, mixing(fits[[name]], x)))
  do.call(rbind, rows)
}
