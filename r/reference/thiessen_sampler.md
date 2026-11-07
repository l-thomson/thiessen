# Drive the sampler one call at a time

**\[experimental\]**

## Usage

``` r
thiessen_sampler(x, y, control = thiessen_control(), seed = NULL)
```

## Arguments

- x:

  A numeric matrix of covariates, one row per observation. A numeric
  vector is taken as one column.

- y:

  The response, one value per row of `x`, in the shapes
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  takes: a numeric vector (labels in {0, 1} under the probit family), a
  two-level or ordered factor, or a
  [`survival::Surv()`](https://rdrr.io/pkg/survival/man/Surv.html). It
  selects the family where `control` names none.

- control:

  An object of class `"thiessen_control"`, from
  [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md).

- seed:

  The seed. `NULL`, the default, draws one from R's stream, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) governs; a whole
  number in `[0, 2^53]` gives the chain that
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  would run first.

## Value

An object of class `"thiessen_sampler"`: an environment holding the
verbs above.

## Details

An outcome family, a censoring scheme or an imputation scheme the
package does not ship is written in R against this loop, with no Rust
and no recompilation. A model is reachable exactly when it is a Gaussian
regression on a response the caller can rewrite each sweep, which covers
every latent-Gaussian data augmentation: probit, tobit, accelerated
failure time, interval-censored and ordinal. An augmentation needing
per-observation weights, such as logistic through Polya-Gamma, is not
reachable, because nothing sets the noise variances. Neither is the
geometry, tessellation membership, cell internals, the inclusion prior
or the proposals.

It follows the updatable sampler object of dbarts and the low-level
interface of stochtree: construct with the configuration, the data and a
seed, then drive the Gibbs loop yourself. Burn-in and thinning are the
caller's loop. Parameters sampled in the caller's loop, cutpoints for
instance, are not in `$finish()`'s draws, so the caller keeps and
diagnoses those.
[`vignette("sampler-api")`](https://l-thomson.github.io/thiessen/r/articles/sampler-api.md)
reimplements the probit family in R and checks it against the built-in
one.

The response is on the caller's scale through an affine map frozen at
construction, so a response outside the training range is legitimate.
The sampler owns its RNG, seeded at construction with the chain-0 seed
of
[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md);
driving the configured schedule by hand reproduces a one-chain fit bit
for bit. The loop cannot rewire tessellation membership or cell
internals. The `burn_in`, `draws` and `thinning` settings of the control
play no part here.

The returned object holds the loop's verbs:

- `$step(n = 1)`:

  Run `n` sweeps of the Gibbs loop.

- `$keep()`:

  Record the current state as a posterior draw.

- `$n_kept()`:

  The number of draws kept so far.

- `$set_response(y)`:

  Replace the response, keeping the tessellations, the cell values and
  sigma^2; the next sweep conditions on it. Takes the shapes
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  takes, checked against the sampler's family: labels in {0, 1} or a
  two-level factor under the probit family, an ordered factor under the
  ordinal family, a `Surv` under the AFT and interval-censored families.
  Under the censored families the latents reset to the new observed
  values.

- `$fitted_values()`:

  The current mean function at the training rows: f(x_i), or c + f(x_i)
  under the probit family.

- `$noise_variances()`:

  The current variance of y given f at each training row: sigma^2 under
  the Gaussian model, 1 under the probit family (the latent scale),
  s^2(x_i) under the heteroscedastic model.

- `$finish()`:

  The fit of the kept draws, as
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  returns one. Consumes the sampler: every later call on it errors.

## See also

[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md),
whose loop this is.

## Examples

``` r
fixture <- matrix(seq(0, 1, length.out = 40), ncol = 1)
response <- 3 * fixture[, 1]^2 - fixture[, 1]

sampler <- thiessen_sampler(fixture, response,
                            thiessen_control(tessellations = 10),
                            seed = 1)
sampler$step(20)
for (draw in seq_len(30)) {
  sampler$step(1)
  sampler$keep()
}
fit <- sampler$finish()
fit$n_draws
#> [1] 30
```
