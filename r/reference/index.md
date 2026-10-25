# Package index

## Fitting

- [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  : Fit an AddiVortes model
- [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)
  : Configuration of a fit

## Fast fits

- [`xaddivortes()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes.md)
  : Fit XAddiVortes, the fast fit of AddiVortes
- [`predict(`*`<xaddivortes>`*`)`](https://l-thomson.github.io/thiessen/r/reference/predict.xaddivortes.md)
  : Predictions of an XAddiVortes fit
- [`summary(`*`<xaddivortes>`*`)`](https://l-thomson.github.io/thiessen/r/reference/summary.xaddivortes.md)
  [`print(`*`<summary.xaddivortes>`*`)`](https://l-thomson.github.io/thiessen/r/reference/summary.xaddivortes.md)
  : Summarise an XAddiVortes fit
- [`print(`*`<xaddivortes>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.xaddivortes.md)
  : Print an XAddiVortes fit
- [`xaddivortes_params()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_params.md)
  : XAddiVortes settings
- [`xaddivortes_defaults()`](https://l-thomson.github.io/thiessen/r/reference/xaddivortes_defaults.md)
  : The core's default XAddiVortes settings

## Outcome families

- [`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md)
  : The Gaussian outcome family
- [`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md)
  : The binary probit outcome family
- [`print(`*`<thiessen_outcome>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.thiessen_outcome.md)
  : Print an outcome family

## Experimental outcome families

Compiled only into a core built with its `experimental` feature; see
`experimental_outcomes`.

- [`tobit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/tobit_outcome.md)
  **\[experimental\]** : The tobit outcome family
- [`aft_outcome()`](https://l-thomson.github.io/thiessen/r/reference/aft_outcome.md)
  **\[experimental\]** : The accelerated failure time outcome family
- [`interval_censored_outcome()`](https://l-thomson.github.io/thiessen/r/reference/interval_censored_outcome.md)
  **\[experimental\]** : The interval-censored outcome family
- [`ordinal_outcome()`](https://l-thomson.github.io/thiessen/r/reference/ordinal_outcome.md)
  **\[experimental\]** : The ordinal outcome family
- [`student_t_outcome()`](https://l-thomson.github.io/thiessen/r/reference/student_t_outcome.md)
  **\[experimental\]** : The Student-t outcome family
- [`laplace_outcome()`](https://l-thomson.github.io/thiessen/r/reference/laplace_outcome.md)
  **\[experimental\]** : The Laplace outcome family
- [`experimental_outcomes`](https://l-thomson.github.io/thiessen/r/reference/experimental_outcomes.md)
  : Outcome families behind the core's experimental feature

## Parameter groups

- [`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)
  : One ensemble of tessellations
- [`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md)
  : The covariate space of the ensembles
- [`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md)
  : The covariate-inclusion prior of the ensembles
- [`cell_params()`](https://l-thomson.github.io/thiessen/r/reference/cell_params.md)
  : The within-cell response surface of the ensembles
- [`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md)
  : The sweep schedule of a fit
- [`print(`*`<term_params>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.params.md)
  [`print(`*`<geometry_params>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.params.md)
  [`print(`*`<structure_params>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.params.md)
  [`print(`*`<cell_params>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.params.md)
  [`print(`*`<general_params>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.params.md)
  [`print(`*`<xaddivortes_params>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.params.md)
  : Print a parameter group
- [`print(`*`<thiessen_control>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.thiessen_control.md)
  : Print a control object

## Experimental component options

Values of the parameter groups compiled only into a core built with its
`experimental` feature; see `experimental_outcomes` for the policy.

- [`soft_membership()`](https://l-thomson.github.io/thiessen/r/reference/soft_membership.md)
  **\[experimental\]** : Soft membership of observations in cells
- [`weighted_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/weighted_inclusion.md)
  **\[experimental\]** : Fixed inclusion weights over the covariates
- [`dart_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/dart_inclusion.md)
  **\[experimental\]** : The DART sparsity prior over the covariates
- [`print(`*`<thiessen_option>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.thiessen_option.md)
  : Print a component option

## Fitted-model methods

- [`predict(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md)
  : Posterior predictions from a fitted model
- [`fitted(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/fitted.thiessen.md)
  [`residuals(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/fitted.thiessen.md)
  [`nobs(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/fitted.thiessen.md)
  : Fitted values, residuals and observation count of a fitted model
- [`sigma(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/sigma.thiessen.md)
  : Residual standard deviation of a fitted model
- [`summary(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/summary.thiessen.md)
  [`print(`*`<summary.thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/summary.thiessen.md)
  : Summarise a fitted model
- [`print(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.thiessen.md)
  : Print a fitted model
- [`plot(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/plot.thiessen.md)
  : Trace plots of a fitted model

## Posterior and rstantools generics

- [`as_draws_df(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/as_draws_df.thiessen.md)
  [`as_draws_array(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/as_draws_df.thiessen.md)
  : Posterior draws of a fitted model
- [`ndraws(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/ndraws.thiessen.md)
  [`nchains(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/ndraws.thiessen.md)
  : Number of draws and chains of a fitted model
- [`posterior_predict(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/posterior_predict.thiessen.md)
  : Draw from the posterior predictive distribution
- [`posterior_epred(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/posterior_epred.thiessen.md)
  : Draw from the posterior distribution of the expected response
- [`log_lik(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/log_lik.thiessen.md)
  : Pointwise log-likelihood of a fitted model
- [`predictive_interval(`*`<thiessen>`*`)`](https://l-thomson.github.io/thiessen/r/reference/predictive_interval.thiessen.md)
  : Central posterior predictive interval
- [`reexports`](https://l-thomson.github.io/thiessen/r/reference/reexports.md)
  [`posterior_predict`](https://l-thomson.github.io/thiessen/r/reference/reexports.md)
  [`posterior_epred`](https://l-thomson.github.io/thiessen/r/reference/reexports.md)
  [`log_lik`](https://l-thomson.github.io/thiessen/r/reference/reexports.md)
  [`predictive_interval`](https://l-thomson.github.io/thiessen/r/reference/reexports.md)
  : Objects exported from other packages

## Diagnostics

- [`thiessen_diagnostics()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_diagnostics.md)
  : Per-draw sampler diagnostics of a fitted model
- [`variable_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/variable_inclusion.md)
  : Variable inclusion proportions of a fitted model

## The sampler API

- [`thiessen_sampler()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_sampler.md)
  **\[experimental\]** : Drive the sampler one call at a time
- [`print(`*`<thiessen_sampler>`*`)`](https://l-thomson.github.io/thiessen/r/reference/print.thiessen_sampler.md)
  : Print a sampler

## Build information

- [`core_version()`](https://l-thomson.github.io/thiessen/r/reference/core_version.md)
  : Version of the core crate
- [`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
  : Whether the core was built with its experimental feature
- [`thiessen-package`](https://l-thomson.github.io/thiessen/r/reference/thiessen-package.md)
  : thiessen: Bayesian Additive Voronoi Tessellations
