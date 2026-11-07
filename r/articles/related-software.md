# Related software

This page places the package beside the ones a reader is likely to have
used, with the argument of each that sets the same quantity, and says
what each fits that this one does not.

## The same quantity in each package

| Quantity | thiessen | AddiVortes | BART | dbarts | stochtree |
|----|----|----|----|----|----|
| ensemble size | `term_params(tessellations = )` | `m` | `ntree` | `ntree` | `num_trees` |
| cell or leaf prior spread | `term_params(k = )` | `k` | `k` | `k` | `k` |
| noise prior degrees of freedom | `gaussian_outcome(nu = )` | `nu` | `sigdf` | `sigdf` |  |
| noise prior quantile | `gaussian_outcome(q = )` | `q` | `sigquant` | `sigquant` |  |
| burn-in sweeps | `general_params(burn_in = )` | `mcmcBurnIn` | `nskip` | `nskip` | `num_burnin` |
| kept draws | `general_params(draws = )` | `totalMCMCIter - mcmcBurnIn` | `ndpost` | `ndpost` | `num_mcmc` |
| thinning | `general_params(thinning = )` |  | `keepevery` | `keepevery` |  |
| chains | `thiessen(chains = )` |  | `mc.wbart(mc.cores = )` | `nchain` | `num_chains` |
| threads | `thiessen(threads = )`, `mc.cores` |  | `mc.cores` | `nthread` | `num_threads` |
| seed | `thiessen(seed = )` | [`set.seed()`](https://rdrr.io/r/base/Random.html) | `seed` | `seed` | `random_seed` |

The three rows with no counterpart in the tree packages are the
tessellation’s own: `sigma_c`, the centre-coordinate scale (`sd` in
AddiVortes); `omega`, the prior mean number of active covariates
(`Omega`); and `lambda_c`, the cell-count prior rate (`LambdaRate`). A
tree has no centres, and its depth prior plays the part of the
cell-count prior.

## What each package fits

[AddiVortes](https://github.com/johnpaulgosling/AddiVortes) is the
authors’ implementation of the method, in R and C++. It is the reference
this package is checked against: the test suite compares posterior
summaries and the comparators harness in the repository checks realised
cell counts and held-out summaries at release. This package adds the
formula interface, the multi-chain diagnostics, the posterior and
rstantools generics, persistence of a fit, and the sampler API; the
models are the same.

[BART](https://cran.r-project.org/package=BART) (Sparapani, Spanbauer
and McCulloch 2021) fits BART for continuous, binary, multinomial,
survival (`abart`, `surv.bart`), competing-risks and recurrent-events
outcomes, with the DART sparsity prior. Of those, this package’s
published surface has the continuous and probit models; the
accelerated-failure-time, Student-t, Laplace and DART variants exist
behind the experimental build
([`?experimental_outcomes`](https://l-thomson.github.io/thiessen/r/reference/experimental_outcomes.md))
until each is calibrated and derived.

[dbarts](https://cran.r-project.org/package=dbarts) (Dorie and others)
fits continuous and binary BART with a sampler object the caller can
drive and modify between sweeps, the pattern this package’s
[`thiessen_sampler()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_sampler.md)
follows.

[stochtree](https://stochtree.ai) (Herren and others) fits BART and
Bayesian causal forests with a low-level interface for custom sampling
loops and a multi-chain page with R-hat and effective sample sizes; its
`bart()` also takes a `variance_forest_params` for a heteroscedastic
model, as `variance_params` does here.

[bartMachine](https://cran.r-project.org/package=bartMachine) (Kapelner
and Bleich 2016) fits continuous and binary BART in Java, with
variable-selection procedures this package does not have.

[rbart](https://cran.r-project.org/package=rbart) (Pratola and others)
is HBART, the heteroscedastic tree model that H-AddiVortes follows;
`rbart::rbart(ntreeh = )` is `variance_params` here.

## The Python package

The same core is wrapped for Python, with the same configuration names,
a scikit-learn estimator pair and an arviz conversion; a fit saved by
one binding is read by the other, and an integer seed gives the same
draws in both. Its documentation is under `python/` in the
[repository](https://github.com/l-thomson/thiessen).

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. *The Annals of Applied Statistics*
4(1), 266-298. <doi:10.1214/09-AOAS285>

Kapelner, A. and Bleich, J. (2016). bartMachine: machine learning with
Bayesian additive regression trees. *Journal of Statistical Software*
70(4), 1-40. <doi:10.18637/jss.v070.i04>

Pratola, M. T., Chipman, H. A., George, E. I. and McCulloch, R. E.
(2020). Heteroscedastic BART via multiplicative regression trees.
*Journal of Computational and Graphical Statistics* 29(2), 405-417.
<doi:10.1080/10618600.2019.1677243>

Sparapani, R., Spanbauer, C. and McCulloch, R. (2021). Nonparametric
machine learning and efficient computation with Bayesian additive
regression trees: the BART R package. *Journal of Statistical Software*
97(1), 1-66. <doi:10.18637/jss.v097.i01>

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871. <doi:10.1080/10618600.2024.2414104>
