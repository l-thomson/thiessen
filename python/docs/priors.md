# Priors and scaling

The response is scaled to [-0.5, 0.5] over its training range and each
Euclidean covariate column to the same interval over its own, following
Chipman, George and McCulloch (2010), s. 2.2.2. The priors below are stated on
that scaled response, so their defaults do not depend on the units of the data.

## The parameters

| Parameter | Where | Default | Paper | Prior |
| --- | --- | --- | --- | --- |
| `tessellations` | `mean_params` | 200 | 200 | ensemble size of the mean function |
| `k` | `TermParams` | 3 | 3 | cell means N(0, sigma_mu^2), sigma_mu = 0.5 / (k sqrt(m)) |
| `nu` | `gaussian()` | 6 | 6 | sigma^2 ~ Inv-Gamma(nu / 2, nu lambda / 2) |
| `q` | `gaussian()` | 0.85 | 0.85 | lambda calibrated so Pr(sigma < sigma_hat) = q |
| `sigma_c` | `GeometryParams` | 0.8 | 0.8 | centre coordinates N(0, sigma_c^2), scaled space |
| `omega` | `StructureParams` | min(3, p) | min(3, p) | dimension count, omega / p per covariate |
| `lambda_c` | `TermParams` | 5 | 25 | cells per tessellation, b - 1 ~ Poisson(lambda_c) |
| `tessellations` | `variance_params` | 0 | 40 | variance ensemble; a positive count is the heteroscedastic model |

`lambda_c` is the one default that departs from the paper: CRAN AddiVortes
takes 5 from 0.6.8 onward, and this package follows the implementation. Pass
the paper's value in the group:

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(100, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=100)

paper = Model(
    mean_params=TermParams(tessellations=25, lambda_c=25.0), burn_in=50, draws=100
).fit(x, y, random_state=1)
print("cells per tessellation:", round(float(paper.cell_counts().mean()), 2))

default = Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=100).fit(
    x, y, random_state=1
)
print("with lambda_c = 5:     ", round(float(default.cell_counts().mean()), 2))
```

`omega` is `None` by default and resolves at fit to min(3, p), so the default
is valid whatever the number of covariates. The resolved value appears on the
fitted model's configuration.

## Correspondence with BART

The mean-function prior is that of BART with tessellations in place of trees,
so the parameters carry over directly.

| thiessen | BART, dbarts | Meaning |
| --- | --- | --- |
| `tessellations` | `ntree` | ensemble size m |
| `k` | `k` | cell-mean prior spread |
| `nu` | `sigdf` | sigma^2 prior degrees of freedom |
| `q` | `sigquant` | sigma^2 prior calibration quantile |
| `lambda_c` | `base`, `power` | cell or node count; a Poisson rate here, a splitting probability there |
| `draws` | `ndpost` | draws kept |
| `burn_in` | `nskip` | burn-in sweeps |
| `thinning` | `keepevery` | thinning interval |
| `offset` | `binaryOffset` | probit offset c |

The cell-count prior is where the two differ in kind: BART places a
probability on splitting a node at a given depth, while AddiVortes places a
Poisson prior on the number of cells.

## Sampling from the prior

`prior_only` switches the likelihood off, so the chain draws the tessellations,
the cell means and sigma^2 from the prior and `predict` gives prior predictive
draws. The response still fixes the scaling and the lambda calibration, so the
prior sampled is the one a fit on the same data would use.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(100, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=100)

prior = Model(
    prior_only=True,
    mean_params=TermParams(tessellations=25),
    burn_in=50,
    draws=100,
).fit(x, y, random_state=1)
print("prior predictive spread:", round(float(prior.predict_draws(x).std()), 4))
```

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART: Bayesian
additive regression trees. *Annals of Applied Statistics*, 4(1), 266-298.
