# arviz

`FittedModel.to_inference_data(X, y)` returns the arviz `DataTree` of a fit,
with the groups of the PyMC and numpyro convention. It needs the `arviz` extra.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)

fitted = Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=100).fit(
    x, y, random_state=1
)
data = fitted.to_inference_data(x, y)
print("groups:", sorted(data.children))
print("posterior:", sorted(data["posterior"].dataset.data_vars))
```

| Group | Variables |
| --- | --- |
| `posterior` | `mu`, the mean function per draw; `sigma` under a model with a global sampled scale; `cell_count`; `dimension_count`; under the experimental items that sample them, `df`, `cutpoint`, `bandwidth`, `inclusion_weight` and `concentration` |
| `posterior_predictive` | `y`, one replicate per draw under the family's own observation model |
| `log_likelihood` | `y`, pointwise per draw |
| `observed_data` | `y`; `time` and `event` under the experimental AFT family; `lower` and `upper` under the experimental interval-censored family |

The chain dimension of every group holds the chains of the fit, so a fit of
one chain has one chain and R-hat is `NaN`. The observation dimension is
labelled, so the arviz summaries and the information-criterion functions work
directly.

```python exec="on" source="above" result="text"
import arviz as az
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)

model = Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=500)
fitted = model.fit(x, y, random_state=1, n_chains=4)
data = fitted.to_inference_data(x, y)
print("chains:", data["posterior"].dataset.sizes["chain"])
print(az.summary(data, var_names=["sigma"], kind="diagnostics"))
```

`fit(n_chains=k)` runs k chains, each with a seed the core derives from the
resolved seed, and pools their draws; the default is four, the number
Vehtari and others (2021) recommend. A fit of two or more chains checks
R-hat and the bulk and tail effective sample sizes of `sigma` and of the mean
function at up to twenty training rows, and warns where R-hat exceeds 1.01 or
an effective sample size falls below 400 (Vehtari, Gelman, Simpson, Carpenter
and Buerkner, 2021). The check needs arviz; a fit without it says so.

```python exec="on" source="above" result="text"
import arviz as az
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)

data = (
    Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=100)
    .fit(x, y, random_state=1)
    .to_inference_data(x, y)
)
print(az.summary(data, var_names=["sigma", "cell_count"]))
```

`sigma` appears under the Gaussian model alone: the probit model's latent
variance is one and the heteroscedastic model's varies with x, so neither has a
scalar to report. Use `predict_variance` for the heteroscedastic variance.

The predictive replicates are drawn in numpy from the fit's resolved seed, so
they are reproducible without being draws of the core. Everything else in the
tree comes from the kept draws.
