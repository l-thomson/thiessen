# Models

The outcome family selects the observation model: `gaussian()` or
`probit()`, with the heteroscedastic extension of the Gaussian family
selected by attaching `variance_params` with a positive tessellation count.
The three are the published method: Stone and Gosling (2025) and CRAN
AddiVortes.

| Model | Selected by | Response | `sigma()` | `predict` returns | `predict_variance` |
| --- | --- | --- | --- | --- | --- |
| gaussian | `gaussian()` | continuous | one sigma per draw | E[y \| x] | sigma^2, constant in x |
| probit | `probit()` | labels in {0, 1} | empty | P(y = 1 \| x) | not applicable |
| heteroscedastic | `gaussian()` + `variance_params` | continuous | empty | E[y \| x] | s^2(x), varying in x |

## Gaussian

y = f(x) + e with e ~ N(0, sigma^2) and f a sum of m tessellations. The sweep
draws sigma^2 from its inverse-gamma conditional.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] ** 2 + rng.normal(scale=0.1, size=120)

fitted = Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=100).fit(
    x, y, random_state=1
)
print("sigma draws:", fitted.sigma().shape)
print("posterior mean sigma:", round(float(fitted.sigma().mean()), 4))
```

## Probit

P(y = 1 | x) = Phi(c + f(x)), with the Albert and Chib (1993) latent
augmentation and unit latent variance. The offset c defaults to Phi^-1(ybar),
the BART `binaryOffset` default, and is resolved at fit. `predict` gives the
posterior mean probability; `predict_latent` gives c + f(x).

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams, probit

rng = np.random.default_rng(0)
x = rng.uniform(size=(150, 2))
y = (x[:, 0] > 0.5).astype(float)

model = Model(
    outcome=probit(),
    mean_params=TermParams(tessellations=25),
    burn_in=50,
    draws=100,
)
fitted = model.fit(x, y, random_state=1)
probabilities = fitted.predict(x)
print(
    "range:", round(float(probabilities.min()), 3), round(float(probabilities.max()), 3)
)
print("latent draws:", fitted.predict_latent(x).shape)
```

`sigma()` is empty, as the latent variance is one. `predict_variance` and
`prediction_interval` raise, as a two-point distribution has neither.

## Heteroscedastic

y = f(x) + e with e ~ N(0, s^2(x)) and s^2 a multiplicative ensemble of
inverse-gamma variance tessellations, sized by the `variance_params`
tessellation count. `predict_variance` gives s^2(x) per draw.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 2))
y = x[:, 0] + rng.normal(scale=0.05 + 0.3 * x[:, 0], size=200)

model = Model(
    mean_params=TermParams(tessellations=25),
    variance_params=TermParams(tessellations=20),
    burn_in=50,
    draws=100,
)
fitted = model.fit(x, y, random_state=1)
variance = fitted.predict_variance(x).mean(axis=0)
order = np.argsort(x[:, 0])
print("variance at low x: ", round(float(variance[order[:20]].mean()), 4))
print("variance at high x:", round(float(variance[order[-20:]].mean()), 4))
```

## Chains and threads

`fit` runs `n_chains=4` chains by default, each with a seed the core derives
from the resolved seed, and pools their draws; four is the number Vehtari and
others (2021) recommend for the R-hat and effective sample size checks a fit
of two or more chains makes. The chains run on `n_threads=1` threads by
default, the scikit-learn convention, so a call that sets nothing pays four
chains on one core. `n_threads=os.cpu_count()` runs the chains on
every core for the same draws, which do not depend on the thread count; on
Friedman #1 with n = 200 and p = 10 the fit is then about 2.7 times faster on
four cores of a 2025 laptop, so it costs about one and a half chains rather
than four.

The default schedule is short for the thresholds: on that data a default fit
reaches a smallest effective sample size of about 290 against the threshold
of 400 and a largest R-hat of about 1.014 against 1.01, so it warns. More
draws per chain, `Model(draws=)`, is the answer; `thinning` is not.

A fit that warned can be continued rather than rerun:
`model.fit(x, y, init=fitted)` starts each chain from one of `fitted`'s kept
draws, so the new chains begin where the old ones ended instead of at the cold
start, and only the starting state differs from a cold-started fit. The
estimators do the same through the scikit-learn convention,
`AddiVortesRegressor(warm_start=True)`, each repeated `fit` continuing every
chain from its own last draw. Chains started from one previous fit start close
together, so a clean R-hat on the continued fit checks that run, not dispersed
starts (Vehtari and others, 2021). The option is experimental and needs a core
built with the `experimental` feature.

```python exec="on" source="above" result="text"
import os
import warnings

import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] ** 2 + rng.normal(scale=0.1, size=120)

model = Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=100)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fitted = model.fit(x, y, random_state=1, n_threads=os.cpu_count())
print("chains:", fitted.n_chains)
print("draws:", fitted.n_draws)
print("warned:", any("converged" in str(w.message) for w in caught))
```

## Column metrics

The geometry's `metric` (`TermParams(geometry=GeometryParams(metric=[...]))`)
sets the distance of each covariate column, one entry per column in column
order. Non-Euclidean columns are not scaled.

| Entry | Squared distance | Centre coordinates |
| --- | --- | --- |
| `'euclidean'` | Euclidean on the column scaled to [-0.5, 0.5] | N(0, sigma_c^2) |
| `{'spherical': {'sphere': k}}` | great-circle angle, radians | N(mid, sd^2), longitude wrapped |
| `'categorical'` | 2 / n^2 per mismatch, n levels | uniform over the levels |

Columns sharing a `sphere` label form one sphere: its latitudes, then its
longitude last. The categorical weight is that of Eskin et al. (2002), CRAN
AddiVortes `metric = "C"`.

## The stable surface

These three models are the published method and follow semantic versioning.
The core's calibration suite covers the configurations listed in
`docs/calibrated.md` in the repository; component options are verified in
isolation, and every other combination of the documented options is valid
to run and is not separately verified. The boundary of what the library
can carry, observation models with an exact conditionally Gaussian
augmentation, is the section of that name in the repository's
`docs/models.md`.
Everything else the core crate adds sits behind its `experimental` Cargo
feature, which a released wheel does not enable, so a configuration or a
saved model naming such an option is rejected with `RequiresFeatureError`.
The further outcome families each have a constructor in
`thiessen.families`, and the component options a constructor in
`thiessen.params` or an entry of `GeometryParams(metric=)`, so naming one
is portable, and an extension accepting them is built from source:

```sh
pip install ./python --config-settings build-args="--features experimental"
```

`thiessen._native.EXPERIMENTAL` reports the setting of the extension in
use. Such a build is outside semantic versioning: the configuration and
the drawn values of a gated item may change in any release. The table of
experimental items and their status is `docs/experimental.md` in the
repository, with the Python entry point of each. A graduated item is
exposed here as any other option, with no separate opt-in.

## The response selects the family

`Model(outcome=None)`, the default, takes the family from the shape of
`y` at `fit`, the rule of `glm` with "not declared" representable: a
numeric array is the Gaussian family, a boolean array or a two-category
pandas `Categorical` the probit family, an ordered `Categorical` the
ordinal family (its categories the category count), a structured array
of a boolean event indicator and a time, the layout of
`sksurv.util.Surv.from_arrays`, the AFT family, and an array of shape
`(n, 2)` of lower and upper bounds the interval-censored family, an
infinite bound for one-sided censoring and an equal pair for an exact
value. A named family is checked against the shape and a mismatch is an
error naming both; the probit family also takes the numbers 0 and 1 and
the ordinal family integer codes 0 to K - 1 with `categories` named.
`FittedModel.log_likelihood`, `to_inference_data`, `Sampler` and
`Sampler.set_response` take the same shapes. `FittedModel.predict_proba`
gives the ordinal category probabilities, and `dfs`, `cutpoints`,
`bandwidths`, `inclusion_weights` and `concentrations` the posterior
quantities the experimental items sample, empty where none is.

## References

Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
polychotomous response data. *Journal of the American Statistical
Association*, 88(422), 669-679.

Eskin, E., Arnold, A., Prerau, M., Portnoy, L. and Stolfo, S. (2002). A
geometric framework for unsupervised anomaly detection. In *Applications of
Data Mining in Computer Security*, 77-101.

Stone, E. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive Voronoi
tessellations. *Journal of Computational and Graphical Statistics*, 34(3),
859-871.
