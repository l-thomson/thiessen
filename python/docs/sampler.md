# The sampler API

!!! note "Experimental"
    This interface may change in a minor release without a deprecation
    cycle. The models it drives and the draws it produces carry the same
    guarantees as `Model.fit`.

`thiessen.sampler.Sampler` is the researcher's interface, after the
updatable sampler object of dbarts and the low-level interface of
stochtree: construct with the configuration, the data and a seed, then
drive the Gibbs loop yourself. Burn-in and thinning are the caller's loop,
and the response may be replaced between sweeps, which is what makes
censored responses, imputation between sweeps, custom likelihoods through
the response, and prototyping of outcome models possible. Anything that is
not an outcome family or a setting goes through this loop.

The response is on the caller's scale through an affine map frozen at
construction, so a response outside the training range is legitimate. The
sampler owns its RNG, seeded at construction with the chain-0 seed of
`fit`, so driving the configured schedule by hand reproduces `fit` bit for
bit; the loop cannot rewire tessellation membership or cell internals.
The constructor and `set_response` take the response shapes `Model.fit`
takes, so a structured survival array drives the AFT family and a
two-column array of bounds the interval-censored family, in an extension
built with the core's `experimental` feature.

## A fit as its own loop

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams
from thiessen.sampler import Sampler

rng = np.random.default_rng(0)
x = rng.uniform(size=(60, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=60)

sampler = Sampler(x, y, mean_params=TermParams(tessellations=10), random_state=1)
sampler.step(20)
for _ in range(40):
    sampler.step(1)
    sampler.keep()
driven = sampler.finish()

model = Model(mean_params=TermParams(tessellations=10), burn_in=20, draws=40)
fitted = model.fit(x, y, random_state=1)
print("matches fit:", bool(np.array_equal(driven.predict(x), fitted.predict(x))))
```

## Imputation between sweeps

A right-censored response can be redrawn from its conditional before each
sweep: read the current mean function and noise level, redraw the censored
values, and hand the response back.

```python exec="on" source="above" result="text"
import numpy as np
from scipy.stats import norm
from thiessen import TermParams
from thiessen.sampler import Sampler

rng = np.random.default_rng(0)
x = rng.uniform(size=(60, 2))
latent = x[:, 0] + rng.normal(scale=0.1, size=60)
limit = np.quantile(latent, 0.8)
censored = latent > limit
y = np.minimum(latent, limit)

sampler = Sampler(x, y, mean_params=TermParams(tessellations=10), random_state=1)
sampler.step(20)
for _ in range(40):
    mean = sampler.fitted_values()
    scale = np.sqrt(sampler.noise_variances())
    working = y.copy()
    # A draw from the normal truncated to the tail above the limit, by
    # inversion of its distribution function.
    at_limit = norm.cdf((limit - mean[censored]) / scale[censored])
    u = at_limit + rng.uniform(size=censored.sum()) * (1.0 - at_limit)
    working[censored] = mean[censored] + scale[censored] * norm.ppf(u)
    sampler.set_response(working)
    sampler.step(1)
    sampler.keep()

fitted = sampler.finish()
print("draws:", fitted.n_draws)
```

## Prototyping an outcome model

An outcome model with a latent Gaussian representation can be prototyped
in the loop before it earns a family: impute the latent from its
conditional given the observed outcome, hand it back, sweep. The probit
model is the worked case, its latent drawn from the normal truncated to
the side its label demands (Albert and Chib 1993).

```python exec="on" source="above" result="text"
import numpy as np
from scipy.stats import norm
from thiessen import Model, TermParams, probit
from thiessen.sampler import Sampler

rng = np.random.default_rng(0)
n = 80
x = rng.uniform(size=(n, 2))
labels = (x[:, 0] + rng.normal(scale=0.3, size=n) > 0.5).astype(float)

z = np.where(labels == 1.0, 0.5, -0.5)
sampler = Sampler(x, z, mean_params=TermParams(tessellations=10), random_state=1)
sampler.step(30)
latent_mean = np.zeros(n)
for _ in range(60):
    f = sampler.fitted_values()
    scale = np.sqrt(sampler.noise_variances())
    # The conditional mass below zero, then a draw from the label's side
    # of it, by inversion.
    at_zero = norm.cdf((0.0 - f) / scale)
    u = rng.uniform(size=n)
    inside = np.where(labels == 1.0, at_zero + u * (1.0 - at_zero), u * at_zero)
    sampler.set_response(f + scale * norm.ppf(inside))
    sampler.step(1)
    latent_mean += sampler.fitted_values()
latent_mean /= 60

family = Model(
    outcome=probit(), mean_params=TermParams(tessellations=10), burn_in=30, draws=60
)
fitted = family.fit(x, labels, random_state=1)
agreement = np.mean((latent_mean > 0.0) == (fitted.predict(x) > 0.5))
print("classification agreement with the probit family:", agreement)
```

The prototype and the family are the same model up to one difference:
the family fixes the latent variance at 1 for identification, while the
prototype keeps sampling sigma^2. Differences of that kind are what
graduate a prototype into an outcome family. A prototype is checked
distributionally, by simulation-based calibration and posterior
summaries within Monte Carlo error, never bitwise against the family.
