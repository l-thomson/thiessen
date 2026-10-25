# Fast fits and fast starts

`XAddiVortesRegressor` fits the AddiVortes model, a sum of Voronoi
tessellations, by a fast point fit in place of Markov chain Monte Carlo. The
columns are screened by mutual information with the response, the tessellations
are built in turn on the running residual, and each round replaces the share
with the smallest Bayes factor by fresh tessellations and refits the rest; a
final refit chooses the damping of the cell values by generalised
cross-validation. The fit has two uses: a fast point estimate, and the starting state of the exact sampler,
`AddiVortesRegressor(start="fast")` or `Model().fit(..., start="fast")`, which
changes where the chains start and not the posterior they target.

Both are experimental and need a core built with the `experimental` feature:

```sh
pip install ./python --config-settings build-args="--features experimental"
```

## A fast fit

```python
import numpy as np
from thiessen.estimators import XAddiVortesRegressor

rng = np.random.default_rng(1)
X = rng.uniform(size=(300, 10))
f = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)
y = f + rng.normal(size=300)
model = XAddiVortesRegressor(random_state=1).fit(X, y)
model.predict(X[:5])
model.train_score_  # after the build, each round and the final refit
```

The fit time grows linearly with `rounds` (default 40); the constructor's
defaults are the core's,
`XAddiVortesRegressor.core_defaults()`. Each setting is checked at `fit`
against the range the core accepts, as scikit-learn estimators check theirs.
The settings accept NumPy scalars, so a `GridSearchCV` over `numpy` grids
reaches the core unchanged.

## Intervals

`sigma_` is the root of the training residual sum of squares over the residual
degrees of freedom of the final refit. Those degrees of freedom do not count the
choice of the cells, so the intervals of `predict_interval` are a rough guide. Split-conformal intervals (Lei and
others, 2018) are calibrated without that assumption; MAPIE provides them
around any scikit-learn regressor:

```python
from mapie.regression import SplitConformalRegressor

conformal = SplitConformalRegressor(
    XAddiVortesRegressor(random_state=1), confidence_level=0.95, prefit=False
)
conformal.fit(X[:150], y[:150]).conformalize(X[150:], y[150:])
prediction, interval = conformal.predict_interval(X[:5])
```

## Fast starts

```python
from thiessen.estimators import AddiVortesRegressor

fast = AddiVortesRegressor(start="fast", fast_settings={"rounds": 20})
fast.fit(X, y)
```

Each chain starts from its own fast fit, after which the configured burn-in and
draws run. `fast_settings` takes the names of `XAddiVortesRegressor`; given
with `start="cold"` it is an error. Under `warm_start=True`, `start` applies to
the first `fit` only; a repeated `fit` continues the previous chains.

The chains start from fast fits, not from the overdispersed points R-hat
assumes (Vehtari and others, 2021), so agreement between them is weaker
evidence of convergence; the convergence warning of such a fit says so.

## References

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018).
Distribution-free predictive inference for regression. Journal of the American
Statistical Association 113(523), 1094-1111.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Buerkner, P.-C. (2021).
Rank-normalization, folding, and localization: an improved R-hat for assessing
convergence of MCMC. Bayesian Analysis 16(2), 667-718.
