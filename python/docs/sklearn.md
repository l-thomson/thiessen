# scikit-learn

`thiessen.estimators` holds three estimators meeting the scikit-learn
contract, so they compose with `Pipeline`, `GridSearchCV`, `cross_val_score`
and `sklearn.inspection`. They need the `sklearn` extra.

`AddiVortesRegressor` fits the Gaussian and heteroscedastic models;
`AddiVortesClassifier` fits the binary probit model. Both take the parameters
of [Priors and scaling](priors.md) as explicit keyword arguments.
`XAddiVortesRegressor`, the experimental fast fit, is described in
[Fast fits and fast starts](fast-fits.md).

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import TermParams
from thiessen.estimators import AddiVortesRegressor

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 3))
y = 3.0 * (x[:, 0] - 0.4) ** 2 + 0.5 * x[:, 1] + rng.normal(scale=0.1, size=200)

model = AddiVortesRegressor(
    mean_params=TermParams(tessellations=25), burn_in=50, draws=100, random_state=1
).fit(x, y)
print("R^2:", round(model.score(x, y), 4))

mean, std = model.predict(x, return_std=True)
print("mean posterior standard deviation:", round(float(std.mean()), 4))
```

`predict(X, return_std=True)` returns the posterior standard deviation of the
mean function over the kept draws, following the `GaussianProcessRegressor`
precedent. `predict_interval` gives the predictive interval for a new
observation.

## Classification

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import TermParams
from thiessen.estimators import AddiVortesClassifier

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 2))
y = np.where(x[:, 0] > 0.5, "high", "low")

model = AddiVortesClassifier(
    mean_params=TermParams(tessellations=25), burn_in=50, draws=100, random_state=1
).fit(x, y)
print("classes:", list(model.classes_))
print("accuracy:", round(model.score(x, y), 4))
print("probabilities:", model.predict_proba(x).shape)
```

The labels are returned as given: `classes_` holds them and `predict` draws
from it. `predict_proba` puts the columns in the order of `classes_`.

## Cross-validation and search

```python exec="on" source="above" result="text"
import numpy as np
from sklearn.model_selection import GridSearchCV
from thiessen import TermParams
from thiessen.estimators import AddiVortesRegressor

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)

search = GridSearchCV(
    AddiVortesRegressor(
        mean_params=TermParams(tessellations=10), burn_in=20, draws=40, random_state=1
    ),
    {"mean_params__lambda_c": [5.0, 25.0], "mean_params__k": [2.0, 3.0]},
    cv=3,
)
search.fit(x, y)
print("best:", search.best_params_)
```

## Partial dependence

Partial dependence and individual conditional expectation come from
`sklearn.inspection`; the estimators implement nothing of their own.

```python exec="on" source="above" result="text"
import numpy as np
from sklearn.inspection import partial_dependence
from thiessen import TermParams
from thiessen.estimators import AddiVortesRegressor

rng = np.random.default_rng(0)
x = rng.uniform(size=(150, 2))
y = 3.0 * (x[:, 0] - 0.4) ** 2 + rng.normal(scale=0.1, size=150)

model = AddiVortesRegressor(
    mean_params=TermParams(tessellations=25), burn_in=50, draws=100, random_state=1
).fit(x, y)
result = partial_dependence(model, x, features=[0], grid_resolution=5)
print("grid:   ", np.round(result["grid_values"][0], 3))
print("average:", np.round(result["average"][0], 3))
```

`PartialDependenceDisplay.from_estimator(model, x, features=[0, 1])` plots the
same quantity.

## Categorical covariates

`categorical_features` follows the `HistGradientBoostingRegressor` shape.
Without it the input is taken as numeric, and the usual route is an explicit
encoder in a `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ColumnTransformer([("g", OneHotEncoder(drop="first"), ["group"])])
```

Naming the columns instead has the estimator apply the same d - 1
treatment-contrast encoding, the first level as reference, as upstream
AddiVortes and `model.matrix`. `'from_dtype'` reads the pandas categorical
dtypes; an index array or a boolean mask names the columns directly.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import GeometryParams, TermParams
from thiessen.estimators import AddiVortesRegressor

rng = np.random.default_rng(0)
n = 150
group = rng.integers(0, 3, size=n).astype(float)
x = np.column_stack([rng.uniform(size=n), group])
y = x[:, 0] + 0.5 * group + rng.normal(scale=0.1, size=n)

expanded = AddiVortesRegressor(
    categorical_features=[1],
    mean_params=TermParams(tessellations=25),
    burn_in=50,
    draws=100,
    random_state=1,
).fit(x, y)
print("columns the core sees:", len(expanded._encoding.core_metric))

codes = AddiVortesRegressor(
    categorical_features=[1],
    mean_params=TermParams(
        tessellations=25,
        geometry=GeometryParams(metric=["euclidean", "categorical"]),
    ),
    burn_in=50,
    draws=100,
    random_state=1,
).fit(x, y)
print("under the categorical metric:", codes._encoding.core_metric)
```

Three levels become two indicator columns under the Euclidean metric, or stay
as one column of codes under the categorical metric, which then takes the Eskin
mismatch weight.

## Chains and threads

The estimators run `n_chains=4` chains, the number the convergence checks
are designed for, on `n_jobs=None` threads, which is one under the joblib
convention, so an estimator that sets nothing pays four chains on one core.
`n_jobs=-1` runs the chains on every core for the same draws; on Friedman #1
with n = 200 and p = 10 the fit is then about 2.7 times faster on four cores
of a 2025 laptop. A fit at the default schedule warns on that data (smallest
effective sample size about 290 against 400, largest R-hat about 1.014
against 1.01); more `draws` per chain is the answer.

```python exec="on" source="above" result="text"
import warnings

import numpy as np
from thiessen import TermParams
from thiessen.estimators import AddiVortesRegressor

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 3))
y = 3.0 * (x[:, 0] - 0.4) ** 2 + 0.5 * x[:, 1] + rng.normal(scale=0.1, size=200)

model = AddiVortesRegressor(
    mean_params=TermParams(tessellations=25), burn_in=50, draws=100, n_jobs=-1
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(x, y)
print("chains:", model.n_chains)
print("R^2:", round(model.score(x, y), 4))
```

## Seeds

`random_state` takes an integer, a `numpy.random.Generator`, a
`numpy.random.RandomState` or `None`. An integer passes through to the core
unchanged, so the estimators and `Model` draw alike:

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams
from thiessen import TermParams
from thiessen.estimators import AddiVortesRegressor

rng = np.random.default_rng(0)
x = rng.uniform(size=(80, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=80)
sweep = {"mean_params": TermParams(tessellations=10), "burn_in": 20, "draws": 40}

through_estimator = AddiVortesRegressor(random_state=1, **sweep).fit(x, y).predict(x)
through_model = Model(**sweep).fit(x, y, random_state=1).predict(x)
print("identical:", bool(np.array_equal(through_estimator, through_model)))
```

`sklearn.utils.check_random_state` is not used, as it maps an integer to a
`RandomState` and would stop an integer passing through. The resolved seed is
on `random_state_` after fit.

## Pickling

The estimators pickle through the core's representation, and `clone` works, so
they survive `joblib` and the usual model-persistence routes.
