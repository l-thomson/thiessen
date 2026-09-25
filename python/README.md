# thiessen (Python)

Python bindings to the `thiessen` Rust crate, an implementation of
AddiVortes: Bayesian regression on a sum of Voronoi tessellations (Stone and
Gosling, 2025, *Journal of Computational and Graphical Statistics* 34(3),
859-871, [doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)).
The documentation, with the API reference, is at
<https://l-thomson.github.io/thiessen/python/>.

## Statement of need

BART (Chipman, George and McCulloch, 2010) partitions the covariate space
with axis-aligned boxes, so a boundary oblique to the axes costs many splits
to approximate. AddiVortes replaces the trees with Voronoi tessellations, so
a cell is a region of the covariate space and an oblique or curved boundary
is reached directly. It keeps the parts of BART that make it usable as a
default: the priors that regularise each component to a weak learner, the
Gibbs sampler, and the posterior summaries.

No implementation of the method existed outside the authors' R code. This
package provides one with a Rust core, a reproducibility contract that fixes
the draws for a seed, and the interfaces a Python user expects: a
scikit-learn estimator pair and an arviz conversion. The models are Gaussian
regression, binary probit and the heteroscedastic variant.

## Installation

The package builds the core crate from source, so a Rust toolchain
(`cargo`, `rustc` >= 1.74) is required until wheels are published.

```sh
pip install "thiessen @ git+https://github.com/l-thomson/thiessen#subdirectory=python"
```

or, from a checkout of the repository, `pip install -e python/`.

## Usage

```python
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 3))
y = 3.0 * (x[:, 0] - 0.4) ** 2 + 0.5 * x[:, 1] + rng.normal(scale=0.1, size=200)

model = Model(mean_params=TermParams(tessellations=50), burn_in=100, draws=200)
fitted = model.fit(x, y, random_state=1)
mean = fitted.predict(x)
lower, upper = fitted.credible_interval(x, level=0.9).T
```

The configuration has four parts, named as the core stores them: an outcome
family from `gaussian()` or `probit()`, one `TermParams` group per ensemble
(`mean_params`, and `variance_params` for the heteroscedastic model), and
the flat run-length settings. Parameters left unset take the core's
defaults.

```python
from thiessen import gaussian

hetero = Model(
    outcome=gaussian(nu=10.0),
    mean_params=TermParams(tessellations=200, lambda_c=25.0),
    variance_params=TermParams(tessellations=40),
)
```

`random_state` takes an integer, a `numpy.random.Generator`, a
`numpy.random.RandomState` or `None`. The same integer, package version and
target reproduce the same draws; the resolved seed is on
`fitted.random_state`.

## scikit-learn

With the `sklearn` extra, `thiessen.estimators` holds two estimators meeting
the scikit-learn contract, so they compose with `Pipeline`, `GridSearchCV`,
`cross_val_score` and `sklearn.inspection`.

```python
from sklearn.inspection import PartialDependenceDisplay
from thiessen.estimators import AddiVortesRegressor

model = AddiVortesRegressor(
    mean_params=TermParams(tessellations=50),
    burn_in=100,
    draws=200,
    random_state=1,
)
model.fit(x, y)
PartialDependenceDisplay.from_estimator(model, x, features=[0, 1])
```

The parameter groups implement `get_params`/`set_params`, so grid search
routes into them with keys of the form `<group>__<parameter>`:

```python
from sklearn.model_selection import GridSearchCV

GridSearchCV(model, {"mean_params__tessellations": [50, 200]})
```

`AddiVortesClassifier` fits the binary probit model and carries
`predict_proba`. An integer `random_state` gives the same draws through either
API, so `AddiVortesRegressor(random_state=1)` and `Model(random_state=1)`
agree.

### Priors

| Parameter | Where | Default | Stone and Gosling (2025) |
| --- | --- | --- | --- |
| `tessellations` | `TermParams` | 200 | 200 |
| `nu` | `gaussian()` | 6 | 6 |
| `q` | `gaussian()` | 0.85 | 0.85 |
| `k` | `TermParams` | 3 | 3 |
| `sigma_c` | `GeometryParams` | 0.8 | 0.8 |
| `omega` | `StructureParams` | min(3, p) | min(3, p) |
| `lambda_c` | `TermParams` | 5 | 25 |

The `lambda_c` default follows AddiVortes >= 0.6.8; pass
`TermParams(lambda_c=25.0)` for the paper's value.

### Categorical covariates

`categorical_features` follows the `HistGradientBoostingRegressor` shape.
Without it the input is taken as numeric, and the usual route is an explicit
encoder:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ColumnTransformer([("g", OneHotEncoder(drop="first"), ["group"])])
```

Naming the columns instead has the estimator apply the same d - 1
treatment-contrast encoding, the first level as reference, as upstream
AddiVortes and `model.matrix`. A column whose entry in the geometry's
`metric` is `'categorical'` passes as integer level codes and takes the
Eskin mismatch weight rather than being expanded.

## arviz

With the `arviz` extra, a fit converts to an arviz `DataTree`:

```python
import arviz as az

data = fitted.to_inference_data(x, y)
az.summary(data, var_names=["sigma"])
az.loo(data)
```

The sampler runs one chain, so every group has a chain dimension of one.

## Models and the stable surface

The families are `gaussian()` and `probit()`, and attaching
`variance_params` with a positive tessellation count to the Gaussian family
selects the heteroscedastic model: the models of Stone and Gosling (2025)
and of CRAN AddiVortes. Everything else the core crate adds
sits behind its `experimental` Cargo feature, which this package does not
enable, so a configuration or a saved model naming such an option is rejected
with the core's message naming the feature. The table of experimental items and
their status is `docs/experimental.md` in the repository. A graduated item is
exposed here as any other option, with no separate opt-in.

## Documentation

The site under `python/docs` covers the models, the priors and their
correspondence with BART, the input-data contract, the scikit-learn and arviz
interfaces, reproducibility and the testing strategy. Build it with:

```sh
pip install -e "python/[sklearn,arviz]" -r python/requirements-docs.txt
cd python && mkdocs serve
```

Every example on the site is executed at build, so the printed output is that
of the version being documented.

## Contributing

`CONTRIBUTING.md` at the root of the repository covers the development setup
and the gates a change must pass. `CODE_OF_CONDUCT.md` applies to every
space of the project.

## Citation

Cite the method as Stone and Gosling (2025). `CITATION.cff` at the root of
the repository carries the software metadata.

## Licence

MIT or Apache-2.0, at your option.
