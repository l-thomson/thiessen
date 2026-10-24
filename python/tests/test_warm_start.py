"""Warm starts from a previous fit: `Model.fit(init=...)` and the
estimators' `warm_start`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from thiessen import Model, RequiresFeatureError, ThiessenError, _native
from thiessen.estimators import AddiVortesClassifier, AddiVortesRegressor

from .conftest import SMALL

experimental = pytest.mark.skipif(
    not _native.EXPERIMENTAL, reason="built without the feature"
)


def quiet(fit, *args, **kwargs):
    """Run a fit, ignoring the short schedule's convergence warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit(*args, **kwargs)


@pytest.mark.skipif(_native.EXPERIMENTAL, reason="built with the feature")
def test_a_warm_start_needs_the_feature(gaussian_fixture):
    x, y = gaussian_fixture
    previous = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)
    with pytest.raises(RequiresFeatureError):
        Model(**SMALL).fit(x, y, random_state=2, n_chains=1, init=previous)


@experimental
def test_a_warm_start_reproduces_and_differs_from_the_cold_start(gaussian_fixture):
    x, y = gaussian_fixture
    model = Model(**SMALL)
    previous = model.fit(x, y, random_state=1, n_chains=1)
    a = quiet(model.fit, x, y, random_state=2, n_chains=2, init=previous)
    b = quiet(model.fit, x, y, random_state=2, n_chains=2, init=previous)
    cold = quiet(model.fit, x, y, random_state=2, n_chains=2)
    np.testing.assert_array_equal(a.predict_draws(x), b.predict_draws(x))
    assert not np.array_equal(a.predict_draws(x), cold.predict_draws(x))


@experimental
def test_each_chain_continues_its_counterpart_by_default(gaussian_fixture):
    x, y = gaussian_fixture
    model = Model(**SMALL)
    previous = quiet(model.fit, x, y, random_state=1, n_chains=2)
    default = quiet(model.fit, x, y, random_state=2, n_chains=2, init=previous)
    last = quiet(
        model.fit,
        x,
        y,
        random_state=2,
        n_chains=2,
        init=previous,
        init_draw=previous.n_draws - 1,
    )
    assert not np.array_equal(default.predict_draws(x), last.predict_draws(x))


@experimental
def test_reusing_the_seed_of_init_warns(gaussian_fixture):
    x, y = gaussian_fixture
    model = Model(**SMALL)
    previous = quiet(model.fit, x, y, random_state=1, n_chains=1)
    with pytest.warns(UserWarning, match="replay its random numbers"):
        model.fit(x, y, random_state=1, n_chains=1, init=previous)


@experimental
def test_too_few_draws_warn_that_the_chains_share_a_start(gaussian_fixture):
    x, y = gaussian_fixture
    model = Model(**SMALL)
    previous = model.fit(x, y, random_state=1, n_chains=1)
    with pytest.warns(UserWarning, match="every chain starts from the same draw"):
        model.fit(x, y, random_state=2, n_chains=3, init=previous, init_draw=1)


@experimental
def test_the_previous_fit_must_be_of_this_model_and_data(gaussian_fixture):
    x, y = gaussian_fixture
    previous = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)
    with pytest.raises(ThiessenError, match="differ from this data"):
        Model(**SMALL).fit(x, y + 1.0, random_state=2, n_chains=1, init=previous)


def test_init_arguments_are_checked(gaussian_fixture):
    x, y = gaussian_fixture
    model = Model(**SMALL)
    previous = model.fit(x, y, random_state=1, n_chains=1)
    with pytest.raises(ValueError, match="FittedModel"):
        model.fit(x, y, random_state=2, n_chains=1, init="previous")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="init_draw needs init"):
        model.fit(x, y, random_state=2, n_chains=1, init_draw=3)
    with pytest.raises(ValueError, match="non-negative integer"):
        model.fit(x, y, random_state=2, n_chains=1, init=previous, init_draw=-1)


@experimental
@pytest.mark.parametrize("estimator", [AddiVortesRegressor, AddiVortesClassifier])
def test_warm_start_continues_each_chain(gaussian_fixture, estimator):
    x, y = gaussian_fixture
    if estimator is AddiVortesClassifier:
        y = (y > np.median(y)).astype(np.float64)
    params = {"random_state": 3, "n_chains": 2, **SMALL}
    warm = estimator(warm_start=True, **params)
    quiet(warm.fit, x, y)
    first = warm._fitted
    quiet(warm.fit, x, y)
    again = estimator(warm_start=True, **params)
    quiet(again.fit, x, y)
    quiet(again.fit, x, y)
    np.testing.assert_array_equal(warm.predict(x), again.predict(x))
    cold = estimator(**params)
    quiet(cold.fit, x, y)
    quiet(cold.fit, x, y)
    assert warm._fitted is not first
    assert warm.random_state_ != 3
    assert not np.array_equal(warm.predict(x), cold.predict(x))


@experimental
def test_warm_start_keeps_the_chain_count(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(warm_start=True, random_state=3, n_chains=2, **SMALL)
    quiet(model.fit, x, y)
    model.set_params(n_chains=3)
    with pytest.raises(ValueError, match="n_chains changed"):
        model.fit(x, y)
