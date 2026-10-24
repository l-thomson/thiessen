"""The XAddiVortes fast fit and the fast start: the feature gate, the
settings, agreement with the core through the cross-binding fixture
(`crates/thiessen/tests/fixtures/xaddivortes/cross_binding.txt`),
prediction and the warm start."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from thiessen import Model, RequiresFeatureError, TermParams, ThiessenError, _native
from thiessen._fast import _fast_json
from thiessen.estimators import AddiVortesRegressor, XAddiVortesRegressor
from thiessen.sampler import Sampler

from .conftest import SEED, SMALL, _fixture

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "crates/thiessen/tests/fixtures/xaddivortes/cross_binding.txt"
)
experimental = pytest.mark.skipif(
    not _native.EXPERIMENTAL, reason="built without the feature"
)
FIFTEEN = TermParams(tessellations=15)
#: Settings under which the 15-tessellation fit builds every slot and runs
#: rounds on the fixture; the default pair screen costs more than a small
#: budget.
WORKING: dict[str, Any] = {"budget": 20.0, "pair_screen": 2}


def quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit(*args, **kwargs)


@pytest.mark.skipif(_native.EXPERIMENTAL, reason="built with the feature")
def test_the_fast_fit_needs_the_feature():
    x, y = _fixture()
    with pytest.raises(RequiresFeatureError):
        XAddiVortesRegressor(mean_params=FIFTEEN).fit(x, y)
    with pytest.raises(RequiresFeatureError):
        Model(**SMALL).fit(x, y, random_state=1, n_chains=1, start="fast")


def test_the_signature_defaults_are_the_cores():
    shown = {
        name: p.default
        for name, p in inspect.signature(XAddiVortesRegressor).parameters.items()
        if name in XAddiVortesRegressor.core_defaults()
    }
    assert shown == XAddiVortesRegressor.core_defaults()


def test_numpy_settings_reach_the_core_as_their_json_types():
    json = _fast_json(
        {"budget": np.float64(5), "centres": np.int64(8), "rescore": np.bool_(True)}
    )
    assert json == '{"budget": 5.0, "centres": 8, "rescore": true}'
    assert _fast_json({"value_scale": 1 / 3}) == '{"value_scale": 0.3333333333333333}'
    with pytest.raises(ValueError, match="unknown"):
        _fast_json({"budjet": 5})
    with pytest.raises(ValueError, match="budget must be finite"):
        _fast_json({"budget": float("inf")})


@pytest.mark.parametrize(
    ("setting", "error"),
    [
        ({"budget": None}, TypeError),
        ({"budget": True}, TypeError),
        ({"budget": "50"}, TypeError),
        ({"centres": 8.0}, TypeError),
        ({"centres": np.float64(8)}, TypeError),
        ({"rescore": 1}, TypeError),
        ({"update_sigma": None}, TypeError),
        ({"budget": 0.0}, ValueError),
        ({"budget": float("inf")}, ValueError),
        ({"budget": float("nan")}, ValueError),
        ({"value_scale": 0.0}, ValueError),
        ({"replace_share": 1.5}, ValueError),
        ({"centres": 0}, ValueError),
        ({"centres": 65_536}, ValueError),
        ({"pair_screen": 1}, ValueError),
        ({"column_bins": 256}, ValueError),
        ({"score_floor": -0.1}, ValueError),
        ({"max_rounds": -1}, ValueError),
        ({"worth": "best"}, ValueError),
        ({"worth": None}, ValueError),
    ],
)
def test_the_estimator_checks_each_setting_at_fit(setting, error):
    x, y = _fixture()
    (name,) = setting
    model = XAddiVortesRegressor(mean_params=FIFTEEN).set_params(**setting)
    with pytest.raises(error, match=name):
        model.fit(x, y)


def test_the_settings_accept_numpy_scalars_and_their_bounds():
    x, y = _fixture()
    model = XAddiVortesRegressor(
        value_scale=1.0, replace_share=0.0, max_rounds=0, worth="squared_values"
    ).set_params(
        budget=np.float64(1e-3), centres=np.int64(65_535), rescore=np.bool_(False)
    )
    if not _native.EXPERIMENTAL:
        with pytest.raises(RequiresFeatureError):
            model.fit(x, y)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(x, y)
    assert model.n_rounds_ == 0


@pytest.mark.parametrize(
    "setting",
    [{"centres": 8.0}, {"centres": -1}, {"rescore": 1}, {"worth": "best"}],
)
def test_the_core_names_a_setting_it_rejects(setting):
    x, y = _fixture()
    (name,) = setting
    with pytest.raises(ThiessenError, match=f"`xaddivortes.{name}`"):
        Model(**SMALL).fit(
            x, y, random_state=1, n_chains=1, start="fast", fast_settings=setting
        )


@experimental
def test_a_budget_that_leaves_tessellations_at_zero_warns():
    x, y = _fixture()
    with pytest.warns(UserWarning, match="left at zero"):
        model = XAddiVortesRegressor(budget=0.5, mean_params=FIFTEEN).fit(x, y)
    assert model.n_zeroed_ == 15


@experimental
def test_the_fast_fit_equals_the_cores_bit_for_bit():
    x, y = _fixture()
    lines = FIXTURE.read_text().splitlines()
    sigma = float(lines[0].removeprefix("sigma "))
    want = np.array([float(v) for v in lines[1:]])
    model = XAddiVortesRegressor(mean_params=FIFTEEN, random_state=SEED).fit(x, y)
    assert model.sigma_ == sigma
    np.testing.assert_array_equal(model.predict(x), want)


@experimental
def test_the_fast_fit_predicts_and_gives_intervals_from_its_sigma():
    x, y = _fixture()
    model = XAddiVortesRegressor(**WORKING, mean_params=FIFTEEN, random_state=3)
    model.fit(x, y)
    assert model.n_zeroed_ == 0 and model.n_rounds_ > 0
    interval = model.predict_interval(x, level=0.95)
    width = interval[:, 1] - interval[:, 0]
    np.testing.assert_allclose(
        width, 2 * NormalDist().inv_cdf(0.975) * model.sigma_, rtol=1e-10
    )
    np.testing.assert_allclose(model._fitted.sigma(), [model.sigma_], rtol=0)
    assert model.train_score_.shape == (model.n_rounds_ + 1,)
    assert model.budget_used_ <= 20
    again = clone(model).fit(x, y)
    np.testing.assert_array_equal(again.predict(x), model.predict(x))


@experimental
def test_a_search_over_numpy_grids_fits():
    from sklearn.model_selection import GridSearchCV, KFold

    rng = np.random.default_rng(0)
    x = rng.uniform(size=(200, 3))
    y = np.sin(6 * x[:, 0]) + x[:, 1] + 0.1 * rng.normal(size=200)
    search = GridSearchCV(
        XAddiVortesRegressor(**WORKING, mean_params=FIFTEEN, random_state=1),
        {"centres": np.arange(4, 7), "rescore": np.array([False, True])},
        cv=KFold(3, shuffle=True, random_state=0),
    )
    search.fit(x, y)
    assert search.best_score_ > 0.5


@experimental
def test_chains_start_from_their_own_fast_fits():
    x, y = _fixture()
    params = {"random_state": 4, "n_chains": 2, **SMALL}
    fast = dict(WORKING)
    a = quiet(AddiVortesRegressor(start="fast", fast_settings=fast, **params).fit, x, y)
    b = quiet(
        AddiVortesRegressor(start="fast", fast_settings=fast, n_jobs=2, **params).fit,
        x,
        y,
    )
    cold = quiet(AddiVortesRegressor(**params).fit, x, y)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    assert not np.array_equal(a.predict(x), cold.predict(x))
    model = Model(**SMALL)
    c = quiet(
        model.fit, x, y, random_state=4, n_chains=2, start="fast", fast_settings=fast
    )
    np.testing.assert_array_equal(c.predict(x), a.predict(x))


def test_the_fast_start_arguments_are_checked():
    x, y = _fixture()
    with pytest.raises(ValueError, match="needs start='fast'"):
        AddiVortesRegressor(fast_settings={"budget": 5.0}, **SMALL).fit(x, y)
    with pytest.raises(ValueError, match="start must be"):
        AddiVortesRegressor(start="lukewarm", **SMALL).fit(x, y)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fast_settings must be"):
        AddiVortesRegressor(start="fast", fast_settings=[1], **SMALL).fit(x, y)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="budget"):
        AddiVortesRegressor(start="fast", fast_settings={"budget": None}, **SMALL).fit(
            x, y
        )
    model = Model(**SMALL)
    previous = quiet(model.fit, x, y, random_state=1, n_chains=1)
    with pytest.raises(ValueError, match="give one"):
        model.fit(x, y, random_state=2, n_chains=1, start="fast", init=previous)


@experimental
def test_the_sampler_takes_a_fast_fit_before_its_first_sweep():
    x, y = _fixture()
    sampler = Sampler(x, y, mean_params=FIFTEEN, random_state=SEED)
    summary = sampler.xaddivortes(WORKING)
    assert summary["zeroed"] == 0 and summary["rounds"] > 0
    assert summary["sigma"] > 0 and len(summary["train_score"]) == summary["rounds"] + 1
    stepped = Sampler(x, y, mean_params=FIFTEEN, random_state=SEED)
    stepped.step(1)
    with pytest.raises(ThiessenError):
        stepped.xaddivortes()


def test_the_fast_settings_are_checked_when_a_warm_start_continues():
    x, y = _fixture()
    model = AddiVortesRegressor(warm_start=True, random_state=1, n_chains=1, **SMALL)
    quiet(model.fit, x, y)
    model.set_params(start="fast", fast_settings={"budjet": 5})
    with pytest.raises(ValueError, match="unknown"):
        model.fit(x, y)


@experimental
def test_the_convergence_warning_of_a_fast_start_says_the_chains_did_not_start_apart():
    x, y = _fixture()
    caveat = "started from their own fast fits"
    regressor = AddiVortesRegressor(
        start="fast", fast_settings=WORKING, random_state=4, n_chains=2, **SMALL
    )
    with pytest.warns(UserWarning, match=caveat):
        regressor.fit(x, y)
    with pytest.warns(UserWarning, match=caveat):
        Model(**SMALL).fit(
            x, y, random_state=4, n_chains=2, start="fast", fast_settings=WORKING
        )
