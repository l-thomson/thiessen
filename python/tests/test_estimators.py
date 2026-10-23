"""The scikit-learn estimators."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from thiessen import GeometryParams, Model, TermParams, ThiessenError, gaussian, probit
from thiessen.estimators import AddiVortesClassifier, AddiVortesRegressor

from .conftest import SMALL

#: The estimator parameters: the configuration groups, the run-length
#: settings, and the settings that are not configuration fields.
PARAMETERS = {
    "outcome",
    "mean_params",
    "variance_params",
    "burn_in",
    "draws",
    "thinning",
    "prior_only",
    "categorical_features",
    "n_chains",
    "n_jobs",
    "random_state",
    "warm_start",
}


def test_the_parameters_are_the_groups_and_the_run_length_settings():
    assert set(AddiVortesClassifier(n_chains=1).get_params(deep=False)) == PARAMETERS
    regressor = AddiVortesRegressor(n_chains=1)
    assert set(regressor.get_params(deep=False)) == PARAMETERS | {
        "start",
        "fast_settings",
    }


def test_deep_params_route_into_the_groups():
    estimator = AddiVortesRegressor(mean_params=TermParams(tessellations=8), n_chains=1)
    deep = estimator.get_params(deep=True)
    assert deep["mean_params__tessellations"] == 8

    estimator.set_params(mean_params__tessellations=16)
    assert estimator.mean_params is not None
    assert estimator.mean_params.tessellations == 16


def test_regressor_fits_and_predicts(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)

    check_is_fitted(model)
    assert model.n_features_in_ == 2
    assert model.random_state_ == 1
    assert model.predict(x).shape == (48,)
    assert model.score(x, y) > 0.0


def test_regressor_return_std(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)

    mean, std = model.predict(x, return_std=True)
    np.testing.assert_allclose(mean, model.predict(x))
    assert std.shape == (48,)
    assert np.all(std > 0.0)


def test_regressor_prediction_interval_brackets_the_mean(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)

    interval = model.predict_interval(x, level=0.9)
    mean = model.predict(x)
    assert np.all(interval[:, 0] <= mean)
    assert np.all(mean <= interval[:, 1])


def test_heteroscedastic_model(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(
        variance_params=TermParams(tessellations=5), random_state=1, **SMALL, n_chains=1
    ).fit(x, y)

    assert model.predict(x).shape == (48,)


def test_probit_is_rejected_by_the_regressor(gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.raises(ValueError, match="AddiVortesClassifier"):
        AddiVortesRegressor(outcome=probit(), **SMALL, n_chains=1).fit(x, y)  # type: ignore[arg-type]


def test_the_classifier_rejects_a_regression_family(probit_fixture):
    x, y = probit_fixture
    with pytest.raises(ValueError, match="AddiVortesRegressor"):
        AddiVortesClassifier(outcome=gaussian(), **SMALL, n_chains=1).fit(x, y)  # type: ignore[arg-type]


def test_classifier_predicts_labels_and_probabilities(probit_fixture):
    x, y = probit_fixture
    model = AddiVortesClassifier(random_state=1, **SMALL, n_chains=1).fit(x, y)

    np.testing.assert_array_equal(model.classes_, [0.0, 1.0])
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (48, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(x))) <= {0.0, 1.0}
    assert model.score(x, y) > 0.5


def test_classifier_keeps_the_caller_labels(probit_fixture):
    x, y = probit_fixture
    labels = np.where(y > 0.5, "high", "low")

    model = AddiVortesClassifier(random_state=1, **SMALL, n_chains=1).fit(x, labels)

    np.testing.assert_array_equal(model.classes_, ["high", "low"])
    assert set(np.unique(model.predict(x))) <= {"high", "low"}


def test_classifier_rejects_more_than_two_classes(gaussian_fixture):
    x, _ = gaussian_fixture
    three = np.array([i % 3 for i in range(48)])
    with pytest.raises(ValueError, match="Only binary classification"):
        AddiVortesClassifier(**SMALL, n_chains=1).fit(x, three)


def test_the_estimator_and_model_agree_for_one_seed(gaussian_fixture):
    x, y = gaussian_fixture

    estimator = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)
    direct = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    np.testing.assert_array_equal(estimator.predict(x), direct.predict(x))


def test_clone_and_get_params_round_trip():
    model = AddiVortesRegressor(
        mean_params=TermParams(tessellations=10, lambda_c=25.0), n_chains=1
    )
    copy = clone(model)

    assert copy.get_params() == model.get_params()
    assert copy.mean_params is not None
    assert copy.mean_params.lambda_c == 25.0
    # The group is cloned through its own parameters, not shared.
    assert copy.mean_params is not model.mean_params


def test_pickle_round_trip_preserves_predictions(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)

    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    assert restored.random_state_ == model.random_state_


def test_pickle_of_an_unfitted_estimator(gaussian_fixture):
    model = AddiVortesRegressor(mean_params=TermParams(tessellations=10), n_chains=1)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params() == model.get_params()


def test_predict_before_fit_is_rejected(gaussian_fixture):
    x, _ = gaussian_fixture
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        AddiVortesRegressor(n_chains=1).predict(x)


def test_predict_with_the_wrong_column_count(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(x[:, :1])


def test_core_errors_surface_as_value_errors(gaussian_fixture):
    x, y = gaussian_fixture
    x = x.copy()
    x[:, 1] = 1.0
    with pytest.raises(ThiessenError):
        AddiVortesRegressor(**SMALL, n_chains=1).fit(x, y)


def test_feature_names_are_stored():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {"a": np.linspace(0.0, 1.0, 30), "b": np.linspace(1.0, 0.0, 30)}
    )
    y = frame["a"].to_numpy() + frame["b"].to_numpy() * 0.5

    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(frame, y)

    np.testing.assert_array_equal(model.feature_names_in_, ["a", "b"])


def test_pipeline_and_cross_validation(gaussian_fixture):
    x, y = gaussian_fixture
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("fit", AddiVortesRegressor(random_state=1, **SMALL, n_chains=1)),
        ]
    )

    scores = cross_val_score(pipeline, x, y, cv=3)

    assert scores.shape == (3,)


def test_grid_search_routes_into_the_groups(gaussian_fixture):
    x, y = gaussian_fixture
    search = GridSearchCV(
        AddiVortesRegressor(random_state=1, **SMALL, n_chains=1),
        {
            "mean_params__lambda_c": [5.0, 25.0],
            "outcome": [gaussian(nu=3.0), gaussian(nu=6.0)],
        },
        cv=2,
    )

    search.fit(x, y)

    assert search.best_params_["mean_params__lambda_c"] in (5.0, 25.0)
    assert search.best_params_["outcome"] in (gaussian(nu=3.0), gaussian(nu=6.0))


def test_partial_dependence_works_through_sklearn(gaussian_fixture):
    from sklearn.inspection import partial_dependence

    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)

    result = partial_dependence(model, x, features=[0], grid_resolution=5)

    assert result["average"].shape == (1, 5)


class TestCategorical:
    """The `categorical_features` parameter."""

    @staticmethod
    def _data():
        n = 60
        codes = np.array([i % 3 for i in range(n)], dtype=float)
        continuous = np.linspace(0.0, 1.0, n)
        x = np.column_stack([continuous, codes])
        y = continuous + 0.5 * codes
        return x, y

    def test_indices_expand_to_treatment_contrasts(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[1], random_state=1, **SMALL, n_chains=1
        ).fit(x, y)

        # Three levels become two indicator columns, the first as reference.
        assert model._encoding.core_metric == ["euclidean"] * 3
        assert model.n_features_in_ == 2
        assert model.predict(x).shape == (60,)

    def test_boolean_mask_is_accepted(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[False, True], random_state=1, **SMALL, n_chains=1
        ).fit(x, y)

        assert model.predict(x).shape == (60,)

    def test_eskin_metric_passes_integer_codes(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[1],
            mean_params=TermParams(
                tessellations=8,
                geometry=GeometryParams(metric=["euclidean", "categorical"]),
            ),
            burn_in=10,
            draws=20,
            random_state=1,
            n_chains=1,
        ).fit(x, y)

        assert model._encoding.core_metric == ["euclidean", "categorical"]
        assert model.predict(x).shape == (60,)

    def test_no_encoding_happens_without_the_parameter(self):
        x, y = self._data()
        model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)

        assert model._encoding.core_metric == ["euclidean", "euclidean"]

    def test_string_levels_are_encoded(self):
        n = 60
        continuous = np.linspace(0.0, 1.0, n)
        letters = np.array(["a", "b", "c"] * (n // 3))
        x = np.empty((n, 2), dtype=object)
        x[:, 0] = continuous
        x[:, 1] = letters
        y = continuous + 0.5 * (letters == "b")

        model = AddiVortesRegressor(
            categorical_features=[1], random_state=1, **SMALL, n_chains=1
        ).fit(x, y)

        assert model.predict(x).shape == (60,)

    def test_an_unseen_level_is_rejected(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[1], random_state=1, **SMALL, n_chains=1
        ).fit(x, y)

        unseen = x.copy()
        unseen[0, 1] = 9.0
        with pytest.raises(ValueError, match="not one of the levels"):
            model.predict(unseen)

    def test_a_single_level_is_rejected(self):
        x, y = self._data()
        constant = x.copy()
        constant[:, 1] = 0.0
        with pytest.raises(ValueError, match="at least two"):
            AddiVortesRegressor(categorical_features=[1], **SMALL, n_chains=1).fit(
                constant, y
            )

    def test_an_out_of_range_index_is_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="outside"):
            AddiVortesRegressor(categorical_features=[5], **SMALL, n_chains=1).fit(x, y)

    def test_an_unknown_string_is_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="from_dtype"):
            AddiVortesRegressor(categorical_features="auto", **SMALL, n_chains=1).fit(
                x, y
            )

    def test_from_dtype_reads_the_pandas_dtypes(self):
        pandas = pytest.importorskip("pandas")
        n = 60
        frame = pandas.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, n),
                "g": pandas.Categorical(["a", "b", "c"] * (n // 3)),
            }
        )
        y = frame["x"].to_numpy() + 0.5 * (frame["g"] == "b")

        model = AddiVortesRegressor(
            categorical_features="from_dtype", random_state=1, **SMALL, n_chains=1
        ).fit(frame, y)

        assert model._encoding.core_metric == ["euclidean"] * 3
        np.testing.assert_array_equal(model.feature_names_in_, ["x", "g"])
        assert model.predict(frame).shape == (60,)

    def test_from_dtype_needs_a_data_frame(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="data frame"):
            AddiVortesRegressor(
                categorical_features="from_dtype", **SMALL, n_chains=1
            ).fit(x, y)


def test_n_jobs_spreads_the_chains_without_changing_the_draws(gaussian_fixture):
    from thiessen.estimators import _resolve_jobs

    x, y = gaussian_fixture
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = AddiVortesRegressor(**SMALL, n_chains=2, random_state=1).fit(x, y)
        threaded = AddiVortesRegressor(
            **SMALL, n_chains=2, n_jobs=2, random_state=1
        ).fit(x, y)
        every_core = AddiVortesRegressor(
            **SMALL, n_chains=2, n_jobs=-1, random_state=1
        ).fit(x, y)

    np.testing.assert_array_equal(threaded.predict(x), serial.predict(x))
    np.testing.assert_array_equal(every_core.predict(x), serial.predict(x))
    assert threaded._fitted.threads == 2
    assert every_core._fitted.threads == _resolve_jobs(-1) >= 1
    assert _resolve_jobs(None) == 1
    with pytest.raises(ValueError):
        _resolve_jobs(0)


def test_n_jobs_is_read_at_each_prediction(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)
    mean = model.predict(x)

    model.set_params(n_jobs=2)

    np.testing.assert_array_equal(model.predict(x), mean)
    assert model._fitted.threads == 2


def test_the_estimators_default_to_four_chains_on_one_thread(gaussian_fixture):
    x, y = gaussian_fixture

    assert AddiVortesRegressor().n_chains == 4
    assert AddiVortesClassifier().n_chains == 4
    assert AddiVortesRegressor().n_jobs is None
    with pytest.warns(UserWarning):
        model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)
    assert model._fitted.n_draws == 4 * SMALL["draws"]
