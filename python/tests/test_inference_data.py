"""The arviz interop."""

from __future__ import annotations

import numpy as np
import pytest
from thiessen import Model, TermParams, _native, laplace, probit, student_t
from thiessen._inference import _replicates

from .conftest import SMALL

pytest.importorskip("arviz")


def test_the_groups_follow_the_convention(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    data = fitted.to_inference_data(x, y)

    assert set(data.children) == {
        "posterior",
        "posterior_predictive",
        "log_likelihood",
        "observed_data",
    }


def test_the_posterior_carries_the_mean_function_and_sigma(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    posterior = fitted.to_inference_data(x, y)["posterior"].dataset

    assert posterior["mu"].dims == ("chain", "draw", "observation")
    assert posterior["mu"].shape == (1, 20, 48)
    assert posterior["sigma"].shape == (1, 20)
    assert posterior["cell_count"].shape == (1, 20)
    assert posterior["dimension_count"].shape == (1, 20)
    np.testing.assert_array_equal(posterior["sigma"].values[0], fitted.sigma())


def test_the_observation_dimension_is_labelled(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    data = fitted.to_inference_data(x, y)

    np.testing.assert_array_equal(
        data["posterior"].dataset.coords["observation"].values, np.arange(48)
    )
    assert data["observed_data"].dataset["y"].dims == ("observation",)


def test_the_observed_data_is_the_response(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    observed = fitted.to_inference_data(x, y)["observed_data"].dataset["y"]

    np.testing.assert_array_equal(observed.values, y)


def test_the_log_likelihood_matches_the_accessor(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    group = fitted.to_inference_data(x, y)["log_likelihood"].dataset["y"]

    assert group.shape == (1, 20, 48)
    np.testing.assert_array_equal(group.values[0], fitted.log_likelihood(x, y))


def test_the_predictive_replicates_are_reproducible(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    first = fitted.to_inference_data(x, y)["posterior_predictive"].dataset["y"]
    second = fitted.to_inference_data(x, y)["posterior_predictive"].dataset["y"]

    np.testing.assert_array_equal(first.values, second.values)
    assert first.shape == (1, 20, 48)


def test_the_probit_model_carries_no_sigma(probit_fixture):
    x, y = probit_fixture
    fitted = Model(outcome=probit(), **SMALL).fit(x, y, random_state=1, n_chains=1)

    posterior = fitted.to_inference_data(x, y)["posterior"].dataset

    assert "sigma" not in posterior
    assert "mu" in posterior


def test_the_probit_replicates_are_labels(probit_fixture):
    x, y = probit_fixture
    fitted = Model(outcome=probit(), **SMALL).fit(x, y, random_state=1, n_chains=1)

    replicates = fitted.to_inference_data(x, y)["posterior_predictive"].dataset["y"]

    assert set(np.unique(replicates.values)) <= {0.0, 1.0}


def test_the_heteroscedastic_model_carries_no_sigma(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(variance_params=TermParams(tessellations=5), **SMALL).fit(
        x, y, random_state=1, n_chains=1
    )

    posterior = fitted.to_inference_data(x, y)["posterior"].dataset

    assert "sigma" not in posterior


def test_a_row_count_mismatch_fails_loudly(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    with pytest.raises(ValueError, match="one label per row"):
        fitted.to_inference_data(x, y[:-1])


def test_arviz_reads_the_result(gaussian_fixture):
    az = pytest.importorskip("arviz")
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    data = fitted.to_inference_data(x, y)
    summary = az.summary(data, var_names=["sigma"])

    assert "sigma" in str(summary)


@pytest.mark.skipif(not _native.EXPERIMENTAL, reason="built without the feature")
def test_the_heavy_tailed_replicates_are_at_sigma(gaussian_fixture):
    x, y = gaussian_fixture

    model = Model(outcome=student_t(df=3.0), **SMALL)
    fitted = model.fit(x, y, random_state=1, n_chains=1)
    latent = fitted._fitted.predict_latent(x)
    sigma = np.asarray(fitted._fitted.sigma())[:, None]
    errors = np.random.default_rng(1).standard_t(3.0, size=latent.shape)
    expected = latent + sigma * errors
    np.testing.assert_array_equal(_replicates(fitted._fitted, x, 1), expected)
    np.testing.assert_allclose(
        sigma[:, 0] * np.sqrt(3.0), np.sqrt(fitted._fitted.predict_variance(x)[:, 0])
    )

    fitted = Model(outcome=laplace(), **SMALL).fit(x, y, random_state=1, n_chains=1)
    latent = fitted._fitted.predict_latent(x)
    sigma = np.asarray(fitted._fitted.sigma())[:, None]
    errors = np.random.default_rng(1).laplace(0.0, 1.0, size=latent.shape)
    expected = latent + sigma * errors
    np.testing.assert_array_equal(_replicates(fitted._fitted, x, 1), expected)

    model = Model(outcome=student_t(df=[2.0, 4.0]), **SMALL)
    fitted = model.fit(x, y, random_state=1, n_chains=1)
    assert np.isfinite(_replicates(fitted._fitted, x, 1)).all()
