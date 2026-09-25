"""Conversion of a fitted model to an arviz `DataTree`.

Requires arviz, the `arviz` extra. The groups follow the PyMC and numpyro
convention: `posterior`, `posterior_predictive`, `log_likelihood` and
`observed_data`, over the chains of the fit.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native
from ._response import Response, _log_likelihood

__all__ = ["_to_inference_data"]

Float = npt.NDArray[np.float64]


def _replicates(fitted: _native.Fitted, design: Float, seed: int) -> Float:
    """Draw one posterior predictive replicate per kept draw.

    The replicates are drawn in numpy from a generator seeded by the fit's
    resolved seed, not by the core, so they are reproducible but are not the
    core's draws. Each family's replicate is its own observation model:
    labels under the probit family, category codes through the cutpoints
    under the ordinal family, a time under the AFT family, a value clipped
    to the limits under the tobit family, and the error of the Student-t
    and Laplace families at the drawn scale.
    """
    rng = np.random.default_rng(seed)
    model = fitted.model
    if model == "probit":
        draws = fitted.predict_draws(design)
        return rng.binomial(1, np.clip(draws, 0.0, 1.0)).astype(np.float64)
    latent = fitted.predict_latent(design)
    n_draws, n_rows = latent.shape
    if model == "ordinal":
        interior = np.asarray(fitted.cutpoints())
        if interior.size == 0:
            interior = np.empty((n_draws, 0))
        # The first cutpoint is fixed at zero; a category is the count of
        # cutpoints below the latent value.
        cutpoints = np.concatenate([np.zeros((n_draws, 1)), interior], axis=1)
        z = latent + rng.standard_normal(latent.shape)
        return (z[:, :, None] > cutpoints[:, None, :]).sum(axis=2).astype(np.float64)
    outcome: dict[str, Any] = json.loads(fitted.config)["outcome"]
    if model in ("student_t", "laplace"):
        # The scale of the t or Laplace error per draw, not the error
        # standard deviation `predict_variance` squares.
        sigma = np.asarray(fitted.sigma())[:, None]
        if model == "student_t":
            dfs = np.asarray(fitted.dfs())
            df = dfs[:, None] if dfs.size else float(outcome["student_t"]["df"])
            return latent + sigma * rng.standard_t(df, size=latent.shape)
        return latent + sigma * rng.laplace(0.0, 1.0, size=latent.shape)
    scale = np.sqrt(fitted.predict_variance(design))
    values = latent + rng.normal(0.0, scale)
    if model == "aft":
        return np.exp(values)
    if model == "tobit":
        limits = outcome["tobit"]
        lower = -np.inf if limits.get("lower") is None else float(limits["lower"])
        upper = np.inf if limits.get("upper") is None else float(limits["upper"])
        return np.clip(values, lower, upper)
    return values


def _observed(response: Response) -> dict[str, npt.NDArray[Any]]:
    """Return the `observed_data` variables of a parsed response."""
    if response.kind == "aft":
        return {"time": response.times, "event": response.events}
    if response.kind == "interval_censored":
        return {"lower": response.lower, "upper": response.upper}
    return {"y": response.y}


def _to_inference_data(
    fitted: _native.Fitted,
    design: Float,
    response: Response,
    seed: int,
    n_chains: int = 1,
) -> Any:
    """Build the `DataTree` of a fitted model over `design` and `response`.

    Parameters
    ----------
    fitted : Fitted
        The native handle.
    design : numpy.ndarray of shape (n_samples, n_features)
        The rows the observation dimension indexes.
    response : Response
        The observed response, parsed.
    seed : int
        The fit's resolved seed, which seeds the predictive replicates.
    n_chains : int, default=1
        The number of chains the pooled draws hold, in chain order.

    Returns
    -------
    xarray.DataTree
        The `posterior`, `posterior_predictive`, `log_likelihood` and
        `observed_data` groups. The replicate and the log-likelihood take
        the name of the observed variable they pair with, ``time`` under
        the AFT family and ``y`` otherwise; an interval-censored
        observation is a pair of bounds, so its replicate has no partner
        and keeps ``y``.

    Raises
    ------
    ImportError
        If arviz is not installed.
    ValueError
        If `design` and `response` disagree on the number of rows.
    """
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "to_inference_data needs arviz; install the `arviz` extra"
        ) from error

    if design.shape[0] != response.n:
        raise ValueError(
            f"X has {design.shape[0]} rows and y has {response.n}; "
            "the observation dimension needs one label per row"
        )

    # The pooled draws are in chain order, so the leading axis splits into
    # chain by iteration.
    def by_chain(draws: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(draws).reshape((n_chains, -1) + np.shape(draws)[1:])

    latent = fitted.predict_latent(design)
    posterior: dict[str, npt.NDArray[np.float64]] = {
        "mu": by_chain(latent),
        "cell_count": by_chain(np.asarray(fitted.cell_counts())),
        "dimension_count": by_chain(np.asarray(fitted.dimension_counts())),
    }
    # Each is empty under the models that do not sample it: sigma under
    # the probit and ordinal models and the heteroscedastic model, the
    # rest outside the experimental item that draws it.
    sampled = {
        "sigma": np.asarray(fitted.sigma()),
        "df": np.asarray(fitted.dfs()),
        "cutpoint": np.asarray(fitted.cutpoints()),
        "bandwidth": np.asarray(fitted.bandwidths()),
        "inclusion_weight": np.asarray(fitted.inclusion_weights()),
        "concentration": np.asarray(fitted.concentrations()),
    }
    for name, draws in sampled.items():
        if draws.size:
            posterior[name] = by_chain(draws)

    observed = _observed(response)
    replicate = "time" if "time" in observed else "y"
    groups = {
        "posterior": posterior,
        "posterior_predictive": {
            replicate: by_chain(_replicates(fitted, design, seed))
        },
        "log_likelihood": {
            replicate: by_chain(np.asarray(_log_likelihood(fitted, design, response)))
        },
        "observed_data": observed,
    }

    return az.from_dict(
        groups,
        coords={"observation": np.arange(design.shape[0])},
        dims={
            "mu": ["observation"],
            "y": ["observation"],
            "time": ["observation"],
            "event": ["observation"],
            "lower": ["observation"],
            "upper": ["observation"],
            "cutpoint": ["cutpoint_index"],
            "bandwidth": ["tessellation"],
            "inclusion_weight": ["feature"],
        },
    )
