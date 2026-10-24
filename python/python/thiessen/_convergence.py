"""Convergence diagnostics of a multi-chain fit.

Rank-normalised split R-hat and the bulk and tail effective sample sizes
(Vehtari, Gelman, Simpson, Carpenter and Buerkner, 2021) of sigma, where
the model has one, and of the mean function at a subsample of the training
rows, computed by arviz at the thresholds arviz documents. Requires arviz,
the `arviz` extra.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native

__all__ = ["_warn_convergence"]

#: Above this R-hat the chains have not mixed.
RHAT_THRESHOLD = 1.01

#: Below this effective sample size the summaries are not reliable.
ESS_THRESHOLD = 400

#: Training rows the mean function is monitored at.
POINTS = 20


def _monitored_rows(n: int, points: int = POINTS) -> npt.NDArray[np.intp]:
    """Return the rows of the design the mean function is monitored at."""
    if n <= points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, points)).astype(np.intp))


def _worst(diagnostic: Any, reduce: Any) -> float:
    """Reduce a diagnostic over every monitored variable.

    `arviz.rhat` and `arviz.ess` return the posterior group itself, one
    value per variable and per index of a vector variable.
    """
    values = np.concatenate(
        [
            np.ravel(np.asarray(variable.values))
            for variable in diagnostic.dataset.data_vars.values()
        ]
    )
    if not np.isfinite(values).any():
        return float("nan")
    return float(reduce(values))


def _diagnostics(
    fitted: _native.Fitted,
    n_chains: int,
    design: npt.NDArray[np.float64],
) -> dict[str, float] | None:
    """Return the worst diagnostics over the monitored variables.

    Returns `None` where arviz is not installed.
    """
    try:
        import arviz as az
    except ImportError:  # pragma: no cover - depends on the install
        return None

    rows = _monitored_rows(design.shape[0])
    latent = fitted.predict_latent(design[rows])
    iterations = latent.shape[0] // n_chains
    posterior: dict[str, npt.NDArray[np.float64]] = {
        "mu": latent.reshape(n_chains, iterations, -1)
    }
    sigma = np.asarray(fitted.sigma())
    if sigma.size:
        posterior["sigma"] = sigma.reshape(n_chains, iterations)
    tree = az.from_dict({"posterior": posterior})
    return {
        "rhat": _worst(az.rhat(tree), np.nanmax),
        "ess_bulk": _worst(az.ess(tree, method="bulk"), np.nanmin),
        "ess_tail": _worst(az.ess(tree, method="tail"), np.nanmin),
    }


def _convergence_message(diagnostics: dict[str, float]) -> str | None:
    """Return the message a fit that has not converged carries."""
    ess = min(diagnostics["ess_bulk"], diagnostics["ess_tail"])
    if not diagnostics["rhat"] > RHAT_THRESHOLD and not ess < ESS_THRESHOLD:
        return None
    return (
        f"the chains may not have converged: largest R-hat "
        f"{diagnostics['rhat']:.3f} (threshold {RHAT_THRESHOLD:.2f}), "
        f"smallest effective sample size {ess:.0f} (threshold "
        f"{ESS_THRESHOLD}); run more draws or more chains"
    )


def _warn_convergence(
    fitted: _native.Fitted,
    n_chains: int,
    design: npt.NDArray[np.float64],
    stacklevel: int,
    warm: bool = False,
) -> None:
    """Warn where a multi-chain fit has not met the thresholds.

    A fit of one chain has no diagnostics to report. Where arviz is not
    installed the fit says so instead of checking. Under a warm start the
    chains began at draws of one previous fit, not at overdispersed
    points, so the message says the diagnostics are a within-run check
    (Vehtari and others, 2021).
    """
    if n_chains < 2:
        return
    diagnostics = _diagnostics(fitted, n_chains, design)
    if diagnostics is None:  # pragma: no cover - depends on the install
        warnings.warn(
            "the convergence diagnostics of a multi-chain fit need arviz; "
            "install the `arviz` extra",
            UserWarning,
            stacklevel=stacklevel,
        )
        return
    message = _convergence_message(diagnostics)
    if message is not None:
        if warm:
            message += (
                "; the chains were warm-started from draws of a previous fit, "
                "so these diagnostics compare chains that did not start apart"
            )
        warnings.warn(message, UserWarning, stacklevel=stacklevel)
