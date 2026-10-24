"""The sampler API: the core's Gibbs loop, driven one call at a time.

.. note::
    This interface is experimental: it may change in a minor release
    without a deprecation cycle. The models it drives and the draws it
    produces carry the same guarantees as `thiessen.Model.fit`.

`Sampler` is the researcher's interface, after the updatable sampler of
dbarts and the low-level interface of stochtree: construct with the
configuration, the data and a seed, then drive the loop yourself.
Burn-in and thinning are the caller's loop, and the response may be
replaced between sweeps, which is what makes censoring, imputation and
custom likelihoods through the response possible. Anything that is not an
outcome family or a setting goes through this loop.

The response passes through the affine map to the model's internal scale
frozen at construction, so a response outside the training range is
legitimate. The sampler owns its RNG, seeded at construction with the
chain-0 seed of `fit`; the loop cannot rewire tessellation membership or
cell internals.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native
from ._arrays import _as_design
from ._config import _config_json
from ._fast import _fast_json
from ._params import Outcome
from ._response import (
    _as_response,
    _check_family,
    _new_sampler,
    _resolve_outcome,
    _sampler_family,
    _set_response,
)
from ._seed import SeedLike, _resolve_seed
from .model import FittedModel
from .params import TermParams

__all__ = ["Sampler"]


class Sampler:
    """The core's Gibbs loop over one chain, driven by the caller.

    Experimental: this interface may change in a minor release without a
    deprecation cycle.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        The design, under the input-data contract of `thiessen.Model.fit`.
    y : array_like
        The response, in the shapes `thiessen.Model.fit` takes: a numeric
        array, a boolean array or ``Categorical`` of labels, an ordered
        ``Categorical``, a structured survival array or a two-column array
        of bounds.
    outcome : Outcome, optional
        The outcome family, from a constructor of `thiessen.families`.
        `None` takes the family the response selects, as
        `thiessen.Model.fit` does.
    mean_params : TermParams, optional
        The ensemble describing the average, as for `thiessen.Model`.
    variance_params : TermParams, optional
        The ensemble describing the spread, as for `thiessen.Model`.
    prior_only : bool, default=False
        Switch the likelihood off, so the loop draws from the prior.
    random_state : int, Generator, RandomState or None, default=None
        The seed, resolved by the rule of `thiessen.Model.fit`. The
        sampler runs the chain that `fit` would run first, so driving the
        configured schedule by hand reproduces ``fit`` bit for bit.

    Attributes
    ----------
    random_state : int
        The resolved seed.

    See Also
    --------
    thiessen.Model : The fitting entry point this loop underlies.

    Notes
    -----
    Burn-in and thinning stay in the caller's loop; the ``burn_in``,
    ``draws`` and ``thinning`` settings play no part here.

    Examples
    --------
    A Gaussian fit written as its own loop:

    >>> import numpy as np
    >>> from thiessen import TermParams
    >>> from thiessen.sampler import Sampler
    >>> x = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    >>> y = 3.0 * x[:, 0] ** 2 - x[:, 0]
    >>> sampler = Sampler(x, y, mean_params=TermParams(tessellations=10),
    ...                   random_state=1)
    >>> sampler.step(20)
    >>> for _ in range(30):
    ...     sampler.step(1)
    ...     sampler.keep()
    >>> sampler.finish().n_draws
    30
    """

    def __init__(
        self,
        X: Any,
        y: Any,
        *,
        outcome: Outcome | None = None,
        mean_params: TermParams | None = None,
        variance_params: TermParams | None = None,
        prior_only: bool = False,
        random_state: SeedLike = None,
    ) -> None:
        response = _as_response(y)
        config = _config_json(
            _resolve_outcome(outcome, response),
            mean_params,
            variance_params,
            burn_in=0,
            draws=1,
            thinning=1,
            prior_only=prior_only,
        )
        seed = _resolve_seed(random_state)
        self.random_state = seed
        self._sampler = _new_sampler(config, _as_design(X), response, seed)

    def step(self, n: int = 1) -> None:
        """Run `n` sweeps of the Gibbs loop.

        Parameters
        ----------
        n : int, default=1
            The number of sweeps.

        Raises
        ------
        ThiessenError
            After `finish`.
        ValueError
            If `n` is negative.
        """
        count = int(n)
        if count != n or count < 0:
            raise ValueError(f"n must be a whole number of sweeps; got {n!r}")
        self._sampler.step(count)

    def xaddivortes(self, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Replace the chain's state by an XAddiVortes fast fit.

        The sweeps that follow run the exact sampler from the fast fit
        instead of the cold start; kept at once, the fit is a one-draw
        model whose noise draw is the fit's sigma. Experimental: needs a
        core built with the ``experimental`` feature.

        Parameters
        ----------
        settings : dict, optional
            XAddiVortes settings, by the names of
            `thiessen.estimators.XAddiVortesRegressor`; an omitted setting
            takes its default.

        Returns
        -------
        dict
            The fit's summary: ``sigma``, the in-sample residual root mean
            square on the response scale; ``rounds``; ``planned_sweeps``;
            ``column_scores``; ``zeroed``; and ``train_score``, the residual
            root mean square after the build and after each round.

        Raises
        ------
        RequiresFeatureError
            If the core was built without the ``experimental`` feature.
        ValueError
            For a setting of unknown name or of a non-finite value.
        ThiessenError
            After `finish`, once the chain has swept or kept a draw, for
            an outcome family other than the Gaussian with a constant
            spread, or for a setting of the wrong type or out of its range,
            naming it.
        """
        raw = self._sampler.xaddivortes(_fast_json(settings))
        summary: dict[str, Any] = json.loads(raw)
        return summary

    def keep(self) -> None:
        """Record the current state as a posterior draw.

        Raises
        ------
        ThiessenError
            After `finish`.
        """
        self._sampler.keep()

    def set_response(self, y: Any) -> None:
        """Replace the response; the next sweep conditions on it.

        The tessellations, the cell values and sigma^2 are kept. The new
        response is on the caller's scale and passes through the affine
        map frozen at construction, so values outside the training range
        are legitimate.

        Parameters
        ----------
        y : array_like
            The new response, in the shape the constructor took for the
            family: the times and event flags under the AFT family, the
            bound pairs under the interval-censored family.

        Raises
        ------
        ThiessenError
            For a row-count mismatch, a non-finite value, a label outside
            {0, 1} under the probit family, or after `finish`.
        ValueError
            For a response the family does not take.
        """
        response = _as_response(y)
        _check_family(_sampler_family(self._sampler), response)
        _set_response(self._sampler, response)

    def fitted_values(self) -> npt.NDArray[np.float64]:
        """Return the current mean function at the training rows.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            f(x_i) on the caller's scale, or c + f(x_i) under the probit
            family.

        Raises
        ------
        ThiessenError
            After `finish`.
        """
        return self._sampler.fitted_values()

    def noise_variances(self) -> npt.NDArray[np.float64]:
        """Return the current variance of y given f at each training row.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            sigma^2 under the Gaussian model, 1 under the probit family
            (the latent scale), s^2(x_i) under the heteroscedastic model.

        Raises
        ------
        ThiessenError
            After `finish`.
        """
        return self._sampler.noise_variances()

    @property
    def n_kept(self) -> int:
        """int: The number of draws kept so far."""
        return int(self._sampler.n_kept)

    @property
    def config(self) -> dict[str, Any]:
        """dict: The resolved configuration, the core's four groups."""
        parsed: dict[str, Any] = json.loads(self._sampler.config)
        return parsed

    def finish(self) -> FittedModel:
        """Return the fitted model of the kept draws.

        Consumes the sampler: every later call on it raises.

        Returns
        -------
        FittedModel
            The kept draws, as `thiessen.Model.fit` returns them.

        Raises
        ------
        ThiessenError
            If no draws were kept, or on a second call.
        """
        return FittedModel(self._sampler.finish(), self.random_state, 1)

    def __repr__(self) -> str:
        try:
            kept = self.n_kept
        except _native.ThiessenError:
            return f"Sampler(finished, random_state={self.random_state})"
        return f"Sampler(n_kept={kept}, random_state={self.random_state})"
