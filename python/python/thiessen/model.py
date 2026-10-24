"""The configuration and the fitted model.

`Model` holds a configuration and fits it; `FittedModel` holds the kept
draws and answers prediction and posterior queries.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native
from ._arrays import _as_design
from ._config import _config_json
from ._convergence import _warn_convergence
from ._fast import _fast_json
from ._inference import _to_inference_data
from ._params import Outcome, Params
from ._response import (
    _as_response,
    _check_family,
    _fit,
    _log_likelihood,
    _resolve_outcome,
)
from ._seed import SeedLike, _resolve_seed
from .params import TermParams

__all__ = ["FittedModel", "Model"]


class Model(Params):
    """An AddiVortes configuration.

    The outcome family and the two ensembles are objects with the names of
    the stored configuration; parameters left unset take the core's
    defaults. The parameters are those of Stone and Gosling (2025), s. 2,
    with the sweep schedule that `fit` runs.

    Parameters
    ----------
    outcome : Outcome, optional
        The outcome family, from `gaussian`, `probit` or one of the
        experimental constructors of `thiessen.families`. `None`, the
        default, takes the family the response selects at `fit`: the
        Gaussian family for a numeric array, the probit family for a
        boolean array or a two-category ``Categorical``, the ordinal
        family for an ordered ``Categorical``, the AFT family for a
        structured survival array and the interval-censored family for a
        two-column array of bounds. A named family is checked against the
        response and a mismatch is an error naming both.
    mean_params : TermParams, optional
        The ensemble describing the average. Default ``TermParams()``,
        whose tessellation count resolves to 200.
    variance_params : TermParams, optional
        The ensemble describing the spread. Default none: the spread is
        constant. A positive tessellation count selects the
        heteroscedastic model and needs the Gaussian family with nu > 2;
        the paper's count is 40.
    burn_in : int, default=200
        Burn-in sweeps discarded.
    draws : int, default=1000
        Posterior draws kept.
    thinning : int, default=1
        Thinning interval; every `thinning`-th sweep after burn-in is
        kept.
    prior_only : bool, default=False
        Switch the likelihood off, so the chain draws from the prior and
        `predict` gives prior predictive draws.

    See Also
    --------
    FittedModel : The result of `fit`.
    thiessen.gaussian, thiessen.probit : The outcome families.
    thiessen.TermParams : One ensemble's parameters.

    Notes
    -----
    Stone, E. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
    Voronoi tessellations. *Journal of Computational and Graphical
    Statistics*, 34(3), 859-871.

    Examples
    --------
    >>> import numpy as np
    >>> from thiessen import Model, TermParams
    >>> x = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    >>> y = 3.0 * x[:, 0] ** 2 - x[:, 0]
    >>> model = Model(mean_params=TermParams(tessellations=10),
    ...               burn_in=20, draws=30)
    >>> model.fit(x, y, random_state=42).predict(x).shape
    (40,)

    The heteroscedastic model attaches a variance ensemble:

    >>> hetero = Model(variance_params=TermParams(tessellations=40))
    """

    def __init__(
        self,
        *,
        outcome: Outcome | None = None,
        mean_params: TermParams | None = None,
        variance_params: TermParams | None = None,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
    ) -> None:
        self.outcome = outcome
        self.mean_params = mean_params
        self.variance_params = variance_params
        self.burn_in = burn_in
        self.draws = draws
        self.thinning = thinning
        self.prior_only = prior_only

    def _json(self) -> str:
        return _config_json(
            self.outcome,
            self.mean_params,
            self.variance_params,
            self.burn_in,
            self.draws,
            self.thinning,
            self.prior_only,
        )

    def validate(self) -> None:
        """Validate the configuration without data.

        Raises
        ------
        ThiessenError
            Naming the field at fault. Checks that need the data, the
            omega <= p bound and the length of ``metric``, run at fit.
        TypeError
            For a group that is not its parameter object.
        """
        _native.validate_config(self._json())

    def fit(
        self,
        X: Any,
        y: Any,
        random_state: SeedLike = None,
        n_chains: int = 4,
        n_threads: int = 1,
        init: FittedModel | None = None,
        init_draw: int | None = None,
        start: str = "cold",
        fast_settings: Mapping[str, Any] | None = None,
    ) -> FittedModel:
        """Fit the model to `X` and `y`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design. Euclidean columns are min-max scaled over their
            training range; spherical columns are coordinates in radians
            and categorical columns are integer level codes, neither
            scaled.
        y : array_like
            The response, one observation per row of `X`: a numeric array
            of shape (n_samples,); a boolean array or a two-category
            pandas ``Categorical`` of labels, or the numbers 0 and 1 under
            a named probit family; an ordered ``Categorical``, or integer
            codes 0 to K - 1 under a named ordinal family; a structured
            array of a boolean event indicator and a time, the layout of
            ``sksurv.util.Surv.from_arrays``, under the AFT family; or an
            array of shape (n_samples, 2) of lower and upper bounds, an
            infinite bound for one-sided censoring and an equal pair for
            an exact value, under the interval-censored family.
        random_state : int, numpy.random.Generator, numpy.random.RandomState or None
            The seed. `None` draws fresh entropy. The resolved seed is on
            the returned object.
        n_chains : int, default=4
            The number of chains to run. Each chain has its own seed,
            derived from the resolved seed in the core, and the draws of
            the chains are pooled.
        n_threads : int, default=1
            The number of threads. The chains are spread over at most this
            many threads, each chain on one thread with its own generator,
            so the draws do not depend on it; the returned model splits
            the rows of a prediction over the same number.
        init : FittedModel, optional
            A previous fit of this model to the same data. Each chain then
            starts from one of its kept draws instead of the cold start;
            only the starting state changes, not the posterior. Without
            `init_draw` and with as many chains as `init`, chain k
            continues from the last draw of chain k of `init`. Otherwise
            chain k (from 0) starts from pooled draw ``init_draw - k``,
            `init_draw` defaulting to the last; with fewer draws than
            chains up to it, every chain starts from it and the fit warns.
            This is stochtree's rule. Experimental: needs a core built with
            the ``experimental`` feature.
        init_draw : int, optional
            The 0-based pooled draw of `init` the first chain starts from.
        start : {'cold', 'fast'}, default='cold'
            Where each chain starts: the sampler's own start, or its own
            XAddiVortes fast fit (`thiessen.estimators.XAddiVortesRegressor`),
            after which the same burn-in and draws run. Only the starting
            state differs. 'fast' needs a numeric response under the
            Gaussian family with a constant spread, cannot be combined with
            `init`, and needs a core built with the ``experimental``
            feature.
        fast_settings : dict, optional
            XAddiVortes settings for ``start='fast'``, by the names of
            `XAddiVortesRegressor`; unset settings take its defaults. Given
            with ``start='cold'`` it is an error.

        Returns
        -------
        FittedModel
            The kept draws of every chain.

        Warns
        -----
        UserWarning
            Where two or more chains ran and R-hat exceeds 1.01 or an
            effective sample size falls below 400 (Vehtari and others,
            2021), or where arviz is not installed to compute them.

        Raises
        ------
        ThiessenError
            For an invalid configuration, or for data the core rejects:
            missing or non-finite values, a constant response, a constant
            column, fewer than two rows, or a row-count mismatch.
        ValueError
            If `n_chains` or `n_threads` is not a positive integer, if
            `init` is not a `FittedModel` or `init_draw` not a
            non-negative integer, or for a response the named family does
            not take.
        ValueError
            For ``start`` other than 'cold' or 'fast', `fast_settings` with
            ``start='cold'``, ``start='fast'`` with `init` or a response
            other than numeric, or a fast setting of the wrong type.
        RequiresFeatureError
            For `init` or ``start='fast'`` in a build without the
            ``experimental`` feature.
        """
        design = _as_design(X)
        response = _as_response(y)
        outcome = _resolve_outcome(self.outcome, response)
        config = _config_json(
            outcome,
            self.mean_params,
            self.variance_params,
            self.burn_in,
            self.draws,
            self.thinning,
            self.prior_only,
        )
        seed = _resolve_seed(random_state)
        chains = _resolve_chains(n_chains)
        threads = _resolve_threads(n_threads)
        previous, draw, draws = _resolve_init(init, init_draw, chains)
        if start not in ("cold", "fast"):
            raise ValueError(f"start must be 'cold' or 'fast'; got {start!r}")
        if start == "cold" and fast_settings is not None:
            raise ValueError("fast_settings sets the fast start; it needs start='fast'")
        if start == "fast" and previous is not None:
            raise ValueError("start='fast' and init both set the start; give one")
        if start == "fast":
            if response.kind != "plain" or response.shape != "numeric":
                raise ValueError(
                    "start='fast' needs a numeric response under the Gaussian family"
                )
            fitted = _native.fit_from_fast(
                config,
                design,
                response.y,
                seed,
                chains,
                threads,
                _fast_json(fast_settings),
            )
        else:
            fitted = _fit(
                config,
                design,
                response,
                seed,
                chains,
                threads,
                init=previous,
                init_draw=draw,
                init_draws=draws,
            )
        _emit_warnings(fitted, stacklevel=3)
        _warn_convergence(
            fitted,
            chains,
            design,
            stacklevel=3,
            warm=previous is not None,
        )
        return FittedModel(fitted, seed, chains, threads)


def _resolve_init(
    init: Any, init_draw: Any, n_chains: int
) -> tuple[_native.Fitted | None, int | None, list[int] | None]:
    """Return the core fit and the 0-based draws a warm start names.

    The third value names each chain's draw: the last of its counterpart
    in `init`, when `init_draw` is unset and the chain counts agree.
    """
    if init is None:
        if init_draw is not None:
            raise ValueError("init_draw needs init")
        return None, None, None
    if not isinstance(init, FittedModel):
        raise ValueError(f"init must be a FittedModel; got {type(init).__name__}")
    if init_draw is None:
        if init.n_chains == n_chains:
            per_chain = init.n_draws // n_chains
            return (
                init._fitted,
                None,
                [(k + 1) * per_chain - 1 for k in range(n_chains)],
            )
        return init._fitted, None, None
    draw = int(init_draw)
    if draw != init_draw or draw < 0:
        raise ValueError(f"init_draw must be a non-negative integer; got {init_draw!r}")
    return init._fitted, draw, None


def _emit_warnings(fitted: _native.Fitted, stacklevel: int) -> None:
    """Re-raise the core's fit-time warnings as `UserWarning`."""
    for message in fitted.warnings:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _resolve_chains(n_chains: Any) -> int:
    """Return `n_chains` as a positive integer."""
    chains = int(n_chains)
    if chains != n_chains or chains < 1:
        raise ValueError(f"n_chains must be a positive integer; got {n_chains!r}")
    return chains


def _resolve_threads(n_threads: Any) -> int:
    """Return `n_threads` as a positive integer."""
    threads = int(n_threads)
    if threads != n_threads or threads < 1:
        raise ValueError(f"n_threads must be a positive integer; got {n_threads!r}")
    return threads


def _rebuild(
    payload: str, seed: int, n_chains: int = 1, n_threads: int = 1
) -> FittedModel:
    """Reconstruct a `FittedModel` from its pickled state."""
    return FittedModel(
        _native.fitted_from_json(payload, n_threads), seed, n_chains, n_threads
    )


class FittedModel:
    """A fitted model: the kept draws and the queries over them.

    Returned by `Model.fit`; not constructed directly. Prediction methods
    take a design with the column count of the fit and return values on the
    caller's scale.

    Attributes
    ----------
    random_state : int
        The resolved seed of the fit.

    See Also
    --------
    Model : The configuration.
    """

    def __init__(
        self,
        fitted: _native.Fitted,
        seed: int,
        n_chains: int = 1,
        n_threads: int = 1,
    ) -> None:
        self._fitted = fitted
        self.random_state = seed
        self._n_chains = n_chains
        self._n_threads = n_threads

    @property
    def n_chains(self) -> int:
        """int: The number of chains the draws were pooled from."""
        return self._n_chains

    @property
    def n_threads(self) -> int:
        """int: The number of threads a prediction splits its rows over.

        Settable: a positive integer, in force from the next prediction.
        """
        return self._n_threads

    @n_threads.setter
    def n_threads(self, n_threads: int) -> None:
        threads = _resolve_threads(n_threads)
        self._fitted.set_threads(threads)
        self._n_threads = threads

    @property
    def model(self) -> str:
        """str: The observation model."""
        return str(self._fitted.model)

    @property
    def config(self) -> dict[str, Any]:
        """dict: The resolved configuration, the core's four groups.

        Every field is set: the outcome family under ``outcome``, the two
        ensembles under ``mean_params`` and ``variance_params``, and the
        sweep schedule under ``general_params``.
        """
        parsed: dict[str, Any] = json.loads(self._fitted.config)
        return parsed

    @property
    def n_draws(self) -> int:
        """int: The number of kept draws over every chain."""
        return int(self._fitted.n_draws)

    @property
    def in_sample_rmse(self) -> float:
        """float: The in-sample root mean squared error.

        The root mean squared error of the posterior mean at the training
        rows.
        """
        return float(self._fitted.in_sample_rmse)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Tuple of str: the fit-time warnings."""
        return tuple(self._fitted.warnings)

    def predict(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the posterior mean at each row of `X`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The posterior mean of f(x), or of P(y = 1 | x) under the probit
            family. Under the AFT family it is f(x) on the log-time scale,
            under the tobit and interval-censored families the uncensored
            f(x), and under the ordinal family the expected category.
        """
        return self._fitted.predict(_as_design(X))

    def predict_proba(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the posterior mean category probabilities at each row.

        The ordinal family only; the scikit-learn name, one column per
        category. Experimental, with the family.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_categories)
            Each row sums to one.

        Raises
        ------
        ThiessenError
            Under a family other than ordinal.
        RequiresFeatureError
            In a build without the core's ``experimental`` feature.
        """
        return self._fitted.predict_category_probabilities(_as_design(X))

    def predict_draws(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the quantity of `predict` for every kept draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            Draw-major.
        """
        return self._fitted.predict_draws(_as_design(X))

    def predict_latent(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the mean function for every kept draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            f(x), or the latent mean c + f(x) under the probit family.
        """
        return self._fitted.predict_latent(_as_design(X))

    def predict_variance(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the variance of y given f for every kept draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            sigma^2 under the Gaussian model, constant across rows; s^2(x)
            under the heteroscedastic model.

        Raises
        ------
        ThiessenError
            Under the probit family, whose latent variance is one.
        """
        return self._fitted.predict_variance(_as_design(X))

    def predict_quantiles(
        self, X: Any, probs: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Return posterior quantiles of the quantity of `predict`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        probs : sequence of float
            Probabilities in (0, 1).

        Returns
        -------
        numpy.ndarray of shape (n_samples, len(probs))
            Type 7 interpolation over the kept draws.

        Raises
        ------
        ThiessenError
            For an empty `probs` or a probability outside (0, 1).
        """
        return self._fitted.predict_quantiles(_as_design(X), [float(p) for p in probs])

    def credible_interval(self, X: Any, level: float = 0.95) -> npt.NDArray[np.float64]:
        """Return the central credible interval for the mean at `level`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        level : float, optional
            Interval level in (0, 1). Default 0.95.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Lower and upper ends. On the probability scale under the probit
            family.
        """
        return self._fitted.credible_interval(_as_design(X), float(level))

    def prediction_interval(
        self, X: Any, level: float = 0.95
    ) -> npt.NDArray[np.float64]:
        """Return the central predictive interval for a new observation.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        level : float, optional
            Interval level in (0, 1). Default 0.95.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Quantiles of the equal-weight mixture over kept draws of
            N(f_d(x), s_d^2(x)).

        Raises
        ------
        ThiessenError
            Under the probit family, which has no continuous predictive
            distribution.
        """
        return self._fitted.prediction_interval(_as_design(X), float(level))

    def log_likelihood(self, X: Any, y: Any) -> npt.NDArray[np.float64]:
        """Return the pointwise log-likelihood per draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        y : array_like
            The response, in the shape `Model.fit` took for the family.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            Draw-major.

        Raises
        ------
        ValueError
            For a response the fitted family does not take.
        """
        response = _as_response(y)
        _check_family(self.model, response)
        return _log_likelihood(self._fitted, _as_design(X), response)

    def sigma(self) -> npt.NDArray[np.float64]:
        """Return sigma per kept draw.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            Empty under the probit family, whose latent variance is one,
            and under the heteroscedastic model, where `predict_variance`
            gives s^2(x).
        """
        return self._fitted.sigma()

    def dfs(self) -> npt.NDArray[np.float64]:
        """Return the error degrees of freedom per kept draw.

        Experimental, with the Student-t family.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            Empty outside the Student-t family with a grid of degrees of
            freedom, where none is sampled.
        """
        return self._fitted.dfs()

    def cutpoints(self) -> npt.NDArray[np.float64]:
        """Return the interior cutpoints per kept draw.

        Experimental, with the ordinal family.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_categories - 2)
            Increasing within a draw, on the latent scale; the first
            cutpoint is fixed at zero and not returned. Empty outside the
            ordinal family and at two categories.
        """
        return self._fitted.cutpoints()

    def bandwidths(self) -> npt.NDArray[np.float64]:
        """Return the soft-membership bandwidth of each mean tessellation.

        Experimental, with `soft_membership`.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_tessellations)
            On the scaled covariate space. Empty under hard membership.
        """
        return self._fitted.bandwidths()

    def inclusion_weights(self) -> npt.NDArray[np.float64]:
        """Return the sampled inclusion weight of each covariate per draw.

        Experimental, with `dart_inclusion`. The prior weight the sampler
        drew, the quantity Linero (2018) reports; not
        `variable_inclusion_proportions`, which counts the usage the
        tessellations realised.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_features)
            Each row sums to one. Empty outside the DART prior.
        """
        return self._fitted.inclusion_weights()

    def concentrations(self) -> npt.NDArray[np.float64]:
        """Return the Dirichlet concentration theta per kept draw.

        Experimental, with `dart_inclusion`.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            Empty outside the DART prior.
        """
        return self._fitted.concentrations()

    def cell_counts(self) -> npt.NDArray[np.float64]:
        """Return the mean cells per mean tessellation, per kept draw.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            The mean over the m tessellations of each draw.
        """
        return self._fitted.cell_counts()

    def dimension_counts(self) -> npt.NDArray[np.float64]:
        """Return the mean active covariates per mean tessellation.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            The mean over the m tessellations of each draw.
        """
        return self._fitted.dimension_counts()

    def variable_inclusion_proportions(self) -> npt.NDArray[np.float64]:
        """Return the share of active dimensions falling on each covariate.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Sums to one (Chipman, George and McCulloch, 2010, s. 5.1).
        """
        return self._fitted.variable_inclusion_proportions()

    def to_inference_data(self, X: Any, y: Any) -> Any:
        """Return the fit as an arviz `DataTree`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The rows the observation dimension indexes, usually the training
            design.
        y : array_like
            The observed response, in the shape `Model.fit` took.

        Returns
        -------
        xarray.DataTree
            The `posterior` group carries `mu`, the mean function per draw,
            the per-draw cell and dimension counts, `sigma` under a model
            with a global sampled scale, and, where the model samples
            them, `df`, `cutpoint`, `bandwidth`, `inclusion_weight` and
            `concentration`; `posterior_predictive` and `log_likelihood`
            carry `y`; `observed_data` carries the response as `y`, or as
            `time` and `event` under the AFT family and `lower` and
            `upper` under the interval-censored family.

        Raises
        ------
        ImportError
            If arviz is not installed.
        ValueError
            If `X` and `y` disagree on the number of rows, or `y` is not
            in the fitted family's shape.

        Notes
        -----
        The chain dimension of every group holds the chains of the fit. The
        predictive replicates are drawn in numpy from the fit's resolved
        seed rather than by the core, each family under its own
        observation model.
        """
        response = _as_response(y)
        _check_family(self.model, response)
        return _to_inference_data(
            self._fitted,
            _as_design(X),
            response,
            self.random_state,
            self._n_chains,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the fitted model to `path`.

        Parameters
        ----------
        path : str or os.PathLike
            The destination. The format is the core's serde representation,
            which reloads bit-exact.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        with open(os.fspath(path), "w", encoding="utf-8") as handle:
            handle.write(self._fitted.to_json())

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        random_state: int = 0,
        n_chains: int = 1,
        n_threads: int = 1,
    ) -> FittedModel:
        """Read a fitted model from `path`.

        Parameters
        ----------
        path : str or os.PathLike
            A file written by `save`.
        random_state : int, default=0
            The seed to report on the loaded object, which the file does not
            carry.
        n_chains : int, default=1
            The number of chains the draws were pooled from, which the file
            does not carry.
        n_threads : int, default=1
            The number of threads a prediction splits its rows over, which
            the file does not carry.

        Returns
        -------
        FittedModel
            The loaded model.

        Raises
        ------
        OSError
            If the file cannot be read.
        ThiessenError
            If the contents are not a fitted model of this crate version.
        """
        with open(os.fspath(path), encoding="utf-8") as handle:
            payload = handle.read()
        threads = _resolve_threads(n_threads)
        return cls(
            _native.fitted_from_json(payload, threads), random_state, n_chains, threads
        )

    def __reduce__(self) -> tuple[Any, tuple[str, int, int, int]]:
        return _rebuild, (
            self._fitted.to_json(),
            self.random_state,
            self._n_chains,
            self._n_threads,
        )

    def __repr__(self) -> str:
        return (
            f"FittedModel(model={self.model!r}, n_chains={self.n_chains}, "
            f"n_draws={self.n_draws}, random_state={self.random_state})"
        )
