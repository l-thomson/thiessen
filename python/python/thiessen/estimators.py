"""scikit-learn estimators over the core.

`AddiVortesRegressor` fits the Gaussian model, with the heteroscedastic
extension when a variance ensemble is attached; `AddiVortesClassifier`
fits the binary probit model. Both meet the scikit-learn estimator
contract, so they compose with `Pipeline`, `GridSearchCV`,
`cross_val_score` and `sklearn.inspection`, and the parameter groups
implement ``get_params``/``set_params``, so grid search routes into them:
``{"mean_params__tessellations": [50, 200]}``.

Requires scikit-learn 1.6 or later, the `sklearn` extra.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _native
from ._arrays import _as_numeric
from ._config import _config_json
from ._convergence import _warn_convergence
from ._encoding import Encoding, columns_of, resolve_mask
from ._seed import SeedLike, _resolve_seed
from .families import Gaussian, Probit
from .model import _emit_warnings, _resolve_chains
from .params import MetricEntry, TermParams


def _cpu_count() -> int:
    """Return the number of CPUs the process may run on, affinity aware."""
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        return int(process_count() or 1)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0)) or 1
    return os.cpu_count() or 1


def _resolve_jobs(n_jobs: int | None) -> int:
    """Resolve `n_jobs` to a thread count under the joblib convention.

    `None` is one thread; a positive count is itself; a negative count is
    that many below the CPUs the process may run on plus one, so -1 is
    every one of them.
    """
    if n_jobs is None:
        return 1
    jobs = int(n_jobs)
    if jobs != n_jobs or jobs == 0:
        raise ValueError(f"n_jobs must be a non-zero integer or None; got {n_jobs!r}")
    if jobs > 0:
        return jobs
    return max(1, _cpu_count() + 1 + jobs)


__all__ = ["AddiVortesClassifier", "AddiVortesRegressor"]


class _BaseAddiVortes(BaseEstimator):
    """Shared configuration, fitting and prediction."""

    _fit_chains: int

    def __init__(
        self,
        *,
        mean_params: TermParams | None = None,
        variance_params: TermParams | None = None,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
        categorical_features: str | Sequence[Any] | None = None,
        n_chains: int = 4,
        n_jobs: int | None = None,
        random_state: SeedLike = None,
        warm_start: bool = False,
    ) -> None:
        self.mean_params = mean_params
        self.variance_params = variance_params
        self.burn_in = burn_in
        self.draws = draws
        self.thinning = thinning
        self.prior_only = prior_only
        self.categorical_features = categorical_features
        self.n_chains = n_chains
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        return tags

    def _user_metric(self) -> Sequence[MetricEntry] | None:
        """Return the metric declared over the input columns."""
        if self.mean_params is None or self.mean_params.geometry is None:
            return None
        return self.mean_params.geometry.metric

    def _config(self, core_metric: list[MetricEntry]) -> str:
        varies = any(entry != "euclidean" for entry in core_metric)
        return _config_json(
            self._outcome(),
            self.mean_params,
            self.variance_params,
            self.burn_in,
            self.draws,
            self.thinning,
            self.prior_only,
            core_metric=core_metric if varies else None,
        )

    def _encode_fit(self, x: Any) -> npt.NDArray[np.float64]:
        """Learn the categorical encoding and return the core's design."""
        columns = columns_of(x)
        mask = resolve_mask(self.categorical_features, x, len(columns))
        self._encoding = Encoding.fit(columns, mask, self._user_metric())
        return self._encoding.transform(columns)

    def _encode_predict(self, x: Any) -> npt.NDArray[np.float64]:
        columns = columns_of(x)
        if len(columns) != self.n_features_in_:
            raise ValueError(
                f"X has {len(columns)} features, but this "
                f"{type(self).__name__} is expecting {self.n_features_in_} "
                "features as input"
            )
        return self._encoding.transform(columns)

    def _validate_fit(self, x: Any, y: Any) -> tuple[npt.NDArray[np.float64], Any]:
        """Validate and encode, setting `n_features_in_`."""
        if self.categorical_features is None:
            x_checked, y_checked = validate_data(
                self, x, y, dtype=np.float64, ensure_min_samples=2
            )
            self._encoding = Encoding.fit(
                columns_of(x_checked),
                np.zeros(x_checked.shape[1], dtype=bool),
                self._user_metric(),
            )
            return x_checked, y_checked
        names = getattr(x, "columns", None)
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(columns_of(x))
        return self._encode_fit(x), np.asarray(y)

    def _validate_predict(self, x: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self)
        self._fitted.set_threads(_resolve_jobs(self.n_jobs))
        if self.categorical_features is None:
            checked = validate_data(self, x, reset=False, dtype=np.float64)
            return np.asarray(checked, dtype=np.float64)
        return self._encode_predict(x)

    def _fit_core(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        seed = _resolve_seed(self.random_state)
        chains = _resolve_chains(self.n_chains)
        threads = _resolve_jobs(self.n_jobs)
        previous = getattr(self, "_fitted", None) if self.warm_start else None
        draws = None
        if previous is not None:
            if getattr(self, "_fit_chains", chains) != chains:
                raise ValueError(
                    "warm_start continues each chain of the previous fit; "
                    f"n_chains changed from {self._fit_chains} to {chains}"
                )
            per_chain = previous.n_draws // chains
            draws = [(k + 1) * per_chain - 1 for k in range(chains)]
        self.random_state_ = seed
        design = np.ascontiguousarray(x, dtype=np.float64)
        self._fitted: _native.Fitted = _native.fit(
            self._config(self._encoding.core_metric),
            design,
            _as_numeric(y),
            seed,
            chains,
            threads,
            init=previous,
            init_draws=draws,
        )
        self._fit_chains = chains
        _emit_warnings(self._fitted, stacklevel=4)
        _warn_convergence(
            self._fitted, chains, design, stacklevel=4, warm=previous is not None
        )

    def _outcome(self) -> Gaussian | Probit:
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        fitted = state.pop("_fitted", None)
        if fitted is not None:
            state["_fitted_json"] = fitted.to_json()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payload = state.pop("_fitted_json", None)
        self.__dict__.update(state)
        if payload is not None:
            self._fitted = _native.fitted_from_json(payload, _resolve_jobs(self.n_jobs))


class AddiVortesRegressor(RegressorMixin, _BaseAddiVortes):
    """AddiVortes regression.

    Bayesian regression on a sum of Voronoi tessellations (Stone and Gosling,
    2025). The posterior mean of f(x) is the prediction.

    Parameters
    ----------
    outcome : Gaussian, optional
        The outcome family, from `thiessen.gaussian`. `None` is
        ``gaussian()``. Binary responses need `AddiVortesClassifier`.
    variance_params : TermParams, optional
        The ensemble describing the spread. `None`, the default, keeps the
        spread constant; a positive tessellation count selects the
        heteroscedastic model, so the residual variance varies with x. The
        ensembles share one covariate space, declared on `mean_params`.
    mean_params : TermParams, optional
        The ensemble describing the average. `None` takes the core's
        defaults: 200 tessellations, k=3, lambda_c=5, Euclidean geometry.
        Grid search routes into it: ``mean_params__tessellations``.
    burn_in : int, default=200
        Burn-in sweeps discarded.
    draws : int, default=1000
        Posterior draws kept.
    thinning : int, default=1
        Every `thinning`-th sweep after burn-in is kept.
    prior_only : bool, default=False
        Switch the likelihood off, so the chain draws from the prior.
    categorical_features : None, 'from_dtype' or array_like, default=None
        The categorical columns. `None` takes the input as numeric; for a
        column that needs encoding, either name it here or encode it yourself
        with ``OneHotEncoder(drop='first')`` in a `ColumnTransformer`.
        'from_dtype' reads the pandas categorical dtypes; an array of indices
        or a boolean mask names the columns. A named column becomes d - 1
        treatment-contrast indicators, the first level as reference, unless
        its entry in the geometry's ``metric`` is ``'categorical'``, in which
        case it passes as integer level codes.
    n_chains : int, default=4
        The number of chains to run. Each chain has its own seed, derived
        from the resolved seed in the core, and the draws of the chains are
        pooled. Two or more chains warn where R-hat exceeds 1.01 or an
        effective sample size falls below 400 (Vehtari and others, 2021),
        which needs arviz.
    n_jobs : int or None, default=None
        The number of threads, under the joblib convention: `None` is one,
        -1 every core. The chains are spread over at most this many
        threads, each chain on one thread with its own generator, so the
        draws do not depend on it; a prediction splits its rows over the
        same number, read again at each call.
    random_state : int, Generator, RandomState or None, default=None
        The seed. An integer passes through to the core unchanged, so
        `AddiVortesRegressor(random_state=1)` and `Model(random_state=1)`
        draw alike.
    warm_start : bool, default=False
        When True, a repeated `fit` continues each chain from its last
        kept draw instead of the cold start, under a seed resolved from
        `random_state` as usual; only the starting state changes, not the
        posterior. The data, the settings and `n_chains` must be those of
        the previous fit. Iterative fitting is not bit-identical to one
        longer fit, because the generator's state is not carried.
        Experimental: needs a core built with the ``experimental``
        feature.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen at fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen at fit, when the input carried them.
    random_state_ : int
        The resolved seed.

    See Also
    --------
    AddiVortesClassifier : The binary probit model.
    thiessen.TermParams : One ensemble's parameters.

    Notes
    -----
    Partial dependence and individual conditional expectation come from
    `sklearn.inspection`; this estimator implements nothing of its own.

    Examples
    --------
    >>> import numpy as np
    >>> from thiessen import TermParams
    >>> from thiessen.estimators import AddiVortesRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(size=(60, 2))
    >>> y = 3.0 * (X[:, 0] - 0.4) ** 2 + 0.5 * X[:, 1]
    >>> model = AddiVortesRegressor(mean_params=TermParams(tessellations=10),
    ...                             burn_in=20, draws=40, random_state=1)
    >>> model.fit(X, y).predict(X).shape
    (60,)

    The paper's cell-count prior is a group parameter:

    >>> paper = AddiVortesRegressor(mean_params=TermParams(lambda_c=25.0))
    """

    def __init__(
        self,
        *,
        outcome: Gaussian | None = None,
        mean_params: TermParams | None = None,
        variance_params: TermParams | None = None,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
        categorical_features: str | Sequence[Any] | None = None,
        n_chains: int = 4,
        n_jobs: int | None = None,
        random_state: SeedLike = None,
        warm_start: bool = False,
    ) -> None:
        super().__init__(
            mean_params=mean_params,
            variance_params=variance_params,
            burn_in=burn_in,
            draws=draws,
            thinning=thinning,
            prior_only=prior_only,
            categorical_features=categorical_features,
            n_chains=n_chains,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
        )
        self.outcome = outcome

    def _outcome(self) -> Gaussian | Probit:
        # The estimator holds no family list of its own: any family object
        # other than probit passes through for the core to validate.
        if self.outcome is None:
            return Gaussian()
        if isinstance(self.outcome, Probit):
            raise ValueError(
                "probit() is a classification family; use AddiVortesClassifier"
            )
        return self.outcome

    def fit(self, X: Any, y: Any) -> AddiVortesRegressor:
        """Fit the model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        y : array_like of shape (n_samples,)
            The response.

        Returns
        -------
        AddiVortesRegressor
            The fitted estimator.

        Raises
        ------
        ThiessenError
            For data the core rejects: missing or non-finite values, a
            constant response, a constant column, or fewer than two rows.
        """
        design, response = self._validate_fit(X, y)
        self._fit_core(design, response)
        return self

    @overload
    def predict(
        self, X: Any, return_std: Literal[False] = False
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def predict(
        self, X: Any, return_std: Literal[True]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

    def predict(
        self, X: Any, return_std: bool = False
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]:
        """Predict the mean function.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        return_std : bool, default=False
            Also return the posterior standard deviation of the mean function
            over the kept draws (`GaussianProcessRegressor` precedent).

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The posterior mean of f(x).
        y_std : ndarray of shape (n_samples,)
            Present when `return_std` is True.
        """
        design = self._validate_predict(X)
        if not return_std:
            return np.asarray(self._fitted.predict(design), dtype=np.float64)
        draws = self._fitted.predict_draws(design)
        return draws.mean(axis=0), draws.std(axis=0)

    def predict_interval(self, X: Any, level: float = 0.95) -> npt.NDArray[np.float64]:
        """Return the central posterior predictive interval.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        level : float, default=0.95
            Interval level in (0, 1).

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Lower and upper ends for a new observation.
        """
        design = self._validate_predict(X)
        return np.asarray(
            self._fitted.prediction_interval(design, float(level)), dtype=np.float64
        )


class AddiVortesClassifier(ClassifierMixin, _BaseAddiVortes):
    """AddiVortes binary classification.

    The probit model P(y = 1 | x) = Phi(c + f(x)) with Albert and Chib (1993)
    augmentation. Two classes only.

    Parameters
    ----------
    outcome : Probit, optional
        The outcome family, from `thiessen.probit`. `None` is ``probit()``,
        whose offset resolves to Phi^-1(ybar) at fit.
    variance_params : TermParams, optional
        Not available: the probit latent scale is fixed at 1 for
        identification, so the core rejects a positive tessellation count
        here.
    mean_params : TermParams, optional
        The ensemble describing the average. `None` takes the core's
        defaults: 200 tessellations, k=3, lambda_c=5, Euclidean geometry.
        Grid search routes into it: ``mean_params__tessellations``.
    burn_in : int, default=200
        Burn-in sweeps discarded.
    draws : int, default=1000
        Posterior draws kept.
    thinning : int, default=1
        Every `thinning`-th sweep after burn-in is kept.
    prior_only : bool, default=False
        Switch the likelihood off, so the chain draws from the prior.
    categorical_features : None, 'from_dtype' or array_like, default=None
        The categorical columns. `None` takes the input as numeric; for a
        column that needs encoding, either name it here or encode it yourself
        with ``OneHotEncoder(drop='first')`` in a `ColumnTransformer`.
        'from_dtype' reads the pandas categorical dtypes; an array of indices
        or a boolean mask names the columns. A named column becomes d - 1
        treatment-contrast indicators, the first level as reference, unless
        its entry in the geometry's ``metric`` is ``'categorical'``, in which
        case it passes as integer level codes.
    n_chains : int, default=4
        The number of chains to run. Each chain has its own seed, derived
        from the resolved seed in the core, and the draws of the chains are
        pooled. Two or more chains warn where R-hat exceeds 1.01 or an
        effective sample size falls below 400 (Vehtari and others, 2021),
        which needs arviz.
    n_jobs : int or None, default=None
        The number of threads, under the joblib convention: `None` is one,
        -1 every core. The chains are spread over at most this many
        threads, each chain on one thread with its own generator, so the
        draws do not depend on it; a prediction splits its rows over the
        same number, read again at each call.
    random_state : int, Generator, RandomState or None, default=None
        The seed. An integer passes through to the core unchanged, so
        `AddiVortesRegressor(random_state=1)` and `Model(random_state=1)`
        draw alike.
    warm_start : bool, default=False
        When True, a repeated `fit` continues each chain from its last
        kept draw instead of the cold start, under a seed resolved from
        `random_state` as usual; only the starting state changes, not the
        posterior. The data, the settings and `n_chains` must be those of
        the previous fit. Iterative fitting is not bit-identical to one
        longer fit, because the generator's state is not carried.
        Experimental: needs a core built with the ``experimental``
        feature.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels.
    n_features_in_ : int
        Number of features seen at fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen at fit, when the input carried them.
    random_state_ : int
        The resolved seed.

    See Also
    --------
    AddiVortesRegressor : The Gaussian and heteroscedastic models.
    thiessen.TermParams : One ensemble's parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from thiessen import TermParams
    >>> from thiessen.estimators import AddiVortesClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(size=(60, 2))
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> model = AddiVortesClassifier(mean_params=TermParams(tessellations=10),
    ...                              burn_in=20, draws=40, random_state=1)
    >>> model.fit(X, y).predict_proba(X).shape
    (60, 2)
    """

    def __init__(
        self,
        *,
        outcome: Probit | None = None,
        mean_params: TermParams | None = None,
        variance_params: TermParams | None = None,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
        categorical_features: str | Sequence[Any] | None = None,
        n_chains: int = 4,
        n_jobs: int | None = None,
        random_state: SeedLike = None,
        warm_start: bool = False,
    ) -> None:
        super().__init__(
            mean_params=mean_params,
            variance_params=variance_params,
            burn_in=burn_in,
            draws=draws,
            thinning=thinning,
            prior_only=prior_only,
            categorical_features=categorical_features,
            n_chains=n_chains,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
        )
        self.outcome = outcome

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _outcome(self) -> Gaussian | Probit:
        if self.outcome is None:
            return Probit()
        if not isinstance(self.outcome, Probit):
            raise ValueError(
                "AddiVortesClassifier fits the probit model; "
                "use AddiVortesRegressor for other families"
            )
        return self.outcome

    def fit(self, X: Any, y: Any) -> AddiVortesClassifier:
        """Fit the model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        y : array_like of shape (n_samples,)
            The class labels, two distinct values.

        Returns
        -------
        AddiVortesClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            If `y` does not hold exactly two classes.
        """
        design, labels = self._validate_fit(X, y)
        target = type_of_target(labels, input_name="y", raise_unknown=True)
        if target != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the "
                f"target is {target}."
            )
        self.classes_ = unique_labels(labels)
        if self.classes_.size < 2:
            raise ValueError(
                "AddiVortesClassifier needs two classes; y holds "
                f"{self.classes_.size} class."
            )
        indicator = (np.asarray(labels) == self.classes_[1]).astype(np.float64)
        self._fit_core(design, indicator)
        return self

    def predict_proba(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the posterior mean class probabilities.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns in the order of `classes_`.
        """
        design = self._validate_predict(X)
        positive = np.asarray(self._fitted.predict(design), dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> npt.NDArray[Any]:
        """Return the class of highest posterior mean probability.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels drawn from `classes_`.
        """
        indices = self.predict_proba(X).argmax(axis=1)
        return np.asarray(self.classes_)[indices]
