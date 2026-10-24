"""The response shapes the package takes and the outcome family each selects.

The shapes are the ones the Python ecosystem already uses, and the rule is
``glm``'s: the response selects the family where none is named, a named
family is checked against the response, and nothing is coerced. A numeric
array selects the Gaussian family, a boolean array or a two-category
``Categorical`` the probit family, an ordered ``Categorical`` the ordinal
family, a structured survival array (the scikit-survival layout) the AFT
family, and a two-column array of bounds the interval-censored family.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native
from ._arrays import _as_numeric
from ._params import Outcome
from .families import Aft, Gaussian, IntervalCensored, Ordinal, Probit

__all__ = [
    "Response",
    "_as_response",
    "_check_family",
    "_family_name",
    "_fit",
    "_log_likelihood",
    "_new_sampler",
    "_resolve_outcome",
    "_sampler_family",
    "_set_response",
]

Float = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]


def _empty() -> Float:
    return np.empty(0, dtype=np.float64)


def _no_flags() -> Bool:
    return np.empty(0, dtype=np.bool_)


@dataclass(frozen=True)
class Response:
    """A parsed response: the core entry point it reaches and its columns.

    Attributes
    ----------
    kind : str
        The core entry point: ``'plain'``, ``'aft'`` or
        ``'interval_censored'``.
    shape : str
        The response shape: ``'numeric'``, ``'binary'``, ``'ordered'``,
        ``'right'`` or ``'interval'``.
    y : numpy.ndarray
        The numeric response of a plain kind; the category codes of a
        ``Categorical``.
    times, events : numpy.ndarray
        The columns of a right-censored response.
    lower, upper : numpy.ndarray
        The columns of an interval-censored response.
    categories : tuple or None
        The categories of a ``Categorical`` response, in order.
    """

    kind: str
    shape: str
    y: Float = field(default_factory=_empty)
    times: Float = field(default_factory=_empty)
    events: Bool = field(default_factory=_no_flags)
    lower: Float = field(default_factory=_empty)
    upper: Float = field(default_factory=_empty)
    categories: tuple[Any, ...] | None = None

    @property
    def family(self) -> str:
        """str: The outcome family the shape selects."""
        return SHAPE_FAMILY[self.shape]

    @property
    def n(self) -> int:
        """int: The number of observations."""
        if self.kind == "aft":
            return len(self.times)
        if self.kind == "interval_censored":
            return len(self.lower)
        return len(self.y)


#: The outcome family each response shape selects when none is named.
SHAPE_FAMILY = {
    "numeric": "gaussian",
    "binary": "probit",
    "ordered": "ordinal",
    "right": "aft",
    "interval": "interval_censored",
}

#: The response shapes each outcome family accepts when named: the families
#: over a real line take a numeric array, the probit family a boolean array
#: or the numbers 0 and 1, the ordinal family an ordered `Categorical` or
#: integer codes, and each censored family the one shape that selects it.
FAMILY_SHAPES = {
    "gaussian": ("numeric",),
    "tobit": ("numeric",),
    "student_t": ("numeric",),
    "laplace": ("numeric",),
    "probit": ("binary", "numeric"),
    "ordinal": ("ordered", "numeric"),
    "aft": ("right",),
    "interval_censored": ("interval",),
}

SHAPE_LABEL = {
    "numeric": "a numeric array",
    "binary": "a boolean or two-category response",
    "ordered": "an ordered Categorical",
    "right": "a structured survival array",
    "interval": "a two-column array of bounds",
}


def _categorical(y: Any) -> Any:
    """Return the pandas categorical behind `y`, or `None`.

    A `Series` of category dtype answers through its ``cat`` accessor and
    a `Categorical` directly; both carry ``codes``, ``categories`` and
    ``ordered``.
    """
    if str(getattr(y, "dtype", "")) != "category":
        return None
    return getattr(y, "cat", y)


def _categorical_response(categorical: Any) -> Response:
    """Parse a pandas categorical: ordered as ordinal codes, two as labels.

    The codes are 0 to K - 1 in category order, the scikit-learn
    ``OrdinalEncoder`` convention.
    """
    codes = np.asarray(categorical.codes)
    if np.any(codes < 0):
        raise ValueError("y must not contain missing values")
    categories = tuple(categorical.categories)
    values = np.ascontiguousarray(codes, dtype=np.float64)
    if categorical.ordered:
        return Response("plain", "ordered", y=values, categories=categories)
    if len(categories) != 2:
        raise ValueError(
            "a Categorical response must have two categories, or be ordered; "
            f"got {len(categories)} unordered categories"
        )
    return Response("plain", "binary", y=values, categories=categories)


def _survival_response(y: Any) -> Response:
    """Parse a structured survival array.

    The layout of ``sksurv.util.Surv.from_arrays``: two fields, the first
    a boolean event indicator and the second the event or censoring time.
    """
    array = np.asarray(y)
    names = list(array.dtype.names or ())
    if (
        len(names) != 2
        or array.dtype[names[0]].kind != "b"
        or array.dtype[names[1]].kind not in "iuf"
    ):
        raise ValueError(
            "a structured survival array must have two fields, a boolean "
            f"event indicator then a numeric time; got fields {tuple(names)}"
        )
    events = np.ascontiguousarray(array[names[0]], dtype=np.bool_)
    times = np.ascontiguousarray(array[names[1]], dtype=np.float64)
    return Response("aft", "right", times=times, events=events)


def _interval_response(bounds: npt.NDArray[Any]) -> Response:
    """Parse an `(n, 2)` array of lower and upper bounds.

    An infinite bound is one-sided censoring and an equal pair an exact
    value, as the core reads them.
    """
    array = np.ascontiguousarray(bounds, dtype=np.float64)
    return Response(
        "interval_censored",
        "interval",
        lower=np.ascontiguousarray(array[:, 0]),
        upper=np.ascontiguousarray(array[:, 1]),
    )


def _as_response(y: Any) -> Response:
    """Parse `y` into the shape the core takes.

    Parameters
    ----------
    y : array_like
        The response as the caller gave it.

    Returns
    -------
    Response
        The parsed response.

    Raises
    ------
    ValueError
        For a shape no family takes, or a missing value.
    """
    categorical = _categorical(y)
    if categorical is not None:
        return _categorical_response(categorical)
    if getattr(getattr(y, "dtype", None), "names", None):
        return _survival_response(y)
    array = np.asarray(y)
    if array.dtype.kind == "b":
        return Response(
            "plain", "binary", y=np.ascontiguousarray(array, dtype=np.float64)
        )
    if array.ndim == 2 and array.shape[1] == 2:
        return _interval_response(array)
    return Response("plain", "numeric", y=_as_numeric(array))


def _family_name(outcome: Outcome) -> str:
    """Return the stored name of `outcome`'s family."""
    return outcome._tag


def _model_family(model: str) -> str:
    """Return the family of a fitted model's name."""
    return "gaussian" if model == "heteroscedastic" else model


def _check_family(family: str, response: Response) -> None:
    """Refuse a family the response does not fit.

    Parameters
    ----------
    family : str
        The stored name of the named family, or of a fitted model.
    response : Response
        The parsed response.

    Raises
    ------
    ValueError
        Naming the shape, the family it selects and the family named.
    """
    family = _model_family(family)
    if response.shape in FAMILY_SHAPES.get(family, ("numeric",)):
        return
    raise ValueError(
        f"The response is {SHAPE_LABEL[response.shape]}, which selects the "
        f"{response.family} family, but outcome names the {family} family."
    )


def _family_of(response: Response) -> Outcome:
    """Return the family the response selects, at its defaults."""
    family = response.family
    if family == "probit":
        return Probit()
    if family == "ordinal":
        return Ordinal(categories=len(response.categories or ()))
    if family == "aft":
        return Aft()
    if family == "interval_censored":
        return IntervalCensored()
    return Gaussian()


def _resolve_outcome(outcome: Outcome | None, response: Response) -> Outcome:
    """Return the outcome family resolved against the response.

    `None` takes the family the response selects; a named family is
    checked against the response and never coerced. The ordinal family
    takes its category count from an ordered `Categorical` where the
    constructor left it unset.

    Parameters
    ----------
    outcome : Outcome or None
        The family as configured.
    response : Response
        The parsed response.

    Returns
    -------
    Outcome
        A named family.

    Raises
    ------
    ValueError
        For a family the response does not fit, an ordinal category count
        disagreeing with the categories, or an ordinal family over integer
        codes with no category count.
    """
    if outcome is None:
        return _family_of(response)
    _check_family(_family_name(outcome), response)
    if not isinstance(outcome, Ordinal):
        return outcome
    if response.categories is None:
        if outcome.categories is None:
            raise ValueError(
                "ordinal() needs categories over integer codes; "
                "an ordered Categorical response carries its own"
            )
        return outcome
    count = len(response.categories)
    if outcome.categories is None:
        return type(outcome)(**{**outcome.get_params(deep=False), "categories": count})
    if outcome.categories != count:
        raise ValueError(
            f"outcome names {outcome.categories} categories but the response "
            f"has {count}"
        )
    return outcome


def _fit(
    config: str,
    design: Float,
    response: Response,
    seed: int,
    n_chains: int,
    n_threads: int,
    init: _native.Fitted | None = None,
    init_draw: int | None = None,
) -> _native.Fitted:
    """Fit through the core entry point the response reaches.

    The chains start from the cold start, or from draws of `init` by the
    core's rule (`thiessen::Init::From`).
    """
    if response.kind == "aft":
        return _native.fit_aft(
            config,
            design,
            response.times,
            response.events,
            seed,
            n_chains,
            n_threads,
            init=init,
            init_draw=init_draw,
        )
    if response.kind == "interval_censored":
        return _native.fit_interval_censored(
            config,
            design,
            response.lower,
            response.upper,
            seed,
            n_chains,
            n_threads,
            init=init,
            init_draw=init_draw,
        )
    return _native.fit(
        config,
        design,
        response.y,
        seed,
        n_chains,
        n_threads,
        init=init,
        init_draw=init_draw,
    )


def _new_sampler(
    config: str, design: Float, response: Response, seed: int
) -> _native.Sampler:
    """Construct the core's sampler over a parsed response."""
    if response.kind == "aft":
        return _native.Sampler.aft(
            config, design, response.times, response.events, seed
        )
    if response.kind == "interval_censored":
        return _native.Sampler.interval_censored(
            config, design, response.lower, response.upper, seed
        )
    return _native.Sampler(config, design, response.y, seed)


def _set_response(sampler: _native.Sampler, response: Response) -> None:
    """Replace a sampler's response with a parsed one."""
    if response.kind == "aft":
        sampler.set_aft_response(response.times, response.events)
    elif response.kind == "interval_censored":
        sampler.set_interval_censored_response(response.lower, response.upper)
    else:
        sampler.set_response(response.y)


def _log_likelihood(fitted: _native.Fitted, design: Float, response: Response) -> Float:
    """Return the pointwise log-likelihood of a parsed response, draw-major."""
    if response.kind == "aft":
        return fitted.log_likelihood_survival(design, response.times, response.events)
    if response.kind == "interval_censored":
        return fitted.log_likelihood_interval_censored(
            design, response.lower, response.upper
        )
    return fitted.log_likelihood(design, response.y)


def _sampler_family(sampler: _native.Sampler) -> str:
    """Return the stored family name of a live sampler's configuration."""
    outcome: dict[str, Any] = json.loads(sampler.config)["outcome"]
    return next(iter(outcome))
