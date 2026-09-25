"""The parameter groups of a configuration.

`TermParams` describes one ensemble; ``mean_params`` and
``variance_params`` each take one. Geometry, structure and cell nest
inside it the way the core's configuration nests them, so every name here
is a name in the stored configuration. The objects implement
``get_params`` and ``set_params``, so `sklearn.model_selection.GridSearchCV`
routes into them with keys of the form ``mean_params__tessellations``.

The component options behind the core's ``experimental`` feature, the
distance metrics beyond the published three, `soft_membership`,
`weighted_inclusion`, `dart_inclusion` and the linear cell basis, follow
the outcome-family idiom: a constructor returns a classed value carrying
its parameters and serialising as the configuration's tagged form. Each
exists in every build; a configuration naming one is rejected with
`RequiresFeatureError` unless the extension was built with
``--features experimental`` (``docs/experimental.md`` in the repository).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np

from ._params import Params, Tagged, _dense

__all__ = [
    "CellParams",
    "DartInclusion",
    "GeometryParams",
    "SoftMembership",
    "StructureParams",
    "TermParams",
    "WeightedInclusion",
    "dart_inclusion",
    "soft_membership",
    "weighted_inclusion",
]

MetricEntry = Union[str, "dict[str, dict[str, Any]]"]

#: The metric entries whose fields all have defaults, accepted as bare
#: strings and tagged for the core, which reads a struct variant as a map.
_TAGGED_ENTRIES = ("manhattan", "cosine")


def _metric_entry(entry: MetricEntry) -> MetricEntry:
    """Return `entry` in the core's tagged form."""
    if isinstance(entry, str) and entry in _TAGGED_ENTRIES:
        return {entry: {}}
    return entry


class Option(Tagged):
    """An experimental component option, the value of a configuration field.

    Subclasses name the tag the field stores the option under and the
    constructor that returns them.
    """

    _constructor: str

    def _display_name(self) -> str:
        return self._constructor


class SoftMembership(Option):
    """Kernel-weighted membership, returned by `soft_membership`. Experimental.

    Parameters
    ----------
    rate : float, default=10.0
        Rate of the exponential prior on the bandwidth, on the scaled
        covariate space.
    """

    _tag = "soft"
    _constructor = "soft_membership"

    def __init__(self, rate: float = 10.0) -> None:
        self.rate = rate


class WeightedInclusion(Option):
    """Fixed inclusion weights, returned by `weighted_inclusion`. Experimental.

    Parameters
    ----------
    weights : sequence of float
        One non-negative weight per covariate column, in column order.
    """

    _tag = "weighted"
    _constructor = "weighted_inclusion"

    def __init__(self, weights: Sequence[float]) -> None:
        self.weights = weights


class DartInclusion(Option):
    """The DART inclusion prior, returned by `dart_inclusion`. Experimental.

    Parameters
    ----------
    a : float, default=0.5
        Shape a of the Beta(a, b) prior on the concentration ratio.
    b : float, default=1.0
        Shape b of the Beta(a, b) prior.
    rho : float, optional
        Scale rho of the concentration. `None` resolves to the covariate
        count at fit.
    """

    _tag = "dart"
    _constructor = "dart_inclusion"

    def __init__(
        self, a: float = 0.5, b: float = 1.0, rho: float | None = None
    ) -> None:
        self.a = a
        self.b = b
        self.rho = rho


def soft_membership(rate: float = 10.0) -> SoftMembership:
    """Return kernel-weighted membership. Experimental.

    The softening of the tree split of Linero and Yang (2018) carried to
    the Voronoi assignment: observation i takes weight proportional to
    exp(-d^2 / (2 tau^2)) in each cell, normalised over the tessellation's
    centres, with tau a per-tessellation bandwidth under an exponential
    prior and updated by a Metropolis step. Constant cell basis and
    constant spread only. The bandwidth draws are `FittedModel.bandwidths`.

    Parameters
    ----------
    rate : float, default=10.0
        Rate of the exponential prior on the bandwidth, on the scaled
        covariate space, so the prior mean bandwidth is a tenth of a
        column's range.

    Returns
    -------
    SoftMembership
        The option, for ``GeometryParams(membership=)``.

    Notes
    -----
    Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles
    that adapt to smoothness and sparsity. *Journal of the Royal
    Statistical Society: Series B*, 80(5), 1087-1110.

    Examples
    --------
    >>> from thiessen import soft_membership
    >>> soft_membership(rate=5.0)
    soft_membership(rate=5.0)
    """
    return SoftMembership(rate=rate)


def weighted_inclusion(weights: Sequence[float]) -> WeightedInclusion:
    """Return fixed inclusion weights. Experimental.

    The covariate of a proposed dimension is drawn in proportion to its
    weight rather than uniformly, the fixed-weight case of the split
    probabilities of Linero (2018); the weights are normalised at fit.

    Parameters
    ----------
    weights : sequence of float
        One non-negative weight per covariate column, in column order,
        not all zero.

    Returns
    -------
    WeightedInclusion
        The option, for ``StructureParams(inclusion=)``.

    Examples
    --------
    >>> from thiessen import weighted_inclusion
    >>> weighted_inclusion([2.0, 1.0])
    weighted_inclusion(weights=[2.0, 1.0])
    """
    return WeightedInclusion(weights=weights)


def dart_inclusion(
    a: float = 0.5, b: float = 1.0, rho: float | None = None
) -> DartInclusion:
    """Return the DART inclusion prior. Experimental.

    The Dirichlet prior on the inclusion weights of Linero (2018): the
    weights are drawn each sweep from a Dirichlet whose concentration
    theta / rho carries a Beta(a, b) prior on theta / (theta + rho), so
    covariates the tessellations do not use lose weight. The sampled
    weights and concentration are `FittedModel.inclusion_weights` and
    `FittedModel.concentrations`.

    This is Linero's (2018) prior carried over from trees. It is not the
    Dirichlet AddiVortes of Stone and Gosling (in preparation), whose subset
    prior draws covariates sequentially without replacement and whose
    weights have an exact Gibbs update by latent augmentation.

    Parameters
    ----------
    a : float, default=0.5
        Shape a of the Beta(a, b) prior.
    b : float, default=1.0
        Shape b of the Beta(a, b) prior.
    rho : float, optional
        Scale rho of the concentration. `None` resolves to the covariate
        count at fit.

    Returns
    -------
    DartInclusion
        The option, for ``StructureParams(inclusion=)``.

    Notes
    -----
    Linero, A. R. (2018). Bayesian regression trees for high-dimensional
    prediction and variable selection. *Journal of the American
    Statistical Association*, 113(522), 626-636.

    Examples
    --------
    >>> from thiessen import dart_inclusion
    >>> dart_inclusion()
    dart_inclusion()
    """
    return DartInclusion(a=a, b=b, rho=rho)


class GeometryParams(Params):
    """The covariate space of one ensemble.

    Parameters
    ----------
    metric : sequence, optional
        The metric of each covariate column, one entry per column in
        column order: ``'euclidean'``, ``'categorical'``, or
        ``{'spherical': {'sphere': k}}`` with `k` the sphere label. `None`
        is Euclidean on every column. Non-Euclidean columns are not
        scaled. The experimental entries, in the form the core stores:
        ``{'minkowski': {'p': 3.0}}``, ``'manhattan'``, ``'cosine'``,
        ``{'gower': {'kind': 'numeric'}}`` or ``'categorical'`` for its
        kind, and ``'mahalanobis'`` with `precision`; the Minkowski,
        Manhattan, cosine and Gower entries take a ``'group'`` label
        (default 0), so the columns sharing a label form one composite
        distance.
    sigma_c : float, default=0.8
        Centre-coordinate prior and proposal standard deviation sigma_c
        in the scaled space.
    membership : str or SoftMembership, optional
        How an observation belongs to a tessellation's cells:
        ``'hard'``, the published rule, or `soft_membership`. `None` is
        ``'hard'``.
    precision : array_like of shape (n_features, n_features), optional
        The precision matrix of the Mahalanobis metric, over the columns
        of the encoded design, required exactly when an entry of `metric`
        is ``'mahalanobis'``; checked at fit to be symmetric and positive
        definite. Experimental, as the metric it serves.
    """

    def __init__(
        self,
        metric: Sequence[MetricEntry] | None = None,
        sigma_c: float = 0.8,
        membership: str | SoftMembership | None = None,
        precision: Any = None,
    ) -> None:
        self.metric = metric
        self.sigma_c = sigma_c
        self.membership = membership
        self.precision = precision

    def _core(self) -> dict[str, Any]:
        membership: Any = self.membership
        if isinstance(membership, SoftMembership):
            membership = membership._core()
        elif membership is not None and not isinstance(membership, str):
            raise TypeError(
                "membership must be 'hard', soft_membership(...) or None; "
                f"got {membership!r}"
            )
        precision = None
        if self.precision is not None:
            # The core takes the matrix row-major as one vector.
            precision = np.asarray(self.precision, dtype=np.float64).reshape(-1)
        metric = None
        if self.metric is not None:
            metric = [_metric_entry(entry) for entry in self.metric]
        return _dense(
            {
                "metric": metric,
                "sigma_c": self.sigma_c,
                "membership": membership,
                "precision": precision,
            }
        )


class StructureParams(Params):
    """The covariate-inclusion prior of one ensemble.

    Parameters
    ----------
    omega : float, optional
        Dimension-count prior parameter omega; omega / p is the prior
        probability of including a covariate. `None` resolves to
        min(3, p) at fit. Must satisfy 0 < omega <= p.
    inclusion : str, WeightedInclusion or DartInclusion, optional
        The prior weight of each covariate: ``'uniform'``, the published
        prior, `weighted_inclusion` or `dart_inclusion`. `None` is
        ``'uniform'``.
    """

    def __init__(
        self,
        omega: float | None = None,
        inclusion: str | WeightedInclusion | DartInclusion | None = None,
    ) -> None:
        self.omega = omega
        self.inclusion = inclusion

    def _core(self) -> dict[str, Any]:
        inclusion: Any = self.inclusion
        if isinstance(inclusion, (WeightedInclusion, DartInclusion)):
            inclusion = inclusion._core()
        elif inclusion is not None and not isinstance(inclusion, str):
            raise TypeError(
                "inclusion must be 'uniform', weighted_inclusion(...), "
                f"dart_inclusion(...) or None; got {inclusion!r}"
            )
        return _dense({"omega": self.omega, "inclusion": inclusion})


class CellParams(Params):
    """The within-cell response surface of one ensemble.

    Parameters
    ----------
    basis : str, optional
        The value a cell holds: ``'constant'``, one value per cell, the
        published basis; or ``'linear'``, a value that tilts across the
        cell, mu + beta' (x_A - c) over the active covariates centred at
        the cell's centre, with the slopes under the cell-value prior. The
        linear basis is experimental, needs every column min-max scaled,
        and applies to the mean ensemble only. `None` is ``'constant'``.

    Examples
    --------
    >>> from thiessen import CellParams
    >>> CellParams(basis="linear")
    CellParams(basis='linear')
    """

    def __init__(self, basis: str | None = None) -> None:
        self.basis = basis

    def _core(self) -> dict[str, Any]:
        return _dense({"basis": self.basis})


class TermParams(Params):
    """One ensemble of tessellations: its size, priors and covariate space.

    Parameters
    ----------
    tessellations : int, optional
        Number of tessellations in the ensemble. `None` resolves at fit
        to 200 as ``mean_params`` and to 0 as ``variance_params``; a
        positive count as ``variance_params`` selects the heteroscedastic
        model (the paper's count is 40).
    k : float, default=3.0
        Cell-value prior spread k: sigma_mu = w / (k sqrt(m)) with the
        half-width w the outcome family owns (Chipman, George and
        McCulloch, 2010, s. 4). The variance ensemble's inverse-gamma
        cells do not use it.
    lambda_c : float, default=5.0
        Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). The
        default follows AddiVortes >= 0.6.8; Stone and Gosling (2025),
        s. 2.3, report 25.
    geometry : GeometryParams, optional
        The covariate space. `None` takes the defaults. The ensembles
        share one covariate space: set it on ``mean_params`` and it
        applies to ``variance_params`` as well.
    structure : StructureParams, optional
        The covariate-inclusion prior. `None` takes the defaults. Shared
        between the ensembles like `geometry`.
    cell : CellParams, optional
        The within-cell response surface. `None` takes the default, the
        constant basis.

    Examples
    --------
    >>> from thiessen import TermParams
    >>> TermParams(tessellations=200, k=3.0)
    TermParams(tessellations=200)
    """

    def __init__(
        self,
        tessellations: int | None = None,
        k: float = 3.0,
        lambda_c: float = 5.0,
        geometry: GeometryParams | None = None,
        structure: StructureParams | None = None,
        cell: CellParams | None = None,
    ) -> None:
        self.tessellations = tessellations
        self.k = k
        self.lambda_c = lambda_c
        self.geometry = geometry
        self.structure = structure
        self.cell = cell

    def _core(self) -> dict[str, Any]:
        group = _dense(
            {
                "tessellations": self.tessellations,
                "k": self.k,
                "lambda_c": self.lambda_c,
            }
        )
        for name, value, kind in (
            ("geometry", self.geometry, GeometryParams),
            ("structure", self.structure, StructureParams),
            ("cell", self.cell, CellParams),
        ):
            if value is None:
                continue
            if not isinstance(value, kind):
                raise TypeError(
                    f"{name} must be {kind.__name__}(...) or None; got {value!r}"
                )
            nested = value._core()
            if nested:
                group[name] = nested
        return group
