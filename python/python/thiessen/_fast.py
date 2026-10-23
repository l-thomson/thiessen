"""XAddiVortes settings: the core's `XAddiVortesParams` as JSON."""

from __future__ import annotations

import json
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _native

#: The XAddiVortes settings in the core's order.
FAST_SETTINGS: tuple[str, ...] = (
    "budget",
    "max_columns",
    "centres",
    "value_scale",
    "main_effect_share",
    "replace_share",
    "max_rounds",
    "pair_screen",
    "column_bins",
    "response_bins",
    "score_floor",
    "tries",
    "rescore",
    "refit_passes",
    "update_sigma",
    "worth",
    "accept_if_better",
)

_FLOATS = frozenset(
    {"budget", "value_scale", "main_effect_share", "replace_share", "score_floor"}
)
_FLAGS = frozenset({"rescore", "update_sigma", "accept_if_better"})


def fast_defaults() -> dict[str, Any]:
    """Return the core's default XAddiVortes settings.

    Returns
    -------
    dict
        Setting name to default value, as the core defines them.
    """
    defaults: dict[str, Any] = json.loads(_native.xaddivortes_defaults())
    return defaults


def _coerce(name: str, value: Any) -> Any:
    """Return `value` as the JSON type of setting `name`.

    NumPy scalars are accepted, so a search over `numpy` grids reaches the
    core; a non-integral count or a non-boolean flag is rejected.
    """
    if name in _FLOATS:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a number; got {value!r}")
        return float(value)
    if name in _FLAGS:
        if not isinstance(value, (bool, np.bool_, numbers.Integral)) or value not in (
            0,
            1,
        ):
            raise ValueError(f"{name} must be a bool; got {value!r}")
        return bool(value)
    if name == "worth":
        return str(value)
    whole = isinstance(value, numbers.Integral) or (
        isinstance(value, numbers.Real) and float(value).is_integer()
    )
    if isinstance(value, (bool, np.bool_)) or not whole:
        raise ValueError(f"{name} must be a whole number; got {value!r}")
    return int(value)


def _fast_json(settings: Mapping[str, Any] | None) -> str:
    """Serialise the settings given; the core fills the rest with its defaults.

    Unknown names are rejected here with the valid names, and out-of-range
    values by the core.
    """
    chosen = {k: v for k, v in (settings or {}).items() if v is not None}
    unknown = sorted(set(chosen) - set(FAST_SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown XAddiVortes setting(s) {unknown}; valid: {list(FAST_SETTINGS)}"
        )
    return json.dumps({name: _coerce(name, value) for name, value in chosen.items()})
