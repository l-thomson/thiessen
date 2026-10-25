"""XAddiVortes settings: the core's `XAddiVortesParams` as JSON."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _native

#: The XAddiVortes settings in the core's order.
FAST_SETTINGS: tuple[str, ...] = (
    "rounds",
    "max_columns",
    "centres",
    "value_scale",
    "replace_share",
)

#: Each setting's type and range in the arguments of
#: `sklearn.utils.check_scalar`: target type, `min_val`, `max_val` and
#: `include_boundaries`. The ranges are those of the core's
#: `XAddiVortesParams::validate`.
FAST_RANGES: dict[str, tuple[type, float | None, float | None, str]] = {
    "rounds": (numbers.Integral, 0, None, "left"),
    "max_columns": (numbers.Integral, 1, None, "left"),
    "centres": (numbers.Integral, 1, 65_535, "both"),
    "value_scale": (numbers.Real, 0.0, 1.0, "right"),
    "replace_share": (numbers.Real, 0.0, 1.0, "both"),
}


def fast_defaults() -> dict[str, Any]:
    """Return the core's default XAddiVortes settings.

    Returns
    -------
    dict
        Setting name to default value, as the core defines them.
    """
    defaults: dict[str, Any] = json.loads(_native.xaddivortes_defaults())
    return defaults


def _check_names(settings: Mapping[str, Any]) -> None:
    """Raise `ValueError` for a name that is not an XAddiVortes setting."""
    unknown = sorted(set(settings) - set(FAST_SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown XAddiVortes setting(s) {unknown}; valid: {list(FAST_SETTINGS)}"
        )


def _fast_json(settings: Mapping[str, Any] | None) -> str:
    """Serialise the settings given; the core fills the rest with its defaults.

    Unknown names and non-finite numbers are rejected here; the types and
    ranges are checked by the core, whose errors name the setting. NumPy
    scalars pass as the Python scalars they hold.

    Raises
    ------
    ValueError
        For an unknown name or a non-finite number.
    """
    chosen = dict(settings or {})
    _check_names(chosen)
    plain = {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in chosen.items()
    }
    for name, value in plain.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
    return json.dumps(plain)
