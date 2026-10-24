"""XAddiVortes settings: the core's `XAddiVortesParams` as JSON."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from typing import Any, Literal

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

#: How a round ranks the tessellations it may replace.
Worth = Literal["gain", "squared_values"]

#: The values of `Worth`.
WORTHS: tuple[str, ...] = ("gain", "squared_values")

#: The settings that are flags.
FAST_FLAGS: tuple[str, ...] = ("rescore", "update_sigma", "accept_if_better")

#: Each numeric setting's type and range in the arguments of
#: `sklearn.utils.check_scalar`: target type, `min_val`, `max_val` and
#: `include_boundaries`. The ranges are those of the core's
#: `XAddiVortesParams::validate`; an infinite `max_val` excludes infinity.
FAST_RANGES: dict[str, tuple[type, float | None, float | None, str]] = {
    "budget": (numbers.Real, 0.0, math.inf, "neither"),
    "max_columns": (numbers.Integral, 1, None, "left"),
    "centres": (numbers.Integral, 1, 65_535, "both"),
    "value_scale": (numbers.Real, 0.0, 1.0, "right"),
    "main_effect_share": (numbers.Real, 0.0, 1.0, "both"),
    "replace_share": (numbers.Real, 0.0, 1.0, "both"),
    "max_rounds": (numbers.Integral, 0, None, "left"),
    "pair_screen": (numbers.Integral, 2, None, "left"),
    "column_bins": (numbers.Integral, 2, 255, "both"),
    "response_bins": (numbers.Integral, 2, 255, "both"),
    "score_floor": (numbers.Real, 0.0, math.inf, "left"),
    "tries": (numbers.Integral, 1, None, "left"),
    "refit_passes": (numbers.Integral, 1, None, "left"),
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
