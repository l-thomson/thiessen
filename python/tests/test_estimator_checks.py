"""The scikit-learn estimator contract.

`parametrize_with_checks` runs scikit-learn's own conformance suite. The
configuration is small so the suite runs in reasonable time; the contract
does not depend on the sweep schedule.
"""

from __future__ import annotations

from typing import Any

from sklearn.utils.estimator_checks import parametrize_with_checks
from thiessen import TermParams, _native
from thiessen.estimators import (
    AddiVortesClassifier,
    AddiVortesRegressor,
    XAddiVortesRegressor,
)

from .conftest import sweep

#: Large enough that the suite's accuracy thresholds are met.
SMALL: dict[str, Any] = sweep(20, 30, 60)

EXPECTED_FAILURES: dict[str, str] = {}


def _expected_failed_checks(estimator: object) -> dict[str, str]:
    return dict(EXPECTED_FAILURES)


CHECKED: list[object] = [AddiVortesRegressor(**SMALL), AddiVortesClassifier(**SMALL)]
if _native.EXPERIMENTAL:
    CHECKED.append(
        XAddiVortesRegressor(rounds=5, mean_params=TermParams(tessellations=10))
    )


@parametrize_with_checks(
    CHECKED,
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
