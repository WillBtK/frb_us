"""Tests for stochastic debt fan charts (small replication counts for speed)."""

from __future__ import annotations

import numpy as np
import pytest

from frbus_shock import PERCENTILES, debt_fan_chart


def test_fan_chart_structure_and_ordering():
    """A small fan run returns ordered percentiles whose band widens over time."""
    r = debt_fan_chart(fiscal_rule="exogenous_taxes", start="2026Q3", horizon=20,
                       nrepl=12, seed=3)
    assert r.nrepl >= 5
    assert set(r.debt_pct) == set(PERCENTILES)
    H = len(r.window)
    for key in ("debt_pct", "primary_pct"):
        band = getattr(r, key)
        # Percentiles are monotone at every quarter.
        assert np.all(band[10] <= band[50] + 1e-9)
        assert np.all(band[50] <= band[90] + 1e-9)
        assert all(len(band[p]) == H for p in PERCENTILES)
    # Macro uncertainty accumulates: the 10–90 band is wider late than early.
    early = r.debt_pct[90][2] - r.debt_pct[10][2]
    late = r.debt_pct[90][-1] - r.debt_pct[10][-1]
    assert late > early
    assert 0.0 <= r.prob_rising <= 1.0
    # Debt/GDP level is a sensible magnitude (tens of percent, not raw dollars).
    assert 20.0 < r.debt_pct[50][0] < 300.0


def test_deterministic_shock_shifts_the_fan():
    """Adding a fiscal expansion raises the central debt/GDP path vs. no shock."""
    base = debt_fan_chart(fiscal_rule="exogenous_taxes", start="2026Q3", horizon=20,
                          nrepl=8, seed=5)
    shocked = debt_fan_chart(shocks=[{"key": "fiscal_spending", "magnitude": 3.0, "duration": 8}],
                             fiscal_rule="exogenous_taxes", start="2026Q3", horizon=20,
                             nrepl=8, seed=5)
    assert shocked.central_debt[-1] > base.central_debt[-1]


def test_fan_rejects_bad_args():
    with pytest.raises(ValueError):
        debt_fan_chart(nrepl=2)
    with pytest.raises(ValueError):
        debt_fan_chart(fiscal_rule="not_a_rule")
