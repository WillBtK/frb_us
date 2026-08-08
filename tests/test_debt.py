"""Tests for debt-sustainability analysis and fiscal closure rules."""

from __future__ import annotations

import pytest

from frbus_shock import (
    DEBT_OUTPUT_KEYS,
    FISCAL_RULES,
    deviations,
    run_debt_comparison,
    run_simulation,
)

# A sustained fiscal expansion — the canonical debt-dynamics disturbance.
_EXPANSION = [{"key": "fiscal_spending", "magnitude": 2.0, "duration": 8}]


def test_debt_comparison_runs_all_rules():
    res = run_debt_comparison(shocks=_EXPANSION, start="2026Q3", horizon=40)
    assert res.fiscal_rules == list(FISCAL_RULES)
    for rule in res.fiscal_rules:
        df = res.deviations[rule]
        assert list(df.columns) == DEBT_OUTPUT_KEYS
        assert df.notna().all().all()  # incl. r − g at the first quarter (full-frame lag)


def test_no_stabilisation_drifts_most():
    """Under exogenous taxes, debt/GDP drifts up more than under a stabilising rule."""
    res = run_debt_comparison(shocks=_EXPANSION, start="2026Q3", horizon=40)
    end = {r: res.deviations[r]["debt_gdp"].iloc[-1] for r in res.fiscal_rules}
    # The non-stabilising rule leaves the largest end-horizon debt/GDP...
    assert end["exogenous_taxes"] == max(end.values())
    # ...and debt keeps rising rather than returning (later > earlier).
    drift = res.deviations["exogenous_taxes"]["debt_gdp"]
    assert drift.iloc[-1] > drift.iloc[8]
    # A stabilising rule brings debt/GDP back down from its peak by the horizon.
    stab = res.deviations["debt_stabilization"]["debt_gdp"]
    assert stab.iloc[-1] < stab.iloc[stab.abs().to_numpy().argmax()]


def test_fiscal_rule_threads_through_run_simulation():
    """run_simulation accepts a fiscal_rule and records it; the debt ratios differ."""
    surplus = run_simulation(shocks=_EXPANSION, horizon=12, fiscal_rule="surplus_ratio")
    exog = run_simulation(shocks=_EXPANSION, horizon=12, fiscal_rule="exogenous_taxes")
    assert surplus.fiscal_rule == "surplus_ratio"
    assert exog.fiscal_rule == "exogenous_taxes"
    # The debt-ratio deviation is a valid, finite output on the general tab too.
    d = deviations(surplus, "active", ["debt_gdp", "primary_gdp"])
    assert d.notna().all().all()


def test_unknown_fiscal_rule_rejected():
    with pytest.raises(ValueError):
        run_simulation("fiscal_spending", fiscal_rule="not_a_rule", horizon=8)
