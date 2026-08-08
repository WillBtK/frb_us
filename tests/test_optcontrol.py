"""Tests for linear-quadratic optimal-control monetary policy (VAR)."""

from __future__ import annotations

import pytest

from frbus_shock import SCENARIO_PRESETS, run_optimal_control


def test_optimal_beats_taylor_beats_held():
    res = run_optimal_control(
        shocks=[{"custom_variable": "eco", "magnitude": -0.01, "duration": 4}],
        expectations="var",
        start="2026Q3",
        horizon=12,
    )
    # Optimal control achieves a lower loss than the Taylor rule, which beats
    # no monetary response.
    assert res.losses["optimal"] < res.losses["taylor"] < res.losses["held"]


def test_linear_approximation_is_accurate():
    res = run_optimal_control(
        shocks=[{"custom_variable": "eco", "magnitude": -0.01, "duration": 4}],
        expectations="var",
        start="2026Q3",
        horizon=12,
    )
    # The closed-form (linear) optimum re-solved through the nonlinear model
    # matches to a few hundredths of a percentage point.
    assert res.approx_error < 0.15


def test_recession_optimal_cuts_rates():
    res = run_optimal_control(
        shocks=[{"custom_variable": "eco", "magnitude": -0.01, "duration": 4}],
        expectations="var",
        start="2026Q3",
        horizon=12,
    )
    assert res.delta.mean() < 0  # a demand-driven recession → ease policy


def test_higher_unemployment_weight_stabilises_unemployment_more():
    """Weighting unemployment more heavily leaves a smaller unemployment gap."""
    base = run_optimal_control(
        shocks=[{"custom_variable": "eco", "magnitude": -0.01, "duration": 4}],
        weights={"inflation": 1.0, "unemployment": 1.0, "smoothing": 0.5},
        expectations="var", start="2026Q3", horizon=12,
    )
    heavy_u = run_optimal_control(
        shocks=[{"custom_variable": "eco", "magnitude": -0.01, "duration": 4}],
        weights={"inflation": 1.0, "unemployment": 3.0, "smoothing": 0.5},
        expectations="var", start="2026Q3", horizon=12,
    )
    # A heavier unemployment weight -> smaller peak unemployment deviation, and a
    # deeper trough in the funds-rate path (more aggressive easing at the bottom).
    assert heavy_u.paths["optimal_u"].abs().max() < base.paths["optimal_u"].abs().max()
    assert heavy_u.delta.min() < base.delta.min()


# Multi-shock scenarios the OCP tab now accepts — every one except those using
# the funds-rate rule itself (the control variable), which the tab filters out.
_OCP_SCENARIOS = [
    name for name, p in SCENARIO_PRESETS.items()
    if all(s["key"] != "monetary" for s in p["shocks"])
]


@pytest.mark.parametrize("name", _OCP_SCENARIOS)
def test_multishock_scenario_optimal_is_the_minimum(name):
    """Optimal control minimises the loss for every OCP-eligible scenario.

    The optimum must weakly beat both the Taylor rule and no response; the
    Taylor-vs-held ordering is *not* guaranteed (for supply shocks, an aggressive
    rule can underperform inaction under a balanced mandate).
    """
    res = run_optimal_control(
        shocks=SCENARIO_PRESETS[name]["shocks"], expectations="var",
        start="2026Q3", horizon=12,
    )
    L = res.losses
    assert L["optimal"] <= L["taylor"] + 1e-9
    assert L["optimal"] <= L["held"] + 1e-9


def test_monetary_scenario_excluded_from_ocp():
    """'Hawkish Fed surprise' uses the rate rule, so it's not an OCP scenario."""
    hawkish = SCENARIO_PRESETS["Hawkish Fed surprise"]
    assert any(s["key"] == "monetary" for s in hawkish["shocks"])
    assert "Hawkish Fed surprise" not in _OCP_SCENARIOS
