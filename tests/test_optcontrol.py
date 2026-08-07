"""Tests for linear-quadratic optimal-control monetary policy (VAR)."""

from __future__ import annotations

from frbus_shock import run_optimal_control


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
