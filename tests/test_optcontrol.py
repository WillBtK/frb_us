"""Tests for linear-quadratic optimal-control monetary policy (VAR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frbus_shock import RULE_LABELS, SCENARIO_PRESETS, run_optimal_control

_RECESSION = [{"custom_variable": "eco", "magnitude": -0.01, "duration": 4}]
_ALL_RULES = ["held", "inertial", "balanced_approach", "taylor", "first_difference"]


def test_optimal_is_the_minimum_loss():
    """The optimum weakly beats every comparator rule (it minimises the loss)."""
    res = run_optimal_control(
        shocks=_RECESSION, comparators=_ALL_RULES, expectations="var",
        start="2026Q3", horizon=12,
    )
    for key in _ALL_RULES:
        assert res.losses["optimal"] <= res.losses[key] + 1e-9
    # For a demand recession, a responding rule beats no response.
    assert res.losses["inertial"] < res.losses["held"]


def test_linear_approximation_is_accurate():
    res = run_optimal_control(shocks=_RECESSION, comparators=_ALL_RULES,
                              expectations="var", start="2026Q3", horizon=12)
    # Closed-form (linear) paths re-solved through the nonlinear model match to a
    # few hundredths of a percentage point — for the optimum *and* the rules.
    assert res.approx_error < 0.15


def test_recession_optimal_cuts_rates():
    res = run_optimal_control(shocks=_RECESSION, expectations="var",
                              start="2026Q3", horizon=12)
    assert res.delta.mean() < 0  # a demand-driven recession → ease policy


def test_higher_unemployment_weight_stabilises_unemployment_more():
    """Weighting unemployment more heavily leaves a smaller unemployment gap."""
    base = run_optimal_control(
        shocks=_RECESSION, weights={"inflation": 1.0, "unemployment": 1.0, "smoothing": 0.5},
        expectations="var", start="2026Q3", horizon=12,
    )
    heavy_u = run_optimal_control(
        shocks=_RECESSION, weights={"inflation": 1.0, "unemployment": 3.0, "smoothing": 0.5},
        expectations="var", start="2026Q3", horizon=12,
    )
    assert heavy_u.paths["optimal_u"].abs().max() < base.paths["optimal_u"].abs().max()
    assert heavy_u.delta.min() < base.delta.min()


def test_comparator_rules_present_and_ordered():
    """Requested comparator columns/losses exist, and rule aggressiveness ranks."""
    res = run_optimal_control(shocks=_RECESSION, comparators=_ALL_RULES,
                              expectations="var", start="2026Q3", horizon=12)
    for key in _ALL_RULES:
        assert key in res.losses
        for suffix in ("rff", "pi", "u"):
            assert f"{key}_{suffix}" in res.paths.columns
    # 'No response' holds the funds rate at baseline (zero deviation).
    assert res.paths["held_rff"].abs().max() < 1e-9
    # A more aggressive output-gap response eases more into a recession:
    # balanced-approach (1.0) cuts deeper than classic Taylor (0.5).
    assert res.paths["balanced_approach_rff"].min() < res.paths["taylor_rff"].min()


def test_taylor_coefficient_of_one_matches_balanced_approach():
    """Taylor with output-gap coefficient 1.0 reproduces the balanced-approach rule."""
    res = run_optimal_control(
        shocks=_RECESSION, comparators=["taylor", "balanced_approach"],
        taylor_coef=1.0, expectations="var", start="2026Q3", horizon=12,
    )
    diff = (res.paths["taylor_rff"] - res.paths["balanced_approach_rff"]).abs().max()
    assert diff < 1e-6


def test_linear_feedback_rules_match_native_switch_solves():
    """The linear-feedback comparator rules reproduce FRB/US's own switch rules."""
    from frbus_shock.model import load_baseline, load_frbus
    from frbus_shock.simulate import _fiscal_baseline_config, _solve_robust

    start = pd.Period("2026Q3", freq="Q"); H = 12; end = start + H - 1
    window = pd.period_range(start, end, freq="Q")
    frbus = load_frbus("var")

    def native(switch):
        d = _fiscal_baseline_config(load_baseline(), start, end, "var")
        for s in ("dmpintay", "dmptay", "dmptlr", "dmpalt", "dmpex", "dmprr"):
            if s in d.columns:
                d.loc[start:end, s] = 0
        d.loc[start:end, switch] = 1
        a = frbus.init_trac(start, end, d); base = a.copy()
        a = a.copy(); a.loc[start:start + 3, "eco_aerr"] += -0.01
        sim = _solve_robust(frbus, start, end, a)
        return (sim.loc[window, "rff"] - base.loc[window, "rff"]).values

    res = run_optimal_control(shocks=_RECESSION, comparators=["inertial", "balanced_approach"],
                              expectations="var", start="2026Q3", horizon=12)
    assert np.max(np.abs(res.paths["inertial_rff"].values - native("dmpintay"))) < 0.05
    assert np.max(np.abs(res.paths["balanced_approach_rff"].values - native("dmptay"))) < 0.05


# Multi-shock scenarios the OCP tab now accepts — every one except those using
# the funds-rate rule itself (the control variable), which the tab filters out.
_OCP_SCENARIOS = [
    name for name, p in SCENARIO_PRESETS.items()
    if all(s["key"] != "monetary" for s in p["shocks"])
]


@pytest.mark.parametrize("name", _OCP_SCENARIOS)
def test_multishock_scenario_optimal_is_the_minimum(name):
    """Optimal control minimises the loss for every OCP-eligible scenario.

    The optimum must weakly beat all comparators; the ordering *among* the rules is
    not guaranteed (for supply shocks, an aggressive rule can underperform inaction
    under a balanced mandate).
    """
    res = run_optimal_control(
        shocks=SCENARIO_PRESETS[name]["shocks"], comparators=_ALL_RULES,
        expectations="var", start="2026Q3", horizon=12,
    )
    for key in res.comparators:
        assert res.losses["optimal"] <= res.losses[key] + 1e-9


def test_monetary_scenario_excluded_from_ocp():
    """'Hawkish Fed surprise' uses the rate rule, so it's not an OCP scenario."""
    hawkish = SCENARIO_PRESETS["Hawkish Fed surprise"]
    assert any(s["key"] == "monetary" for s in hawkish["shocks"])
    assert "Hawkish Fed surprise" not in _OCP_SCENARIOS
