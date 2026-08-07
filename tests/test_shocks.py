"""Behavioural tests for the shock catalogue and the policy mechanism.

These assert the *sign* and basic sanity of each shock's response and, most
importantly, that "funds rate held at baseline" really holds the funds rate at
baseline. They are intentionally sign/qualitative (not golden-number) so they
stay robust across data-vintage refreshes, while ``test_validation.py`` pins the
exact numbers against the demo.
"""

from __future__ import annotations

import pytest

from frbus_shock import deviations, run_simulation, summary_table
from frbus_shock.shocks import CATALOGUE

# One representative quarter to read the response at (index into the window).
Q4 = 3


def _dev(result, scenario, var, q=Q4):
    return deviations(result, scenario)[var].iloc[q]


def test_funds_rate_hold_pins_rff():
    """The held scenario must keep the funds-rate deviation at ~0 throughout."""
    result = run_simulation("fiscal_spending", magnitude=1.0, duration=8, horizon=12)
    held_rff = deviations(result, "held")["rff"].abs().max()
    assert held_rff < 1e-6  # basis points of a percent — effectively exact
    # ...while the active rule actually moves the rate.
    active_rff = deviations(result, "active")["rff"].abs().max()
    assert active_rff > 0.01


def test_fiscal_spending_is_expansionary():
    result = run_simulation("fiscal_spending", magnitude=1.0, duration=8, horizon=12)
    assert _dev(result, "active", "hggdp") > 0  # growth rises on impact-ish
    # No monetary offset -> larger output effect when the rate is held.
    assert _dev(result, "held", "hggdp") >= _dev(result, "active", "hggdp") - 1e-9


def test_tax_hike_is_contractionary():
    result = run_simulation("tax", magnitude=1.0, duration=8, horizon=12)
    # A tax increase lowers the *level* of GDP: cumulative growth deviation < 0.
    cum = deviations(result, "active")["hggdp"].iloc[: Q4 + 1].sum()
    assert cum < 0


def test_oil_shock_is_stagflationary():
    result = run_simulation("oil", magnitude=10.0, duration=4, horizon=12)
    assert _dev(result, "active", "picnia") > 0  # inflation up
    assert _dev(result, "active", "hggdp") < 0  # output down


def test_productivity_shock_signs():
    result = run_simulation("productivity", magnitude=1.0, duration=1, horizon=12)
    assert _dev(result, "active", "hggdp") > 0  # output up
    assert _dev(result, "active", "picnia") < 0  # disinflationary


def test_term_premium_is_contractionary():
    result = run_simulation("term_premium", magnitude=100.0, duration=4, horizon=12)
    assert _dev(result, "active", "hggdp") < 0


def test_custom_shock_runs():
    """A custom add-factor shock (consumption) should run and move GDP."""
    result = run_simulation(
        custom_variable="eco", magnitude=0.01, duration=4, horizon=8
    )
    assert deviations(result, "active")["hggdp"].abs().max() > 0


def test_summary_table_peak_and_horizons():
    result = run_simulation("term_premium", magnitude=100.0, duration=4, horizon=16)
    tbl = summary_table(result, ["hggdp", "rff", "xgdp"], horizons=(4, 8, 99))
    # Expected columns exist.
    assert {"peak", "peak_quarter", "@4q", "@8q", "@99q"}.issubset(tbl.columns)
    # Horizons beyond the shown window are NaN.
    assert tbl["@99q"].isna().all()
    # The held funds-rate peak is ~0 (rate pinned at baseline).
    held_rff = tbl[
        tbl["variable"].str.contains("Federal funds")
        & tbl["scenario"].str.contains("Without")
    ]
    assert abs(float(held_rff["peak"].iloc[0])) < 1e-6
    # Peak magnitude is at least as large as the effect at any finite horizon.
    for _, row in tbl.iterrows():
        assert abs(row["peak"]) + 1e-9 >= abs(row["@4q"])
    # xgdp is a level -> reported as a percent deviation.
    assert (tbl["unit"] == "%").any()


@pytest.mark.slow
def test_mce_expectations_solve_and_hold():
    """Model-consistent expectations solve, and the funds-rate hold still pins rff."""
    result = run_simulation(
        "term_premium", magnitude=100.0, duration=4, expectations="mce", horizon=12
    )
    # Forward-looking solve produced finite paths...
    assert deviations(result, "active").notna().all().all()
    # ...a contractionary financial shock still lowers output...
    assert _dev(result, "active", "hggdp") < 0
    # ...and the held scenario pins the funds rate at baseline.
    assert deviations(result, "held")["rff"].abs().max() < 1e-6


@pytest.mark.parametrize("key", sorted(CATALOGUE))
def test_every_catalogue_shock_solves(key):
    """Every catalogue shock solves under VAR at its default settings."""
    spec = CATALOGUE[key]
    result = run_simulation(
        key, magnitude=spec.default_magnitude, duration=spec.default_duration, horizon=8
    )
    # Output frames are populated and finite.
    for scen in ("active", "held"):
        dev = deviations(result, scen)
        assert dev.notna().all().all()
