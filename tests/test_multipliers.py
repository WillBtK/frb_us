"""Tests for the fiscal-multiplier module.

Qualitative/range checks (robust across data-vintage refreshes): purchases carry
larger multipliers than tax cuts, and accommodation (funds rate held) raises the
cumulative multiplier relative to the active rule.
"""

from __future__ import annotations

from frbus_shock import INSTRUMENTS, multiplier_table, run_multiplier


def test_purchases_multiplier_range_and_ordering():
    tbl = multiplier_table(
        ["fed_purchases", "transfers", "personal_tax_cut"],
        expectations="var",
        start="2026Q3",
        horizon=12,
    )
    active = tbl[tbl["scenario"].str.startswith("With")].set_index("instrument")
    purch = active.loc["Federal purchases", "1 year"]
    tax = active.loc["Personal tax cut", "1 year"]
    transf = active.loc["Federal transfers", "1 year"]
    # Government purchases multiplier is a sensible ~0.5-1.2.
    assert 0.4 < purch < 1.3
    # Purchases > transfers > tax cut (the standard ordering).
    assert purch > transf > tax > 0


def test_accommodation_raises_multiplier():
    res = run_multiplier("fed_purchases", expectations="var", start="2026Q3", horizon=12)
    # By 3 years, holding the funds rate gives a larger cumulative multiplier.
    assert res.held.iloc[11] > res.active.iloc[11]
    # Impact multiplier is ~identical (policy reacts with a lag).
    assert abs(res.held.iloc[0] - res.active.iloc[0]) < 0.05


def test_every_instrument_solves():
    tbl = multiplier_table(list(INSTRUMENTS), expectations="var", start="2026Q3", horizon=12)
    assert tbl.notna().all().all()  # all horizons within the 12-quarter window
    assert len(tbl) == 2 * len(INSTRUMENTS)  # active + held per instrument
