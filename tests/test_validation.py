"""Validate the vendored, patched model against the Fed's own demo.

``demos/example1.py`` from the PyFRB/US package applies a 100 bp shock to the
funds-rate rule (``rffintay_aerr += 1``) at 2040Q1 under VAR expectations and
solves. Reproducing its funds-rate and GDP paths to tight tolerance is the
anchor that lets us trust every other shock — and it is the guard that the
scipy/sympy modernisation patches did not change the model's answers.

The golden values below were captured from the vendored model and match the
behaviour documented for the demo (a 100 bp impact that decays over ~2 years,
with real GDP falling a few tenths of a percent).
"""

from __future__ import annotations

import pandas as pd
import pytest

# Golden deviations for the first 8 quarters (see module docstring).
GOLDEN_RFF_BP = [100.0113, 82.6314, 66.4751, 50.8255, 36.842, 24.3446, 13.5329, 4.288]
GOLDEN_XGDP_PCT = [
    0.00079,
    -0.15566,
    -0.24333,
    -0.36776,
    -0.40956,
    -0.45105,
    -0.46869,
    -0.47916,
]


def _run_example1(var_model, baseline):
    start = pd.Period("2040Q1", freq="Q")
    end = start + 23
    data = baseline.copy()
    data.loc[start:end, "dfpdbt"] = 0
    data.loc[start:end, "dfpsrp"] = 1
    with_adds = var_model.init_trac(start, end, data)
    with_adds.loc[start, "rffintay_aerr"] += 1
    sim = var_model.solve(start, end, with_adds)
    rff_bp = (sim.loc[start:end, "rff"] - with_adds.loc[start:end, "rff"]) * 100
    xgdp_pct = (sim.loc[start:end, "xgdp"] / with_adds.loc[start:end, "xgdp"] - 1) * 100
    return rff_bp.tolist(), xgdp_pct.tolist()


def test_example1_reproduces_golden_paths(var_model, baseline):
    rff_bp, xgdp_pct = _run_example1(var_model, baseline)
    for got, want in zip(rff_bp[:8], GOLDEN_RFF_BP):
        assert got == pytest.approx(want, abs=0.05)
    for got, want in zip(xgdp_pct[:8], GOLDEN_XGDP_PCT):
        assert got == pytest.approx(want, abs=0.005)


def test_example1_qualitative(var_model, baseline):
    """Sanity: 100 bp impact, monotone decay, contractionary GDP effect."""
    rff_bp, xgdp_pct = _run_example1(var_model, baseline)
    assert rff_bp[0] == pytest.approx(100.0, abs=0.5)  # ~100 bp impact
    assert all(b >= -1.0 for b in rff_bp[:8])  # rate stays up then decays
    assert rff_bp[1] < rff_bp[0]  # decays
    assert min(xgdp_pct[:8]) < -0.2  # GDP clearly falls


def test_run_simulation_matches_demo(var_model, baseline):
    """The public ``run_simulation`` 'monetary' shock == the raw demo."""
    from frbus_shock import run_simulation

    result = run_simulation(
        "monetary", magnitude=100.0, duration=1, expectations="var", horizon=8
    )
    dev_rff_bp = (result.active["rff"] - result.baseline["rff"]) * 100
    for got, want in zip(dev_rff_bp.tolist(), GOLDEN_RFF_BP):
        assert got == pytest.approx(want, abs=0.05)
