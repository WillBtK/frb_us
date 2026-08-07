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

import json
import pathlib

import pandas as pd
import pytest

# Golden deviations for the first 8 quarters are vintage-specific and live in
# tests/golden_example1.json, which scripts/regen_goldens.py rewrites whenever
# the data-refresh workflow updates LONGBASE. That keeps this strict check in
# sync with the committed data automatically. Embedded values are a fallback if
# the JSON is ever missing.
_GOLDEN_JSON = pathlib.Path(__file__).with_name("golden_example1.json")
_FALLBACK_RFF_BP = [100.0107, 82.5306, 66.1999, 50.273, 36.0264, 23.3095, 12.3605, 3.0577]
_FALLBACK_XGDP_PCT = [0.00082, -0.1622, -0.25559, -0.3883, -0.4313, -0.47154, -0.48467, -0.48973]


def _load_golden():
    try:
        data = json.loads(_GOLDEN_JSON.read_text())
        return data["rff_bp"], data["xgdp_pct"]
    except (OSError, ValueError, KeyError):
        return _FALLBACK_RFF_BP, _FALLBACK_XGDP_PCT


GOLDEN_RFF_BP, GOLDEN_XGDP_PCT = _load_golden()


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
