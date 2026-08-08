"""Fiscal-multiplier estimation, in the spirit of CBO's fiscal-multiplier work.

The **output multiplier** of a fiscal instrument at horizon ``h`` is the
cumulative change in real GDP divided by the cumulative fiscal impulse, both in
real (chained-2012) dollars::

    multiplier(h) = sum_{t=0..h} ΔGDP_t  /  sum_{t=0..h} Δimpulse_t

Different instruments carry different multipliers, and — the key policy point —
the multiplier is larger when monetary policy **accommodates** (the funds rate
held at baseline) than when the active rule leans against the shock. Both cases
are computed, reusing the same funds-rate-hold mechanism as the shock analysis.

Instruments and their fiscal impulse (all validated to solve with sensible,
literature-consistent multipliers — purchases > transfers > tax cut, each rising
with accommodation):

* Federal purchases  (EGFE) — impulse = Δreal federal purchases
* S&L purchases      (EGSE) — impulse = Δreal state & local purchases
* Federal transfers  (GTR)  — impulse = Δreal transfer payments
* Personal tax cut   (TRP)  — impulse = −Δreal personal tax receipts (TPN),
  deflated by the GDP price index (a rate *cut* is the stimulus)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import policy
from .model import load_baseline, load_frbus
from .simulate import (
    _MCE_MIN_SOLVE_QUARTERS,
    _MCE_TERMINAL_BUFFER,
    _fiscal_baseline_config,
    _solve_robust,
)


@dataclass(frozen=True)
class FiscalInstrument:
    """A fiscal instrument and how to measure its dollar impulse."""

    key: str
    label: str
    lever: str  # the ``*_aerr`` add factor to shock
    magnitude: float  # stimulus size in the lever's native add-factor units
    duration: int  # quarters the impulse is applied
    impulse_var: str  # variable whose Δ (real $) is the fiscal impulse
    impulse_nominal: bool  # deflate impulse_var by the GDP price index?
    impulse_sign: float  # +1 spending/transfers; −1 tax (revenue fall = stimulus)
    note: str


# +1% of the relevant real spending component; −0.5 pp personal tax rate (a cut);
# +2% of transfers. These are moderate, well-inside-linear impulses.
INSTRUMENTS: Dict[str, FiscalInstrument] = {
    "fed_purchases": FiscalInstrument(
        "fed_purchases", "Federal purchases", "egfe_aerr", 0.01, 8,
        "egfe", False, +1.0, "Government purchases — the classic high multiplier."),
    "sl_purchases": FiscalInstrument(
        "sl_purchases", "State & local purchases", "egse_aerr", 0.01, 8,
        "egse", False, +1.0, "State & local purchases."),
    "transfers": FiscalInstrument(
        "transfers", "Federal transfers", "gtr_aerr", 0.02, 8,
        "gtr", False, +1.0, "Transfer payments — smaller multiplier (partly saved)."),
    "personal_tax_cut": FiscalInstrument(
        "personal_tax_cut", "Personal tax cut", "trp_aerr", -0.005, 8,
        "tpn", True, -1.0, "A personal income-tax-rate cut — smallest multiplier."),
    "corporate_tax_cut": FiscalInstrument(
        "corporate_tax_cut", "Corporate tax cut", "trci_aerr", -0.005, 8,
        "tcin", True, -1.0, "A corporate income-tax-rate cut — works through investment."),
}

# Horizons reported (index into the window; 0 = impact quarter).
DEFAULT_HORIZON_LABELS = {0: "Impact", 3: "1 year", 7: "2 years", 11: "3 years"}


def get_instrument(key: str) -> FiscalInstrument:
    try:
        return INSTRUMENTS[key]
    except KeyError as exc:
        raise KeyError(f"unknown instrument '{key}'; choose {sorted(INSTRUMENTS)}") from exc


@dataclass
class MultiplierResult:
    instrument: FiscalInstrument
    expectations: str
    start: pd.Period
    window: pd.PeriodIndex
    # cumulative multiplier path, indexed by quarter, per scenario
    active: pd.Series
    held: pd.Series
    # underlying real-$ paths (for export/inspection)
    dgdp_active: pd.Series
    dimpulse_active: pd.Series


def _cumulative_multiplier(dgdp: pd.Series, dimpulse: pd.Series) -> pd.Series:
    """Cumulative ΔGDP / cumulative impulse at each horizon."""
    return dgdp.cumsum() / dimpulse.cumsum()


def run_multiplier(
    instrument_key: str,
    expectations: str = "var",
    start: str = "2026Q3",
    horizon: int = 12,
) -> MultiplierResult:
    """Compute an instrument's cumulative output multiplier, active vs. held."""
    inst = get_instrument(instrument_key)
    if expectations not in ("var", "mce"):
        raise ValueError("expectations must be 'var' or 'mce'")

    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    if expectations == "mce":
        solve_end = max(disp_end + _MCE_TERMINAL_BUFFER, start_p + _MCE_MIN_SOLVE_QUARTERS)
    else:
        solve_end = disp_end

    data = _fiscal_baseline_config(load_baseline(), start_p, solve_end, expectations)
    frbus = load_frbus(expectations)
    base = frbus.init_trac(start_p, solve_end, data)

    sl = slice(start_p, disp_end)
    defl = (base.loc[sl, "pgdp"] / 100.0) if inst.impulse_nominal else 1.0

    def _impulse(sim: pd.DataFrame) -> pd.Series:
        raw = sim.loc[sl, inst.impulse_var] - base.loc[sl, inst.impulse_var]
        return inst.impulse_sign * (raw / defl)

    def _dgdp(sim: pd.DataFrame) -> pd.Series:
        return sim.loc[sl, "xgdp"] - base.loc[sl, "xgdp"]

    def _one(held: bool):
        df = base.copy()
        if held:
            df = policy.apply_funds_rate_hold(df, base, start_p, solve_end)
        last = start_p + (inst.duration - 1)
        df.loc[start_p:last, inst.lever] += inst.magnitude
        sim = _solve_robust(frbus, start_p, solve_end, df)
        return _dgdp(sim), _impulse(sim)

    dY_a, dG_a = _one(False)
    dY_h, dG_h = _one(True)
    window = pd.period_range(start_p, disp_end, freq="Q")
    return MultiplierResult(
        instrument=inst,
        expectations=expectations,
        start=start_p,
        window=window,
        active=_cumulative_multiplier(dY_a, dG_a),
        held=_cumulative_multiplier(dY_h, dG_h),
        dgdp_active=dY_a,
        dimpulse_active=dG_a,
    )


def multiplier_table(
    instrument_keys: Sequence[str],
    expectations: str = "var",
    start: str = "2026Q3",
    horizon: int = 12,
    horizon_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Tidy table of cumulative multipliers by instrument, scenario, and horizon.

    Columns: ``instrument``, ``scenario``, then one column per horizon label
    (e.g. Impact / 1 year / 2 years / 3 years).
    """
    labels = horizon_labels or DEFAULT_HORIZON_LABELS
    rows: List[dict] = []
    for key in instrument_keys:
        res = run_multiplier(key, expectations, start, horizon)
        for scen_label, series in (
            ("With policy response (active rule)", res.active),
            ("Accommodative (funds rate held)", res.held),
        ):
            row = {"instrument": res.instrument.label, "scenario": scen_label}
            for idx, lab in labels.items():
                row[lab] = round(float(series.iloc[idx]), 3) if idx < len(series) else None
            rows.append(row)
    return pd.DataFrame(rows)
