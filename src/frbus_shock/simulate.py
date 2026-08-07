"""Core simulation: run one shock with and without a monetary-policy response.

``run_simulation`` is the single entry point the dashboard (and tests) call. It
returns a :class:`SimResult` holding the common no-shock baseline plus the two
shocked solutions (active rule / funds rate held), from which deviation paths
for GDP growth, unemployment, PCE inflation, and the funds rate are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import policy
from .model import load_baseline, load_frbus
from .shocks import ShockSpec, custom_shock, get_shock, with_defaults

# Imported after model.py (which wires up sys.path for the vendored package).
from pyfrbus.exceptions import ComputationError, ConvergenceError  # type: ignore  # noqa: E402

# MCE (forward-looking) needs lead room past the displayed window so the terminal
# conditions — leads returning to baseline — do not distort it. Rather than always
# solving to a long fixed horizon, we solve to a fixed buffer *beyond the display
# window*, with an absolute floor. This keeps the common 20-quarter case light
# (a ~40-quarter solve) while still giving ~5 years of terminal buffer.
#
# The stacked-time MCE Jacobian and peak memory scale strongly with this horizon
# (≈740 MB at 40q vs ≈1150 MB at 60q), yet the displayed deviations are unchanged
# to <0.01 pp across 40/48/60 — so the shorter, display-scaled horizon is both
# faster and much easier on a memory-limited runtime (e.g. Streamlit Cloud).
_MCE_TERMINAL_BUFFER = 20  # quarters of lead room past the displayed window
_MCE_MIN_SOLVE_QUARTERS = 40  # absolute floor (10 years)

DEFAULT_START = "2040Q1"
DEFAULT_HORIZON = 20  # quarters displayed (5 years)


@dataclass
class SimResult:
    """Everything a caller needs to chart or export one run."""

    shock: ShockSpec
    magnitude: float
    duration: int
    expectations: str
    start: pd.Period
    end: pd.Period  # last displayed quarter (inclusive)
    baseline: pd.DataFrame
    active: pd.DataFrame  # shock + active policy rule (with response)
    held: pd.DataFrame  # shock + funds rate held at baseline (without response)

    @property
    def window(self) -> pd.PeriodIndex:
        return pd.period_range(self.start, self.end, freq="Q")


def _fiscal_baseline_config(
    data: pd.DataFrame, start: pd.Period, end: pd.Period, expectations: str
) -> pd.DataFrame:
    """Apply the standard FRB/US baseline policy configuration.

    Matches the shipped demos: debt-stabilisation off, surplus-ratio targeting
    on. Under MCE we also fix ``rstar`` (``drstar=0``), as in ``example2.py``.
    """
    data.loc[start:end, "dfpdbt"] = 0
    data.loc[start:end, "dfpsrp"] = 1
    if expectations == "mce" and "drstar" in data.columns:
        data.loc[start:end, "drstar"] = 0
    return data


def _solve_robust(frbus, start, end, data) -> pd.DataFrame:
    """Solve, falling back to the sparse Newton solver if the default diverges."""
    try:
        return frbus.solve(start, end, data)
    except (ConvergenceError, ComputationError):
        # The User Guide notes VAR sims can fail on the default solver; the
        # sparse Newton method often converges where it does not.
        return frbus.solve(start, end, data, options={"newton": "newton"})


def run_simulation(
    shock_key: Optional[str] = None,
    magnitude: Optional[float] = None,
    duration: Optional[int] = None,
    expectations: str = "var",
    start: str = DEFAULT_START,
    horizon: int = DEFAULT_HORIZON,
    custom_variable: Optional[str] = None,
    custom_label: Optional[str] = None,
) -> SimResult:
    """Run one shock twice — active rule and funds rate held — vs. baseline.

    Parameters
    ----------
    shock_key:
        A key from :data:`frbus_shock.shocks.CATALOGUE`. Ignored if
        ``custom_variable`` is given.
    magnitude, duration:
        Shock size (in the shock's user unit) and how many quarters it is held
        on. ``None`` falls back to the shock's defaults.
    expectations:
        ``"var"`` or ``"mce"``.
    start:
        First simulation quarter, e.g. ``"2040Q1"``.
    horizon:
        Number of quarters to display (>= 1).
    custom_variable, custom_label:
        Shock an arbitrary ``<variable>_aerr`` instead of a catalogue entry.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 quarter")
    if expectations not in ("var", "mce"):
        raise ValueError("expectations must be 'var' or 'mce'")

    # Resolve the shock specification and its magnitude/duration.
    if custom_variable:
        spec = custom_shock(custom_variable, magnitude or 0.0, duration or 1, custom_label)
        mag = spec.default_magnitude if magnitude is None else magnitude
        dur = spec.default_duration if duration is None else duration
    else:
        if not shock_key:
            raise ValueError("provide either shock_key or custom_variable")
        spec, mag, dur = with_defaults(get_shock(shock_key), magnitude, duration)

    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    if expectations == "mce":
        solve_end = max(disp_end + _MCE_TERMINAL_BUFFER, start_p + _MCE_MIN_SOLVE_QUARTERS)
    else:
        solve_end = disp_end

    # Baseline dataset + standard policy configuration.
    data = load_baseline()
    data = _fiscal_baseline_config(data, start_p, solve_end, expectations)

    frbus = load_frbus(expectations)

    # Add factors so the model reproduces baseline exactly over the window.
    baseline_adds = frbus.init_trac(start_p, solve_end, data)

    # --- Scenario 1: active policy rule (with monetary response) ---
    active_in = policy.apply_active_rule(baseline_adds.copy(), start_p, solve_end)
    active_in = spec.apply(active_in, mag, dur, start_p)
    active = _solve_robust(frbus, start_p, solve_end, active_in)

    # --- Scenario 2: funds rate held at baseline (no monetary response) ---
    held_in = policy.apply_funds_rate_hold(
        baseline_adds.copy(), baseline_adds, start_p, solve_end
    )
    held_in = spec.apply(held_in, mag, dur, start_p)
    held = _solve_robust(frbus, start_p, solve_end, held_in)

    sl = slice(start_p, disp_end)
    return SimResult(
        shock=spec,
        magnitude=mag,
        duration=dur,
        expectations=expectations,
        start=start_p,
        end=disp_end,
        baseline=baseline_adds.loc[sl].copy(),
        active=active.loc[sl].copy(),
        held=held.loc[sl].copy(),
    )
