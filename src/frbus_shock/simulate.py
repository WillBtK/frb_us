"""Core simulation: run one shock with and without a monetary-policy response.

``run_simulation`` is the single entry point the dashboard (and tests) call. It
returns a :class:`SimResult` holding the common no-shock baseline plus the two
shocked solutions (active rule / funds rate held), from which deviation paths
for GDP growth, unemployment, PCE inflation, and the funds rate are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from . import policy
from .model import load_baseline, load_frbus
from .shocks import ShockSpec, custom_shock, get_shock, with_defaults

# Imported after model.py (which wires up sys.path for the vendored package).
from pyfrbus.exceptions import ComputationError, ConvergenceError  # type: ignore  # noqa: E402

# MCE (forward-looking) needs lead room past the displayed window so the terminal
# conditions — leads returning to baseline — do not distort it. We solve to a
# fixed buffer *beyond the display window*, with an absolute floor, rather than to
# a long fixed horizon. The stacked-time MCE Jacobian and peak memory scale
# strongly with this horizon, yet the displayed deviations barely move: over a
# 12-quarter window, hggdp differs by <0.008 pp across 28/40/60-quarter solves —
# even for the most persistent shocks (productivity, fiscal) — while peak memory
# falls from ≈1150 MB (60q) to ≈590 MB (28q). A 16-quarter buffer past the shown
# window keeps the default 12-quarter (3-year) view at a ~28-quarter solve, which
# fits comfortably on a memory-limited runtime (e.g. Streamlit Cloud) and is far
# faster and less variable.
_MCE_TERMINAL_BUFFER = 16  # quarters of lead room past the displayed window
_MCE_MIN_SOLVE_QUARTERS = 24  # absolute floor (6 years)

DEFAULT_START = "2040Q1"
DEFAULT_HORIZON = 12  # quarters displayed (3 years) — shocks have largely played out


@dataclass
class ShockRequest:
    """One shock in a (possibly multi-shock) scenario."""

    spec: ShockSpec
    magnitude: float
    duration: int

    def describe(self) -> str:
        return f"{self.spec.label} {self.magnitude:g} {self.spec.user_unit}, {self.duration}q"


@dataclass
class SimResult:
    """Everything a caller needs to chart or export one run."""

    requests: List[ShockRequest]  # one or more shocks applied together
    expectations: str
    start: pd.Period
    end: pd.Period  # last displayed quarter (inclusive)
    baseline: pd.DataFrame
    active: pd.DataFrame  # shock(s) + active policy rule (with response)
    held: pd.DataFrame  # shock(s) + funds rate held at baseline (without response)
    policy_rule: str = "inertial"  # which rule the active scenario follows

    @property
    def window(self) -> pd.PeriodIndex:
        return pd.period_range(self.start, self.end, freq="Q")

    # Backward-compatible single-shock accessors (first shock).
    @property
    def shock(self) -> ShockSpec:
        return self.requests[0].spec

    @property
    def magnitude(self) -> float:
        return self.requests[0].magnitude

    @property
    def duration(self) -> int:
        return self.requests[0].duration

    @property
    def label(self) -> str:
        return " + ".join(r.describe() for r in self.requests)


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
    shocks: Optional[Sequence[dict]] = None,
    policy_rule: str = "inertial",
) -> SimResult:
    """Run a scenario twice — active rule and funds rate held — vs. baseline.

    Parameters
    ----------
    shock_key:
        A key from :data:`frbus_shock.shocks.CATALOGUE`. Ignored if
        ``custom_variable`` or ``shocks`` is given.
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
    shocks:
        Apply **several shocks together**. A sequence of dicts, each with either
        ``key`` (a catalogue key) or ``custom_variable``, plus optional
        ``magnitude``/``duration``/``label``. Takes precedence over the
        single-shock arguments. All shocks share the same start/expectations and
        are added to the same dataset, so their effects combine.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 quarter")
    if expectations not in ("var", "mce"):
        raise ValueError("expectations must be 'var' or 'mce'")
    if policy_rule not in policy.ACTIVE_RULES:
        raise ValueError(
            f"policy_rule must be one of {list(policy.ACTIVE_RULES)}, got '{policy_rule}'"
        )

    requests = _resolve_requests(
        shocks, shock_key, magnitude, duration, custom_variable, custom_label
    )

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

    def _apply_all(df: pd.DataFrame) -> pd.DataFrame:
        for req in requests:
            df = req.spec.apply(df, req.magnitude, req.duration, start_p)
        return df

    # --- Scenario 1: active policy rule (with monetary response) ---
    # The inertial rule is the baseline default (already tracked); any other rule is
    # init_trac'd under its own switch so it, too, reproduces the baseline exactly
    # (zero no-shock deviation) before the shock is applied. Baseline *levels* are
    # identical across rules, so ``baseline_adds`` stays the deviation reference.
    if policy_rule == policy.DEFAULT_RULE:
        active_in = policy.apply_active_rule(baseline_adds.copy(), start_p, solve_end)
    else:
        ruled = policy.set_active_rule(data, policy_rule, start_p, solve_end)
        active_in = frbus.init_trac(start_p, solve_end, ruled)
    active = _solve_robust(frbus, start_p, solve_end, _apply_all(active_in))

    # --- Scenario 2: funds rate held at baseline (no monetary response) ---
    held_in = policy.apply_funds_rate_hold(
        baseline_adds.copy(), baseline_adds, start_p, solve_end
    )
    held = _solve_robust(frbus, start_p, solve_end, _apply_all(held_in))

    sl = slice(start_p, disp_end)
    return SimResult(
        requests=requests,
        expectations=expectations,
        start=start_p,
        end=disp_end,
        baseline=baseline_adds.loc[sl].copy(),
        active=active.loc[sl].copy(),
        held=held.loc[sl].copy(),
        policy_rule=policy_rule,
    )


def _resolve_requests(
    shocks, shock_key, magnitude, duration, custom_variable, custom_label
) -> List[ShockRequest]:
    """Build the list of ShockRequests from either ``shocks`` or legacy args."""
    if shocks:
        out: List[ShockRequest] = []
        for s in shocks:
            cv = s.get("custom_variable")
            if cv:
                spec = custom_shock(cv, s.get("magnitude") or 0.0, s.get("duration") or 1,
                                    s.get("label"))
            else:
                key = s.get("key") or s.get("shock_key")
                if not key:
                    raise ValueError("each shock needs a 'key' or 'custom_variable'")
                spec = get_shock(key)
            spec, mag, dur = with_defaults(spec, s.get("magnitude"), s.get("duration"))
            out.append(ShockRequest(spec, mag, dur))
        if not out:
            raise ValueError("shocks list is empty")
        return out

    if custom_variable:
        spec = custom_shock(custom_variable, magnitude or 0.0, duration or 1, custom_label)
        mag = spec.default_magnitude if magnitude is None else magnitude
        dur = spec.default_duration if duration is None else duration
        return [ShockRequest(spec, mag, dur)]

    if not shock_key:
        raise ValueError("provide shock_key, custom_variable, or shocks")
    spec, mag, dur = with_defaults(get_shock(shock_key), magnitude, duration)
    return [ShockRequest(spec, mag, dur)]
