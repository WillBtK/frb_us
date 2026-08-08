"""Debt-sustainability analysis: one fiscal shock under alternative closure rules.

The core question for debt sustainability is *whether debt returns to its baseline
path after a shock, and how* — which is governed by the government's fiscal closure
rule (see :data:`frbus_shock.policy.FISCAL_RULES`). This module runs a shock under
each selected closure rule and reports the deviation paths of the key fiscal ratios
so the "stabilises vs. drifts" contrast is visible directly.

It is VAR-only and built for **long horizons** (debt/GDP evolves over years, not the
12-quarter shock-propagation window): a common no-shock baseline is solved once, and
the shocked economy is solved once per closure rule (each ``init_trac``\\ ed under its
own switches so it reproduces that baseline exactly before the shock).

A note for interpretation: FRB/US gives the *mechanical* debt dynamics — endogenous
interest rates via the policy rule and term structure, and their feedback into debt
service — but it does **not** endogenise a sovereign-risk premium (rising debt →
wider term premium → higher debt service). Represent that channel by adding the
term-premium lever to the shock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from . import policy
from .model import load_baseline, load_frbus
from .outputs import OUTPUT_BY_KEY
from .simulate import ShockRequest, _fiscal_baseline_config, _resolve_requests, _solve_robust

DEFAULT_HORIZON = 40  # quarters displayed (10 years) — debt dynamics are slow


def _r_minus_g(frame: pd.DataFrame) -> pd.Series:
    """Effective interest rate on the debt minus nominal GDP growth (annualised pp).

    ``r`` = annualised net interest paid over last quarter's debt stock; ``g`` =
    annualised nominal GDP growth. Their difference is the classic driver of debt
    dynamics — when r − g > 0, debt/GDP tends to rise absent primary surpluses.
    """
    r = 400.0 * frame["gfintn"] / frame["gfdbtnp"].shift(1)
    g = 400.0 * (frame["xgdpn"] / frame["xgdpn"].shift(1) - 1.0)
    return r - g


# Debt-sustainability outputs shown on the tab: (key, label, unit).
DEBT_OUTPUTS: List[Tuple[str, str, str]] = [
    ("debt_gdp", "Federal debt held by public (% of GDP)", "pp of GDP"),
    ("primary_gdp", "Primary balance (% of GDP)", "pp of GDP"),
    ("budget_gdp", "Federal budget balance (% of GDP)", "pp of GDP"),
    ("interest_gdp", "Net interest / debt service (% of GDP)", "pp of GDP"),
    ("r_minus_g", "Interest rate − growth (r − g)", "pp"),
]
DEBT_OUTPUT_KEYS = [k for k, _, _ in DEBT_OUTPUTS]
DEBT_LABELS: Dict[str, str] = {k: lbl for k, lbl, _ in DEBT_OUTPUTS}
DEBT_UNITS: Dict[str, str] = {k: u for k, _, u in DEBT_OUTPUTS}


def _level(frame: pd.DataFrame, key: str) -> pd.Series:
    if key == "r_minus_g":
        return _r_minus_g(frame)
    return OUTPUT_BY_KEY[key].level(frame)


@dataclass
class DebtResult:
    start: pd.Period
    window: pd.PeriodIndex
    fiscal_rules: List[str]
    policy_rule: str
    requests: List[ShockRequest]
    # {fiscal_rule: DataFrame(index=window, columns=DEBT_OUTPUT_KEYS)} of deviations
    deviations: Dict[str, pd.DataFrame]


def run_debt_comparison(
    shocks: Sequence[dict],
    fiscal_rules: Sequence[str] = tuple(policy.FISCAL_RULES),
    policy_rule: str = "inertial",
    start: str = "2026Q3",
    horizon: int = DEFAULT_HORIZON,
) -> DebtResult:
    """Run one shock under each fiscal closure rule; return debt-ratio deviations."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 quarter")
    if policy_rule not in policy.ACTIVE_RULES:
        raise ValueError(f"unknown policy_rule '{policy_rule}'")
    rules = [r for r in fiscal_rules if r in policy.FISCAL_RULES]
    if not rules:
        raise ValueError("no valid fiscal_rules given")

    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    window = pd.period_range(start_p, disp_end, freq="Q")
    frbus = load_frbus("var")
    requests = _resolve_requests(shocks, None, None, None, None, None)

    # Common no-shock baseline (default fiscal + inertial monetary) — the deviation
    # reference. Baseline *levels* are identical across closure rules, so a single
    # baseline serves them all.
    base_data = _fiscal_baseline_config(
        load_baseline(), start_p, disp_end, "var", policy.DEFAULT_FISCAL_RULE
    )
    baseline = frbus.init_trac(start_p, disp_end, base_data)

    deviations: Dict[str, pd.DataFrame] = {}
    for rule in rules:
        data = _fiscal_baseline_config(load_baseline(), start_p, disp_end, "var", rule)
        if policy_rule != policy.DEFAULT_RULE:
            data = policy.set_active_rule(data, policy_rule, start_p, disp_end)
        adds = frbus.init_trac(start_p, disp_end, data)
        for req in requests:
            adds = req.spec.apply(adds, req.magnitude, req.duration, start_p)
        sim = _solve_robust(frbus, start_p, disp_end, adds)

        df = pd.DataFrame(index=window)
        for key in DEBT_OUTPUT_KEYS:
            # Compute levels on the full frames (so r − g's lag is clean), then slice.
            df[key] = (_level(sim, key) - _level(baseline, key)).loc[window].values
        deviations[rule] = df

    return DebtResult(
        start=start_p, window=window, fiscal_rules=rules, policy_rule=policy_rule,
        requests=requests, deviations=deviations,
    )
