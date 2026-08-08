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

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import policy
from .feedback import fresh_frbus, solve_with_feedback
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
    feedback: Tuple[float, float] = (0.0, 0.0)  # (beta_debt, beta_deficit) bps/pp
    # {fiscal_rule: bool} — did the sovereign-risk feedback reach a fixed point?
    converged: Dict[str, bool] = field(default_factory=dict)
    # Baseline (no-shock) *levels* of each debt output over the window — lets the UI
    # show the actual debt/GDP path (level = baseline_levels + deviations).
    baseline_levels: pd.DataFrame = field(default_factory=pd.DataFrame)

    def levels(self, rule: str) -> pd.DataFrame:
        """Actual level path of each debt output under ``rule`` (baseline + deviation)."""
        return self.baseline_levels.add(self.deviations[rule], fill_value=0.0)


def run_debt_comparison(
    shocks: Sequence[dict],
    fiscal_rules: Sequence[str] = tuple(policy.FISCAL_RULES),
    policy_rule: str = "inertial",
    start: str = "2026Q3",
    horizon: int = DEFAULT_HORIZON,
    feedback: Tuple[float, float] = (0.0, 0.0),
) -> DebtResult:
    """Run one shock under each fiscal closure rule; return debt-ratio deviations.

    ``feedback`` = ``(beta_debt, beta_deficit)`` in bps of the 10-year rate per 1pp of
    debt/GDP and primary-deficit/GDP. When either is positive, a sovereign-risk
    feedback is iterated to a fixed point per rule (see :mod:`frbus_shock.feedback`),
    and each rule's convergence is recorded — non-convergence flags a debt spiral.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 quarter")
    if policy_rule not in policy.ACTIVE_RULES:
        raise ValueError(f"unknown policy_rule '{policy_rule}'")
    rules = [r for r in fiscal_rules if r in policy.FISCAL_RULES]
    if not rules:
        raise ValueError("no valid fiscal_rules given")

    beta_debt, beta_deficit = feedback
    use_feedback = beta_debt > 0 or beta_deficit > 0

    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    window = pd.period_range(start_p, disp_end, freq="Q")
    frbus = load_frbus("var")
    fb_frbus = fresh_frbus() if use_feedback else None
    requests = _resolve_requests(shocks, None, None, None, None, None)

    # Common no-shock baseline (default fiscal + inertial monetary) — the deviation
    # reference. Baseline *levels* are identical across closure rules, so a single
    # baseline serves them all.
    base_data = _fiscal_baseline_config(
        load_baseline(), start_p, disp_end, "var", policy.DEFAULT_FISCAL_RULE
    )
    baseline = frbus.init_trac(start_p, disp_end, base_data)

    deviations: Dict[str, pd.DataFrame] = {}
    converged: Dict[str, bool] = {}
    for rule in rules:
        data = _fiscal_baseline_config(load_baseline(), start_p, disp_end, "var", rule)
        if policy_rule != policy.DEFAULT_RULE:
            data = policy.set_active_rule(data, policy_rule, start_p, disp_end)
        adds = frbus.init_trac(start_p, disp_end, data)
        for req in requests:
            adds = req.spec.apply(adds, req.magnitude, req.duration, start_p)

        if use_feedback:
            sim, _sim0, conv, _iters, _tp = solve_with_feedback(
                fb_frbus, start_p, disp_end, adds, baseline, beta_debt, beta_deficit
            )
            converged[rule] = conv
        else:
            sim = _solve_robust(frbus, start_p, disp_end, adds)
            converged[rule] = True

        df = pd.DataFrame(index=window)
        for key in DEBT_OUTPUT_KEYS:
            # Compute levels on the full frames (so r − g's lag is clean), then slice.
            df[key] = (_level(sim, key) - _level(baseline, key)).loc[window].values
        deviations[rule] = df

    base_levels = pd.DataFrame(
        {key: _level(baseline, key).loc[window].values for key in DEBT_OUTPUT_KEYS},
        index=window,
    )
    return DebtResult(
        start=start_p, window=window, fiscal_rules=rules, policy_rule=policy_rule,
        requests=requests, deviations=deviations, feedback=(beta_debt, beta_deficit),
        converged=converged, baseline_levels=base_levels,
    )


# --------------------------------------------------------------------------- #
# Clean deficit-shock scenario (intuitive % of GDP input)                     #
# --------------------------------------------------------------------------- #
def _solve_debt_shock(frbus, start_p, disp_end, adds, baseline, gtrd_path,
                      beta_debt, beta_deficit, max_iter=6, tol=0.03, blowup=1000.0):
    """Solve with transfers (``gtrd``) exogenised at ``gtrd_path`` (the fiscal shock),
    optionally iterating the sovereign-risk term-premium feedback to a fixed point.
    Returns ``(sim, converged)``."""
    window = pd.period_range(start_p, disp_end, freq="Q")
    use_fb = beta_debt > 0 or beta_deficit > 0
    exog = ["gtrd", "rg10p"] if use_fb else ["gtrd"]
    base_debt = _debt_level(baseline, window)
    base_def = _primary_deficit_level(baseline, window)
    base_tp = adds.loc[window, "rg10p"].to_numpy()

    frbus.exogenize(exog)
    try:
        add = np.zeros(len(window))
        prev = None
        converged = True
        sim = None
        for _it in range(1, (max_iter if use_fb else 1) + 1):
            data = adds.copy()
            data.loc[start_p:disp_end, "gtrd"] = gtrd_path
            if use_fb:
                data.loc[start_p:disp_end, "rg10p"] = base_tp + add
            sim = _solve_robust(frbus, start_p, disp_end, data)
            if not use_fb:
                break
            ddev = _debt_level(sim, window) - base_debt
            if not np.all(np.isfinite(ddev)) or np.max(np.abs(ddev)) > blowup:
                converged = False
                break
            if prev is not None and np.max(np.abs(ddev - prev)) < tol:
                converged = True
                break
            prev = ddev
            fdev = _primary_deficit_level(sim, window) - base_def
            add = (beta_debt / 100.0) * ddev + (beta_deficit / 100.0) * fdev
        return sim, converged
    finally:
        frbus.exogenize([])


def _debt_level(frame, window):
    return (100.0 * frame.loc[window, "gfdbtnp"] / frame.loc[window, "xgdpn"]).to_numpy()


def _primary_deficit_level(frame, window):
    num = frame.loc[window, "gfsrpn"] + frame.loc[window, "gfintn"]
    return (-100.0 * num / frame.loc[window, "xgdpn"]).to_numpy()


def run_debt_scenario(
    deficit_pct: float,
    deficit_years: int,
    fiscal_rules: Sequence[str] = tuple(policy.FISCAL_RULES),
    policy_rule: str = "inertial",
    start: str = "2026Q3",
    horizon: int = DEFAULT_HORIZON,
    feedback: Tuple[float, float] = (0.0, 0.0),
) -> DebtResult:
    """Debt paths from a **sustained deficit shock of ``deficit_pct`` % of GDP**.

    The shock is a rise in federal transfers held for ``deficit_years`` years,
    implemented by exogenising ``gtrd`` (the transfers/GDP gap) so the impulse is an
    intuitive, near-exact share of GDP — not a hard-to-read add-factor magnitude. It
    is run under each fiscal closure rule (which sets how *taxes* respond), with the
    optional sovereign-risk feedback.
    """
    if horizon < 1 or deficit_years < 1:
        raise ValueError("horizon and deficit_years must be at least 1")
    if policy_rule not in policy.ACTIVE_RULES:
        raise ValueError(f"unknown policy_rule '{policy_rule}'")
    rules = [r for r in fiscal_rules if r in policy.FISCAL_RULES]
    if not rules:
        raise ValueError("no valid fiscal_rules given")

    beta_debt, beta_deficit = feedback
    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    window = pd.period_range(start_p, disp_end, freq="Q")
    n_shock = min(deficit_years * 4, horizon)

    # Baseline levels — from the shared (never-exogenised) cached model.
    base_data = _fiscal_baseline_config(
        load_baseline(), start_p, disp_end, "var", policy.DEFAULT_FISCAL_RULE
    )
    baseline = load_frbus("var").init_trac(start_p, disp_end, base_data)
    base_levels = pd.DataFrame(
        {key: _level(baseline, key).loc[window].values for key in DEBT_OUTPUT_KEYS},
        index=window,
    )

    deviations: Dict[str, pd.DataFrame] = {}
    converged: Dict[str, bool] = {}
    for rule in rules:
        # A fresh model per rule — exogenising then discarding avoids the corrupted
        # state that reusing an exogenised instance leaves for the next init_trac.
        frbus = fresh_frbus()
        data = _fiscal_baseline_config(load_baseline(), start_p, disp_end, "var", rule)
        if policy_rule != policy.DEFAULT_RULE:
            data = policy.set_active_rule(data, policy_rule, start_p, disp_end)
        adds = frbus.init_trac(start_p, disp_end, data)
        # Transfers path: baseline + deficit_pct (% of GDP) for the shock years.
        gtrd_path = adds.loc[start_p:disp_end, "gtrd"].copy()
        gtrd_path.iloc[:n_shock] = gtrd_path.iloc[:n_shock].values + deficit_pct / 100.0

        sim, conv = _solve_debt_shock(
            frbus, start_p, disp_end, adds, baseline, gtrd_path.values,
            beta_debt, beta_deficit,
        )
        converged[rule] = conv
        df = pd.DataFrame(index=window)
        for key in DEBT_OUTPUT_KEYS:
            df[key] = (_level(sim, key) - _level(baseline, key)).loc[window].values
        deviations[rule] = df

    return DebtResult(
        start=start_p, window=window, fiscal_rules=rules, policy_rule=policy_rule,
        requests=[], deviations=deviations, feedback=(beta_debt, beta_deficit),
        converged=converged, baseline_levels=base_levels,
    )
