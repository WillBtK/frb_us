"""Stochastic debt fan charts — debt/GDP under macroeconomic uncertainty.

Wraps FRB/US's ``stochsim`` (block-bootstrap of the model's 52 estimated equation
residuals) to produce **debt-sustainability fan charts**: percentile bands around
the debt/GDP path that show how macroeconomic uncertainty propagates into the debt
trajectory — the standard IMF/CBO debt-sustainability visual.

The residual pool is the historical equation errors over a chosen window (default
1975Q1–2019Q4, excluding the COVID outliers, which otherwise dominate the draws).
Each replication is a full VAR solve (~0.7s), so a run of ``nrepl`` draws is on the
order of a minute or two; the app caches results and warns before running.

As with the deterministic Debt Sustainability tab, FRB/US does **not** endogenise a
sovereign-risk premium — the fan reflects real-economy and financial shocks, not a
debt-triggered blowout in the term premium. Add the term-premium lever to the
deterministic shock to layer that in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import policy
from .feedback import fresh_frbus, solve_with_feedback
from .model import load_baseline, load_frbus
from .simulate import _fiscal_baseline_config, _resolve_requests, _solve_robust

# Imported after model.py wires up sys.path for the vendored package.
from pyfrbus.stochsim import stochsim  # type: ignore  # noqa: E402

DEFAULT_HORIZON = 40  # quarters (10 years)
DEFAULT_NREPL = 100
DEFAULT_RESID = ("1975q1", "2019q4")  # historical residual pool (ex-COVID)
PERCENTILES = [10, 25, 50, 75, 90]  # 50% (25–75) and 80% (10–90) bands + median


@dataclass
class FanResult:
    start: pd.Period
    window: pd.PeriodIndex
    fiscal_rule: str
    policy_rule: str
    nrepl: int  # replications that converged and were used
    resid_window: tuple
    baseline_debt: np.ndarray  # no-shock deterministic debt/GDP level (%)
    central_debt: np.ndarray  # deterministic (shocked) central path (%)
    debt_pct: Dict[int, np.ndarray]  # percentile -> debt/GDP level path (%)
    central_primary: np.ndarray
    primary_pct: Dict[int, np.ndarray]
    prob_rising: float  # P(debt/GDP at horizon > its starting level)
    feedback: tuple = (0.0, 0.0)  # (beta_debt, beta_deficit) bps/pp
    feedback_converged: bool = True  # did the central-path feedback settle?
    requests: list = field(default_factory=list)


def _debt_gdp(frame: pd.DataFrame, window) -> np.ndarray:
    return (100.0 * frame.loc[window, "gfdbtnp"] / frame.loc[window, "xgdpn"]).to_numpy()


def _primary(frame: pd.DataFrame, window) -> np.ndarray:
    num = frame.loc[window, "gfsrpn"] + frame.loc[window, "gfintn"]
    return (100.0 * num / frame.loc[window, "xgdpn"]).to_numpy()


def debt_fan_chart(
    shocks: Sequence[dict] = (),
    fiscal_rule: str = "surplus_ratio",
    policy_rule: str = "inertial",
    start: str = "2026Q3",
    horizon: int = DEFAULT_HORIZON,
    nrepl: int = DEFAULT_NREPL,
    seed: int = 12345,
    resid_start: str = DEFAULT_RESID[0],
    resid_end: str = DEFAULT_RESID[1],
    feedback: tuple = (0.0, 0.0),
) -> FanResult:
    """Stochastic debt/GDP fan chart around a (optionally shocked) baseline.

    ``feedback`` = ``(beta_debt, beta_deficit)`` in bps/pp adds a sovereign-risk
    feedback: the term premium is set on the *deterministic central path* (a fixed
    point) and imposed across the draws — a central-path approximation that carries
    the debt-spiral amplification into the whole fan without a fixed point per draw.
    """
    if nrepl < 5:
        raise ValueError("nrepl must be at least 5")
    if fiscal_rule not in policy.FISCAL_RULES:
        raise ValueError(f"unknown fiscal_rule '{fiscal_rule}'")
    if policy_rule not in policy.ACTIVE_RULES:
        raise ValueError(f"unknown policy_rule '{policy_rule}'")

    beta_debt, beta_deficit = feedback
    use_feedback = beta_debt > 0 or beta_deficit > 0

    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    window = pd.period_range(start_p, disp_end, freq="Q")
    frbus = load_frbus("var")

    data = _fiscal_baseline_config(load_baseline(), start_p, disp_end, "var", fiscal_rule)
    if policy_rule != policy.DEFAULT_RULE:
        data = policy.set_active_rule(data, policy_rule, start_p, disp_end)
    # init_trac over the residual history + sim window so historical residuals exist.
    with_adds = frbus.init_trac(resid_start, disp_end, data)

    baseline_debt = _debt_gdp(with_adds, window)  # no-shock reference (tracked baseline)

    # Deterministic central path (with the shock, no stochastic draws).
    shocked = with_adds.copy()
    requests = _resolve_requests(shocks, None, None, None, None, None) if shocks else []
    for req in requests:
        shocked = req.spec.apply(shocked, req.magnitude, req.duration, start_p)

    fb_converged = True
    stoch_frbus = frbus
    if use_feedback:
        # Central-path feedback: iterate the term premium to a fixed point, then
        # impose that path (exogenised) across the stochastic draws.
        fb_frbus = fresh_frbus()
        central, _s0, fb_converged, _it, tp_path = solve_with_feedback(
            fb_frbus, start_p, disp_end, shocked, with_adds, beta_debt, beta_deficit
        )
        shocked = shocked.copy()
        shocked.loc[window, "rg10p"] = tp_path
        fb_frbus.exogenize(["rg10p"])
        stoch_frbus = fb_frbus
    else:
        central = _solve_robust(frbus, start_p, disp_end, shocked)

    # Stochastic replications (block-bootstrap of residuals over the sim window).
    nextra = max(5, nrepl // 8)
    sols = stochsim(
        stoch_frbus, nrepl, shocked, start_p, disp_end, resid_start, resid_end,
        multiproc=False, nextra=nextra, seed=seed, options=None,
    )
    if use_feedback:
        stoch_frbus.exogenize([])  # tidy (instance is discarded regardless)
    ok = [s for s in sols if not isinstance(s, str)][:nrepl]
    if len(ok) < 5:
        raise RuntimeError(f"only {len(ok)} replications converged; try fewer/smaller shocks")

    debt = np.array([_debt_gdp(s, window) for s in ok])       # n × H
    prim = np.array([_primary(s, window) for s in ok])
    debt_pct = {p: np.percentile(debt, p, axis=0) for p in PERCENTILES}
    primary_pct = {p: np.percentile(prim, p, axis=0) for p in PERCENTILES}

    prob_rising = float(np.mean(debt[:, -1] > baseline_debt[0]))

    return FanResult(
        start=start_p, window=window, fiscal_rule=fiscal_rule, policy_rule=policy_rule,
        nrepl=len(ok), resid_window=(resid_start, resid_end),
        baseline_debt=baseline_debt, central_debt=_debt_gdp(central, window),
        debt_pct=debt_pct, central_primary=_primary(central, window),
        primary_pct=primary_pct, prob_rising=prob_rising,
        feedback=(beta_debt, beta_deficit), feedback_converged=fb_converged,
        requests=requests,
    )
