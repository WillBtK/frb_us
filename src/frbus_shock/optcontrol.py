"""Linear-quadratic optimal-control monetary policy, à la the FEDS Note.

Given a shock, this finds the funds-rate path that minimises a quadratic loss in
the inflation and unemployment **deviations from baseline** (i.e. optimal
stabilisation of the shock), plus a rate-smoothing penalty::

    L = w_pi Σ_t π_dev_t² + w_u Σ_t u_dev_t² + w_smooth Σ_t (Δδ_t)²

It uses the standard linear-quadratic method (how LINVER / the Fed do optimal
control): the model's funds-rate impulse responses give a linear map from the
funds-rate path δ to the inflation/unemployment deviations, so the loss is
quadratic in δ and its minimiser has a closed form::

    δ* = −(MᵀM + w_smooth DᵀD)⁻¹ Mᵀ c

The **optimal** path is always computed. It is compared against a choice of
**comparator policy rules**, evaluated in the same linearised model (again, the
LINVER approach): each simple rule — inertial, balanced-approach, Taylor (with a
selectable output-gap coefficient), or first-difference — is a linear feedback on
the inflation and output-gap deviations, so its implied funds-rate path solves a
small linear system built from the same impulse-response matrices. "No response"
is just δ = 0 (the funds rate held at baseline). Every path is then re-solved
through the full nonlinear model for accurate outcomes; the linear-feedback rules
reproduce the model's own switch-based rules to a few hundredths of a pp.

Under **VAR** the impulse-response matrix is Toeplitz (one extra solve); under
**MCE**, anticipation matters, so it is built one column per control quarter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .model import load_baseline, load_frbus
from .simulate import (
    _MCE_MIN_SOLVE_QUARTERS,
    _MCE_TERMINAL_BUFFER,
    _fiscal_baseline_config,
    _resolve_requests,
    _solve_robust,
)

PI_VAR = "picnia"  # inflation measure stabilised (loss)
U_VAR = "lur"  # unemployment measure stabilised (loss)
RULE_PI = "picxfe"  # core PCE (q/q ann.); 4-quarter-averaged for the rules' inflation term
RULE_YG = "xgap2"  # output gap used by the rules

DEFAULT_WEIGHTS = {"inflation": 1.0, "unemployment": 1.0, "smoothing": 0.5}

# Comparator policy rules the optimum can be shown against. "held" = no response.
# Each simple rule is a linear feedback on the 4-quarter inflation deviation and
# the output-gap deviation (see ``_rule_delta``), matching the Fed's published
# rule set (Monetary Policy Report). Order is the display order.
RULE_LABELS: Dict[str, str] = {
    "held": "No response (funds rate held)",
    "inertial": "Inertial rule (Fed default)",
    "balanced_approach": "Balanced-approach rule",
    "taylor": "Taylor rule",
    "first_difference": "First-difference rule",
}
DEFAULT_COMPARATORS = ("inertial", "held")
DEFAULT_TAYLOR_COEF = 0.5  # classic Taylor puts 0.5 on the output gap (BA puts 1.0)


@dataclass
class OCPResult:
    start: pd.Period
    window: pd.PeriodIndex
    weights: Dict[str, float]
    expectations: str
    delta: np.ndarray  # optimal funds-rate deviation path (pp)
    comparators: List[str]  # comparator rule keys shown alongside the optimum
    taylor_coef: float
    # deviation-from-baseline paths (pp): cols {scen}_rff/_pi/_u for scen in
    # ("optimal", *comparators)
    paths: pd.DataFrame
    losses: Dict[str, float]  # keyed by "optimal" and each comparator
    approx_error: float  # max |linear prediction − nonlinear solve| over shown paths


def _set_funds_path(df, base, start, end, delta_win, window):
    """Return ``df`` with the funds rate pinned to baseline + delta over ``window``."""
    out = df.copy()
    out.loc[start:end, "dmpintay"] = 0
    out.loc[start:end, "dmpex"] = 1
    out.loc[start:end, "rff_trac"] = 0
    out.loc[start:end, "rffrule_trac"] = 0
    rff = base.loc[start:end, "rff"].copy()
    rff.loc[window] = base.loc[window, "rff"].values + delta_win
    out.loc[start:end, "rfffix"] = rff
    return out


def _avg4(H: int) -> np.ndarray:
    """H×H operator: 4-quarter trailing average (pre-window quarters treated as 0)."""
    A = np.zeros((H, H))
    for t in range(H):
        for k in range(4):
            if t - k >= 0:
                A[t, t - k] = 0.25
    return A


def _rule_delta(key, taylor_coef, g_pi4, A_pi4, g_xg, A_xg, H):
    """Funds-rate deviation path implied by a comparator rule (linear feedback).

    Every rule is written in deviation-from-baseline form (the baseline tracks its
    own rule, so there is no constant term). The nominal rate responds 1.5× to the
    inflation deviation (the ``+π`` level term plus the 0.5 gap coefficient) except
    the first-difference rule, which responds only through its 0.1 gap coefficient.
    """
    I = np.eye(H)
    Lag = np.eye(H, k=-1)  # x_{t-1}, x_{-1} = 0
    if key == "held":
        return np.zeros(H)
    if key == "balanced_approach":
        return np.linalg.solve(I - (1.5 * A_pi4 + 1.0 * A_xg), 1.5 * g_pi4 + 1.0 * g_xg)
    if key == "taylor":
        g = float(taylor_coef)
        return np.linalg.solve(I - (1.5 * A_pi4 + g * A_xg), 1.5 * g_pi4 + g * g_xg)
    if key == "inertial":
        L = 0.85
        return np.linalg.solve(
            I - L * Lag - (1 - L) * (1.5 * A_pi4 + 1.0 * A_xg),
            (1 - L) * (1.5 * g_pi4 + 1.0 * g_xg),
        )
    if key == "first_difference":
        D4 = I - np.eye(H, k=-4)  # y_t − y_{t−4}
        return np.linalg.solve(
            I - Lag - 0.1 * A_pi4 - 0.1 * (D4 @ A_xg),
            0.1 * g_pi4 + 0.1 * (D4 @ g_xg),
        )
    raise ValueError(f"unknown comparator rule '{key}'")


def _responses(frbus, start, solve_end, base, shocked, window, H, expectations, vars_):
    """Disturbance g_v and funds-rate impulse-response matrix A_v for each var.

    ``g_v`` is the deviation under the funds rate held (the disturbance); ``A_v`` is
    the response to a unit funds-rate impulse — Toeplitz under VAR (one solve), or
    built column by column under MCE (one solve per control quarter).
    """
    def dev(sim, v):
        return (sim.loc[window, v] - base.loc[window, v]).values

    zero = np.zeros(H)
    held = _solve_robust(
        frbus, start, solve_end, _set_funds_path(shocked, base, start, solve_end, zero, window)
    )
    g = {v: dev(held, v) for v in vars_}
    A = {v: np.zeros((H, H)) for v in vars_}
    if expectations == "var":
        imp = np.zeros(H); imp[0] = 1.0
        h = _solve_robust(
            frbus, start, solve_end, _set_funds_path(base.copy(), base, start, solve_end, imp, window)
        )
        hv = {v: dev(h, v) for v in vars_}
        for v in vars_:
            for t in range(H):
                for s in range(t + 1):
                    A[v][t, s] = hv[v][t - s]
    else:
        for s in range(H):
            imp = np.zeros(H); imp[s] = 1.0
            c = _solve_robust(
                frbus, start, solve_end,
                _set_funds_path(base.copy(), base, start, solve_end, imp, window),
            )
            for v in vars_:
                A[v][:, s] = dev(c, v)
    return g, A


def run_optimal_control(
    shocks: Sequence[dict],
    weights: Optional[Dict[str, float]] = None,
    expectations: str = "var",
    start: str = "2026Q3",
    horizon: int = 12,
    comparators: Sequence[str] = DEFAULT_COMPARATORS,
    taylor_coef: float = DEFAULT_TAYLOR_COEF,
) -> OCPResult:
    """Optimal funds-rate path for a shock, vs. selected comparator policy rules."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if expectations not in ("var", "mce"):
        raise ValueError("expectations must be 'var' or 'mce'")
    comparators = [c for c in comparators if c in RULE_LABELS]
    unknown = [c for c in comparators if c not in RULE_LABELS]
    if unknown:
        raise ValueError(f"unknown comparator rule(s): {unknown}")
    requests = _resolve_requests(shocks, None, None, None, None, None)

    start_p = pd.Period(start, freq="Q")
    disp_end = start_p + (horizon - 1)
    if expectations == "mce":
        solve_end = max(disp_end + _MCE_TERMINAL_BUFFER, start_p + _MCE_MIN_SOLVE_QUARTERS)
    else:
        solve_end = disp_end
    window = pd.period_range(start_p, disp_end, freq="Q")
    H = horizon

    data = _fiscal_baseline_config(load_baseline(), start_p, solve_end, expectations)
    frbus = load_frbus(expectations)
    base = frbus.init_trac(start_p, solve_end, data)

    shocked = base.copy()
    for r in requests:
        shocked = r.spec.apply(shocked, r.magnitude, r.duration, start_p)

    # Impulse responses for the loss variables (π, u) and the rule variables
    # (core inflation, output gap) — all from the same solves.
    g, A = _responses(
        frbus, start_p, solve_end, base, shocked, window, H, expectations,
        [PI_VAR, U_VAR, RULE_PI, RULE_YG],
    )
    g_pi, A_pi = g[PI_VAR], A[PI_VAR]
    g_u, A_u = g[U_VAR], A[U_VAR]
    avg4 = _avg4(H)
    g_pi4, A_pi4 = avg4 @ g[RULE_PI], avg4 @ A[RULE_PI]
    g_xg, A_xg = g[RULE_YG], A[RULE_YG]

    # Closed-form LQ optimum over the funds-rate path.
    wpi, wu, wd = w["inflation"], w["unemployment"], w["smoothing"]
    M = np.vstack([np.sqrt(wpi) * A_pi, np.sqrt(wu) * A_u])
    c = np.concatenate([np.sqrt(wpi) * g_pi, np.sqrt(wu) * g_u])
    D = np.eye(H) - np.eye(H, k=-1)
    delta_opt = -np.linalg.solve(M.T @ M + wd * (D.T @ D), M.T @ c)

    def _loss(pi, u, rff):
        dd = np.diff(np.concatenate([[0.0], rff]))
        return float(wpi * np.sum(pi ** 2) + wu * np.sum(u ** 2) + wd * np.sum(dd ** 2))

    def _nonlinear(delta):
        sim = _solve_robust(
            frbus, start_p, solve_end,
            _set_funds_path(shocked, base, start_p, solve_end, delta, window),
        )
        return (
            (sim.loc[window, "rff"] - base.loc[window, "rff"]).values,
            (sim.loc[window, PI_VAR] - base.loc[window, PI_VAR]).values,
            (sim.loc[window, U_VAR] - base.loc[window, U_VAR]).values,
        )

    cols: Dict[str, np.ndarray] = {}
    losses: Dict[str, float] = {}
    approx = 0.0

    # Optimal (always) — nonlinear-verified, with its linear-approximation error.
    o_rff, o_pi, o_u = _nonlinear(delta_opt)
    cols["optimal_rff"], cols["optimal_pi"], cols["optimal_u"] = o_rff, o_pi, o_u
    losses["optimal"] = _loss(o_pi, o_u, o_rff)
    approx = max(approx, float(np.max(np.abs((g_pi + A_pi @ delta_opt) - o_pi))),
                 float(np.max(np.abs((g_u + A_u @ delta_opt) - o_u))))

    # Comparator rules.
    for key in comparators:
        delta = _rule_delta(key, taylor_coef, g_pi4, A_pi4, g_xg, A_xg, H)
        if key == "held":
            r_rff, r_pi, r_u = np.zeros(H), g_pi, g_u
        else:
            r_rff, r_pi, r_u = _nonlinear(delta)
            approx = max(approx, float(np.max(np.abs((g_pi + A_pi @ delta) - r_pi))),
                         float(np.max(np.abs((g_u + A_u @ delta) - r_u))))
        cols[f"{key}_rff"], cols[f"{key}_pi"], cols[f"{key}_u"] = r_rff, r_pi, r_u
        losses[key] = _loss(r_pi, r_u, r_rff)

    paths = pd.DataFrame(cols, index=window)
    return OCPResult(
        start=start_p, window=window, weights=w, expectations=expectations,
        delta=delta_opt, comparators=list(comparators), taylor_coef=float(taylor_coef),
        paths=paths, losses=losses, approx_error=float(approx),
    )
