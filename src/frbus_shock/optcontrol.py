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

where M stacks the (weighted) impulse-response matrices and c the (weighted)
"disturbance" — the deviations under the shock with the funds rate held.

Under **VAR** expectations the dynamics are shift-invariant, so the
impulse-response matrix is Toeplitz and needs a single extra solve. Under
**MCE**, anticipation of future policy matters, so the full matrix is built with
one solve per control quarter (slower). The result is a *linear approximation*;
its accuracy is checked by re-solving the nonlinear model along δ* (the error is
a few hundredths of a pp for moderate shocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import policy
from .model import load_baseline, load_frbus
from .simulate import (
    _MCE_MIN_SOLVE_QUARTERS,
    _MCE_TERMINAL_BUFFER,
    _fiscal_baseline_config,
    _resolve_requests,
    _solve_robust,
)

PI_VAR = "picnia"  # inflation measure stabilised
U_VAR = "lur"  # unemployment measure stabilised

DEFAULT_WEIGHTS = {"inflation": 1.0, "unemployment": 1.0, "smoothing": 0.5}


@dataclass
class OCPResult:
    start: pd.Period
    window: pd.PeriodIndex
    weights: Dict[str, float]
    expectations: str
    delta: np.ndarray  # optimal funds-rate deviation path (pp)
    # deviation-from-baseline paths (pp) for each scenario
    paths: pd.DataFrame  # cols: {scen}_rff/_pi/_u for scen in optimal/taylor/held
    losses: Dict[str, float]
    approx_error: float  # max |linear prediction − nonlinear solve| along δ*


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


def _dev(sim, base, window):
    return (
        (sim.loc[window, PI_VAR] - base.loc[window, PI_VAR]).values,
        (sim.loc[window, U_VAR] - base.loc[window, U_VAR]).values,
    )


def run_optimal_control(
    shocks: Sequence[dict],
    weights: Optional[Dict[str, float]] = None,
    expectations: str = "var",
    start: str = "2026Q3",
    horizon: int = 12,
) -> OCPResult:
    """Compute the loss-minimising funds-rate path for a shock, vs Taylor / held."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if expectations not in ("var", "mce"):
        raise ValueError("expectations must be 'var' or 'mce'")
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

    # Shocked dataset (shocks combine additively).
    shocked = base.copy()
    for r in requests:
        shocked = r.spec.apply(shocked, r.magnitude, r.duration, start_p)

    # Disturbance g: inflation/unemployment deviations under the funds rate held.
    zero = np.zeros(H)
    g_pi, g_u = _dev(
        _solve_robust(frbus, start_p, solve_end, _set_funds_path(shocked, base, start_p, solve_end, zero, window)),
        base, window,
    )

    # Funds-rate impulse-response matrices A_pi, A_u (H x H).
    A_pi = np.zeros((H, H))
    A_u = np.zeros((H, H))
    if expectations == "var":
        # Shift-invariant: one impulse at q0, then Toeplitz.
        imp = np.zeros(H); imp[0] = 1.0
        h_pi, h_u = _dev(
            _solve_robust(frbus, start_p, solve_end, _set_funds_path(base.copy(), base, start_p, solve_end, imp, window)),
            base, window,
        )
        for t in range(H):
            for s in range(t + 1):
                A_pi[t, s] = h_pi[t - s]
                A_u[t, s] = h_u[t - s]
    else:
        # MCE: one solve per control quarter (anticipation → not shift-invariant).
        for s in range(H):
            imp = np.zeros(H); imp[s] = 1.0
            c_pi, c_u = _dev(
                _solve_robust(frbus, start_p, solve_end, _set_funds_path(base.copy(), base, start_p, solve_end, imp, window)),
                base, window,
            )
            A_pi[:, s] = c_pi
            A_u[:, s] = c_u

    # Closed-form LQ minimiser.
    wpi, wu, wd = w["inflation"], w["unemployment"], w["smoothing"]
    M = np.vstack([np.sqrt(wpi) * A_pi, np.sqrt(wu) * A_u])
    c = np.concatenate([np.sqrt(wpi) * g_pi, np.sqrt(wu) * g_u])
    D = np.eye(H) - np.eye(H, k=-1)  # first difference
    delta = -np.linalg.solve(M.T @ M + wd * (D.T @ D), M.T @ c)

    # Optimal path — nonlinear solve along δ* (accurate) + prediction error.
    opt_sim = _solve_robust(frbus, start_p, solve_end, _set_funds_path(shocked, base, start_p, solve_end, delta, window))
    o_pi, o_u = _dev(opt_sim, base, window)
    pred_pi, pred_u = g_pi + A_pi @ delta, g_u + A_u @ delta
    approx_error = float(max(np.max(np.abs(pred_pi - o_pi)), np.max(np.abs(pred_u - o_u))))
    o_rff = (opt_sim.loc[window, "rff"] - base.loc[window, "rff"]).values

    # Taylor rule (the baseline active rule).
    tay = _solve_robust(frbus, start_p, solve_end, shocked)
    t_pi, t_u = _dev(tay, base, window)
    t_rff = (tay.loc[window, "rff"] - base.loc[window, "rff"]).values

    def _loss(pi, u, rff):
        dd = np.diff(np.concatenate([[0.0], rff]))
        return float(wpi * np.sum(pi ** 2) + wu * np.sum(u ** 2) + wd * np.sum(dd ** 2))

    paths = pd.DataFrame(
        {
            "optimal_rff": o_rff, "optimal_pi": o_pi, "optimal_u": o_u,
            "taylor_rff": t_rff, "taylor_pi": t_pi, "taylor_u": t_u,
            "held_rff": zero, "held_pi": g_pi, "held_u": g_u,
        },
        index=window,
    )
    losses = {
        "optimal": _loss(o_pi, o_u, o_rff),
        "taylor": _loss(t_pi, t_u, t_rff),
        "held": _loss(g_pi, g_u, zero),
    }
    return OCPResult(start_p, window, w, expectations, delta, paths, losses, approx_error)
