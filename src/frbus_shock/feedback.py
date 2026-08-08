"""Sovereign-risk feedback: debt / deficit → bond yields → debt dynamics.

FRB/US does not endogenise a sovereign-risk premium — its term premium does not
respond to the debt or the deficit. This module bolts that channel on as a
**calibrated feedback overlay**: the 10-year term premium is set to its
no-feedback value *plus* a premium proportional to the debt/GDP and primary-deficit
deviations, and the model is re-solved to a **fixed point** (rising rates → higher
debt service → higher debt → higher rates …).

The elasticities are the standard empirical ones (bps of the 10-year rate per
percentage point):

* **CBO / Gamber–Seliski (2019)** — ≈2 bps per 1pp of debt/GDP (the literature-review
  central estimate CBO uses in its long-term projections). Debt-level channel only.
* **Engen–Hubbard (2004)** — ≈3 bps per 1pp of debt/GDP.
* **Laubach (2009)** — ≈3–4 bps per 1pp of projected debt/GDP *and* ≈25 bps per 1pp
  of the deficit/GDP (the deficit/expectations channel is the larger one).

The deficit channel is keyed to the **primary** deficit (budget balance ex-interest)
so it does not double-count the interest-cost spiral already carried by the debt
channel and FRB/US's own interest dynamics.

Mechanism: the term premium ``rg10p`` is *exogenised* (on a fresh, un-cached model
so the shared model is untouched) at ``no-feedback path + feedback``, and the fixed
point is iterated. Non-convergence — the premium and debt chasing each other upward
without settling — is reported as an **unstable debt spiral**, which is itself the
signal of interest.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .model import MODEL_XML
from .simulate import _solve_robust

# Imported after model.py wires up sys.path for the vendored package.
from pyfrbus.frbus import Frbus  # type: ignore  # noqa: E402

# key -> (beta_debt bps/pp, beta_deficit bps/pp, label, tooltip)
FEEDBACK_PRESETS: Dict[str, Tuple[float, float, str, str]] = {
    "off": (0.0, 0.0, "Off (no feedback)",
            "Debt does not feed back into interest rates — the FRB/US default."),
    "cbo": (2.0, 0.0, "CBO / Gamber–Seliski (2019)",
            "≈2 bps on the 10-year rate per 1pp of debt/GDP — CBO's literature-review "
            "central estimate, used in its long-term projections. Debt-level channel."),
    "engen_hubbard": (3.0, 0.0, "Engen–Hubbard (2004)",
                      "≈3 bps per 1pp of debt/GDP. Debt-level channel."),
    "laubach": (3.5, 25.0, "Laubach (2009)",
                "≈3.5 bps per 1pp of debt/GDP plus ≈25 bps per 1pp of the (primary) "
                "deficit/GDP — the deficit/expectations channel is the larger one."),
}
DEFAULT_FEEDBACK = "cbo"


def fresh_frbus() -> Frbus:
    """A dedicated (un-cached) VAR model instance — safe to ``exogenize`` in place."""
    return Frbus(str(MODEL_XML), mce=False)


def _debt_gdp(frame: pd.DataFrame, window) -> np.ndarray:
    return (100.0 * frame.loc[window, "gfdbtnp"] / frame.loc[window, "xgdpn"]).to_numpy()


def _primary_deficit_gdp(frame: pd.DataFrame, window) -> np.ndarray:
    # Primary deficit = −(budget balance + net interest) / GDP; positive = deficit.
    num = frame.loc[window, "gfsrpn"] + frame.loc[window, "gfintn"]
    return (-100.0 * num / frame.loc[window, "xgdpn"]).to_numpy()


def solve_with_feedback(
    frbus: Frbus,
    start_p: pd.Period,
    end_p: pd.Period,
    shocked: pd.DataFrame,
    baseline: pd.DataFrame,
    beta_debt_bps: float,
    beta_deficit_bps: float,
    max_iter: int = 6,
    tol: float = 0.03,
    blowup: float = 1000.0,
):
    """Solve ``shocked`` with a debt/deficit → term-premium feedback (fixed point).

    ``frbus`` must be a dedicated instance (see :func:`fresh_frbus`) — its ``rg10p``
    equation is exogenised during the iteration and restored on exit. ``baseline`` is
    the no-shock reference for the debt/deficit deviations. Returns
    ``(sim, sim_no_feedback, converged, iters, term_premium_path)``.
    """
    window = pd.period_range(start_p, end_p, freq="Q")
    base_debt = _debt_gdp(baseline, window)
    base_def = _primary_deficit_gdp(baseline, window)

    # No-feedback solve — its term premium (incl. any direct term-premium shock) is
    # the path the feedback adds on top of.
    sim0 = _solve_robust(frbus, start_p, end_p, shocked)
    base_tp = sim0.loc[window, "rg10p"].to_numpy()

    frbus.exogenize(["rg10p"])
    try:
        used_add = np.zeros(len(window))
        prev = None
        converged = False
        sim = sim0
        iters = 0
        for iters in range(1, max_iter + 1):
            data = shocked.copy()
            data.loc[start_p:end_p, "rg10p"] = base_tp + used_add
            sim = _solve_robust(frbus, start_p, end_p, data)
            ddev = _debt_gdp(sim, window) - base_debt
            if not np.all(np.isfinite(ddev)) or np.max(np.abs(ddev)) > blowup:
                converged = False
                break
            if prev is not None and np.max(np.abs(ddev - prev)) < tol:
                converged = True
                break
            prev = ddev
            fdev = _primary_deficit_gdp(sim, window) - base_def
            used_add = (beta_debt_bps / 100.0) * ddev + (beta_deficit_bps / 100.0) * fdev
        return sim, sim0, converged, iters, base_tp + used_add
    finally:
        frbus.exogenize([])
