"""Monetary-policy configuration: active rule vs. funds rate held at baseline.

This is the crux of the with-response / without-response comparison, so the
mechanism is documented here in full.

FRB/US chooses the funds rate through a *switch-weighted* rule. In the model
XML::

    rffrule = dmpex*rfffix + dmprr*(...) + dmptay*rfftay
              + dmptlr*rfftlr + dmpintay*rffintay + dmpalt*rffalt
    rff     = (1-dmptrsh)*max(rffrule, rffmin) + dmptrsh*(...)

Each ``dmp*`` is a 0/1 switch selecting one rule. In the shipped LONGBASE
baseline **``dmpintay = 1``** (the inertial Taylor rule) and every other switch
is 0, so the funds rate follows the inertial Taylor rule and reacts to the
shock. That is the **active-rule** case — we change nothing.

To **hold the funds rate at its baseline path** (no monetary response) we use
the model's own exogenous-rate switch ``dmpex`` rather than a bespoke hack:

1. ``dmpintay = 0`` and ``dmpex = 1`` over the simulation window, so
   ``rffrule = rfffix`` (the fixed, pre-determined funds-rate path).
2. ``rfffix`` is set to the *baseline* ``rff`` path.
3. The funds-rate tracking add factors ``rff_trac`` and ``rffrule_trac``
   (created by ``init_trac`` so the baseline reproduces exactly) are zeroed over
   the window, otherwise they would re-inject the baseline rule's residual as a
   spurious offset once the rule is switched off.

The result: ``rff`` is pinned to the baseline path (subject to the ``rffmin``
ZLB floor) no matter what the shock does to inflation or the output gap. This
was verified to hold the funds-rate deviation at exactly 0.000 bp under a demand
shock while the active rule moved it by tens of basis points
(``tests/test_shocks.py::test_funds_rate_hold``).

An equivalent alternative is ``frbus.exogenize(["rff"])``; we prefer the
``dmpex`` switch because it is the model-intrinsic mechanism and preserves the
``max(·, rffmin)`` ZLB structure.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

# The policy-rule switches and funds-rate tracking residuals we touch.
_RULE_SWITCHES = ("dmpintay", "dmptay", "dmptlr", "dmpalt", "dmprr", "dmpex")
_FUNDS_TRACS = ("rff_trac", "rffrule_trac")


# Selectable monetary-policy rules for the *active* ("with response") scenario.
# Each is one of FRB/US's own reaction-function switches, so every scenario is an
# exact, endogenous nonlinear solve (the rule reacts inside the solver). Values:
# (switch, display label, explanatory tooltip). Order is the dropdown order.
ACTIVE_RULES: Dict[str, Tuple[str, str, str]] = {
    "inertial": (
        "dmpintay", "Inertial (Taylor) rule",
        "FRB/US's default and the closest to observed Fed behaviour. The funds rate "
        "moves only partway to its prescription each quarter (0.85 weight on last "
        "quarter's rate), where the prescription responds 0.5 to the inflation gap "
        "and 1.0 to the output gap. Gradual and history-dependent.",
    ),
    "balanced_approach": (
        "dmptay", "Balanced-approach rule",
        "The dual-mandate rule with no inertia: it responds 0.5 to the inflation gap "
        "and 1.0 to the output gap, adjusting fully each quarter. More aggressive and "
        "front-loaded than the inertial rule — this is the Fed's balanced-approach rule.",
    ),
    "unemployment_gap": (
        "dmptlr", "Taylor rule (unemployment gap)",
        "Like the balanced-approach rule but measures economic slack with the "
        "unemployment gap (u* − u) instead of the output gap — FRB/US's estimated "
        "unemployment-based reaction function.",
    ),
    "estimated_ma": (
        "dmpalt", "Estimated historical rule",
        "An empirically estimated reaction function fit to the historical funds rate, "
        "with its own inertia (two rate lags) and responses to the output gap and core "
        "inflation. A descriptive 'as the Fed has actually behaved' benchmark rather "
        "than a prescriptive rule.",
    ),
}
DEFAULT_RULE = "inertial"


# Selectable *fiscal* closure rules — how the government stabilises the budget and
# debt. Each sets FRB/US's fiscal-policy switches (dfpsrp / dfpdbt / dfpex). This is
# the core lever for debt-sustainability analysis: it decides whether, and how,
# debt returns to its baseline path after a shock. Values: (switch settings, label,
# tooltip). Order is the dropdown order.
FISCAL_RULES: Dict[str, Tuple[Dict[str, int], str, str]] = {
    "exogenous_taxes": (
        {"dfpsrp": 0, "dfpdbt": 0, "dfpex": 1}, "No fiscal response (deficit-financed)",
        "Tax rates do NOT respond to the deficit or the debt — the government simply "
        "borrows to cover the shock, so debt/GDP can climb without returning. This is "
        "the 'is the path sustainable?' benchmark.",
    ),
    "surplus_ratio": (
        {"dfpsrp": 1, "dfpdbt": 0, "dfpex": 0}, "Gradual correction (surplus target)",
        "FRB/US's default. Tax rates adjust slowly to nudge the budget surplus toward a "
        "share-of-GDP target, so debt is stabilised — but only gradually.",
    ),
    "debt_stabilization": (
        {"dfpsrp": 0, "dfpdbt": 1, "dfpex": 0}, "Active stabilisation (debt target)",
        "Tax rates adjust to pull the debt-to-GDP ratio back to its baseline path — the "
        "government leans directly and firmly against debt (debt/GDP can briefly "
        "overshoot below baseline as taxes rise).",
    ),
}
DEFAULT_FISCAL_RULE = "surplus_ratio"


def apply_active_rule(data: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.DataFrame:
    """Return ``data`` unchanged — the baseline already runs the inertial rule.

    Present for symmetry/readability at the call site (the ``inertial`` default).
    """
    return data


def set_fiscal_rule(
    data: pd.DataFrame, rule_key: str, start: pd.Period, end: pd.Period
) -> pd.DataFrame:
    """Return ``data`` with the fiscal-policy switches set to the named closure rule.

    Selects one of :data:`FISCAL_RULES` by writing its ``dfp*`` switches over the
    window. The caller then ``init_trac``\\ s under this configuration so the rule
    reproduces the baseline exactly before the shock is applied.
    """
    if rule_key not in FISCAL_RULES:
        raise KeyError(
            f"unknown fiscal rule '{rule_key}'; choose one of {list(FISCAL_RULES)}"
        )
    out = data.copy()
    for switch, value in FISCAL_RULES[rule_key][0].items():
        if switch in out.columns:
            out.loc[start:end, switch] = value
    return out


def set_active_rule(
    data: pd.DataFrame, rule_key: str, start: pd.Period, end: pd.Period
) -> pd.DataFrame:
    """Return ``data`` with the policy-rule switches set to the named rule.

    Selects one of :data:`ACTIVE_RULES` by turning its ``dmp*`` switch on and every
    other rule switch off over the window. The caller then ``init_trac``\\ s under
    this configuration so the chosen rule reproduces the baseline exactly (zero
    no-shock deviation) before the shock is applied.
    """
    if rule_key not in ACTIVE_RULES:
        raise KeyError(
            f"unknown policy rule '{rule_key}'; choose one of {list(ACTIVE_RULES)}"
        )
    switch = ACTIVE_RULES[rule_key][0]
    out = data.copy()
    for s in _RULE_SWITCHES:
        if s in out.columns:
            out.loc[start:end, s] = 0
    out.loc[start:end, switch] = 1
    return out


def apply_funds_rate_hold(
    data: pd.DataFrame,
    baseline: pd.DataFrame,
    start: pd.Period,
    end: pd.Period,
) -> pd.DataFrame:
    """Configure ``data`` so the funds rate is held at its baseline path.

    Parameters
    ----------
    data:
        The dataset to modify (already carrying ``init_trac`` add factors).
    baseline:
        The baseline-with-adds dataset, used to read the baseline ``rff`` path
        that the funds rate should be pinned to.
    start, end:
        Simulation window (inclusive).
    """
    out = data.copy()
    out.loc[start:end, "dmpintay"] = 0
    out.loc[start:end, "dmpex"] = 1
    out.loc[start:end, "rfffix"] = baseline.loc[start:end, "rff"]
    for trac in _FUNDS_TRACS:
        if trac in out.columns:
            out.loc[start:end, trac] = 0
    return out


# Public constant so the UI / docs can name the mechanism.
HOLD_MECHANISM = (
    "dmpex=1, dmpintay=0, rfffix=baseline rff, rff_trac=rffrule_trac=0"
)
