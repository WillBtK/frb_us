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

import pandas as pd

# The policy-rule switches and funds-rate tracking residuals we touch.
_RULE_SWITCHES = ("dmpintay", "dmptay", "dmptlr", "dmpalt", "dmprr", "dmpex")
_FUNDS_TRACS = ("rff_trac", "rffrule_trac")


def apply_active_rule(data: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.DataFrame:
    """Return ``data`` unchanged — the baseline already runs the active rule.

    Present for symmetry/readability at the call site.
    """
    return data


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
