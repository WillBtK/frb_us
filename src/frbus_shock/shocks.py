"""Catalogue of supported shocks and how each maps onto an FRB/US lever.

Every shock is applied to a model equation's **add factor** (the ``*_aerr``
series), which is the FRB/US-idiomatic way to perturb an equation without
touching its coefficients. The magnitude the *user* enters is in an intuitive
unit (%, percentage points, basis points); ``unit_to_model`` converts that to
the add factor's native unit.

The specific levers, their units, and the sign of their effect were confirmed
empirically against the vendored model (see ``tests/test_shocks.py``) rather
than assumed:

* fiscal spending  -> ``egfe_aerr``  (federal real purchases; log points)
* personal tax     -> ``trp_aerr``   (average personal tax *rate*; a fraction)
* oil / energy     -> ``poilr_aerr`` (relative oil price; log points)
* productivity     -> ``mfpt_aerr``  (trend multifactor productivity; log points)
* financial / term -> ``rbbbp_aerr`` (BBB corporate risk/term premium; pp)
* monetary (ref.)  -> ``rffintay_aerr`` (policy-rule residual; pp)
* custom           -> any ``<var>_aerr`` the user names
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class ShockSpec:
    """A named shock and everything needed to apply it to the dataset."""

    key: str
    label: str
    description: str
    column: str  # the ``*_aerr`` add-factor series to perturb
    user_unit: str  # human-facing unit for the magnitude field
    unit_to_model: float  # multiply the user magnitude by this to get model units
    default_magnitude: float
    default_duration: int  # quarters the add factor is held on
    sign_note: str

    def apply(
        self,
        data: pd.DataFrame,
        magnitude: float,
        duration: int,
        start: pd.Period,
    ) -> pd.DataFrame:
        """Return a copy of ``data`` with this shock applied in place of a view.

        The add factor is *added* to whatever tracking value ``init_trac`` put
        there, so the perturbation is measured relative to baseline. The shock
        is held on for ``duration`` quarters starting at ``start`` (a sustained
        level shift, not a single-quarter blip).
        """
        if duration < 1:
            raise ValueError("duration must be at least 1 quarter")
        out = data.copy()
        if self.column not in out.columns:
            raise KeyError(
                f"add-factor column '{self.column}' not found in dataset; "
                f"is '{self.key}' a valid shock for this model vintage?"
            )
        last = start + (duration - 1)
        out.loc[start:last, self.column] += magnitude * self.unit_to_model
        return out


# Percentage / percentage-point -> fractional or log-point add factor.
_PCT = 0.01

CATALOGUE: Dict[str, ShockSpec] = {
    "fiscal_spending": ShockSpec(
        key="fiscal_spending",
        label="Fiscal — federal spending",
        description=(
            "A change in real federal government purchases (EGFE), entered as a "
            "percent of that spending. Positive = fiscal expansion."
        ),
        column="egfe_aerr",
        user_unit="% of federal purchases",
        unit_to_model=_PCT,  # log point ~= fractional change
        default_magnitude=1.0,
        default_duration=8,
        sign_note="Higher spending raises GDP; the active rule leans against it.",
    ),
    "tax": ShockSpec(
        key="tax",
        label="Fiscal — personal tax rate",
        description=(
            "A change in the average personal income tax rate (TRP), entered in "
            "percentage points. Positive = tax increase (contractionary)."
        ),
        column="trp_aerr",
        user_unit="percentage points of the tax rate",
        unit_to_model=_PCT,  # TRP is a fraction, so 1pp -> 0.01
        default_magnitude=1.0,
        default_duration=8,
        sign_note=(
            "A tax hike lowers GDP. Kept modest by design — large, sustained "
            "tax-rate moves stress the model's fiscal-closure block."
        ),
    ),
    "oil": ShockSpec(
        key="oil",
        label="Oil / energy price",
        description=(
            "A change in the relative price of imported oil (POILR), entered as "
            "a percent. Positive = an oil-price spike (a supply shock)."
        ),
        column="poilr_aerr",
        user_unit="% change in the oil price",
        unit_to_model=_PCT,
        default_magnitude=10.0,
        default_duration=4,
        sign_note="Raises inflation and lowers GDP (stagflationary).",
    ),
    "productivity": ShockSpec(
        key="productivity",
        label="Productivity (trend MFP)",
        description=(
            "A change in the level of trend multifactor productivity (MFPT), "
            "entered as a percent. Positive = a favourable supply shock."
        ),
        column="mfpt_aerr",
        user_unit="% change in MFP level",
        unit_to_model=_PCT,
        default_magnitude=1.0,
        default_duration=1,
        sign_note="Raises GDP and lowers inflation; the rule eases.",
    ),
    "term_premium": ShockSpec(
        key="term_premium",
        label="Financial — corporate risk/term premium",
        description=(
            "A change in the BBB corporate bond risk/term premium (RBBBP), "
            "entered in basis points. Positive = tighter financial conditions."
        ),
        column="rbbbp_aerr",
        user_unit="basis points",
        unit_to_model=_PCT,  # 100 bp -> 1.0 pp
        default_magnitude=100.0,
        default_duration=4,
        sign_note="A wider premium tightens conditions and lowers GDP.",
    ),
    "monetary": ShockSpec(
        key="monetary",
        label="Monetary policy shock (reference)",
        description=(
            "A direct shock to the policy rule (RFFINTAY), entered in basis "
            "points. This is the shock used in the Fed's own demo/validation "
            "case. With the funds rate held, this shock has little meaning."
        ),
        column="rffintay_aerr",
        user_unit="basis points",
        unit_to_model=_PCT,
        default_magnitude=100.0,
        default_duration=1,
        sign_note="A surprise tightening lowers GDP (the example1 demo).",
    ),
}


def get_shock(key: str) -> ShockSpec:
    try:
        return CATALOGUE[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown shock '{key}'; choose one of {sorted(CATALOGUE)} or use "
            f"custom_shock()"
        ) from exc


def custom_shock(
    variable: str,
    magnitude: float,
    duration: int,
    label: Optional[str] = None,
) -> ShockSpec:
    """Build a shock spec for an arbitrary model variable's add factor.

    ``variable`` is the *endogenous variable name* (e.g. ``eco``); the shock is
    applied to ``<variable>_aerr`` in the variable's own native units
    (``unit_to_model == 1``).
    """
    var = variable.strip().lower()
    col = var if var.endswith("_aerr") else f"{var}_aerr"
    base = var[:-5] if var.endswith("_aerr") else var
    return ShockSpec(
        key=f"custom:{base}",
        label=label or f"Custom — {base}",
        description=f"Direct add-factor shock to '{base}' ({col}), in native model units.",
        column=col,
        user_unit="model units (native)",
        unit_to_model=1.0,
        default_magnitude=magnitude,
        default_duration=duration,
        sign_note="User-defined; interpret with care.",
    )


def with_defaults(spec: ShockSpec, magnitude: Optional[float], duration: Optional[int]):
    """Fill in a spec's default magnitude/duration when the caller omits them."""
    mag = spec.default_magnitude if magnitude is None else magnitude
    dur = spec.default_duration if duration is None else duration
    return replace(spec, default_magnitude=mag, default_duration=dur), mag, dur
