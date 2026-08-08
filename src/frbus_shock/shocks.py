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
    group: str = "Other"  # display grouping in the UI

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

def _spec(key, label, group, column, user_unit, unit_to_model, mag, dur, sign_note, desc):
    return ShockSpec(
        key=key, label=label, group=group, column=column, user_unit=user_unit,
        unit_to_model=unit_to_model, default_magnitude=mag, default_duration=dur,
        sign_note=sign_note, description=desc,
    )


# Friendly-named shock library. Keys are stable (used by tests / saved runs);
# labels are what the dashboard shows. Levers, units, and signs were all
# confirmed empirically against the model (see tests/test_shocks.py).
CATALOGUE: Dict[str, ShockSpec] = {s.key: s for s in [
    # --- Demand ---
    _spec("consumption", "Household consumption", "Demand", "eco_aerr",
          "% of that spending", _PCT, 1.0, 4,
          "A spending boost raises GDP and inflation.",
          "A change in real household spending on non-durables and services."),
    _spec("durables", "Consumer durables spending", "Demand", "ecd_aerr",
          "%", _PCT, 2.0, 4, "Higher durables demand raises GDP.",
          "A change in real consumer spending on durable goods."),
    _spec("housing", "Residential investment (housing)", "Demand", "eh_aerr",
          "%", _PCT, 2.0, 4, "More homebuilding raises GDP.",
          "A change in real residential investment."),
    _spec("business_investment", "Business investment", "Demand", "ebfi_aerr",
          "%", _PCT, 2.0, 4, "Stronger capex raises GDP.",
          "A change in real business fixed investment."),
    _spec("exports", "Exports", "Demand", "ex_aerr",
          "%", _PCT, 2.0, 4, "Stronger exports raise GDP.",
          "A change in real exports (e.g. stronger foreign demand)."),
    _spec("imports", "Imports (ex. oil)", "Demand", "emo_aerr",
          "%", _PCT, 2.0, 4, "Higher imports subtract from GDP.",
          "A change in real non-oil imports."),
    _spec("fiscal_spending", "Federal government spending", "Demand", "egfe_aerr",
          "% of federal purchases", _PCT, 1.0, 8,
          "Higher spending raises GDP; the active rule leans against it.",
          "A change in real federal government purchases. Positive = expansion."),
    # --- Prices & supply ---
    _spec("oil", "Oil / energy price", "Prices & supply", "poilr_aerr",
          "% change in the oil price", _PCT, 10.0, 4,
          "Raises inflation and lowers GDP (stagflationary).",
          "A change in the relative price of imported oil."),
    _spec("core_prices", "Core consumer prices", "Prices & supply", "picxfe_aerr",
          "pp (annualised inflation impulse)", 1.0, 0.5, 4,
          "A cost-push impulse: raises inflation, small drag on GDP.",
          "A direct impulse to core PCE inflation (a cost-push shock)."),
    _spec("import_prices", "Import prices (ex. oil)", "Prices & supply", "pmo_aerr",
          "%", _PCT, 5.0, 4, "Raises inflation; mild GDP effects.",
          "A change in the price of non-oil imports."),
    _spec("house_prices", "House prices", "Prices & supply", "phouse_aerr",
          "%", _PCT, 5.0, 4, "Higher house prices lift wealth and demand.",
          "A change in the house-price index."),
    _spec("productivity", "Productivity (trend MFP)", "Prices & supply", "mfpt_aerr",
          "% change in MFP level", _PCT, 1.0, 1,
          "Raises GDP and lowers inflation; the rule eases.",
          "A change in the level of trend multifactor productivity."),
    # --- Financial ---
    _spec("term_premium", "Corporate bond risk premium", "Financial", "rbbbp_aerr",
          "basis points", _PCT, 100.0, 4,
          "A wider premium tightens conditions and lowers GDP.",
          "A change in the BBB corporate bond risk/term premium."),
    _spec("term_premium_10y", "10-year Treasury term premium", "Financial", "rg10p_aerr",
          "basis points", _PCT, 100.0, 4, "Higher long rates lower GDP.",
          "A change in the 10-year Treasury term premium."),
    _spec("equity_premium", "Equity risk premium", "Financial", "reqp_aerr",
          "basis points", _PCT, 100.0, 4,
          "A higher premium lowers equity wealth and demand.",
          "A change in the required equity risk premium (an equity-price shock)."),
    _spec("mortgage_rate", "Mortgage rate", "Financial", "rme_aerr",
          "basis points", _PCT, 100.0, 4, "Dearer mortgages weigh on housing/GDP.",
          "A change in the conventional mortgage rate."),
    # --- External / global ---
    _spec("exchange_rate", "Real exchange rate (broad $)", "External / global", "fpxr_aerr",
          "% (up = stronger $)", _PCT, 5.0, 4,
          "A stronger dollar trims net exports and inflation.",
          "A direct change in the broad real exchange rate. Positive = appreciation."),
    _spec("global_growth", "Global growth (foreign output gap)", "External / global",
          "fxgap_aerr", "pp/quarter impulse", 1.0, 1.0, 4,
          "Stronger world demand lifts US exports and GDP; the dollar softens.",
          "An impulse to the foreign output gap (world, export-weighted). Positive = a "
          "global upswing — it raises US exports and GDP, improves the current account "
          "and weakens the dollar; negative = a global growth scare. The gap is "
          "persistent, so the impulse compounds: ~1pp of gap for a single-quarter hit, "
          "~4½pp held four quarters (foreign GDP moves about one-for-one with the gap)."),
    _spec("global_rates", "Global long-term interest rates", "External / global",
          "frl10_aerr", "basis points", _PCT, 50.0, 4,
          "Higher foreign yields weaken the dollar, lifting net exports and import prices.",
          "A change in the foreign (G10) 10-year interest rate. Higher foreign yields "
          "shrink the US–foreign rate differential, so the dollar depreciates — "
          "boosting net exports but raising import prices (imported inflation)."),
    _spec("foreign_inflation", "Foreign inflation (G10)", "External / global",
          "fpi10_aerr", "pp (annualised)", 1.0, 1.0, 4,
          "Transmits to the US mainly via a weaker dollar and dearer imports.",
          "A change in G10 foreign consumer-price inflation. It raises foreign policy "
          "rates and weakens the dollar, feeding through to US import prices; the "
          "direct effect on US GDP and PCE inflation is small (select the exchange-rate "
          "and import-price outputs to see the main channel)."),
    # --- Fiscal & monetary ---
    _spec("tax", "Personal tax rate", "Fiscal & monetary", "trp_aerr",
          "percentage points of the tax rate", _PCT, 1.0, 8,
          "A tax hike lowers GDP. Kept modest — large sustained moves stress "
          "the fiscal-closure block.",
          "A change in the average personal income tax rate. Positive = hike."),
    _spec("monetary", "Monetary policy shock (funds-rate rule)", "Fiscal & monetary",
          "rffintay_aerr", "basis points", _PCT, 100.0, 1,
          "A surprise tightening lowers GDP (the example1 demo). Meaningless "
          "when the funds rate is held.",
          "A direct shock to the policy rule. This is the Fed demo/validation case."),
]}


# Named multi-shock scenarios a cross-asset strategist reasons in — each is a
# list of {key, magnitude, duration} applied together. Signs/levers all come from
# the catalogue above (empirically checked); magnitudes are illustrative and fully
# editable once loaded into the dashboard.
SCENARIO_PRESETS: Dict[str, dict] = {
    "Global growth scare": {
        "blurb": "A worldwide demand downturn with risk-off financial conditions: "
        "weaker foreign growth drags US exports while credit and equity premia widen.",
        "shocks": [
            {"key": "global_growth", "magnitude": -1.0, "duration": 4},
            {"key": "term_premium", "magnitude": 75.0, "duration": 4},
            {"key": "equity_premium", "magnitude": 150.0, "duration": 4},
        ],
    },
    "Global inflation surprise": {
        "blurb": "A broad-based cost-push shock: an oil spike plus higher foreign "
        "inflation and a domestic core-price impulse — the stagflationary case.",
        "shocks": [
            {"key": "oil", "magnitude": 30.0, "duration": 4},
            {"key": "foreign_inflation", "magnitude": 1.5, "duration": 4},
            {"key": "core_prices", "magnitude": 0.5, "duration": 4},
        ],
    },
    "Dollar funding squeeze": {
        "blurb": "A flight-to-safety dollar spike with a sharp tightening of risk "
        "conditions: the broad dollar jumps as credit and equity premia blow out.",
        "shocks": [
            {"key": "exchange_rate", "magnitude": 6.0, "duration": 4},
            {"key": "term_premium", "magnitude": 100.0, "duration": 4},
            {"key": "equity_premium", "magnitude": 200.0, "duration": 4},
        ],
    },
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
        group="Advanced (raw variable)",
    )


def with_defaults(spec: ShockSpec, magnitude: Optional[float], duration: Optional[int]):
    """Fill in a spec's default magnitude/duration when the caller omits them."""
    mag = spec.default_magnitude if magnitude is None else magnitude
    dur = spec.default_duration if duration is None else duration
    return replace(spec, default_magnitude=mag, default_duration=dur), mag, dur
