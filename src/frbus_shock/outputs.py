"""Turn a :class:`SimResult` into deviation-from-baseline paths and exports.

Outputs are drawn from a grouped catalogue of FRB/US variables (all confirmed
against ``model.xml``). Each variable carries a **transform** that determines how
its deviation from baseline is expressed:

* ``diff`` — the variable is already a rate / percent (growth rates, inflation
  rates, unemployment, interest rates, the output gap), so the deviation is a
  simple difference ``sim - baseline`` in **percentage points (pp)**.
* ``pct`` — the variable is a level (real GDP, consumption, investment), so the
  deviation is a **percent** change ``100 * (sim / baseline - 1)``.

The four headline defaults (GDP growth, unemployment, PCE inflation, funds rate)
are unchanged; the rest are opt-in via the dashboard's output selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .simulate import SimResult


@dataclass(frozen=True)
class OutputVar:
    """One selectable output variable.

    Most variables map directly onto a single FRB/US series (``key``). Some are
    **derived** — a ratio (e.g. current account as a share of GDP) or a spread
    (e.g. the BBB–Treasury credit spread) computed from several series. For those,
    ``derive`` is a function of a solved DataFrame returning the level series; the
    ``transform`` is then applied to that derived level exactly as for a raw one.
    ``unit_override`` lets a derived ``diff`` variable label its unit precisely
    (e.g. "pp of GDP").
    """

    key: str  # stable identifier (a FRB/US variable name, or a derived-series key)
    label: str
    group: str
    transform: str  # "diff" (-> pp) or "pct" (-> %)
    derive: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    unit_override: Optional[str] = None

    @property
    def unit(self) -> str:
        if self.unit_override is not None:
            return self.unit_override
        return "pp" if self.transform == "diff" else "%"

    def level(self, frame: pd.DataFrame) -> pd.Series:
        """The variable's level series in one solved scenario/baseline frame."""
        if self.derive is not None:
            return self.derive(frame)
        return frame[self.key]


# --- Derived-series builders (ratios and spreads computed from a solved frame) ---
def _share_of_gdp(numerator: str) -> Callable[[pd.DataFrame], pd.Series]:
    """A current-$ flow/stock as a percent of nominal GDP."""
    return lambda f: 100.0 * f[numerator] / f["xgdpn"]


def _net_exports_share() -> Callable[[pd.DataFrame], pd.Series]:
    return lambda f: 100.0 * (f["exn"] - f["emn"]) / f["xgdpn"]


def _spread(a: str, b: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda f: f[a] - f[b]


# Ordered catalogue, grouped. Order here is the display order within each group.
OUTPUT_CATALOGUE: List[OutputVar] = [
    # --- Activity & spending ---
    OutputVar("hggdp", "GDP growth (annual rate)", "Activity & spending", "diff"),
    OutputVar("xgap2", "Output gap", "Activity & spending", "diff"),
    OutputVar("lur", "Unemployment rate", "Activity & spending", "diff"),
    OutputVar("lfpr", "Labour force participation rate", "Activity & spending", "diff"),
    OutputVar("leh", "Civilian employment", "Activity & spending", "pct"),
    OutputVar("xgdp", "Real GDP", "Activity & spending", "pct"),
    OutputVar("xgdpn", "Nominal GDP", "Activity & spending", "pct"),
    OutputVar("ecnia", "Household consumption (PCE)", "Activity & spending", "pct"),
    OutputVar("ecd", "Consumer durables spending", "Activity & spending", "pct"),
    OutputVar("eh", "Residential investment (housing)", "Activity & spending", "pct"),
    OutputVar("ebfi", "Business fixed investment", "Activity & spending", "pct"),
    OutputVar("ex", "Exports", "Activity & spending", "pct"),
    OutputVar("emo", "Imports (ex. oil)", "Activity & spending", "pct"),
    OutputVar("yh", "Real household income (after tax)", "Activity & spending", "pct"),
    # --- Inflation & wages ---
    OutputVar("picnia", "PCE inflation (q/q annualised)", "Inflation & wages", "diff"),
    OutputVar("picxfe", "Core PCE inflation", "Inflation & wages", "diff"),
    OutputVar("pic4", "PCE inflation (4-quarter)", "Inflation & wages", "diff"),
    OutputVar("picx4", "Core PCE inflation (4-quarter)", "Inflation & wages", "diff"),
    OutputVar("pcpi", "CPI inflation", "Inflation & wages", "diff"),
    OutputVar("pcpix", "Core CPI inflation", "Inflation & wages", "diff"),
    OutputVar("pigdp", "GDP-deflator inflation", "Inflation & wages", "diff"),
    OutputVar("pieci", "Compensation growth (ECI)", "Inflation & wages", "diff"),
    # --- Interest rates ---
    OutputVar("rff", "Federal funds rate", "Interest rates", "diff"),
    OutputVar("rrff", "Real federal funds rate", "Interest rates", "diff"),
    OutputVar("rg5", "5-year Treasury yield", "Interest rates", "diff"),
    OutputVar("rg10", "10-year Treasury yield", "Interest rates", "diff"),
    OutputVar("rg30", "30-year Treasury yield", "Interest rates", "diff"),
    OutputVar("rbbb", "BBB corporate bond yield", "Interest rates", "diff"),
    OutputVar("rg10p", "10-year term premium", "Interest rates", "diff"),
    OutputVar("rg30p", "30-year term premium", "Interest rates", "diff"),
    OutputVar("rme", "Mortgage rate (30-year)", "Interest rates", "diff"),
    OutputVar("rcar", "New-car loan rate", "Interest rates", "diff"),
    # Derived spreads — what curve / credit strategists actually watch.
    OutputVar("slope_10y_ff", "Yield-curve slope (10y − funds)", "Interest rates",
              "diff", derive=_spread("rg10", "rff")),
    OutputVar("slope_10y_5y", "Yield-curve slope (10y − 5y)", "Interest rates",
              "diff", derive=_spread("rg10", "rg5")),
    OutputVar("spread_bbb_10y", "Credit spread (BBB − 10y Treasury)", "Interest rates",
              "diff", derive=_spread("rbbb", "rg10")),
    # --- Financial & external ---
    OutputVar("req", "Real equity return (expected)", "Financial & external", "diff"),
    OutputVar("reqp", "Equity risk premium", "Financial & external", "diff"),
    OutputVar("fpxr", "Real exchange rate (broad; up = stronger $)", "Financial & external", "pct"),
    OutputVar("fpx", "Nominal exchange rate (broad)", "Financial & external", "pct"),
    OutputVar("phouse", "House prices", "Financial & external", "pct"),
    OutputVar("fgdp", "Foreign real GDP (world)", "Financial & external", "pct"),
    # External balances (current-$ flows/stocks as a share of nominal GDP).
    OutputVar("cab_gdp", "Current-account balance (% of GDP)", "Financial & external",
              "diff", derive=_share_of_gdp("fcbn"), unit_override="pp of GDP"),
    OutputVar("nx_gdp", "Trade balance (% of GDP)", "Financial & external",
              "diff", derive=_net_exports_share(), unit_override="pp of GDP"),
    OutputVar("niip_gdp", "Net international investment position (% of GDP)",
              "Financial & external", "diff", derive=_share_of_gdp("fnin"),
              unit_override="pp of GDP"),
    OutputVar("netii_gdp", "Net investment income (% of GDP)", "Financial & external",
              "diff", derive=_share_of_gdp("fynin"), unit_override="pp of GDP"),
    # --- Government ---
    OutputVar("gfdbtn", "Federal debt (stock)", "Government", "pct"),
    OutputVar("debt_gdp", "Federal debt held by public (% of GDP)", "Government",
              "diff", derive=_share_of_gdp("gfdbtnp"), unit_override="pp of GDP"),
    OutputVar("budget_gdp", "Federal budget balance (% of GDP)", "Government",
              "diff", derive=_share_of_gdp("gfsrpn"), unit_override="pp of GDP"),
    OutputVar("primary_gdp", "Primary balance (% of GDP)", "Government", "diff",
              derive=lambda f: 100.0 * (f["gfsrpn"] + f["gfintn"]) / f["xgdpn"],
              unit_override="pp of GDP"),
    OutputVar("interest_gdp", "Net interest / debt service (% of GDP)", "Government",
              "diff", derive=_share_of_gdp("gfintn"), unit_override="pp of GDP"),
]

OUTPUT_BY_KEY: Dict[str, OutputVar] = {v.key: v for v in OUTPUT_CATALOGUE}

# Group display order.
OUTPUT_GROUPS: List[str] = [
    "Activity & spending",
    "Inflation & wages",
    "Interest rates",
    "Financial & external",
    "Government",
]

# The four headline defaults (unchanged behaviour).
DEFAULT_OUTPUTS: List[str] = ["hggdp", "lur", "picnia", "rff"]

# Backward-compatible {key: (label, unit)} mapping for the default set.
OUTPUT_VARS: Dict[str, tuple] = {
    k: (OUTPUT_BY_KEY[k].label, OUTPUT_BY_KEY[k].unit) for k in DEFAULT_OUTPUTS
}

# The two scenarios, in display order.
SCENARIOS = {
    "active": "With policy response (active rule)",
    "held": "Without response (funds rate held)",
}


def _resolve_vars(variables: Optional[Sequence[str]]) -> List[str]:
    keys = list(variables) if variables is not None else list(DEFAULT_OUTPUTS)
    unknown = [k for k in keys if k not in OUTPUT_BY_KEY]
    if unknown:
        raise KeyError(f"unknown output variable(s): {unknown}")
    return keys


def deviations(
    result: SimResult,
    scenario: str,
    variables: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Deviation-from-baseline paths for one scenario.

    Returns a DataFrame indexed by quarter with one column per requested output
    variable. Rate variables give a percentage-point difference; level variables
    give a percent deviation (see module docstring). ``variables=None`` uses the
    four headline defaults.
    """
    if scenario not in ("active", "held"):
        raise ValueError("scenario must be 'active' or 'held'")
    sim = getattr(result, scenario)
    out = pd.DataFrame(index=result.window)
    for key in _resolve_vars(variables):
        var = OUTPUT_BY_KEY[key]
        sim_level = var.level(sim)
        base_level = var.level(result.baseline)
        if var.transform == "pct":
            out[key] = 100.0 * (sim_level / base_level - 1.0)
        else:  # diff
            out[key] = sim_level - base_level
    return out


def deviation_panel(
    result: SimResult, variables: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Tidy long-format panel of every scenario × variable deviation.

    Columns: ``quarter``, ``date``, ``variable``, ``label``, ``group``, ``unit``,
    ``scenario``, ``scenario_label``, ``deviation``. Ideal for charting and CSV.
    """
    keys = _resolve_vars(variables)
    rows = []
    for scen, scen_label in SCENARIOS.items():
        dev = deviations(result, scen, keys)
        for key in keys:
            var = OUTPUT_BY_KEY[key]
            for q, value in dev[key].items():
                rows.append(
                    {
                        "quarter": str(q),
                        "date": q.to_timestamp().date().isoformat(),
                        "variable": key,
                        "label": var.label,
                        "group": var.group,
                        "unit": var.unit,
                        "scenario": scen,
                        "scenario_label": scen_label,
                        "deviation": float(value),
                    }
                )
    return pd.DataFrame(rows)


def levels_panel(
    result: SimResult, variables: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Wide panel of baseline and scenario *levels* for each requested variable."""
    frames = {}
    for key in _resolve_vars(variables):
        var = OUTPUT_BY_KEY[key]
        frames[f"{key}_baseline"] = var.level(result.baseline).values
        frames[f"{key}_active"] = var.level(result.active).values
        frames[f"{key}_held"] = var.level(result.held).values
    panel = pd.DataFrame(frames)
    panel.index = [str(q) for q in result.window]
    panel.index.name = "quarter"
    return panel


def summary_table(
    result: SimResult,
    variables: Optional[Sequence[str]] = None,
    horizons: Sequence[int] = (4, 8),
) -> pd.DataFrame:
    """Peak-effect and selected-horizon summary, per variable and scenario.

    For each requested variable and each scenario (active / held), reports:

    * ``peak`` — the largest-magnitude deviation over the shown window, signed,
      and ``peak_quarter`` — the quarter in which it occurs;
    * ``@{h}q`` — the deviation ``h`` quarters after the shock hits (``h = 0`` is
      the impact quarter). Horizons beyond the shown window come back as NaN.

    Units follow each variable (percentage points for rates/inflation, percent
    for level variables); the ``unit`` column records which.
    """
    keys = _resolve_vars(variables)
    window = list(result.window)
    n = len(window)
    horizons = [int(h) for h in horizons]
    rows = []
    for scen, scen_label in SCENARIOS.items():
        dev = deviations(result, scen, keys)
        for key in keys:
            var = OUTPUT_BY_KEY[key]
            series = dev[key]
            peak_idx = series.abs().idxmax()  # Period of the largest |deviation|
            row = {
                "variable": var.label,
                "unit": var.unit,
                "scenario": scen_label,
                "peak": round(float(series.loc[peak_idx]), 3),
                "peak_quarter": str(peak_idx),
            }
            for h in horizons:
                row[f"@{h}q"] = round(float(series.iloc[h]), 3) if 0 <= h < n else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def run_metadata(result: SimResult) -> Dict[str, object]:
    """Compact, serialisable description of the run (for CSV headers / captions)."""
    shocks = [
        {
            "shock": r.spec.label,
            "shock_key": r.spec.key,
            "lever": r.spec.column,
            "magnitude": r.magnitude,
            "magnitude_unit": r.spec.user_unit,
            "duration_quarters": r.duration,
        }
        for r in result.requests
    ]
    meta: Dict[str, object] = {
        "scenario": result.label,
        "n_shocks": len(result.requests),
        "shocks": shocks,
        "expectations": result.expectations,
        "policy_rule": getattr(result, "policy_rule", "inertial"),
        "start": str(result.start),
        "end": str(result.end),
    }
    # Flat single-shock keys for back-compat / simple captions (first shock).
    meta.update({k: v for k, v in shocks[0].items()})
    return meta


def to_csv_bytes(
    result: SimResult, variables: Optional[Sequence[str]] = None
) -> bytes:
    """CSV export of the deviation panel with a metadata comment header."""
    meta = run_metadata(result)
    header = "\n".join(f"# {k}: {v}" for k, v in meta.items())
    body = deviation_panel(result, variables).to_csv(index=False)
    return (header + "\n" + body).encode("utf-8")
