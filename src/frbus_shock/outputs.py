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
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .simulate import SimResult


@dataclass(frozen=True)
class OutputVar:
    """One selectable output variable."""

    key: str  # FRB/US variable name
    label: str
    group: str
    transform: str  # "diff" (-> pp) or "pct" (-> %)

    @property
    def unit(self) -> str:
        return "pp" if self.transform == "diff" else "%"


# Ordered catalogue, grouped. Order here is the display order within each group.
OUTPUT_CATALOGUE: List[OutputVar] = [
    # --- Activity ---
    OutputVar("hggdp", "GDP growth (annual rate)", "Activity", "diff"),
    OutputVar("xgap2", "Output gap", "Activity", "diff"),
    OutputVar("lur", "Unemployment rate", "Activity", "diff"),
    OutputVar("xgdp", "Real GDP (level)", "Activity", "pct"),
    OutputVar("ecnia", "Real consumption (PCE)", "Activity", "pct"),
    OutputVar("ebfi", "Business fixed investment", "Activity", "pct"),
    # --- Inflation ---
    OutputVar("picnia", "PCE inflation (q/q annualised)", "Inflation", "diff"),
    OutputVar("picxfe", "Core PCE inflation", "Inflation", "diff"),
    OutputVar("pic4", "PCE inflation (4-quarter)", "Inflation", "diff"),
    OutputVar("picx4", "Core PCE inflation (4-quarter)", "Inflation", "diff"),
    OutputVar("pcpi", "CPI inflation", "Inflation", "diff"),
    OutputVar("pcpix", "Core CPI inflation", "Inflation", "diff"),
    # --- Interest rates ---
    OutputVar("rff", "Federal funds rate", "Interest rates", "diff"),
    OutputVar("rrff", "Real federal funds rate", "Interest rates", "diff"),
    OutputVar("rg5", "5-year Treasury yield", "Interest rates", "diff"),
    OutputVar("rg10", "10-year Treasury yield", "Interest rates", "diff"),
    OutputVar("rg30", "30-year Treasury yield", "Interest rates", "diff"),
    OutputVar("rbbb", "BBB corporate bond yield", "Interest rates", "diff"),
    OutputVar("rg10p", "10-year term premium", "Interest rates", "diff"),
]

OUTPUT_BY_KEY: Dict[str, OutputVar] = {v.key: v for v in OUTPUT_CATALOGUE}

# Group display order.
OUTPUT_GROUPS: List[str] = ["Activity", "Inflation", "Interest rates"]

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
        if var.transform == "pct":
            out[key] = 100.0 * (sim[key] / result.baseline[key] - 1.0)
        else:  # diff
            out[key] = sim[key] - result.baseline[key]
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
        frames[f"{key}_baseline"] = result.baseline[key]
        frames[f"{key}_active"] = result.active[key]
        frames[f"{key}_held"] = result.held[key]
    panel = pd.DataFrame(frames)
    panel.index = [str(q) for q in result.window]
    panel.index.name = "quarter"
    return panel


def run_metadata(result: SimResult) -> Dict[str, object]:
    """Compact, serialisable description of the run (for CSV headers / captions)."""
    return {
        "shock": result.shock.label,
        "shock_key": result.shock.key,
        "lever": result.shock.column,
        "magnitude": result.magnitude,
        "magnitude_unit": result.shock.user_unit,
        "duration_quarters": result.duration,
        "expectations": result.expectations,
        "start": str(result.start),
        "end": str(result.end),
    }


def to_csv_bytes(
    result: SimResult, variables: Optional[Sequence[str]] = None
) -> bytes:
    """CSV export of the deviation panel with a metadata comment header."""
    meta = run_metadata(result)
    header = "\n".join(f"# {k}: {v}" for k, v in meta.items())
    body = deviation_panel(result, variables).to_csv(index=False)
    return (header + "\n" + body).encode("utf-8")
