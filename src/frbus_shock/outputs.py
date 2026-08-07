"""Turn a :class:`SimResult` into deviation-from-baseline paths and exports.

The four headline outputs the dashboard reports, with their FRB/US variable
names (confirmed against ``model.xml``):

===================  ========  ==================================================
Output               Variable  Definition
===================  ========  ==================================================
GDP growth           hggdp     Growth rate of real GDP (annual rate, %)
Unemployment rate    lur       Civilian unemployment rate (%)
PCE inflation        picnia    PCE inflation (q/q annualised, %)
Federal funds rate   rff       Federal funds rate (%)
===================  ========  ==================================================

All four are already expressed in percent / annualised-percent, so a deviation
from baseline is a simple difference and its unit is **percentage points**.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .simulate import SimResult

# variable -> (short label, unit)
OUTPUT_VARS: Dict[str, tuple] = {
    "hggdp": ("GDP growth", "pp (annual rate)"),
    "lur": ("Unemployment rate", "pp"),
    "picnia": ("PCE inflation", "pp (annualised)"),
    "rff": ("Federal funds rate", "pp"),
}

# The two scenarios, in display order.
SCENARIOS = {
    "active": "With policy response (active rule)",
    "held": "Without response (funds rate held)",
}


def deviations(result: SimResult, scenario: str) -> pd.DataFrame:
    """Deviation-from-baseline paths for one scenario.

    Returns a DataFrame indexed by quarter with one column per output variable,
    each ``scenario - baseline`` in percentage points.
    """
    if scenario not in ("active", "held"):
        raise ValueError("scenario must be 'active' or 'held'")
    sim = getattr(result, scenario)
    out = pd.DataFrame(index=result.window)
    for var in OUTPUT_VARS:
        out[var] = sim[var] - result.baseline[var]
    return out


def deviation_panel(result: SimResult) -> pd.DataFrame:
    """Tidy long-format panel of every scenario × variable deviation.

    Columns: ``quarter``, ``date``, ``variable``, ``label``, ``unit``,
    ``scenario``, ``scenario_label``, ``deviation``. Ideal for charting and CSV.
    """
    rows = []
    for scen, scen_label in SCENARIOS.items():
        dev = deviations(result, scen)
        for var, (label, unit) in OUTPUT_VARS.items():
            for q, value in dev[var].items():
                rows.append(
                    {
                        "quarter": str(q),
                        "date": q.to_timestamp().date().isoformat(),
                        "variable": var,
                        "label": label,
                        "unit": unit,
                        "scenario": scen,
                        "scenario_label": scen_label,
                        "deviation": float(value),
                    }
                )
    return pd.DataFrame(rows)


def levels_panel(result: SimResult) -> pd.DataFrame:
    """Wide panel of baseline and scenario *levels* for each output variable."""
    frames = {}
    for var in OUTPUT_VARS:
        frames[f"{var}_baseline"] = result.baseline[var]
        frames[f"{var}_active"] = result.active[var]
        frames[f"{var}_held"] = result.held[var]
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


def to_csv_bytes(result: SimResult) -> bytes:
    """CSV export of the deviation panel with a metadata comment header."""
    meta = run_metadata(result)
    header = "\n".join(f"# {k}: {v}" for k, v in meta.items())
    body = deviation_panel(result).to_csv(index=False)
    return (header + "\n" + body).encode("utf-8")
