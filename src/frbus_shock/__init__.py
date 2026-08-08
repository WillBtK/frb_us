"""FRB/US shock-analysis toolkit.

A thin, tested wrapper over the Federal Reserve's PyFRB/US platform that runs a
single macro shock under a chosen expectations assumption, both *with* an active
monetary-policy rule and *without* one (funds rate held at baseline), and
reports deviation-from-baseline paths for GDP growth, unemployment, PCE
inflation, and the federal funds rate.

Typical use::

    from frbus_shock import run_simulation, deviation_panel
    result = run_simulation("fiscal_spending", magnitude=1.0, duration=8,
                            expectations="var")
    panel = deviation_panel(result)
"""

from __future__ import annotations

from .model import (
    MODEL_VINTAGE,
    data_vintage,
    expectations_choices,
    load_baseline,
    load_frbus,
)
from .outputs import (
    DEFAULT_OUTPUTS,
    OUTPUT_BY_KEY,
    OUTPUT_CATALOGUE,
    OUTPUT_GROUPS,
    OUTPUT_VARS,
    OutputVar,
    SCENARIOS,
    deviation_panel,
    deviations,
    levels_panel,
    run_metadata,
    summary_table,
    to_csv_bytes,
)
from .multipliers import (
    INSTRUMENTS,
    FiscalInstrument,
    MultiplierResult,
    get_instrument,
    multiplier_table,
    run_multiplier,
)
from .optcontrol import DEFAULT_WEIGHTS, OCPResult, run_optimal_control
from .shocks import (
    CATALOGUE,
    SCENARIO_PRESETS,
    ShockSpec,
    custom_shock,
    get_shock,
)
from .simulate import (
    DEFAULT_HORIZON,
    DEFAULT_START,
    ShockRequest,
    SimResult,
    run_simulation,
)

__all__ = [
    "run_simulation",
    "SimResult",
    "DEFAULT_START",
    "DEFAULT_HORIZON",
    "CATALOGUE",
    "SCENARIO_PRESETS",
    "ShockSpec",
    "ShockRequest",
    "get_shock",
    "custom_shock",
    "multiplier_table",
    "run_multiplier",
    "get_instrument",
    "INSTRUMENTS",
    "FiscalInstrument",
    "MultiplierResult",
    "run_optimal_control",
    "OCPResult",
    "DEFAULT_WEIGHTS",
    "deviations",
    "deviation_panel",
    "levels_panel",
    "run_metadata",
    "summary_table",
    "to_csv_bytes",
    "OUTPUT_VARS",
    "OUTPUT_CATALOGUE",
    "OUTPUT_BY_KEY",
    "OUTPUT_GROUPS",
    "DEFAULT_OUTPUTS",
    "OutputVar",
    "SCENARIOS",
    "load_frbus",
    "load_baseline",
    "expectations_choices",
    "data_vintage",
    "MODEL_VINTAGE",
]
