"""Loading the FRB/US model and the LONGBASE baseline dataset.

Both loads are expensive (the ``Frbus`` constructor does symbolic
differentiation; LONGBASE is a multi-megabyte CSV), so results are memoised.
The app layers Streamlit's own cache on top of these for cross-run reuse.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import pandas as pd

from . import paths  # noqa: F401  (side effect: puts vendored pyfrbus on sys.path)
from .paths import LONGBASE_TXT, MODEL_XML, VINTAGE_JSON

# Imported after paths wires up sys.path.
from pyfrbus.frbus import Frbus  # type: ignore  # noqa: E402
from pyfrbus.load_data import load_data  # type: ignore  # noqa: E402

# Expectations -> MCE argument for the Frbus constructor.
#   "var" : backward-looking VAR expectations (Frbus default, mce=None)
#   "mce" : model-consistent (rational) expectations, mcap+wp block
_MCE_ARG = {"var": None, "mce": "mcap+wp"}


def expectations_choices() -> dict:
    """Human labels for the supported expectations assumptions."""
    return {
        "var": "VAR-based (backward-looking)",
        "mce": "Model-consistent (rational, mcap+wp)",
    }


@lru_cache(maxsize=4)
def load_frbus(expectations: str = "var") -> Frbus:
    """Build (and cache) an ``Frbus`` model object for the given expectations.

    ``expectations`` is ``"var"`` or ``"mce"``.
    """
    if expectations not in _MCE_ARG:
        raise ValueError(
            f"expectations must be one of {sorted(_MCE_ARG)}, got {expectations!r}"
        )
    return Frbus(str(MODEL_XML), mce=_MCE_ARG[expectations])


@lru_cache(maxsize=1)
def _load_baseline_cached() -> pd.DataFrame:
    return load_data(str(LONGBASE_TXT))


def load_baseline() -> pd.DataFrame:
    """Return a fresh **copy** of the LONGBASE baseline dataset.

    A copy is returned so callers can mutate freely without corrupting the
    cached original.
    """
    return _load_baseline_cached().copy()


def data_vintage() -> Optional[dict]:
    """Return the recorded data-vintage metadata, or ``None`` if unavailable."""
    try:
        with open(VINTAGE_JSON) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# The vendored model equations vintage. The data vintage lives in VINTAGE.json
# and refreshes independently (the Fed updates the data more often than the
# equations).
MODEL_VINTAGE = "PyFRB/US 1.0.0 (model.xml equations)"
