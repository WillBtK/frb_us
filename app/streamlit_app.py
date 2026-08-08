"""FRB/US dashboard entry point — the navigation router.

Streamlit Community Cloud deploys this file. It sets the page config once and
routes to the four views under ``app/views/`` with friendly sidebar labels
(Shock Analysis / Fiscal Multipliers / Optimal Control / Debt Sustainability).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``frbus_shock`` package importable for every view.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

st.set_page_config(page_title="FRB/US Dashboard", page_icon="📈", layout="wide")

# Page sources are given relative to this entry file's directory (app/).
pages = [
    st.Page("views/shock_analysis.py", title="Shock Analysis", icon="📈", default=True),
    st.Page("views/fiscal_multipliers.py", title="Fiscal Multipliers", icon="💵"),
    st.Page("views/optimal_control.py", title="Optimal Control", icon="🎯"),
    st.Page("views/debt_sustainability.py", title="Debt Sustainability", icon="🏛️"),
]

st.navigation(pages).run()
