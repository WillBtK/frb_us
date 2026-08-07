"""Fiscal-multiplier tab — output multipliers by instrument and monetary response.

Estimates FRB/US output multipliers (cumulative ΔGDP per $ of fiscal impulse) for
government purchases, transfers, and a personal tax cut, both with the active
policy rule and with the funds rate held (accommodative) — the comparison the
CBO / fiscal-multiplier literature emphasises.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # app/pages/ -> repo root
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from frbus_shock import INSTRUMENTS, data_vintage, multiplier_table  # noqa: E402
from frbus_shock.multipliers import DEFAULT_HORIZON_LABELS  # noqa: E402

st.set_page_config(page_title="FRB/US Fiscal Multipliers", page_icon="💵", layout="wide")

_ACTIVE_COLOR = "#1f4e79"
_HELD_COLOR = "#c1440e"


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _cached_table(instrument_keys, expectations, start):
    return multiplier_table(
        list(instrument_keys), expectations=expectations, start=start, horizon=12
    )


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
st.sidebar.title("Fiscal multipliers")

inst_keys = st.sidebar.multiselect(
    "Instruments",
    options=list(INSTRUMENTS),
    default=list(INSTRUMENTS),
    format_func=lambda k: INSTRUMENTS[k].label,
    help="Each is shocked by a moderate impulse; the multiplier is ΔGDP per $ of "
    "that impulse, so its size barely matters.",
)
for _k in inst_keys:
    st.sidebar.caption(f"↳ {INSTRUMENTS[_k].label}: {INSTRUMENTS[_k].note}")

expectations = st.sidebar.radio(
    "Expectations",
    options=["var", "mce"],
    format_func=lambda k: {"var": "VAR (fast)", "mce": "Model-consistent (slower)"}[k],
)
if expectations == "mce":
    st.sidebar.warning(
        "MCE solves every instrument twice under the forward-looking solver — "
        "expect this to take a while for several instruments."
    )

_today_q = str(pd.Period(pd.Timestamp.today(), freq="Q"))
start = st.sidebar.selectbox(
    "Start quarter",
    options=[_today_q, "2035Q1", "2040Q1"],
    help="When the fiscal impulse begins, on the model's projection baseline.",
)

run = st.sidebar.button("▶ Compute multipliers", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
st.title("💵 Fiscal Multipliers")
st.markdown(
    "The **output multiplier** at a horizon is the *cumulative* change in real "
    "GDP per dollar of fiscal impulse (both in real 2012\\$):"
)
st.latex(r"\text{multiplier}(h)=\frac{\sum_{t=0}^{h}\Delta \text{GDP}_t}{\sum_{t=0}^{h}\Delta \text{impulse}_t}")
st.markdown(
    "Multipliers differ by instrument — **purchases > transfers > tax cuts** — "
    "and are **larger when monetary policy accommodates** (the funds rate held at "
    "baseline) than when the active rule leans against the stimulus. "
    "In the spirit of CBO's fiscal-multiplier analysis."
)

vintage = data_vintage() or {}
st.caption(
    f"Model: PyFRB/US 1.0.0 · Data range {vintage.get('first_obs','?')}–"
    f"{vintage.get('last_obs','?')}. Results are deviations from a stylised "
    "projection baseline (not a forecast)."
)

if run:
    if not inst_keys:
        st.warning("Pick at least one instrument.")
        st.stop()
    with st.spinner("Solving FRB/US for each instrument (active rule and held)…"):
        try:
            tbl = _cached_table(tuple(inst_keys), expectations, start)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Computation failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["mult_tbl"] = tbl
    st.session_state["mult_meta"] = {"expectations": expectations, "start": start}

tbl = st.session_state.get("mult_tbl")
if tbl is None:
    st.info("👈 Choose instruments and press **Compute multipliers**.")
    st.stop()

meta = st.session_state.get("mult_meta", {})
st.subheader(
    f"Cumulative output multipliers · {meta.get('expectations','var').upper()} "
    f"expectations · from {meta.get('start','')}"
)
st.dataframe(tbl, use_container_width=True, hide_index=True)

# --- Grouped bar chart at a chosen horizon --------------------------------- #
horizon_cols = [c for c in tbl.columns if c not in ("instrument", "scenario")]
h_label = st.selectbox("Chart horizon", horizon_cols, index=min(1, len(horizon_cols) - 1))

instruments = list(dict.fromkeys(tbl["instrument"]))
active = tbl[tbl["scenario"].str.startswith("With")].set_index("instrument")[h_label]
held = tbl[tbl["scenario"].str.startswith("Accommod")].set_index("instrument")[h_label]

fig = go.Figure()
fig.add_bar(x=instruments, y=[active.get(i) for i in instruments],
            name="Active rule", marker_color=_ACTIVE_COLOR)
fig.add_bar(x=instruments, y=[held.get(i) for i in instruments],
            name="Accommodative (held)", marker_color=_HELD_COLOR)
fig.update_layout(
    barmode="group",
    height=440,
    yaxis_title=f"Multiplier at {h_label.lower()}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=40, r=20, t=50, b=40),
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "⬇ Multipliers (CSV)",
    data=tbl.to_csv(index=False).encode("utf-8"),
    file_name=f"frbus_multipliers_{meta.get('expectations','var')}_{meta.get('start','')}.csv",
    mime="text/csv",
)

with st.expander("How to read this"):
    st.markdown(
        "- **Purchases** (federal, state & local) have the largest multipliers "
        "(~0.8–1.0): the spending is GDP directly, plus induced private demand.\n"
        "- **Transfers** (~0.25–0.45) and **tax cuts** (~0.15–0.45) are smaller — "
        "part of the extra income is saved rather than spent.\n"
        "- Every multiplier is **larger when the funds rate is held** "
        "(accommodative), because the active rule otherwise raises rates and "
        "offsets some of the stimulus — the effective-lower-bound intuition.\n"
        "- Multipliers here are *cumulative* through the horizon; the impact "
        "(quarter-0) multiplier is the same with or without a monetary response, "
        "since policy reacts with a lag."
    )
