"""Optimal-control monetary policy tab (linear-quadratic).

Given a shock, finds the funds-rate path that minimises a quadratic loss over the
inflation and unemployment deviations (plus a rate-smoothing penalty), and
compares it to the Taylor rule and to no monetary response. In the spirit of the
FEDS Note "Optimal-Control Monetary Policy in the FRB/US Model".
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from frbus_shock import CATALOGUE, data_vintage, run_optimal_control  # noqa: E402

st.set_page_config(page_title="FRB/US Optimal Control", page_icon="🎯", layout="wide")

_OPT = "#1f7a3d"      # green — optimal
_TAY = "#1f4e79"      # blue — Taylor
_HELD = "#c1440e"     # rust — no response


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _cached_ocp(shocks_spec, weights, expectations, start, horizon):
    shocks = [{"key": k, "magnitude": m, "duration": d} for k, m, d in shocks_spec]
    res = run_optimal_control(
        shocks=shocks, weights=dict(weights), expectations=expectations,
        start=start, horizon=horizon,
    )
    return {
        "paths": res.paths,
        "delta": list(res.delta),
        "losses": res.losses,
        "approx_error": res.approx_error,
        "window": [str(q) for q in res.window],
        "weights": dict(res.weights),
    }


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
st.sidebar.title("Optimal control")

_DEMAND = [k for k, s in CATALOGUE.items() if s.group in ("Demand", "Prices & supply")]
shock_key = st.sidebar.selectbox(
    "Shock to stabilise",
    options=_DEMAND,
    index=_DEMAND.index("consumption") if "consumption" in _DEMAND else 0,
    format_func=lambda k: CATALOGUE[k].label,
    help="The disturbance the policymaker faces. Optimal control finds the "
    "funds-rate path that best offsets its effect on inflation and unemployment.",
)
spec = CATALOGUE[shock_key]
default_sign = -1.0 if shock_key in ("consumption", "durables", "housing", "business_investment") else 1.0
st.sidebar.caption(spec.description)
magnitude = st.sidebar.number_input(
    f"Magnitude ({spec.user_unit})",
    value=float(spec.default_magnitude) * default_sign,
    step=abs(float(spec.default_magnitude)) / 4 or 0.1,
    help="Negative demand shocks create a recession (the classic OCP case).",
)
duration = st.sidebar.slider("Duration (quarters on)", 1, 12, int(spec.default_duration))

st.sidebar.divider()
st.sidebar.markdown("**Loss weights**")
w_pi = st.sidebar.slider("Inflation gap", 0.0, 3.0, 1.0, 0.1)
w_u = st.sidebar.slider("Unemployment gap", 0.0, 3.0, 1.0, 0.1)
w_sm = st.sidebar.slider("Rate smoothing", 0.0, 2.0, 0.5, 0.1,
                         help="Penalty on quarter-to-quarter funds-rate changes.")

expectations = st.sidebar.radio(
    "Expectations", options=["var", "mce"],
    format_func=lambda k: {"var": "VAR (fast)", "mce": "Model-consistent (slow)"}[k],
)
if expectations == "mce":
    st.sidebar.warning(
        "MCE optimal control builds the full anticipation matrix — one solve per "
        "quarter — so it can take a couple of minutes."
    )
horizon = st.sidebar.slider("Horizon (quarters)", 8, 12, 12)

run = st.sidebar.button("▶ Optimise policy", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
st.title("🎯 Optimal-Control Monetary Policy")
st.markdown(
    "The policymaker chooses the **funds-rate path that minimises a quadratic "
    "loss** over the inflation and unemployment deviations from baseline (plus a "
    "rate-smoothing penalty), and we compare it to the **Taylor rule** and to "
    "**no monetary response**:"
)
st.latex(r"L=w_\pi\sum_t \pi^{\,dev\,2}_t + w_u\sum_t u^{\,dev\,2}_t + w_{\text{sm}}\sum_t(\Delta i_t)^2")
st.caption(
    "Solved by the linear-quadratic method (the model's impulse responses give a "
    "linear map from the funds-rate path to outcomes, so the optimum is "
    "closed-form). A linear approximation — its error vs. a full nonlinear solve "
    "is reported below. In the spirit of the FEDS Note on optimal-control policy."
)

vintage = data_vintage() or {}
st.caption(
    f"Model PyFRB/US 1.0.0 · data {vintage.get('first_obs','?')}–{vintage.get('last_obs','?')}. "
    "Deviations from a stylised baseline, not a forecast."
)

if run:
    with st.spinner("Building impulse responses and optimising the funds-rate path…"):
        try:
            out = _cached_ocp(
                ((shock_key, float(magnitude), int(duration)),),
                (("inflation", w_pi), ("unemployment", w_u), ("smoothing", w_sm)),
                expectations, start := "2026Q3", int(horizon),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimisation failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["ocp"] = out

out = st.session_state.get("ocp")
if out is None:
    st.info("👈 Pick a shock and loss weights, then press **Optimise policy**.")
    st.stop()

paths = out["paths"]
x = [pd.Period(q, freq="Q").to_timestamp() for q in out["window"]]
losses = out["losses"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Loss — optimal", f"{losses['optimal']:.2f}")
c2.metric("Loss — Taylor rule", f"{losses['taylor']:.2f}",
          delta=f"{losses['optimal'] - losses['taylor']:+.2f}", delta_color="inverse")
c3.metric("Loss — no response", f"{losses['held']:.2f}",
          delta=f"{losses['optimal'] - losses['held']:+.2f}", delta_color="inverse")
c4.metric("Linear-approx error", f"{out['approx_error']:.3f} pp")

titles = ["Federal funds rate (pp dev.)", "PCE inflation (pp dev.)", "Unemployment (pp dev.)"]
fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.07)
series = [("rff", 1), ("pi", 2), ("u", 3)]
for suffix, col in series:
    for scen, color, name in (("optimal", _OPT, "Optimal control"),
                              ("taylor", _TAY, "Taylor rule"),
                              ("held", _HELD, "No response (held)")):
        fig.add_trace(
            go.Scatter(x=x, y=paths[f"{scen}_{suffix}"], name=name, line=dict(color=color, width=2.5),
                       legendgroup=scen, showlegend=(col == 1)),
            row=1, col=col,
        )
    fig.add_hline(y=0, line=dict(color="#999", width=1), row=1, col=col)
fig.update_layout(
    height=420, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.15, x=0),
    margin=dict(l=40, r=20, t=70, b=30),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The optimal path typically moves the funds rate **earlier and more "
    "aggressively** than the Taylor rule, achieving smaller inflation and "
    "unemployment gaps (a lower loss)."
)

st.download_button(
    "⬇ Paths (CSV)",
    data=paths.assign(quarter=out["window"]).to_csv(index=False).encode("utf-8"),
    file_name=f"frbus_optimal_control_{shock_key}_{expectations}.csv",
    mime="text/csv",
)

with st.expander("Method & caveats"):
    st.markdown(
        "- **Linear-quadratic optimal control:** the funds-rate impulse responses "
        "give a linear map from the rate path to inflation/unemployment, so the "
        "quadratic loss is minimised in closed form. Under VAR the response matrix "
        "is Toeplitz (one extra solve); under MCE it is built one column per "
        "quarter to capture anticipation.\n"
        "- **Linear approximation:** exact for the linearised model; the reported "
        "error is the max gap vs. re-solving the full nonlinear model along the "
        "optimal path (a few hundredths of a pp for moderate shocks).\n"
        "- **Loss is in deviations from baseline** (optimal *stabilisation* of the "
        "shock). Raising the unemployment weight makes policy lean harder against "
        "unemployment; raising smoothing makes the rate path gentler."
    )
