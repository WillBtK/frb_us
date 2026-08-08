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
_VIEWS = Path(__file__).resolve().parent
if str(_VIEWS) not in sys.path:
    sys.path.insert(0, str(_VIEWS))

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import _shock_controls as sc  # noqa: E402
from frbus_shock import data_vintage, run_optimal_control  # noqa: E402

_OPT = "#1f7a3d"      # green — optimal
_TAY = "#1f4e79"      # blue — Taylor
_HELD = "#c1440e"     # rust — no response


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _cached_ocp(shocks_spec, weights, expectations, start, horizon):
    shocks = []
    for kind, name, mag, dur in shocks_spec:
        if kind == "custom":
            shocks.append({"custom_variable": name, "magnitude": mag, "duration": dur})
        else:
            shocks.append({"key": name, "magnitude": mag, "duration": dur})
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
st.title("🎯 Optimal-Control Monetary Policy")
st.caption(
    "The funds-rate path that minimises a quadratic loss over the inflation and "
    "unemployment gaps (plus rate smoothing), vs. the Taylor rule and no "
    "response — linear-quadratic method, in the spirit of the FEDS Note."
)

# Shocks to stabilise: the full multi-instrument set shared with the Shock
# Analysis tab, EXCEPT the monetary lever — the funds rate is the control
# variable here, so a shock to the policy rule is inert / incoherent (and any
# named scenario that uses it is dropped from the loader automatically).
_OCP_GROUP_ORDER = [
    "Demand", "Prices & supply", "Financial", "External / global", "Fiscal & monetary",
]
_OCP_EXCLUDE = {"monetary"}
_OCP_OPTIONS, _ocp_label = sc.shock_options(_OCP_GROUP_ORDER, exclude_keys=_OCP_EXCLUDE)

# Loss-weight presets from the literature: (inflation, unemployment, smoothing).
_PRESETS = {
    "Balanced dual mandate (Fed / Yellen 2012)": (1.0, 1.0, 0.5),
    "Inflation-focused (hawkish)": (2.0, 1.0, 0.5),
    "Employment-focused (dovish)": (1.0, 2.0, 0.5),
    "Aggressive (little rate smoothing)": (1.0, 1.0, 0.1),
    "Gradualist (heavy rate smoothing)": (1.0, 1.0, 1.5),
}
_DEFAULT_PRESET = "Balanced dual mandate (Fed / Yellen 2012)"

# --- Scenario loader (same presets as Shock Analysis, minus rate-rule ones) ---
sc.render_scenario_loader("ocp", exclude_keys=_OCP_EXCLUDE)

# --- Settings row: shocks count | expectations | horizon | loss-weight preset ---
# Seed the weight sliders' state once, then let a newly-chosen preset update them.
for _k, _v in (("ocp_wpi", 1.0), ("ocp_wu", 1.0), ("ocp_wsm", 0.5)):
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("ocp_nsh", 1)

_s = st.columns(4)
n_shocks = _s[0].selectbox(
    "Number of shocks", [1, 2, 3, 4], key="ocp_nsh",
    help="The disturbance the policymaker faces — several levers combine into one.",
)
expectations = _s[1].selectbox(
    "Expectations", options=["var", "mce"], key="ocp_exp",
    format_func=lambda k: {"var": "VAR (fast)", "mce": "Model-consistent (slow)"}[k],
    help="VAR is the fast default (Toeplitz impulse responses). MCE builds the "
    "full anticipation matrix — one solve per quarter, a couple of minutes.",
)
horizon = _s[2].selectbox("Horizon (q)", [8, 9, 10, 11, 12], index=4, key="ocp_h")
st.session_state.setdefault("ocp_lwpreset", _DEFAULT_PRESET)
preset = _s[3].selectbox(
    "Loss-weight preset", ["Custom"] + list(_PRESETS), key="ocp_lwpreset",
    help="Named weightings spanning the mainstream range — see 'How to choose "
    "the weights' below. Pick one, then fine-tune the sliders if you like.",
)
if preset != "Custom" and st.session_state.get("_ocp_preset") != preset:
    (st.session_state["ocp_wpi"],
     st.session_state["ocp_wu"],
     st.session_state["ocp_wsm"]) = _PRESETS[preset]
st.session_state["_ocp_preset"] = preset

# --- Editable per-shock rows (the disturbance to stabilise) ---
shock_specs = sc.render_shock_rows("ocp", n_shocks, _OCP_OPTIONS, _ocp_label)

# --- The three loss weights (driven by the preset, editable) ---
_r3 = st.columns(3)
w_pi = _r3[0].slider("Inflation-gap weight", 0.0, 3.0, step=0.1, key="ocp_wpi")
w_u = _r3[1].slider("Unemployment-gap weight", 0.0, 3.0, step=0.1, key="ocp_wu")
w_sm = _r3[2].slider("Rate-smoothing weight", 0.0, 2.0, step=0.1, key="ocp_wsm",
                     help="Penalty on quarter-to-quarter funds-rate changes.")

if expectations == "mce":
    st.caption("⏳ MCE builds the full anticipation matrix — one solve per quarter, a couple of minutes.")
st.caption(
    "The classic OCP case is a **contractionary** disturbance (a recession the "
    "Fed leans against) — flip a magnitude's sign, or load a downside scenario, "
    "for that. A shock to the funds-rate rule itself is excluded here."
)

with st.expander("How to choose the weights"):
    st.markdown(
        "The loss penalises squared deviations of **inflation** and **unemployment** "
        "from baseline (target), plus squared **quarter-to-quarter changes in the "
        "funds rate**. Only the *relative* weights matter for the optimal path.\n\n"
        "- **Balanced dual mandate (default).** Equal weight on the inflation and "
        "unemployment gaps — the canonical Fed case, used in Yellen's 2012 "
        "optimal-control speeches and the FRB/US FEDS Note; it treats the two legs "
        "of the statutory dual mandate symmetrically.\n"
        "- **More weight on inflation (hawkish).** Prioritises price stability and "
        "keeping inflation expectations anchored — welfare-based New-Keynesian "
        "analyses often imply inflation variability is especially costly. The "
        "optimum tolerates a larger unemployment gap to return inflation faster.\n"
        "- **More weight on unemployment (dovish).** Prioritises closing "
        "labour-market slack (the human cost of unemployment, hysteresis risk). The "
        "optimum eases more and accepts a little more inflation deviation.\n"
        "- **Rate-smoothing weight.** The Fed moves rates gradually in practice. A "
        "higher weight gives a gentler, more inertial funds-rate path; a lower "
        "weight gives a more front-loaded, aggressive optimum.\n\n"
        "There is no single 'right' answer — the presets bracket the mainstream "
        "range. Start from *Balanced* and lean toward whichever objective you judge "
        "more costly."
    )

run = st.button("▶ Optimise policy", type="primary")
st.divider()


if run:
    if not shock_specs:
        st.warning("Add at least one shock to stabilise.")
        st.stop()
    with st.spinner("Building impulse responses and optimising the funds-rate path…"):
        try:
            out = _cached_ocp(
                tuple(shock_specs),
                (("inflation", w_pi), ("unemployment", w_u), ("smoothing", w_sm)),
                expectations, "2026Q3", int(horizon),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimisation failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["ocp"] = out
    st.session_state["ocp_meta"] = {
        "expectations": expectations,
        "stem": (shock_specs[0][1].replace(":", "_")
                 if len(shock_specs) == 1 else f"{len(shock_specs)}shocks"),
    }

out = st.session_state.get("ocp")
if out is None:
    st.info("Configure the shock(s) and loss weights above, then press **Optimise policy**.")
    st.stop()
_ocp_meta = st.session_state.get("ocp_meta", {"expectations": "var", "stem": "shock"})

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
    file_name=f"frbus_optimal_control_{_ocp_meta['stem']}_{_ocp_meta['expectations']}.csv",
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

# --- Footnote — loss + vintage ---
st.divider()
_vintage = data_vintage() or {}
st.caption(
    "Loss L = w_π·Σπ² + w_u·Σu² + w_sm·Σ(Δi)² in deviations from baseline. "
    f"Model PyFRB/US 1.0.0 · data {_vintage.get('first_obs', '?')}–"
    f"{_vintage.get('last_obs', '?')}. Deviations from a stylised baseline — not a "
    "forecast; the optimum is a linear approximation (error shown above)."
)
