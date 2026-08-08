"""Debt fan-chart tab — debt/GDP under macroeconomic uncertainty (stochastic).

Runs FRB/US's stochastic simulation (block-bootstrap of the 52 estimated equation
residuals) to draw a fan of debt/GDP paths around a baseline or a chosen fiscal
scenario — the standard IMF/CBO debt-sustainability visual. Each draw is a full VAR
solve (~0.7s), so a run takes a minute or two; results are cached.
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

import _shock_controls as sc  # noqa: E402
from frbus_shock import (  # noqa: E402
    ACTIVE_RULES,
    FISCAL_RULES,
    data_vintage,
    debt_fan_chart,
)

_SHOCK_GROUPS = ["Fiscal & monetary", "Demand", "Prices & supply", "Financial", "External / global"]
_SHOCK_OPTIONS, _shock_label = sc.shock_options(_SHOCK_GROUPS, exclude_keys={"monetary"})
_BAND = "#1f4e79"  # blue


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def _cached_fan(shocks_spec, fiscal_rule, policy_rule, start, horizon, nrepl, feedback):
    shocks = []
    for kind, name, mag, dur in shocks_spec:
        if kind == "custom":
            shocks.append({"custom_variable": name, "magnitude": mag, "duration": dur})
        else:
            shocks.append({"key": name, "magnitude": mag, "duration": dur})
    r = debt_fan_chart(
        shocks=shocks, fiscal_rule=fiscal_rule, policy_rule=policy_rule,
        start=start, horizon=horizon, nrepl=nrepl, feedback=feedback,
    )
    return {
        "window": [str(q) for q in r.window],
        "baseline": list(r.baseline_debt),
        "central": list(r.central_debt),
        "debt_pct": {int(p): list(v) for p, v in r.debt_pct.items()},
        "central_primary": list(r.central_primary),
        "primary_pct": {int(p): list(v) for p, v in r.primary_pct.items()},
        "prob_rising": r.prob_rising,
        "nrepl": r.nrepl,
        "resid_window": r.resid_window,
        "fiscal_rule": r.fiscal_rule,
        "feedback": tuple(r.feedback),
        "feedback_converged": r.feedback_converged,
    }


st.title("📊 Debt Fan Charts")
st.caption(
    "The distribution of federal **debt/GDP** paths under macroeconomic uncertainty, "
    "from FRB/US's stochastic simulation (a block-bootstrap of the model's 52 "
    "estimated equation residuals) — the IMF/CBO debt-sustainability fan chart."
)

# --- Settings ---
st.session_state.setdefault("fan_nsh", 0)
_s = st.columns(4)
n_shocks = _s[0].selectbox(
    "Deterministic shocks", [0, 1, 2], key="fan_nsh",
    help="0 = a fan around the baseline (pure macro uncertainty). Add a shock to "
    "centre the fan on a fiscal scenario.",
)
fiscal_rule = _s[1].selectbox(
    "Fiscal response", list(FISCAL_RULES),
    index=list(FISCAL_RULES).index("surplus_ratio"),
    format_func=lambda k: FISCAL_RULES[k][1], key="fan_frule",
    help="How the government stabilises debt. Compare a stabilising rule (a bounded "
    "fan) with 'no fiscal response' (a fan that widens and drifts up).",
)
policy_rule = _s[2].selectbox(
    "Monetary response", list(ACTIVE_RULES), format_func=lambda k: ACTIVE_RULES[k][1],
    key="fan_mrule",
)
horizon = _s[3].selectbox(
    "Horizon", [20, 40, 60], index=1, key="fan_h",
    format_func=lambda q: f"{q} q ({q // 4} yrs)",
)

_r2 = st.columns([1, 3])
nrepl = _r2[0].selectbox(
    "Draws", [50, 100, 200], index=1, key="fan_nrepl",
    format_func=lambda n: f"{n} (~{round(n * 0.7)}s)",
    help="More draws = smoother bands but a longer run. Each is a full model solve.",
)
_r2[1].caption(
    "⏳ Stochastic simulation runs the model once per draw — expect **roughly "
    f"{round(int(nrepl) * 0.7)}s** (longer at long horizons). The result is cached, "
    "so re-viewing is instant. Residuals are drawn from 1975Q1–2019Q4 (ex-COVID)."
)

feedback = sc.render_feedback_control("fan")

shock_specs = sc.render_shock_rows("fan", n_shocks, _SHOCK_OPTIONS, _shock_label)

run = st.button("▶ Run fan chart", type="primary")
st.divider()


if run:
    with st.spinner(f"Running {nrepl} stochastic replications…"):
        try:
            out = _cached_fan(tuple(shock_specs), fiscal_rule, policy_rule, start := "2026Q3",
                              int(horizon), int(nrepl), tuple(feedback))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Fan chart failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["fan"] = out

out = st.session_state.get("fan")
if out is None:
    st.info("Choose a fiscal closure rule (and optionally a shock), then press **Run "
            "fan chart**. The first run takes a minute or two.")
    st.stop()

_fb = out.get("feedback", (0.0, 0.0))
if _fb[0] or _fb[1]:
    _msg = (f"🔁 Sovereign-risk feedback on (debt {_fb[0]:g} bps/pp, deficit {_fb[1]:g} "
            "bps/pp), calibrated on the deterministic central path and imposed across "
            "the draws.")
    if out.get("feedback_converged", True):
        st.info(_msg)
    else:
        st.error(_msg + "  ⚠️ **Unstable debt spiral** — the central-path feedback did "
                 "not settle; read the fan as *diverging*.")

x = [pd.Period(q, freq="Q").to_timestamp() for q in out["window"]]
pct = out["debt_pct"]
H = len(x) - 1


def _fan_figure(pctiles, central, baseline, title, ylab):
    fig = go.Figure()
    # 80% band (10–90).
    fig.add_trace(go.Scatter(x=x, y=pctiles[90], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=pctiles[10], fill="tonexty", fillcolor="rgba(31,78,121,0.15)",
                             line=dict(width=0), name="10–90%", hoverinfo="skip"))
    # 50% band (25–75).
    fig.add_trace(go.Scatter(x=x, y=pctiles[75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=pctiles[25], fill="tonexty", fillcolor="rgba(31,78,121,0.30)",
                             line=dict(width=0), name="25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=pctiles[50], line=dict(color=_BAND, width=2.5), name="Median"))
    if central is not None:
        fig.add_trace(go.Scatter(x=x, y=central, line=dict(color="#c1440e", width=2, dash="dash"),
                                 name="Central (no draws)"))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=x, y=baseline, line=dict(color="#888", width=1.5, dash="dot"),
                                 name="Baseline (no shock)"))
    fig.update_layout(
        height=430, template="plotly_white", hovermode="x unified",
        title=dict(text=title, x=0.01, font=dict(size=15)),
        yaxis_title=ylab,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=55, r=20, t=60, b=30),
    )
    return fig


st.plotly_chart(
    _fan_figure(pct, out["central"], out["baseline"],
                f"Federal debt held by the public — {FISCAL_RULES[out['fiscal_rule']][1]}",
                "% of GDP"),
    use_container_width=True,
)

# Fiscal-risk readout.
_last = out["window"][-1]
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Median debt/GDP @ {_last}", f"{pct[50][H]:.1f}%")
m2.metric("80% range @ horizon", f"{pct[10][H]:.0f}–{pct[90][H]:.0f}%")
m3.metric("P(debt/GDP rising)", f"{out['prob_rising']:.0%}")
m4.metric("Replications", f"{out['nrepl']}")
st.caption(
    "The **median** is the central tendency across draws; the shaded bands are the "
    "50% (25–75%) and 80% (10–90%) probability ranges. Under a stabilising fiscal "
    "rule the fan stays bounded; under **no stabilisation** it widens and drifts up "
    "— the fiscal-risk signature. Switch the closure rule above to compare."
)

with st.expander("Primary balance fan"):
    st.plotly_chart(
        _fan_figure(out["primary_pct"], out["central_primary"], None,
                    "Primary balance (budget balance ex. interest)", "% of GDP"),
        use_container_width=True,
    )

# Export the percentile bands.
_df = pd.DataFrame({"quarter": out["window"], "baseline": out["baseline"], "central": out["central"]})
for p in (10, 25, 50, 75, 90):
    _df[f"debt_p{p}"] = out["debt_pct"][p]
st.download_button(
    "⬇ Fan percentiles (CSV)", data=_df.to_csv(index=False).encode("utf-8"),
    file_name="frbus_debt_fan.csv", mime="text/csv",
)

with st.expander("Method & caveats"):
    st.markdown(
        "- **Stochastic simulation.** Each replication solves the full model with a "
        "block-bootstrap draw of the 52 estimated equation residuals over the "
        "simulation window; the fan is the distribution across draws. Residuals are "
        "drawn from **1975Q1–2019Q4**, excluding the COVID outliers that would "
        "otherwise dominate.\n"
        "- **Levels, not deviations.** The chart shows the debt/GDP *level* — the "
        "median tends to sit a little above the deterministic 'central' line because "
        "debt dynamics are convex (adverse draws raise debt more than favourable "
        "draws lower it).\n"
        "- **Not a forecast.** The baseline is the Fed's SEP-consistent projection, "
        "not a prediction; read the fan as *plausible dispersion around a stylised "
        "path*, not an official debt-sustainability analysis.\n"
        "- **Sovereign risk is not endogenous** — the fan reflects macro shocks, not "
        "a debt-triggered term-premium blowout. Add the term-premium lever to the "
        "deterministic shock to layer that channel in.\n"
        "- **VAR expectations**, single-threaded for memory safety."
    )

# --- Footnote ---
st.divider()
_vintage = data_vintage() or {}
st.caption(
    f"Stochastic sim over {out['resid_window'][0]}–{out['resid_window'][1]} residuals · "
    f"{out['nrepl']} draws · Model PyFRB/US 1.0.0 · data {_vintage.get('first_obs','?')}–"
    f"{_vintage.get('last_obs','?')}. Deviations from a stylised baseline — not a forecast."
)
