"""Debt-sustainability tab — one fiscal shock under alternative closure rules.

Shows whether federal debt/GDP returns to baseline after a shock, and how, under
each of FRB/US's fiscal closure rules (surplus-ratio targeting, debt-ratio
stabilisation, or no stabilisation). Built for long horizons, since debt dynamics
play out over years. VAR-only — the appropriate, fast choice for long-run debt
paths (and it keeps a long stacked-time MCE solve off a memory-limited runtime).
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
from frbus_shock import (  # noqa: E402
    ACTIVE_RULES,
    DEBT_LABELS,
    DEBT_UNITS,
    FISCAL_RULES,
    data_vintage,
    run_debt_comparison,
)

# One colour per fiscal closure rule (rust = the non-stabilising, drifting case).
_RULE_COLORS = {
    "surplus_ratio": "#1f7a3d",       # green
    "debt_stabilization": "#1f4e79",  # blue
    "exogenous_taxes": "#c1440e",     # rust
}
_SHOCK_GROUPS = ["Fiscal & monetary", "Demand", "Prices & supply", "Financial", "External / global"]
_SHOCK_OPTIONS, _shock_label = sc.shock_options(_SHOCK_GROUPS, exclude_keys={"monetary"})


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _cached_debt(shocks_spec, fiscal_rules, policy_rule, start, horizon, feedback):
    shocks = []
    for kind, name, mag, dur in shocks_spec:
        if kind == "custom":
            shocks.append({"custom_variable": name, "magnitude": mag, "duration": dur})
        else:
            shocks.append({"key": name, "magnitude": mag, "duration": dur})
    res = run_debt_comparison(
        shocks=shocks, fiscal_rules=list(fiscal_rules), policy_rule=policy_rule,
        start=start, horizon=horizon, feedback=feedback,
    )
    return {
        "deviations": {r: res.deviations[r] for r in res.fiscal_rules},
        "fiscal_rules": list(res.fiscal_rules),
        "window": [str(q) for q in res.window],
        "policy_rule": res.policy_rule,
        "converged": dict(res.converged),
        "feedback": tuple(res.feedback),
    }


st.title("🏛️ Debt Sustainability")
st.caption(
    "Run a fiscal shock and see whether federal **debt/GDP** returns to its baseline "
    "path — and how — under each of FRB/US's fiscal closure rules. The contrast "
    "between *stabilising* and *non-stabilising* rules is the debt-sustainability "
    "question."
)

# --- Settings row ---
st.session_state.setdefault("debt_nsh", 1)
st.session_state.setdefault("debt_type_0", "fiscal_spending")  # a fiscal shock by default
_s = st.columns(4)
n_shocks = _s[0].selectbox(
    "Number of shocks", [1, 2], key="debt_nsh",
    help="The fiscal disturbance. Add a second lever (e.g. a term-premium spike) to "
    "represent market repricing of debt.",
)
policy_rule = _s[1].selectbox(
    "Monetary response", list(ACTIVE_RULES), format_func=lambda k: ACTIVE_RULES[k][1],
    key="debt_mrule",
    help="The monetary-policy rule in force (it drives interest rates, hence debt "
    "service). Held common across the fiscal rules so only fiscal policy differs.",
)
start = _s[2].selectbox("Start quarter", ["2026Q3", "2030Q1", "2035Q1"], key="debt_start")
horizon = _s[3].selectbox(
    "Horizon", [20, 40, 60, 80], index=1, key="debt_h",
    format_func=lambda q: f"{q} q ({q // 4} yrs)",
    help="Debt/GDP evolves slowly — long horizons show whether it stabilises or drifts.",
)

fiscal_rules = st.multiselect(
    "Fiscal closure rules to compare", list(FISCAL_RULES),
    default=list(FISCAL_RULES), format_func=lambda k: FISCAL_RULES[k][1], key="debt_frules",
    help="How the government stabilises the budget/debt. Compare them side by side:\n\n"
    + "\n\n".join(f"**{lbl}** — {tip}" for _sw, lbl, tip in FISCAL_RULES.values()),
)

feedback = sc.render_feedback_control("debt")

shock_specs = sc.render_shock_rows("debt", n_shocks, _SHOCK_OPTIONS, _shock_label)

run = st.button("▶ Run debt analysis", type="primary")
st.divider()


if run:
    if not shock_specs:
        st.warning("Add at least one shock.")
        st.stop()
    if not fiscal_rules:
        st.warning("Pick at least one fiscal closure rule to compare.")
        st.stop()
    _spin = ("Solving the shock under each fiscal closure rule"
             + (" with sovereign-risk feedback…" if feedback[0] or feedback[1] else "…"))
    with st.spinner(_spin):
        try:
            out = _cached_debt(tuple(shock_specs), tuple(fiscal_rules), policy_rule,
                               start, int(horizon), tuple(feedback))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Debt analysis failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["debt"] = out

out = st.session_state.get("debt")
if out is None:
    st.info("Pick a fiscal shock and closure rules above, then press **Run debt analysis**.")
    st.stop()

rules = out["fiscal_rules"]
devs = out["deviations"]
x = [pd.Period(q, freq="Q").to_timestamp() for q in out["window"]]

# Sovereign-risk feedback status: flag any closure rule whose feedback did not settle.
_fb = out.get("feedback", (0.0, 0.0))
if _fb[0] or _fb[1]:
    _spiralled = [FISCAL_RULES[r][1] for r in rules if not out.get("converged", {}).get(r, True)]
    _cap = (f"🔁 Sovereign-risk feedback on (debt {_fb[0]:g} bps/pp, deficit "
            f"{_fb[1]:g} bps/pp) — debt/deficit feeds back into the 10-year rate.")
    if _spiralled:
        st.error(_cap + "  ⚠️ **Unstable debt spiral** — the feedback did not reach a "
                 f"fixed point under: {', '.join(_spiralled)}. Debt and yields chase each "
                 "other upward; read those paths as *diverging*, not a settled level.")
    else:
        st.info(_cap + "  All shown rules reached a stable fixed point.")


def _line(key, height=320):
    fig = go.Figure()
    for rule in rules:
        fig.add_trace(go.Scatter(
            x=x, y=devs[rule][key], name=FISCAL_RULES[rule][1],
            line=dict(color=_RULE_COLORS.get(rule, "#666"), width=2.5),
        ))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.update_layout(
        height=height, template="plotly_white", hovermode="x unified",
        title=dict(text=f"{DEBT_LABELS[key]} — deviation ({DEBT_UNITS[key]})", x=0.01, font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


# Headline: debt/GDP.
st.plotly_chart(_line("debt_gdp", height=380), use_container_width=True)
st.caption(
    "A path that **returns toward zero** is a stabilising rule; one that **keeps "
    "drifting up** signals an unsustainable trajectory under that closure rule."
)

# End-of-horizon debt/GDP — the stabilises-vs-drifts read.
def _signed_peak(series):
    """Largest-magnitude (signed) deviation over the horizon."""
    return float(series.iloc[series.abs().to_numpy().argmax()])


_last = out["window"][-1]
summary = pd.DataFrame([
    {
        "Fiscal closure rule": FISCAL_RULES[r][1],
        f"Debt/GDP dev @ {_last}": round(float(devs[r]["debt_gdp"].iloc[-1]), 2),
        "Peak debt/GDP dev": round(_signed_peak(devs[r]["debt_gdp"]), 2),
        "Peak primary-balance dev": round(_signed_peak(devs[r]["primary_gdp"]), 2),
    }
    for r in rules
])
st.dataframe(summary, use_container_width=True, hide_index=True)

# The supporting fiscal ratios, 2×2.
grid = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"{DEBT_LABELS[k]} ({DEBT_UNITS[k]})"
                    for k in ("primary_gdp", "budget_gdp", "interest_gdp", "r_minus_g")],
    vertical_spacing=0.13, horizontal_spacing=0.08,
)
for i, key in enumerate(("primary_gdp", "budget_gdp", "interest_gdp", "r_minus_g")):
    row, col = i // 2 + 1, i % 2 + 1
    for rule in rules:
        grid.add_trace(
            go.Scatter(x=x, y=devs[rule][key], name=FISCAL_RULES[rule][1],
                       line=dict(color=_RULE_COLORS.get(rule, "#666"), width=2.2),
                       legendgroup=rule, showlegend=(i == 0)),
            row=row, col=col,
        )
    grid.add_hline(y=0, line=dict(color="#999", width=1), row=row, col=col)
grid.update_layout(
    height=620, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    margin=dict(l=40, r=20, t=70, b=30),
)
st.plotly_chart(grid, use_container_width=True)

# Export.
_csv = pd.concat({r: devs[r] for r in rules}, axis=1)
_csv.index = out["window"]
st.download_button(
    "⬇ Debt paths (CSV)", data=_csv.to_csv().encode("utf-8"),
    file_name=f"frbus_debt_sustainability_{start}.csv", mime="text/csv",
)

with st.expander("How to read this / caveats"):
    st.markdown(
        "- **Deviations from baseline.** Every line is the shock's effect *relative to "
        "the projection baseline* under that fiscal rule; zero means the shock has "
        "been fully absorbed. Debt/GDP returning to zero = stabilised; a persistent "
        "positive drift = an unsustainable path under that rule.\n"
        "- **Primary balance** (budget balance excluding interest) is the "
        "sustainability-relevant fiscal stance; **net interest / debt service** is the "
        "cost of carrying the debt; **r − g** is the interest-rate-minus-growth gap "
        "that drives debt dynamics.\n"
        "- **Sovereign risk is not endogenous.** FRB/US gives the mechanical debt "
        "dynamics (endogenous rates and their feedback into debt service) but does "
        "*not* widen the term premium as debt rises. To represent a market repricing "
        "of sovereign risk, add the **term-premium** lever as a second shock.\n"
        "- **VAR expectations**, long horizon. Debt paths are a long-run, "
        "backward-looking exercise; this tab is deliberately VAR-only."
    )

# --- Footnote — vintage ---
st.divider()
_vintage = data_vintage() or {}
st.caption(
    "Ratios are current-$ fiscal aggregates over nominal GDP (federal debt held by "
    f"the public, gfdbtnp). Model PyFRB/US 1.0.0 · data {_vintage.get('first_obs', '?')}–"
    f"{_vintage.get('last_obs', '?')}. Deviations from a stylised baseline — not a "
    "forecast or an official debt projection."
)
