"""Debt-sustainability tab — a deficit shock under alternative fiscal responses.

You set a **deficit shock as a share of GDP** (how much bigger the deficit gets,
and for how many years). The tab shows the **actual federal debt/GDP path** — in
levels, anchored to the Fed's projection — under three assumptions about how the
government responds: no response, a gradual correction, or active stabilisation.
Debt/GDP that keeps climbing signals an unsustainable path.

VAR-only and long-horizon, the right choice for slow debt dynamics.
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
    DEBT_SOURCES,
    FISCAL_RULES,
    data_vintage,
    run_debt_scenario,
)

# One colour per fiscal response (rust = the non-stabilising, drifting case).
_RULE_COLORS = {
    "surplus_ratio": "#1f7a3d",       # green
    "debt_stabilization": "#1f4e79",  # blue
    "exogenous_taxes": "#c1440e",     # rust
}
_BASE = "#888"


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _cached_scenario(deficit_pct, deficit_years, fiscal_rules, policy_rule, start,
                     horizon, feedback, source, permanent):
    res = run_debt_scenario(
        deficit_pct=deficit_pct, deficit_years=deficit_years,
        fiscal_rules=list(fiscal_rules), policy_rule=policy_rule,
        start=start, horizon=horizon, feedback=feedback, source=source,
        permanent=permanent,
    )
    return {
        "window": [str(q) for q in res.window],
        "fiscal_rules": list(res.fiscal_rules),
        "baseline": {k: list(res.baseline_levels[k]) for k in res.baseline_levels},
        "levels": {r: {k: list(res.levels(r)[k]) for k in res.baseline_levels}
                   for r in res.fiscal_rules},
        "converged": dict(res.converged),
        "feedback": tuple(res.feedback),
    }


st.title("🏛️ Debt Sustainability")
st.markdown(
    "**Set a deficit shock** — a federal spending rise or tax cut, sized as a share of "
    "GDP — then see the **federal debt/GDP path** it produces under three assumptions "
    "about how the government responds. A path that **climbs and stays up** is "
    "unsustainable; one that **returns toward its starting level** is stabilised."
)

# --- What the shock is (source + size) ---
_c = st.columns([2, 1, 1, 1])
source = _c[0].selectbox(
    "Deficit source", list(DEBT_SOURCES), format_func=lambda k: DEBT_SOURCES[k][3],
    key="debt_src",
    help="Which federal lever widens the deficit. All are exogenised so the impulse "
    "is a clean share of GDP.\n\n(Personal tax cuts and S&L purchases are on the "
    "**Fiscal Multipliers** tab, not here: the debt-response rules below act *through* "
    "personal income taxes, and state & local spending doesn't touch the federal "
    "budget — so neither is a clean *federal-debt* lever.)",
)
deficit_pct = _c[1].number_input(
    "Size (% of GDP)", value=2.0, step=0.5, min_value=-5.0, max_value=10.0,
    help="Positive = bigger deficit (spending up / taxes down); negative = "
    "consolidation.",
)
deficit_years = _c[2].number_input(
    "Held for (years)", value=5, min_value=1, max_value=20, step=1,
    help="How long the change lasts before reverting (ignored if 'permanent').",
)
horizon = _c[3].selectbox(
    "Horizon", [20, 40, 60, 80], index=1, key="debt_h",
    format_func=lambda q: f"{q // 4} years", help="Debt/GDP evolves slowly — a long "
    "horizon shows whether it stabilises or drifts.",
)

_c2 = st.columns([1, 2, 1])
permanent = _c2[0].checkbox(
    "Permanent", value=False, key="debt_perm",
    help="Hold the change for the whole horizon (a lasting policy change, like a "
    "permanent tax cut) rather than reverting after the years above.",
)
policy_rule = _c2[1].selectbox(
    "Monetary response", list(ACTIVE_RULES), format_func=lambda k: ACTIVE_RULES[k][1],
    key="debt_mrule", help="The monetary-policy rule (it sets interest rates, hence "
    "debt-service costs). Held common across the fiscal responses.",
)

fiscal_rules = st.multiselect(
    "Government's fiscal response (compare side by side)", list(FISCAL_RULES),
    default=list(FISCAL_RULES), format_func=lambda k: FISCAL_RULES[k][1], key="debt_frules",
    help="How taxes respond to the shock — the crux of debt sustainability:\n\n"
    + "\n\n".join(f"**{lbl}** — {tip}" for _sw, lbl, tip in FISCAL_RULES.values()),
)

feedback = sc.render_feedback_control("debt")

run = st.button("▶ Run debt analysis", type="primary")
st.divider()


if run:
    if not fiscal_rules:
        st.warning("Pick at least one fiscal response to compare.")
        st.stop()
    _spin = ("Solving the deficit shock under each fiscal response"
             + (" with sovereign-risk feedback…" if feedback[0] or feedback[1] else "…"))
    with st.spinner(_spin):
        try:
            out = _cached_scenario(float(deficit_pct), int(deficit_years),
                                   tuple(fiscal_rules), policy_rule, "2026Q3",
                                   int(horizon), tuple(feedback), source, bool(permanent))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Debt analysis failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["debt"] = out

out = st.session_state.get("debt")
if out is None:
    st.info("Set a deficit shock and fiscal responses above, then press **Run debt "
            "analysis**.")
    st.stop()

rules = out["fiscal_rules"]
lv = out["levels"]
base = out["baseline"]
x = [pd.Period(q, freq="Q").to_timestamp() for q in out["window"]]

# Sovereign-risk feedback badge.
_fb = out.get("feedback", (0.0, 0.0))
if _fb[0] or _fb[1]:
    _spiralled = [FISCAL_RULES[r][1] for r in rules if not out.get("converged", {}).get(r, True)]
    _cap = (f"🔁 Sovereign-risk feedback on (debt {_fb[0]:g} bps/pp, deficit {_fb[1]:g} "
            "bps/pp) — debt/deficit feeds back into the 10-year rate.")
    if _spiralled:
        st.error(_cap + "  ⚠️ **Unstable debt spiral** under: " + ", ".join(_spiralled)
                 + " — debt and yields chase each other upward; read those paths as "
                 "*diverging*, not a settled level.")
    else:
        st.info(_cap)


def _level_fig(key, height=380, show_base=True):
    fig = go.Figure()
    if show_base:
        fig.add_trace(go.Scatter(x=x, y=base[key], name="Baseline (no shock)",
                                 line=dict(color=_BASE, width=1.5, dash="dot")))
    for rule in rules:
        fig.add_trace(go.Scatter(
            x=x, y=lv[rule][key], name=FISCAL_RULES[rule][1],
            line=dict(color=_RULE_COLORS.get(rule, "#666"), width=2.6),
        ))
    fig.update_layout(
        height=height, template="plotly_white", hovermode="x unified",
        title=dict(text=f"{DEBT_LABELS[key]}", x=0.01, font=dict(size=15)),
        yaxis_title="% of GDP" if key != "r_minus_g" else "pp (annualised)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=55, r=20, t=60, b=30),
    )
    return fig


# Headline: the actual debt/GDP level path.
st.plotly_chart(_level_fig("debt_gdp"), use_container_width=True)
st.caption(
    "Federal debt held by the public, **% of GDP** — the actual level, starting from "
    "the Fed's current projection. A line that keeps climbing (rust) is an "
    "unsustainable path; lines that turn back toward the baseline are stabilised."
)

# Where does debt/GDP end up? (levels, not deviations)
_last = out["window"][-1]
_start_lvl = base["debt_gdp"][0]
summary = pd.DataFrame([
    {
        "Fiscal response": FISCAL_RULES[r][1],
        "Debt/GDP now": f"{_start_lvl:.0f}%",
        f"Peak": f"{max(lv[r]['debt_gdp']):.0f}%",
        f"Debt/GDP @ {_last}": f"{lv[r]['debt_gdp'][-1]:.0f}%",
        "Change over horizon": f"{lv[r]['debt_gdp'][-1] - _start_lvl:+.0f} pp",
    }
    for r in rules
])
st.dataframe(summary, use_container_width=True, hide_index=True)

# Supporting fiscal ratios, 2×2 (levels).
grid = make_subplots(
    rows=2, cols=2,
    subplot_titles=[DEBT_LABELS[k] for k in ("primary_gdp", "budget_gdp", "interest_gdp", "r_minus_g")],
    vertical_spacing=0.13, horizontal_spacing=0.08,
)
for i, key in enumerate(("primary_gdp", "budget_gdp", "interest_gdp", "r_minus_g")):
    row, col = i // 2 + 1, i % 2 + 1
    grid.add_trace(go.Scatter(x=x, y=base[key], name="Baseline", legendgroup="base",
                              showlegend=(i == 0), line=dict(color=_BASE, width=1.3, dash="dot")),
                   row=row, col=col)
    for rule in rules:
        grid.add_trace(
            go.Scatter(x=x, y=lv[rule][key], name=FISCAL_RULES[rule][1],
                       line=dict(color=_RULE_COLORS.get(rule, "#666"), width=2.2),
                       legendgroup=rule, showlegend=(i == 0)),
            row=row, col=col,
        )
grid.update_layout(
    height=620, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    margin=dict(l=40, r=20, t=70, b=30),
)
st.plotly_chart(grid, use_container_width=True)

# Export.
_cols = {"quarter": out["window"], "baseline_debt_gdp": base["debt_gdp"]}
for r in rules:
    for k in ("debt_gdp", "primary_gdp", "budget_gdp", "interest_gdp", "r_minus_g"):
        _cols[f"{r}__{k}"] = lv[r][k]
st.download_button(
    "⬇ Debt paths (CSV)", data=pd.DataFrame(_cols).to_csv(index=False).encode("utf-8"),
    file_name="frbus_debt_sustainability.csv", mime="text/csv",
)

with st.expander("How to read this / caveats"):
    st.markdown(
        "- **Levels, anchored to the projection.** Every line is the *actual* ratio "
        "(debt, deficit, interest — all % of GDP), starting from the Fed's current "
        "projected level, so you read where debt/GDP *goes*, not just a deviation.\n"
        "- **The fiscal response is about taxes.** The deficit shock is a spending "
        "rise; the three responses differ in whether (and how fast) *taxes* adjust to "
        "pay for it. 'No response' borrows the whole thing; the others raise taxes to "
        "bring debt back.\n"
        "- **Primary balance** (budget ex-interest) is the fiscal stance; "
        "**net interest** is the carrying cost; **r − g** (interest rate minus growth) "
        "is what makes debt self-stabilise or spiral.\n"
        "- **Sovereign risk is optional.** Turn on the feedback above to let rising "
        "debt push up bond yields (and so debt service) — see its tooltip for the "
        "CBO/Laubach elasticities.\n"
        "- **Not a forecast.** The baseline is the Fed's SEP-consistent projection, "
        "not a prediction; read results as *plausible paths*, not an official "
        "debt-sustainability analysis."
    )

# --- Footnote ---
st.divider()
_vintage = data_vintage() or {}
st.caption(
    "Federal debt held by the public (gfdbtnp) as a share of nominal GDP. Deficit "
    "shock applied via federal transfers. Model PyFRB/US 1.0.0 · data "
    f"{_vintage.get('first_obs', '?')}–{_vintage.get('last_obs', '?')}. VAR "
    "expectations; deviations from a stylised baseline — not a forecast."
)
