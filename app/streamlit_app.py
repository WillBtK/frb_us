"""FRB/US shock-analysis dashboard (Streamlit Community Cloud entry point).

Pick a shock and its size, an expectations assumption, and a start quarter, then
run a fresh FRB/US simulation. The app charts the deviation-from-baseline paths
for GDP growth, unemployment, PCE inflation, and the federal funds rate, in two
scenarios — the active policy rule (with response) and the funds rate held at
its baseline path (without response) — and lets you export the run as CSV or a
chart image.

Deploy: point Streamlit Community Cloud at this repo with
``app/streamlit_app.py`` as the main file. No backend, no secrets, no system
packages required (see requirements.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``frbus_shock`` package importable when run by Streamlit Cloud, which
# executes this file directly (so ``src`` is not on the path yet).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from frbus_shock import (  # noqa: E402
    CATALOGUE,
    MODEL_VINTAGE,
    OUTPUT_VARS,
    data_vintage,
    deviations,
    expectations_choices,
    run_metadata,
    run_simulation,
    to_csv_bytes,
)

st.set_page_config(
    page_title="FRB/US Shock Analysis",
    page_icon="📈",
    layout="wide",
)

# Scenario colours (colour-blind friendly, distinct in light & dark).
_ACTIVE_COLOR = "#1f4e79"  # blue — with response
_HELD_COLOR = "#c1440e"  # rust — without response


# --------------------------------------------------------------------------- #
# Cached model runner — Streamlit reuses results across identical parameters.  #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def _cached_run(
    shock_key,
    magnitude,
    duration,
    expectations,
    start,
    horizon,
    custom_variable,
):
    """Run a simulation and return the deviation panels + metadata (picklable)."""
    result = run_simulation(
        shock_key=shock_key,
        magnitude=magnitude,
        duration=duration,
        expectations=expectations,
        start=start,
        horizon=horizon,
        custom_variable=custom_variable or None,
    )
    return {
        "meta": run_metadata(result),
        "active": deviations(result, "active"),
        "held": deviations(result, "held"),
        "csv": to_csv_bytes(result),
        "window": [str(q) for q in result.window],
    }


def _dates(window):
    return [pd.Period(q, freq="Q").to_timestamp() for q in window]


def _build_figure(active: pd.DataFrame, held: pd.DataFrame, window) -> go.Figure:
    """2×2 grid of deviation charts, one per output variable."""
    labels = [f"{lbl} ({unit})" for lbl, unit in OUTPUT_VARS.values()]
    fig = make_subplots(rows=2, cols=2, subplot_titles=labels, vertical_spacing=0.14)
    x = _dates(window)
    positions = {var: (i // 2 + 1, i % 2 + 1) for i, var in enumerate(OUTPUT_VARS)}
    for var, (row, col) in positions.items():
        show_legend = var == list(OUTPUT_VARS)[0]
        fig.add_trace(
            go.Scatter(
                x=x, y=active[var], name="With response (active rule)",
                line=dict(color=_ACTIVE_COLOR, width=2.5),
                legendgroup="active", showlegend=show_legend,
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=held[var], name="Without response (funds rate held)",
                line=dict(color=_HELD_COLOR, width=2.5, dash="dash"),
                legendgroup="held", showlegend=show_legend,
            ),
            row=row, col=col,
        )
        fig.add_hline(y=0, line=dict(color="#999", width=1), row=row, col=col)
        fig.update_yaxes(title_text="pp dev.", row=row, col=col)
    fig.update_layout(
        height=680,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=90, b=30),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — controls                                                          #
# --------------------------------------------------------------------------- #
st.sidebar.title("Shock configuration")

shock_labels = {k: v.label for k, v in CATALOGUE.items()}
shock_labels["__custom__"] = "Custom variable…"
shock_key = st.sidebar.selectbox(
    "Shock type",
    options=list(shock_labels),
    format_func=lambda k: shock_labels[k],
)

custom_variable = None
if shock_key == "__custom__":
    custom_variable = st.sidebar.text_input(
        "FRB/US variable (endogenous name)",
        value="eco",
        help="The shock is applied to <variable>_aerr in native model units.",
    )
    default_mag, default_dur, unit_label = 0.01, 4, "model units (native)"
    st.sidebar.caption("Custom shocks are in native add-factor units — see docs.")
else:
    spec = CATALOGUE[shock_key]
    default_mag, default_dur, unit_label = (
        spec.default_magnitude,
        spec.default_duration,
        spec.user_unit,
    )
    st.sidebar.caption(spec.description)
    st.sidebar.caption(f"↳ {spec.sign_note}")

magnitude = st.sidebar.number_input(
    f"Magnitude ({unit_label})", value=float(default_mag), step=abs(float(default_mag)) / 4 or 0.1
)
duration = st.sidebar.slider("Duration (quarters shock is held on)", 1, 20, int(default_dur))

st.sidebar.divider()

exp_choices = expectations_choices()
expectations = st.sidebar.radio(
    "Expectations assumption",
    options=list(exp_choices),
    format_func=lambda k: exp_choices[k],
    help="VAR = backward-looking. MCE = model-consistent / rational (slower).",
)
if expectations == "mce":
    st.sidebar.warning("Model-consistent expectations solve over a long horizon and take longer.")

start = st.sidebar.selectbox(
    "Start quarter",
    options=["2040Q1", "2035Q1", "2045Q1", "2030Q1"],
    help="Simulations run off the model's projection baseline; a date well into "
    "the projection avoids edge effects.",
)
horizon = st.sidebar.slider("Horizon (quarters shown)", 8, 40, 20)

run_clicked = st.sidebar.button("▶ Run simulation", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main panel                                                                  #
# --------------------------------------------------------------------------- #
st.title("📈 FRB/US Shock Analysis")
st.markdown(
    "Compare how a macroeconomic shock plays out **with** an active monetary-"
    "policy rule versus **without** a monetary response (the federal funds rate "
    "held at its baseline path). Built on the Federal Reserve's "
    "[FRB/US model](https://www.federalreserve.gov/econres/us-models-about.htm)."
)

vintage = data_vintage() or {}
cols = st.columns(3)
cols[0].metric("Model vintage", MODEL_VINTAGE.split(" (")[0])
cols[1].metric("Data range", f"{vintage.get('first_obs', '?')} – {vintage.get('last_obs', '?')}")
cols[2].metric("Series in dataset", vintage.get("n_variables", "?"))

st.info(
    "**Caveat:** the baseline projection in the Fed's dataset follows the FOMC's "
    "Summary of Economic Projections where available and a model-guided "
    "extrapolation beyond it. That extrapolation **is not a forecast** — treat "
    "results as *deviations from a stylised baseline*, not predictions.",
    icon="⚠️",
)

if run_clicked:
    with st.spinner("Running FRB/US — solving baseline, active rule, and held-rate scenarios…"):
        try:
            out = _cached_run(
                None if shock_key == "__custom__" else shock_key,
                float(magnitude),
                int(duration),
                expectations,
                start,
                int(horizon),
                custom_variable,
            )
        except Exception as exc:  # noqa: BLE001 — surface solver/convergence errors
            st.error(
                f"Simulation failed: {type(exc).__name__}: {exc}\n\n"
                "Try a smaller magnitude or shorter duration — large sustained "
                "shocks can prevent the model from converging."
            )
            st.stop()

    meta = out["meta"]
    st.subheader(f"{meta['shock']} — {meta['magnitude']} {meta['magnitude_unit']}, "
                 f"{meta['duration_quarters']}q, {meta['expectations'].upper()} expectations")

    fig = _build_figure(out["active"], out["held"], out["window"])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Solid = with policy response (active rule). Dashed = without response "
        "(funds rate held at baseline). All values are deviations from baseline "
        "in percentage points."
    )

    # ------------------------------ Exports ------------------------------- #
    st.subheader("Export this run")
    ec1, ec2, ec3 = st.columns(3)
    stem = f"frbus_{meta['shock_key'].replace(':', '_')}_{meta['expectations']}_{meta['start']}"

    ec1.download_button(
        "⬇ Deviations (CSV)",
        data=out["csv"],
        file_name=f"{stem}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    try:
        png = fig.to_image(format="png", scale=2, width=1100, height=680)
        ec2.download_button(
            "⬇ Chart (PNG)",
            data=png,
            file_name=f"{stem}.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception:  # noqa: BLE001 — kaleido missing/unavailable
        ec2.info("PNG export needs the `kaleido` package.")

    ec3.download_button(
        "⬇ Chart (HTML)",
        data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
        file_name=f"{stem}.html",
        mime="text/html",
        use_container_width=True,
    )

    with st.expander("Show deviation table"):
        tidy = pd.concat(
            {"active": out["active"], "held": out["held"]}, axis=1
        )
        tidy.index = out["window"]
        st.dataframe(tidy.round(4), use_container_width=True)

    with st.expander("Run details / reproducibility"):
        st.json(meta)
        st.caption(
            "Funds-rate hold mechanism: dmpex=1, dmpintay=0, rfffix=baseline rff, "
            "rff_trac=rffrule_trac=0 (the model's exogenous-rate switch)."
        )
else:
    st.markdown(
        "👈 Configure a shock in the sidebar and press **Run simulation**. "
        "Each run solves the FRB/US model fresh, so expect a few seconds "
        "(longer under model-consistent expectations)."
    )
    st.markdown(
        "See [`docs/use_cases.md`](https://github.com/willbtk/long_r_star/blob/"
        "main/docs/use_cases.md) for the range of analyses FRB/US supports and "
        "which sit beyond this first-pass build."
    )
