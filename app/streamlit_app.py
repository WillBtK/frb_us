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

import math
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
    DEFAULT_OUTPUTS,
    MODEL_VINTAGE,
    OUTPUT_BY_KEY,
    OUTPUT_CATALOGUE,
    OUTPUT_GROUPS,
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
    variables,
):
    """Run a simulation and return the deviation panels + metadata (picklable).

    ``variables`` is a tuple of output keys; it is part of the cache key so a
    different output selection re-slices without necessarily re-solving (the
    heavy solve is itself memoised at the library level by run parameters).
    """
    result = run_simulation(
        shock_key=shock_key,
        magnitude=magnitude,
        duration=duration,
        expectations=expectations,
        start=start,
        horizon=horizon,
        custom_variable=custom_variable or None,
    )
    keys = list(variables)
    return {
        "meta": run_metadata(result),
        "active": deviations(result, "active", keys),
        "held": deviations(result, "held", keys),
        "csv": to_csv_bytes(result, keys),
        "window": [str(q) for q in result.window],
        "variables": keys,
    }


def _dates(window):
    return [pd.Period(q, freq="Q").to_timestamp() for q in window]


def _build_figure(active: pd.DataFrame, held: pd.DataFrame, window, variables) -> go.Figure:
    """Grid of deviation charts, one per selected output variable (2 columns)."""
    cols = 2 if len(variables) > 1 else 1
    rows = math.ceil(len(variables) / cols)
    titles = [
        f"{OUTPUT_BY_KEY[v].label} ({OUTPUT_BY_KEY[v].unit})" for v in variables
    ]
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.10 if rows <= 2 else 0.06
    )
    x = _dates(window)
    for i, var in enumerate(variables):
        row, col = i // cols + 1, i % cols + 1
        show_legend = i == 0
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
        fig.update_yaxes(title_text=f"{OUTPUT_BY_KEY[var].unit} dev.", row=row, col=col)
    fig.update_layout(
        height=max(340, 300 * rows),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=90, b=30),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — controls                                                          #
# --------------------------------------------------------------------------- #
def _summary_from_out(out, horizons):
    """Peak effect (+ quarter) and effect at chosen horizons, from cached frames.

    Computed from the already-solved deviation frames, so changing the horizons
    re-renders instantly without re-running the model.
    """
    n = len(out["window"])
    quarters = out["window"]
    scen_frames = [
        ("With policy response (active rule)", out["active"]),
        ("Without response (funds rate held)", out["held"]),
    ]
    rows = []
    for scen_label, dev in scen_frames:
        for key in out["variables"]:
            var = OUTPUT_BY_KEY[key]
            s = dev[key]
            pos = int(s.abs().to_numpy().argmax())  # index of largest |deviation|
            row = {
                "Variable": var.label,
                "Unit": var.unit,
                "Scenario": scen_label,
                "Peak": round(float(s.iloc[pos]), 3),
                "Peak quarter": quarters[pos],
            }
            for h in horizons:
                row[f"@{h}q"] = round(float(s.iloc[h]), 3) if 0 <= h < n else None
            rows.append(row)
    return pd.DataFrame(rows)


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

def _start_quarter_options():
    """Chronologically-sorted start quarters, defaulting to the current quarter.

    The baseline is a fixed-vintage FRB/US projection, so "now" means the current
    quarter *within that projection* — not data conditioned on realised recent
    history. By ~2025 the baseline is already on its long-run path, so every
    option here solves cleanly; "long-run" matches the Fed demos' 2040Q1
    convention for textbook-clean impulse responses.
    """
    today_q = pd.Period(pd.Timestamp.today(), freq="Q")
    proj_start = pd.Period("2022Q1", freq="Q")  # permissive floor for start options
    vint = data_vintage() or {}
    try:
        last_obs = pd.Period(vint.get("last_obs", "2176Q1"), freq="Q")
    except Exception:  # noqa: BLE001
        last_obs = pd.Period("2176Q1", freq="Q")
    near = min(max(today_q, proj_start), last_obs - 40)  # leave room for the horizon
    longrun = pd.Period("2040Q1", freq="Q")
    cands = [near, near + 4, near + 8, near + 20, longrun]
    opts = sorted({p for p in cands if proj_start <= p <= last_obs - 8})
    return [str(p) for p in opts], str(near), str(longrun)


_START_OPTS, _START_NEAR, _START_LONGRUN = _start_quarter_options()


def _fmt_start(s):
    if s == _START_NEAR:
        return f"{s} — now"
    if s == _START_LONGRUN:
        return f"{s} — long-run baseline"
    return s


start = st.sidebar.selectbox(
    "Start quarter (when the shock hits)",
    options=_START_OPTS,
    index=_START_OPTS.index(_START_NEAR),
    format_func=_fmt_start,
    help="'Now' starts the shock in the current quarter. The baseline carries "
    "historical actuals up to its jump-off (see the data range above) and a "
    "model-guided projection beyond; 'now' typically sits just past the jump-off. "
    "Every option solves cleanly; 'long-run' matches the Fed demos' 2040Q1 "
    "convention for textbook-clean impulse responses.",
)
horizon = st.sidebar.slider("Horizon (quarters shown)", 8, 40, 20)

st.sidebar.divider()

# --- Output variable selector (grouped) ---
st.sidebar.markdown("**Output variables**")
_selected_outputs: list = []
with st.sidebar.expander(
    "Choose what to chart", expanded=False
):
    st.caption(
        "Rates & inflation show percentage-point deviations; levels (GDP, "
        "consumption, investment) show percent deviations from baseline."
    )
    for _group in OUTPUT_GROUPS:
        _group_vars = [v for v in OUTPUT_CATALOGUE if v.group == _group]
        _defaults = [v.key for v in _group_vars if v.key in DEFAULT_OUTPUTS]
        _picked = st.multiselect(
            _group,
            options=[v.key for v in _group_vars],
            default=_defaults,
            format_func=lambda k: f"{OUTPUT_BY_KEY[k].label} ({OUTPUT_BY_KEY[k].unit})",
            key=f"outsel_{_group}",
        )
        _selected_outputs.extend(_picked)

# Preserve catalogue order; fall back to the four defaults if nothing is picked.
selected_outputs = [v.key for v in OUTPUT_CATALOGUE if v.key in _selected_outputs]
if not selected_outputs:
    selected_outputs = list(DEFAULT_OUTPUTS)
st.sidebar.caption(f"{len(selected_outputs)} variable(s) selected")

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
    "results as *deviations from a stylised baseline*, not predictions. The "
    "dataset carries historical actuals up to its jump-off and a projection "
    "beyond (see the data range above); it is refreshed from the Fed by CI, but "
    "the projection past the jump-off is still not a forecast.",
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
                tuple(selected_outputs),
            )
        except Exception as exc:  # noqa: BLE001 — surface solver/convergence errors
            st.error(
                f"Simulation failed: {type(exc).__name__}: {exc}\n\n"
                "Try a smaller magnitude or shorter duration — large sustained "
                "shocks can prevent the model from converging."
            )
            st.stop()
    # Persist so table/horizon controls re-render without re-solving.
    st.session_state["run"] = out

out = st.session_state.get("run")

if out is None:
    st.markdown(
        "👈 Configure a shock in the sidebar and press **Run simulation**. "
        "Each run solves the FRB/US model fresh, so expect a few seconds "
        "(longer under model-consistent expectations)."
    )
    st.markdown(
        "See [`docs/use_cases.md`](https://github.com/willbtk/frb_us/blob/"
        "main/docs/use_cases.md) for the range of analyses FRB/US supports and "
        "which sit beyond this first-pass build."
    )
else:
    meta = out["meta"]
    st.subheader(f"{meta['shock']} — {meta['magnitude']} {meta['magnitude_unit']}, "
                 f"{meta['duration_quarters']}q, {meta['expectations'].upper()} expectations")
    st.caption("Showing the most recent run (parameters above). Press **Run "
               "simulation** to apply sidebar changes.")

    fig = _build_figure(out["active"], out["held"], out["window"], out["variables"])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Solid = with policy response (active rule). Dashed = without response "
        "(funds rate held at baseline). Deviations from baseline: percentage "
        "points for rates/inflation, percent for level variables."
    )

    # -------------------- Summary: peak + effect at horizon -------------------- #
    st.subheader("Summary — peak effect and effect at a chosen horizon")
    _n = len(out["window"])
    _hopts = [h for h in (1, 2, 4, 6, 8, 12, 16, 20, 24) if h < _n]
    _hdefault = [h for h in (4, 8) if h in _hopts] or _hopts[:2]
    horizons = st.multiselect(
        "Effect this many quarters after the shock hits (0 = impact quarter)",
        options=_hopts,
        default=_hdefault,
        help="Peak = the largest-magnitude deviation within the shown horizon, "
        "and the quarter it occurs. The @Nq columns read the deviation N quarters "
        "after the shock hits.",
    )
    summary = _summary_from_out(out, sorted({int(h) for h in horizons}))
    st.dataframe(summary, use_container_width=True, hide_index=True)

    stem = f"frbus_{meta['shock_key'].replace(':', '_')}_{meta['expectations']}_{meta['start']}"
    st.download_button(
        "⬇ Summary table (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_summary.csv",
        mime="text/csv",
    )

    # ------------------------------ Exports ------------------------------- #
    st.subheader("Export this run")
    ec1, ec2, ec3 = st.columns(3)

    ec1.download_button(
        "⬇ Full paths (CSV)",
        data=out["csv"],
        file_name=f"{stem}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    try:
        png = fig.to_image(format="png", scale=2, width=1100, height=fig.layout.height or 680)
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

    with st.expander("Show full deviation table"):
        tidy = pd.concat({"active": out["active"], "held": out["held"]}, axis=1)
        tidy.index = out["window"]
        st.dataframe(tidy.round(4), use_container_width=True)

    with st.expander("Run details / reproducibility"):
        st.json(meta)
        st.caption(
            "Funds-rate hold mechanism: dmpex=1, dmpintay=0, rfffix=baseline rff, "
            "rff_trac=rffrule_trac=0 (the model's exogenous-rate switch)."
        )
