"""Shared multi-shock configuration UI for the dashboard tabs.

Renders the **scenario loader** and the editable **per-shock rows** (type /
magnitude / duration) once, so the Shock Analysis and Optimal Control tabs stay
in sync — both read the same shock catalogue and named scenarios from the
``frbus_shock`` package. Widget state is namespaced by ``prefix`` so the two tabs
never collide in Streamlit's (app-wide) session state.

A tab supplies its own ``group_order`` and optional ``exclude_keys``; the Optimal
Control tab, for instance, excludes the ``monetary`` lever (the funds rate is its
control variable, so a shock to the policy rule is inert / incoherent there) and
any scenario that uses it is dropped from the loader automatically.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import streamlit as st

from frbus_shock import CATALOGUE, SCENARIO_PRESETS

CUSTOM = "__custom__"

# One shock row, as consumed by the cached runners: (kind, name, magnitude, duration).
ShockSpec = Tuple[str, str, float, int]


def shock_options(
    group_order: Sequence[str],
    exclude_keys: Sequence[str] = (),
    allow_custom: bool = True,
) -> Tuple[List[str], Callable[[str], str]]:
    """Return ``(options, label_func)`` for a shock-type selectbox.

    Catalogue keys come first, ordered by ``group_order`` (then any stragglers),
    with ``exclude_keys`` removed; the raw-variable escape hatch is appended last
    when ``allow_custom``.
    """
    exclude = set(exclude_keys)
    keys = [k for g in group_order for k, s in CATALOGUE.items()
            if s.group == g and k not in exclude]
    keys += [k for k in CATALOGUE if k not in keys and k not in exclude]
    options = keys + ([CUSTOM] if allow_custom else [])

    def label(k: str) -> str:
        if k == CUSTOM:
            return "Advanced — raw FRB/US variable…"
        s = CATALOGUE[k]
        return f"{s.group} · {s.label}"

    return options, label


def available_scenarios(exclude_keys: Sequence[str] = ()) -> dict:
    """Scenario presets whose every shock lever is allowed for this tab."""
    exclude = set(exclude_keys)
    return {
        name: preset
        for name, preset in SCENARIO_PRESETS.items()
        if all(s["key"] not in exclude for s in preset["shocks"])
    }


def render_scenario_loader(prefix: str, exclude_keys: Sequence[str] = ()) -> None:
    """Render the 'Load a scenario' row; on click, populate the shock widgets.

    Writes ``{prefix}_nsh`` and ``{prefix}_{type,mag,dur}_{i}`` then reruns, so the
    shock rows below pick the scenario up as editable defaults.
    """
    presets = available_scenarios(exclude_keys)
    if not presets:
        return
    cols = st.columns([3, 1])
    name = cols[0].selectbox(
        "Load a scenario (optional)", ["—"] + list(presets), key=f"{prefix}_scn",
        help="Named multi-shock scenarios. Load one to populate the shocks below, "
        "then edit freely before running.",
    )
    if name != "—":
        cols[0].caption(f"↳ {presets[name]['blurb']}")
    if cols[1].button(
        "Load scenario", key=f"{prefix}_scnbtn",
        disabled=name == "—", use_container_width=True,
    ):
        shocks = SCENARIO_PRESETS[name]["shocks"]
        st.session_state[f"{prefix}_nsh"] = len(shocks)
        for i, s in enumerate(shocks):
            st.session_state[f"{prefix}_type_{i}"] = s["key"]
            st.session_state[f"{prefix}_mag_{i}"] = float(s["magnitude"])
            st.session_state[f"{prefix}_dur_{i}"] = int(s["duration"])
        st.rerun()


def render_shock_rows(
    prefix: str,
    n_shocks: int,
    options: Sequence[str],
    label_func: Callable[[str], str],
) -> List[ShockSpec]:
    """Render ``n_shocks`` rows of (type | magnitude | duration); return the specs.

    Widget state is seeded once (via ``setdefault``) rather than passed as a
    default each run, so a scenario load can write these keys without tripping
    Streamlit's default-vs-session-state warning.
    """
    non_custom = [o for o in options if o != CUSTOM]
    specs: List[ShockSpec] = []
    for i in range(int(n_shocks)):
        cols = st.columns([3, 1, 1])
        st.session_state.setdefault(
            f"{prefix}_type_{i}", non_custom[min(i, len(non_custom) - 1)]
        )
        key = cols[0].selectbox(
            f"Shock {i + 1}" if n_shocks > 1 else "Shock",
            options=list(options), format_func=label_func, key=f"{prefix}_type_{i}",
        )
        if key == CUSTOM:
            name = cols[0].text_input(
                "FRB/US variable", value="eco", key=f"{prefix}_var_{i}",
                help="Shocks <variable>_aerr in native model units — for power users.",
            )
            dmag, ddur, unit, kind = 0.01, 4, "native units", "custom"
        else:
            spec = CATALOGUE[key]
            dmag, ddur, unit, kind = (
                spec.default_magnitude, spec.default_duration, spec.user_unit, "catalogue",
            )
            name = key
        st.session_state.setdefault(f"{prefix}_mag_{i}", float(dmag))
        st.session_state.setdefault(f"{prefix}_dur_{i}", int(ddur))
        mag = cols[1].number_input(
            f"Magnitude ({unit})", step=abs(float(dmag)) / 4 or 0.1, key=f"{prefix}_mag_{i}",
        )
        dur = cols[2].number_input(
            "Duration (q)", min_value=1, max_value=20, step=1, key=f"{prefix}_dur_{i}",
        )
        specs.append((kind, name, float(mag), int(dur)))
        if key != CUSTOM:
            st.caption(f"↳ {CATALOGUE[key].description}")
    return specs
