# FRB/US Shock Analysis Dashboard

An interactive dashboard for exploring US macroeconomic shocks with the Federal
Reserve Board's **FRB/US** model (via [PyFRB/US](https://www.federalreserve.gov/econres/us-models-python.htm)).
Pick a shock, an expectations assumption, and a monetary-policy setting; the app
runs a fresh FRB/US simulation and charts the deviation-from-baseline paths for
GDP growth, unemployment, PCE inflation, and the federal funds rate — **with** an
active policy rule and **without** a monetary response (funds rate held at
baseline) side by side.

> ⚠️ This is an independent project. It is **not** affiliated with or endorsed by
> the Federal Reserve. FRB/US and its data are public-domain works of the Fed.

---

## What it does

- **Shocks** — a friendly-named library (19 shocks, each mapped to an FRB/US
  add-factor lever and empirically sign-checked), grouped as **Demand** (household
  consumption, durables, housing, business investment, exports, imports, federal
  spending), **Prices & supply** (oil, core prices, import prices, house prices,
  productivity), **Financial** (corporate & 10y term premia, equity premium,
  mortgage rate, exchange rate), and **Fiscal & monetary** (tax rate, policy-rule
  shock) — plus an advanced raw-variable option. **Up to three shocks can be
  applied together** (e.g. an oil-price spike *and* a fiscal expansion).
- **Expectations:** VAR-based (backward-looking) or **model-consistent /
  rational** (MCE, `mcap+wp` block).
- **Monetary policy:** the active inertial Taylor rule vs. the funds rate held at
  its baseline path — the with-response / without-response comparison.
- **Outputs:** a grouped, selectable menu (defaulting to GDP growth, unemployment,
  PCE inflation, and the funds rate). Add **Activity** (output gap; real GDP,
  consumption, investment as % deviations), **Inflation** (core PCE, CPI/core CPI,
  4-quarter measures), **Interest rates** (real funds rate; 5y/10y/30y
  Treasury, BBB corporate, 10y term premium), and **Financial & external** (real
  expected equity return, equity risk premium, real exchange rate, house prices).
  Rates/inflation are shown in percentage points, levels as percent deviations
  from baseline.
- **Export:** download any run as CSV (deviation panel) or PNG/HTML chart, since
  each run is computationally non-trivial.
- **Fiscal Multipliers page** (a second tab): output multipliers (ΔGDP per $ of
  fiscal impulse) for federal & state/local purchases, transfers, and a personal
  tax cut, with the active rule vs. an accommodative (funds-rate-held) response —
  in the spirit of CBO's fiscal-multiplier work.
- **Optimal-Control page** (a third tab): the funds-rate path that minimises a
  quadratic loss over the inflation and unemployment gaps (with adjustable
  weights), vs. the Taylor rule and no response — solved by the linear-quadratic
  method, in the spirit of the FEDS Note on optimal-control policy.

## Vintages

| | Vintage | Source of truth |
|---|---|---|
| **Model / equations** | PyFRB/US 1.0.0 (`third_party/pyfrbus/models/model.xml`) | <https://www.federalreserve.gov/econres/us-models-python.htm> |
| **Data (LONGBASE)** | see `data/VINTAGE.json` (`sha256`, obs range, variable count) | Fed *data-only* package; auto-refreshed by CI |

The **model** equations change rarely and are bumped manually (they change the
model's behaviour). The **data** updates more often and is refreshed
automatically — see [Data refresh](#data-refresh).

> **Note — deliberate model/data vintage split.** The vendored model is the
> 2022-era PyFRB/US 1.0.0; the data is refreshed to the current Fed vintage
> (historical actuals now run through the recent past, then a projection). The
> current data drops three variables the 2022 model still names
> (`dmpgen`/`rffgen`, the generalized policy rule, and `pcstar`) — none are used
> by this app's scenarios, and the package solves cleanly against the newer data
> (the demo-validation test is re-baselined to the committed vintage and runs in
> CI). If you later vendor a newer `model.xml`, re-generate the golden values in
> `tests/test_validation.py`.

### ⚠️ The baseline is not a forecast

Simulations run off the projection baseline embedded in the Fed's dataset. Per
the Fed's own `data/README.TXT`, that baseline follows the median FOMC **Summary
of Economic Projections (SEP)** where available and a **model-guided
extrapolation** beyond the SEP horizon; the funds rate mechanically follows the
inertial Taylor rule past that horizon. **This SEP-consistent extrapolation is
not itself a forecast.** Read every result as a *deviation from a stylised
baseline*, not as a prediction of the economy.

## How the "funds rate held at baseline" case works

This is the crux of the comparison, so it is implemented with the model's own
switches rather than a hack (full detail in `src/frbus_shock/policy.py`):

- The baseline runs the **inertial Taylor rule** (`dmpintay = 1`); this is the
  *active-rule* (with-response) case.
- To **hold the funds rate**, we set `dmpintay = 0`, `dmpex = 1` (the model's
  exogenous-funds-rate switch, which makes `rffrule = rfffix`), point `rfffix` at
  the baseline `rff` path, and zero the funds-rate tracking add factors
  (`rff_trac`, `rffrule_trac`) so no residual leaks back in. The funds rate is
  then pinned to baseline (subject to the `rffmin` ZLB floor) regardless of what
  the shock does to inflation or the output gap. Verified to hold the funds-rate
  deviation at 0.000 bp in `tests/test_shocks.py`.

## Repository layout

```
app/streamlit_app.py         Streamlit entry point — navigation router (deploy this)
app/views/                   The three tabs (via st.navigation)
  ├─ shock_analysis.py       Shock Analysis
  ├─ fiscal_multipliers.py   Fiscal Multipliers
  └─ optimal_control.py      Optimal Control
src/frbus_shock/             Simulation library wrapping PyFRB/US
  ├─ shocks.py               Shock library (levers, units, defaults, groups)
  ├─ policy.py               Active rule vs. funds-rate-held mechanism
  ├─ model.py                Cached model + baseline loading, vintages
  ├─ simulate.py             run_simulation(): baseline + 2 scenarios, multi-shock
  ├─ outputs.py              Deviation panels, summary table, CSV/chart export
  ├─ multipliers.py          Fiscal output multipliers
  └─ optcontrol.py           Linear-quadratic optimal control
third_party/pyfrbus/         Vendored, patched PyFRB/US (public domain)
  ├─ pyfrbus/                Model platform code (3 compat patches)
  ├─ models/model.xml        FRB/US equations
  └─ PATCHES.md              What was patched and why
data/                        LONGBASE.TXT + VINTAGE.json
tests/                       Demo-validation + shock sign tests (CI)
docs/use_cases.md            How FRB/US is actually used, and scope choices
.github/workflows/           CI + automated data refresh
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

No system packages are required — the vendored model is patched to solve with
SciPy's sparse solver instead of `scikit-umfpack`/SuiteSparse (see
`third_party/pyfrbus/PATCHES.md`).

Run the tests (includes reproducing the Fed's `example1` demo exactly):

```bash
pip install -r requirements-dev.txt
pytest -q            # add -m "not slow" to skip the MCE test
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already wired to `origin`).
2. On <https://share.streamlit.io>, **New app** → pick this repo and branch.
3. **Main file path:** `app/streamlit_app.py`.
4. **Advanced settings → Python version: 3.11** (the pinned stack targets 3.11).
5. Deploy. Dependencies come from `requirements.txt`; there is no backend, no
   secrets, and nothing to install at the system level.

Because each run solves the model fresh (a few seconds under VAR, longer under
MCE), the app caches identical runs and offers CSV/PNG export so a result can be
saved rather than recomputed.

## Data refresh

`.github/workflows/refresh-data.yml` checks the Fed's *data-only* package weekly
(and on demand). It's designed to be **light and unattended**:

- A `HEAD` request compares the package's `Last-Modified` against the one last
  ingested (`source_last_modified` in `data/VINTAGE.json`); on a quiet week that
  single request is the whole job — no download.
- When the package is newer it downloads, and commits only if `LONGBASE.TXT`'s
  content hash actually changed.
- On a real change it also **re-baselines the strict validation goldens**
  (`scripts/regen_goldens.py` → `tests/golden_example1.json`) in the same run,
  so an automatic data update never leaves CI stale. `tests/test_validation.py`
  reads those goldens.

The **model/equations** package is deliberately **not** on this cadence —
bumping it changes the model's behaviour and is a manual, reviewed step (after
which, run `python scripts/regen_goldens.py` and commit the result).

To force a check now: **Actions → Refresh LONGBASE data → Run workflow**.

## Use cases and scope

See **[`docs/use_cases.md`](docs/use_cases.md)** for how FRB/US has actually been
used — the shipped demos, documented Fed-internal applications (optimal control,
policy-rule and ELB exercises), and external research — and for an explicit note
on which capabilities (custom/optimal rules under MCE, stochastic simulation,
trajectory matching, threshold forward guidance) sit **beyond this first-pass
build**, so the dashboard's scope is a deliberate choice.

## Provenance & licence

The original code in this repository (the `frbus_shock` package, the app, tests,
and workflows) is **© 2026 WillBtK — All Rights Reserved** (`LICENSE`): the
source is public for viewing, but no licence to use, copy, modify, or distribute
it is granted. The vendored FRB/US model, PyFRB/US platform, and LONGBASE dataset
are **public-domain** works of the Federal Reserve Board and remain freely usable
under their own terms. When you can reach `federalreserve.gov`, prefer
re-vendoring PyFRB/US and the data from the
[official page](https://www.federalreserve.gov/econres/us-models-python.htm).
