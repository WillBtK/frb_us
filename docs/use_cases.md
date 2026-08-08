# How FRB/US is actually used — and what this dashboard chooses to support

This document surveys how the Federal Reserve Board's **FRB/US** model has been
used in practice, to make the dashboard's scope a *deliberate* choice rather than
an accident of what was easiest to wrap. It distinguishes:

- **(a)** the standard demo use cases shipped with PyFRB/US,
- **(b)** documented Fed-internal applications, and
- **(c)** external academic / policy research,

and closes with an explicit list of capabilities that sit **beyond a first-pass
build**.

FRB/US is a large-scale (≈380-variable) estimated general-equilibrium model of
the US economy, in use at the Board since 1996 for forecasting, policy-option
analysis, and research. Its defining feature is **switchable expectations**:
backward-looking **VAR-based** expectations, or forward-looking
**model-consistent (rational) expectations (MCE)**.
Overview: <https://www.federalreserve.gov/econres/us-models-about.htm> ·
Python platform: <https://www.federalreserve.gov/econres/us-models-python.htm>

---

## (a) Standard demo use cases shipped with PyFRB/US

These are the example programs in the package's `demos/` folder. They define the
"blessed" surface of the API, and this dashboard is built directly on the
patterns in `example1`/`example2` (single add-factor shock, VAR vs. MCE, active
rule vs. exogenous funds rate).

| Demo | What it demonstrates | Relation to this dashboard |
|------|----------------------|----------------------------|
| **`example1.py`** | A **100 bp monetary-policy shock** (`rffintay_aerr += 1`) under **VAR** expectations, solved with `init_trac` + `solve`. | **Directly supported** and used as the CI validation anchor (`tests/test_validation.py`). |
| **`example2.py`** | The same shock under **model-consistent expectations** (`mce="mcap+wp"`), with `rstar` endogenous in the long run (`drstar`). | **Supported** — the MCE expectations toggle. |
| **`example3.py`** | **Threshold-based forward guidance**: switches to a non-inertial Taylor rule (`dmptay`), enables thresholds (`dmptrsh`) on unemployment (`lurtrsh`) and inflation (`pitrsh`), and rolls historical residuals forward. | **Beyond first pass** (see below) — thresholds/state-contingent guidance. |
| **`example4.py`** | **Trajectory matching with `mcontrol`**: forces `xgdp`, `lur`, `picxfe`, `rff`, `rg10` onto a specified path (e.g. an SPF-based scenario) by freely moving instrument add factors. | **Beyond first pass** — scenario/nowcast matching. |
| **`stochsim.py`** | **Stochastic simulation**: draws sequences of historical residuals to build fan charts / distributions of outcomes (`frbus.stochsim`). | **Implemented** — the **Debt Fan Charts** tab (`stoch.py`) uses it for debt/GDP probability bands. |

The dashboard's "funds rate held at baseline" case uses the model's own
**`dmpex`** (exogenous funds rate) switch — the same monetary-policy switch
machinery these demos toggle (`dmpintay`, `dmptay`, `dmptrsh`, …) — documented in
`src/frbus_shock/policy.py`.

---

## (b) Documented Fed-internal applications

- **"The FRB/US Model: A Tool for Macroeconomic Policy Analysis"** (Brayton,
  Laubach, Reifschneider; FEDS Notes, April 2014). The canonical description of
  how the Board uses FRB/US: alternative monetary-policy rules, the role of VAR
  vs. model-consistent expectations, and the construction of the SEP-consistent
  baseline (the same baseline this dashboard perturbs). Notably, it is the source
  of the caveat — repeated in the dataset's own `README.TXT` — that the baseline
  projection follows the FOMC's **Summary of Economic Projections** where
  available and a **model-guided extrapolation** beyond, and is **not itself a
  forecast**.
  <https://www.federalreserve.gov/econres/notes/feds-notes/frbus-model-tool-for-macroeconomic-policy-analysis-20140403.html>

- **Optimal-control monetary policy** ("Optimal-Control Monetary Policy in the
  FRB/US Model", FEDS Notes, Nov 2014). A search procedure solves for the
  funds-rate path that minimises a **quadratic loss** (weighted squared
  deviations of inflation and unemployment from target, plus rate-smoothing),
  contrasting the full-commitment optimal-control path against simple Taylor-type
  rules and showing sensitivity to the loss weights and starting conditions.
  Requires a *custom policy objective*, not just a rule switch.
  <https://www.federalreserve.gov/econresdata/notes/feds-notes/2014/optimal-control-monetary-policy-in-frbus-20141121.html>

- **Makeup strategies / average-inflation targeting.** FRB/US simulations of
  price-level targeting, average-inflation targeting, and temporary price-level
  targeting versus an inertial Taylor rule, quantifying the stabilisation gains
  when the strategies are credible, and their robustness to how inflation
  expectations respond ("Alternative Strategies: How Do They Work? How Might They
  Help?", FEDS 2020-068,
  <https://www.federalreserve.gov/econres/feds/files/2020068pap.pdf>; and "How
  Robust Are Makeup Strategies…?", FEDS 2020-069,
  <https://www.federalreserve.gov/econres/feds/files/2020069pap.pdf>). See also
  Bernanke, Kiley & Roberts, "Monetary Policy Strategies for a Low-Rate
  Environment" (FEDS 2019-009,
  <https://www.federalreserve.gov/econres/feds/files/2019009pap.pdf>).

- **Effective-lower-bound (ELB) risk.** FRB/US stochastic simulation is the
  Board's standard tool for quantifying ELB risk — e.g. "Model-Based Measures of
  ELB Risk" runs 20,000 stochastic paths to produce funds-rate fan charts and the
  probability of hitting the ELB within three years
  (<https://www.federalreserve.gov/econres/notes/feds-notes/model-based-measures-of-elb-risk-20170823.html>).
  The residual-bootstrap methodology is documented in "A New Procedure for
  Generating the Stochastic Simulations in FRB/US"
  (<https://www.federalreserve.gov/econres/notes/feds-notes/new-procedure-for-generating-the-stochastic-simulations-in-frb-us-20190307.html>).
  This lineage runs from Reifschneider–Williams, "Three Lessons for Monetary
  Policy in a Low-Inflation Era" (2000,
  <https://www.federalreserve.gov/econres/feds/three-lessons-for-monetary-policy-in-a-low-inflation-era.htm>)
  and Chung et al. (2012,
  <https://www.frbsf.org/wp-content/uploads/wp11-01bk.pdf>) through the strategy
  reviews; it leans heavily on MCE, on stochastic simulation, and on the
  threshold/`dmptrsh` machinery seen in `example3`.

- **Alternative simple policy rules.** The model ships several rule variants
  (`rfftay` output-gap Taylor, `rffintay` inertial, `rfftlr` with an unemployment
  gap, `rffalt` estimated/MA rule), selectable via the `dmp*` switches. Board
  analyses routinely compare macro outcomes across these rules.

---

## (c) External academic and policy research

FRB/US (in EViews, and increasingly the Python port) is widely used outside the
Board because the model and data are public domain. Representative uses:

- **Fiscal multipliers.** Estimating the output effects of government-spending
  and tax changes, and how those multipliers depend on the monetary-policy
  response and the ELB — precisely the *with-response vs. without-response*
  contrast this dashboard foregrounds. FRB/US-type structural multipliers are
  used as a benchmark in fiscal-multiplier methodology, e.g. CBO's working paper
  on the fiscal multiplier
  (<https://www.cbo.gov/sites/default/files/114th-congress-2015-2016/workingpaper/49925-FiscalMultiplier_1.pdf>).
- **Alternative monetary rules and low-rate frameworks.** Kiley & Roberts,
  "Monetary Policy in a Low Interest Rate World" (Brookings Papers, 2017,
  <https://www.brookings.edu/wp-content/uploads/2017/08/kileytextsp17bpea.pdf>)
  and the NBER-distributed "Monetary Policy Strategies for the Federal Reserve"
  (w26657, <https://www.nber.org/system/files/working_papers/w26657/w26657.pdf>)
  use FRB/US to assess ELB frequency and makeup/alternative rules.
- **Forward guidance.** Studies of state-contingent (threshold) guidance and its
  transmission — e.g. reforming state-based forward guidance via a wage-growth
  threshold, analysed with FRB/US simulations of the `dmptrsh` mechanism
  (arXiv:2008.08705, <https://arxiv.org/abs/2008.08705>).
- **Novel couplings.** FRB/US used as the environment for a reinforcement-learning
  fiscal-policy search ("Fiscal Policy Towards Optimizing Macroeconomic Indicators
  by Integrating FRB/US with Reinforcement Learning", Computational Economics 2026,
  <https://link.springer.com/article/10.1007/s10614-026-11330-x>).
- **Community ports.** An R port (`rfrbus`) reproduces the same demo suite outside
  the Fed, evidence of external reuse
  (<https://r-consortium.org/posts/us-federal-reserve-quarterly-model-in-r/>).

(External applications typically require the "beyond first pass" capabilities
below — custom rules under MCE, `mcontrol`, or `stochsim` — rather than the
single-shock deviations this dashboard computes.)

---

## Beyond a first-pass build (explicit scope choices)

The dashboard deliberately supports **single deterministic add-factor shocks**,
under **VAR or MCE** expectations, comparing the **active inertial Taylor rule**
against the **funds rate held at baseline**. The following are *supported by
FRB/US itself* but intentionally **out of scope for this first pass**, each with
the reason and the API it would need:

1. **Custom / optimal policy rules under MCE.** Beyond the built-in `dmp*` rule
   switches, defining a bespoke reaction function or an optimal-control objective
   needs `append_replace`/`exogenize` plus a loss-minimisation loop, and is far
   more sensitive to convergence and terminal conditions. *(API: custom
   equations + MCE solves.)*
2. **Stochastic simulation & fan charts** (`frbus.stochsim`). Hundreds–thousands
   of replications over drawn historical residuals — valuable for uncertainty
   bands but far too heavy for an interactive, per-click Streamlit run.
3. **Trajectory-matching scenarios** (`frbus.mcontrol`, as in `example4`).
   Forcing variables onto a specified path (e.g. an SPF or SEP scenario) is a
   different UX — the user supplies target paths, not a single shock.
4. **Threshold / state-contingent forward guidance** (`dmptrsh`, `lurtrsh`,
   `pitrsh`, as in `example3`). Powerful but adds several interacting controls
   and its own tracking-residual bookkeeping.
5. **ELB/ZLB nonlinearity as a first-class control.** The `rffmin` floor is
   present in the model and respected, but the dashboard does not expose ELB
   scenarios (e.g. deliberately pushing the funds rate to the floor and studying
   makeup dynamics) as a configurable experiment.
6. **Multi-shock and historical counterfactuals.** Combining several shocks, or
   re-running history under alternative policy, needs a scenario builder rather
   than the single-lever picker here.

Each of these is a natural extension: the wrapper in `src/frbus_shock/` already
loads the model under either expectations regime and manipulates the same
add-factor / policy-switch surface these capabilities build on.

### A note on LINVER (and why this dashboard uses the nonlinear model)

The Fed also distributes **LINVER**, a *linearised* version of FRB/US. Because it
is linear, simulations are matrix operations — orders of magnitude faster than
the nonlinear model — which is why LINVER is the Fed's engine for **stochastic
simulation** (thousands of draws) and **optimal-control** policy. For this
dashboard's job — a handful of *single deterministic* shock paths — the nonlinear
model is already fast (VAR runs in seconds), and it is more accurate: it respects
the effective-lower-bound `max()` and other nonlinearities that a linearisation
drops, which matter for larger shocks. LINVER is also a separate package, not part
of PyFRB/US. So the deliberate choice here is the **nonlinear model** (VAR, or MCE
when anticipation matters). LINVER would become the right tool *if* the scope grew
into the fan-chart / optimal-control territory listed above.

---

*Sources are linked inline. The authoritative home for FRB/US documentation,
FEDS Notes, and the model/data packages is the Federal Reserve Board:
<https://www.federalreserve.gov/econres/us-documentation-papers.htm>.*
