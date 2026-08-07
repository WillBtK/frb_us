# Vendored PyFRB/US — provenance and patches

## What this is

`third_party/pyfrbus/` contains a **vendored copy** of the Federal Reserve
Board's **PyFRB/US** simulation platform (the `pyfrbus/` Python package plus
`models/model.xml`, the FRB/US model equations).

PyFRB/US is a work of the U.S. Government and is in the **public domain** (see
the upstream `LICENSE`: *"written and prepared by a U.S. Government employee on
official time, and therefore it is in the public domain and not subject to
copyright. There are no limitations whatsoever on its use."*).

* **Upstream, authoritative source:** <https://www.federalreserve.gov/econres/us-models-python.htm>
* **Vintage vendored here:** PyFRB/US `1.0.0` (model equations `model.xml`).
  This copy was obtained from a public-domain GitHub mirror of the Fed package
  because the build environment used to assemble this repository could not
  reach `federalreserve.gov` directly. **Treat the Fed site as the source of
  truth** and re-vendor from the official ZIP when you can.

## Why it is patched

The upstream `1.0.0` package pins a 2018–2022-era scientific stack
(`pandas==0.25.3`, `sympy==1.3`, `symengine==0.8.1`, Python 3.6) and depends on
`scikit-umfpack` (a native build over SuiteSparse). None of that installs on a
current Python 3.11 / Streamlit Community Cloud runtime. Three **surgical,
behaviour-preserving** patches make the *unmodified model equations* solve on a
modern, wheels-only stack. The model math is untouched.

| File | Change | Reason |
|------|--------|--------|
| `pyfrbus/newton.py` | `from scikits import umfpack` → `try scikits.umfpack.spsolve else scipy.sparse.linalg.spsolve`; the two `umfpack.spsolve(` call sites now use the resolved `_spsolve`. | Removes the SuiteSparse/`scikit-umfpack` native dependency. `scipy.sparse.linalg.spsolve` is a drop-in and produces **bit-for-bit identical** simulation output on the validation demo (verified). |
| `pyfrbus/symbolic.py` | In `take_symengine_partial`, the SymPy fail-over now converts the SymEngine expression with `._sympy_()` before differentiating, and retries with real-typed symbols. | SymEngine ≥ 0.11 returns an **unevaluated** `Derivative(Max(...))` for the funds-rate `max()`/`min()`/`abs()` kinks; passing a raw SymEngine expr to `sympy.diff` left it unevaluated (→ `NameError: Derivative` at Jacobian eval). Converting to native SymPy yields the correct `Heaviside`/`sign` derivatives. |
| `pyfrbus/run_jac.py` | Add `sign = numpy.sign` to the Jacobian-eval namespace. | `abs()` derivatives now evaluate to `sign(...)`, which must resolve when the Jacobian entry strings are `eval`'d. |

Every patch site is marked in-code. To reproduce them against a fresh upstream
checkout, see `scripts/apply_pyfrbus_patches.py` if present, or re-apply the
three edits above.

## Validation

The patched package reproduces the Fed's own **`demos/example1.py`** (a 100 bp
funds-rate shock) exactly — see `tests/test_validation.py`, which runs in CI.
