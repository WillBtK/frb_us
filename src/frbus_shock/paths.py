"""Filesystem locations for the vendored FRB/US model and data.

Importing this module makes the vendored PyFRB/US package importable
(``import pyfrbus``) by putting ``third_party/pyfrbus`` on ``sys.path``.
Keeping this in one place means the app, the library, and the tests all agree
on where the model equations and the LONGBASE dataset live.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root = two levels up from this file (src/frbus_shock/paths.py -> repo/)
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

VENDOR_ROOT: Path = REPO_ROOT / "third_party" / "pyfrbus"
MODEL_XML: Path = VENDOR_ROOT / "models" / "model.xml"

DATA_DIR: Path = REPO_ROOT / "data"
LONGBASE_TXT: Path = DATA_DIR / "LONGBASE.TXT"
VINTAGE_JSON: Path = DATA_DIR / "VINTAGE.json"


def ensure_pyfrbus_importable() -> None:
    """Put the vendored PyFRB/US package on ``sys.path`` (idempotent)."""
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        # Insert at front so the vendored, patched copy wins over any global one.
        sys.path.insert(0, vendor)


# Do it on import — this module exists precisely to wire up the vendored package.
ensure_pyfrbus_importable()
