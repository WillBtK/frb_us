"""Shared pytest fixtures.

Model construction and baseline loading are expensive, so they are built once
per test session and reused. Matplotlib is forced to a headless backend so the
vendored ``sim_lib`` plotting imports never try to open a display in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, before any pyplot import

# Make the ``frbus_shock`` package importable without an install step.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def var_model():
    from frbus_shock.model import load_frbus

    return load_frbus("var")


@pytest.fixture(scope="session")
def baseline():
    from frbus_shock.model import load_baseline

    return load_baseline()


def pytest_configure(config):
    # Keep numpy/threads modest so CI runners don't oversubscribe.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
