"""Runs the dashboard logic tests (tests/js/*.test.mjs, Node's built-in test runner) from pytest."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_coach_core_js():
    r = subprocess.run(["node", "--test", "tests/js/coach-core.test.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
