"""Find a Python interpreter that has optional heavy packages (pymatgen, matplotlib, Pillow).

The analysis itself needs only numpy.  pymatgen (symmetrisation) and matplotlib + Pillow (animations) can live in a
separate environment; helpers run in that interpreter as subprocesses.  Search order: ``$DISTORTKIT_PYTHON``,
``$DISTORTKIT_PMG_PYTHON``, the current interpreter, ``~/.venvs/distortkit``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def find_python_with(modules):
    """Path of an interpreter that can import every module in ``modules`` (a tuple of names), or None."""
    cands = [os.environ.get("DISTORTKIT_PYTHON"), os.environ.get("DISTORTKIT_PMG_PYTHON"), sys.executable,
             os.path.expanduser("~/.venvs/distortkit/bin/python"), os.path.expanduser("~/.venvs/isodistort/bin/python")]
    code = "import " + ", ".join(modules)
    for py in dict.fromkeys(c for c in cands if c):
        if os.path.exists(py):
            try:
                if subprocess.run([py, "-c", code], capture_output=True, timeout=180).returncode == 0:
                    return py
            except (OSError, subprocess.TimeoutExpired):
                continue
    return None


def find_renderer_python():
    return find_python_with(("matplotlib", "PIL"))
