"""Entry point for running the phonon symmetry analysis inside a web page (Pyodide); JSON in, JSON out.

The viewer (viewer/js/symworker.js) loads Pyodide and numpy, unpacks this package, and calls ``analyze`` with one q-point of the
band structure that is open.  Nothing here is specific to the browser: it is the same ``phonons.analyze_qpoint`` the command line
uses, so the two cannot disagree.  Parent structures and their stars are cached between calls, so analysing further q-points of
the same file only pays for the new stars.
"""
from __future__ import annotations

import json

import numpy as np

from .phonons import PhononSymmetry, analyze_qpoint
from .structure import Structure

_cache = {}


def analyze(request_json, progress=None):
    """``request_json``: lattice, frac, species, masses, q, freq (nbands), eig (nbands*natom*3*2 reals: re, im), qi, settings.

    ``progress(stage, fraction)`` is called while a long star is being worked out.  Returns the JSON of ``analyze_qpoint``.
    """
    req = json.loads(request_json)
    st = req.get("settings", {})
    key = json.dumps([req["lattice"], req["frac"], req["species"], st.get("max_cells"), st.get("max_dim"), st.get("labels"),
                      st.get("tol")])
    if key not in _cache:
        parent = Structure(req["lattice"], req["frac"], req["species"])
        _cache.clear()                                   # one structure at a time is enough, and stars can be large
        _cache[key] = PhononSymmetry(parent, tol=st.get("tol", 1e-3), max_cells=st.get("max_cells", 16),
                                     max_dim=st.get("max_dim", 6), labels=st.get("labels", True))
    ps = _cache[key]
    ps.progress = progress
    natom = len(req["species"])
    freq = np.array(req["freq"], float)
    z = np.array(req["eig"], float).reshape(len(freq), natom, 3, 2)
    eig = z[..., 0] + 1j * z[..., 1]
    masses = np.array(req["masses"], float) if st.get("mass_scale", True) else None
    res = analyze_qpoint(ps, np.array(req["q"], float), eig, freq, q_index=req.get("qi", 0), masses=masses,
                         basis_phase=st.get("basis_phase", True))
    return json.dumps(res)
