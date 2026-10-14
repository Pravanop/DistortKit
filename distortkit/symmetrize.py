"""Give a structure that has no symmetry information explicit symmetry, as a CIF.

A POSCAR (or JSON) holds only a cell and atoms.  ``symmetrize`` runs pymatgen's ``SpacegroupAnalyzer`` to find
the space group, writes the idealised structure as a CIF with explicit symmetry operations, and returns a
structure ready for the rest of the analysis.  pymatgen (and numpy 2 / Python 3.10+) can live in a separate
virtual environment: the worker is run as a subprocess in that interpreter.  If pymatgen cannot be found the
built-in detector (``symmetry.find_symmetry`` + ``spacegroup.identify``) does the same job.

Interpreter search order: ``$DISTORTKIT_PMG_PYTHON``, the current interpreter, ``~/.venvs/distortkit``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .structure import Structure
from .symmetry import find_symmetry, op_to_string, to_primitive

DEFAULT_TOLERANCES = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1)

_WORKER = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
req = json.load(sys.stdin)
s = Structure(Lattice(req["lattice"]), req["species"], req["frac"])
out = {"by_tol": {}, "table": []}
for tol in req["tolerances"]:
    try:
        a = SpacegroupAnalyzer(s, symprec=tol, angle_tolerance=req["angle_tolerance"])
        out["table"].append([tol, a.get_space_group_number(), a.get_space_group_symbol()])
        out["by_tol"][repr(tol)] = {"number": a.get_space_group_number(), "symbol": a.get_space_group_symbol(),
            "point_group": a.get_point_group_symbol(),
            "ops": [[op.rotation_matrix.tolist(), op.translation_vector.tolist()] for op in a.get_symmetry_operations()]
            if req.get("want_ops", True) else []}
    except Exception as e:
        out["table"].append([tol, None, str(e)[:60]])
if req.get("standardize"):
    a = SpacegroupAnalyzer(s, symprec=req["symprec"], angle_tolerance=req["angle_tolerance"])
    ref = a.get_refined_structure()
    out["cif"] = str(CifWriter(ref, symprec=req["symprec"]))
    out["number"], out["symbol"], out["point_group"] = a.get_space_group_number(), a.get_space_group_symbol(), a.get_point_group_symbol()
print(json.dumps(out))
'''


@dataclass
class SymmetryResult:
    structure: Structure                 # idealised structure with symmetry (parsed back from the CIF)
    cif_text: str
    number: int
    symbol: str
    point_group: str = ""
    method: str = "builtin"              # "pymatgen" or "builtin"
    table: list = field(default_factory=list)      # [(symprec, number, symbol)] symmetry vs tolerance
    cif_path: str = None
    notes: list = field(default_factory=list)


def find_pymatgen_python():
    """Path of a Python interpreter that can import pymatgen, or None (see ``external.find_python_with``)."""
    from .external import find_python_with
    return find_python_with(("pymatgen.symmetry.analyzer",))


def _run_pymatgen(py, structure, symprec, angle_tolerance, standardize, tolerances, want_ops=True):
    tols = sorted({float(t) for t in tolerances} | ({float(symprec)} if symprec != "auto" else set()))
    req = {
        "lattice": structure.lattice.tolist(), "species": list(structure.species),
        "frac": structure.frac.tolist(), "symprec": symprec if symprec != "auto" else 1e-2,
        "angle_tolerance": angle_tolerance, "standardize": standardize, "tolerances": tols, "want_ops": want_ops,
    }
    r = subprocess.run([py, "-c", _WORKER], input=json.dumps(req), capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("pymatgen failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
    return json.loads(r.stdout.strip().splitlines()[-1])


def tolerance_table(structure, tolerances=DEFAULT_TOLERANCES, angle_tolerance=5.0, method="auto"):
    """Space group found at several symmetry tolerances: [(symprec, number or None, symbol)]."""
    py = find_pymatgen_python() if method in ("auto", "pymatgen") else None
    if py:
        return [tuple(x) for x in _run_pymatgen(py, structure, 1e-3, angle_tolerance, False, tolerances, False)["table"]]
    from .spacegroup import identify
    out = []
    prim = None
    for tol in tolerances:
        try:
            prim, _ = to_primitive(structure, tol)
            ops = find_symmetry(prim, tol)
            sg = identify(ops, prim.lattice)
            out.append((tol, sg.number, sg.symbol))
        except Exception as e:                                    # noqa: BLE001
            out.append((tol, None, str(e)[:50]))
    return out


# ------------------------------------------------------------------ built-in symmetrisation
def idealize(structure, tol=1e-3, iterations=6, ops=None):
    """Average atomic positions and the lattice metric over the detected symmetry operations.

    Operations are found at tolerance ``tol``; their translations are refined by least squares over all atoms and
    the positions are then replaced by their group average, iterated until the operations hold to rounding.
    """
    ops = find_symmetry(structure, tol) if ops is None else ops
    A = structure.lattice
    sp = np.array(structure.species)
    frac = structure.frac.copy()
    for it in range(max(iterations, 200)):
        acc = np.zeros_like(frac)
        for R, t0 in ops:
            Y = frac @ R.T + t0
            diff = Y[:, None, :] - frac[None, :, :]
            diff -= np.rint(diff)
            d = np.linalg.norm(diff @ A, axis=2)
            d[sp[:, None] != sp[None, :]] = np.inf
            j = d.argmin(axis=1)                                   # atom a is carried onto atom j[a]
            shift = np.rint(Y - frac[j])
            t = np.mean(frac[j] + shift - frac @ R.T, axis=0)      # least-squares translation
            Rinv = np.rint(np.linalg.inv(R)).astype(int)
            back = (frac[j] + shift - t) @ Rinv.T                  # R^-1 (x_j + shift - t): estimate of x_a
            acc += back - np.rint(back - frac)
        new = acc / len(ops)
        moved = np.abs(new - frac).max()
        frac = new
        if moved < 1e-12:
            break
    G = A @ A.T
    rots = {tuple(R.ravel()): R for R, _ in ops}
    Gs = sum(R.T @ G @ R for R in rots.values()) / len(rots)       # metric averaged over the point group
    L = np.linalg.cholesky(Gs)                                     # rows are lattice vectors
    ideal = Structure(L, frac % 1.0, list(structure.species))
    return ideal, find_symmetry(ideal, max(tol * 0.1, 1e-6))


def _orbits(structure, ops, tol):
    A = structure.lattice
    n = len(structure)
    sp = np.array(structure.species)
    seen, reps = set(), []
    for a in range(n):
        if a in seen:
            continue
        reps.append(a)
        for R, t in ops:
            y = R @ structure.frac[a] + t
            d = (y[None, :] - structure.frac)
            d -= np.rint(d)
            dist = np.linalg.norm(d @ A, axis=1)
            dist[sp != sp[a]] = np.inf
            seen.add(int(dist.argmin()))
    return reps


def cif_with_symmetry(structure, ops, tol=1e-3, number=None, symbol=None, title="distortkit"):
    """CIF text for ``structure`` listing its asymmetric unit and the explicit symmetry operations ``ops``."""
    a, b, c, al, be, ga = structure.lattice_parameters
    out = [f"data_{title}"]
    if symbol:
        out.append(f"_symmetry_space_group_name_H-M '{symbol}'")
    if number:
        out.append(f"_symmetry_Int_Tables_number {number}")
    out += [f"_cell_length_a {a:.8f}", f"_cell_length_b {b:.8f}", f"_cell_length_c {c:.8f}",
            f"_cell_angle_alpha {al:.6f}", f"_cell_angle_beta {be:.6f}", f"_cell_angle_gamma {ga:.6f}",
            "loop_", "_symmetry_equiv_pos_as_xyz"]
    out += [f"'{op_to_string(R, t, decimals=8)}'" for R, t in ops]
    out += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
            "_atom_site_fract_z"]
    counts = {}
    for i in _orbits(structure, ops, tol):
        sp = structure.species[i]
        counts[sp] = counts.get(sp, 0) + 1
        x = structure.frac[i]
        out.append(f"{sp}{counts[sp]} {sp} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}")
    return "\n".join(out) + "\n"


def choose_symprec(table, max_symprec=0.1):
    """Tolerance giving the highest point-group order among ``table`` entries up to ``max_symprec`` (smallest wins ties)."""
    from . import isodata
    G = isodata.space_groups()
    best = None
    for tol, n, _ in table:
        if n is None or tol > max_symprec:
            continue
        order = len(G[n].ops) // (1 + len(G[n].centering))
        key = (order, -tol)
        if best is None or key > best[0]:
            best = (key, tol)
    return best[1] if best else 1e-2


def symmetrize(structure, symprec=1e-3, angle_tolerance=5.0, method="auto", cif_path=None,
               tolerances=DEFAULT_TOLERANCES, standardize=False):
    """Find the symmetry of ``structure``, write an idealised CIF with explicit operations, return the result.

    ``method``: ``"pymatgen"``, ``"builtin"`` or ``"auto"`` (pymatgen if it can be found).  ``symprec="auto"``
    scans ``tolerances`` and uses the one with the highest symmetry up to 0.1 A (the choice is reported in
    ``SymmetryResult.notes``).  By default the symmetry is assigned in *your* setting and origin (positions are
    idealised in place); ``standardize=True`` returns pymatgen's refined standard-setting structure instead.
    """
    from .io import structure_from_cif_text
    from .spacegroup import identify
    py = find_pymatgen_python() if method in ("auto", "pymatgen") else None
    if method == "pymatgen" and not py:
        raise RuntimeError("pymatgen was requested but no interpreter with pymatgen was found "
                           "(set DISTORTKIT_PMG_PYTHON to a Python that has it)")
    notes = []
    if py:
        res = _run_pymatgen(py, structure, symprec, angle_tolerance, standardize, tolerances)
        table = [tuple(x) for x in res["table"]]
        if symprec == "auto":
            symprec = choose_symprec(table)
            notes.append(f"symprec chosen automatically: {symprec:g} A (highest symmetry up to 0.1 A)")
        if standardize:
            if "cif" not in res or res.get("number") is None:
                res = _run_pymatgen(py, structure, symprec, angle_tolerance, True, tolerances)
            cif = res["cif"]
            out = SymmetryResult(structure_from_cif_text(cif, symprec), cif, res["number"], res["symbol"],
                                 res.get("point_group", ""), "pymatgen", table, cif_path)
            notes.append("standard-setting structure from pymatgen: the origin may differ from your input, which "
                         "changes origin-dependent irrep labels (+/-)")
        else:
            info = res["by_tol"].get(repr(float(symprec)))
            if info is None:
                raise RuntimeError(f"pymatgen could not determine the symmetry at symprec {symprec:g} A")
            start = [(np.rint(np.array(R)).astype(int), np.array(t, dtype=float)) for R, t in info["ops"]]
            ideal, _ = idealize(structure, symprec, ops=start)
            ops = find_symmetry(ideal, 1e-5)
            cif = cif_with_symmetry(ideal, ops, 1e-5, info["number"], info["symbol"])
            out = SymmetryResult(structure_from_cif_text(cif, 1e-4), cif, info["number"], info["symbol"],
                                 info.get("point_group", ""), "pymatgen", table, cif_path)
    else:
        if method == "auto":
            notes.append("pymatgen not found: used the built-in symmetry detector")
        table = tolerance_table(structure, tolerances, method="builtin")
        if symprec == "auto":
            symprec = choose_symprec(table)
            notes.append(f"symprec chosen automatically: {symprec:g} A (highest symmetry up to 0.1 A)")
        ideal, ops = idealize(structure, symprec)
        prim, _ = to_primitive(ideal, symprec)
        sg = identify(find_symmetry(prim, symprec), prim.lattice)
        cif = cif_with_symmetry(ideal, ops, symprec, sg.number, sg.symbol)
        out = SymmetryResult(structure_from_cif_text(cif, 1e-4), cif, sg.number, sg.symbol, sg.point_group,
                             "builtin", table, cif_path)
    out.notes = notes
    if cif_path:
        with open(cif_path, "w") as fh:
            fh.write(out.cif_text)
    return out
