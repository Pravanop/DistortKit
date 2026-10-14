"""Checkers: sanity checks on inputs and on the result of a parent/child comparison.

Every check returns ``Finding`` objects with a level (``error``, ``warning`` or ``info``), a stable ``code`` and
a human-readable message, so they can be printed, filtered, or turned into an exit status by the CLI.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .structure import Structure
from .symmetry import find_symmetry, to_primitive

ELEMENTS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb "
    "Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr D T".split()
)


@dataclass
class Finding:
    level: str          # "error" | "warning" | "info"
    code: str
    message: str

    def __str__(self):
        return f"[{self.level.upper():7s}] {self.code}: {self.message}"


def has_errors(findings):
    return any(f.level == "error" for f in findings)


def format_findings(findings, min_level="info"):
    order = {"info": 0, "warning": 1, "error": 2}
    return "\n".join(str(f) for f in findings if order[f.level] >= order[min_level])


# ====================================================================== input files
def symmetry_info_in_file(path):
    """Does the file carry explicit symmetry operations?  (True only for a CIF listing more than the identity.)"""
    if not path.lower().endswith(".cif"):
        return False
    from .io import _parse_cif
    try:
        with open(path) as fh:
            _, loops = _parse_cif(fh.read())
    except (OSError, ValueError):
        return False
    for header, rows in loops:
        for key in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
            if key in header and len(rows) > 1:
                return True
    return False


def check_file(path):
    f = []
    name = os.path.basename(path)
    if not os.path.isfile(path):
        return [Finding("error", "file-missing", f"{path} does not exist")]
    if os.path.getsize(path) == 0:
        return [Finding("error", "file-empty", f"{path} is empty")]
    low = path.lower()
    if not (low.endswith((".json", ".cif")) or "poscar" in low or "contcar" in low or low.endswith(".vasp")):
        f.append(Finding("info", "file-format", "format guessed from content: treated as POSCAR"))
    if not symmetry_info_in_file(path):
        f.append(Finding("info", "no-symmetry-info",
                         f"{name}: no explicit symmetry operations in the file; symmetry is detected from the structure"))
    return f


# ====================================================================== structure
def _min_distance(s):
    """Smallest interatomic distance in the periodic structure (Angstrom) and the offending pair."""
    from .spacegroup import reduce_basis
    U = reduce_basis(s.lattice)
    A = U @ s.lattice
    x = s.frac @ np.linalg.inv(U).T
    x = x - np.floor(x)
    best, pair = np.inf, None
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    n = len(x)
    for i in range(n):
        d = (x[None, :, :] - x[i][None, None, :] + shifts[:, None, :])
        dist = np.linalg.norm(d @ A, axis=2)
        if True:
            dist[:, i] = np.where((shifts == 0).all(axis=1), np.inf, dist[:, i])
        k = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[k] < best:
            best, pair = float(dist[k]), (i, int(k[1]))
    return best, pair


def check_structure(s: Structure, name="structure"):
    f = []
    if len(s) == 0:
        return [Finding("error", "no-atoms", f"{name} has no atoms")]
    for sp in set(s.species):
        sym = re.match(r"[A-Z][a-z]?", sp)
        if not sym or sym.group(0) not in ELEMENTS:
            f.append(Finding("warning", "species-name", f"{name}: '{sp}' is not a chemical element symbol"))
    vol = np.linalg.det(s.lattice)
    if abs(vol) < 1e-6:
        return f + [Finding("error", "singular-cell", f"{name}: lattice vectors are linearly dependent")]
    if vol < 0:
        f.append(Finding("info", "left-handed", f"{name}: lattice is left-handed; it is flipped internally"))
    a, b, c, al, be, ga = s.lattice_parameters
    if min(a, b, c) < 1.5:
        f.append(Finding("warning", "short-cell", f"{name}: a cell vector is only {min(a, b, c):.2f} A long"))
    if min(al, be, ga) < 30 or max(al, be, ga) > 150:
        f.append(Finding("info", "skewed-cell", f"{name}: very skewed cell angles; the cell is reduced internally"))
    if (s.frac < 0).any() or (s.frac >= 1).any():
        f.append(Finding("info", "wrapped", f"{name}: fractional coordinates outside [0,1) are wrapped"))
    if len(s) > 1:
        dmin, pair = _min_distance(s)
        if dmin < 0.3:
            f.append(Finding("error", "overlap",
                             f"{name}: atoms {pair[0] + 1} ({s.species[pair[0]]}) and {pair[1] + 1} "
                             f"({s.species[pair[1]]}) are only {dmin:.3f} A apart (duplicate atoms?)"))
        elif dmin < 0.9:
            f.append(Finding("warning", "close-atoms", f"{name}: closest atoms are {dmin:.2f} A apart"))
    f += [Finding("warning", "file-note", f"{name}: {n}") for n in s.notes]
    return f


def check_parent_symmetry(parent: Structure, tol=1e-3, tolerances=(1e-4, 1e-3, 1e-2, 5e-2, 1e-1)):
    """Is the parent really symmetric at the tolerance in use, and how fragile is that?"""
    from .spacegroup import identify
    f = []
    prim, _ = to_primitive(parent, tol)
    ops = find_symmetry(prim, tol)
    f.append(Finding("info", "parent-ops", f"parent has {len(ops)} symmetry operations per primitive cell at tolerance {tol:g} A"))
    if len(ops) == 1:
        f.append(Finding("warning", "parent-p1",
                         "parent has no symmetry at this tolerance (space group P1): it is probably not the "
                         "high-symmetry reference; increase the tolerance or symmetrise it"))
    counts = {}
    for t in tolerances:
        try:
            p2, _ = to_primitive(parent, t)
            counts[t] = len(find_symmetry(p2, t))
        except Exception:                                           # noqa: BLE001
            counts[t] = None
    here = counts.get(tol, len(ops))
    higher = [(t, n) for t, n in counts.items() if n and n > here and t > tol]
    if higher:
        t, n = higher[0]
        f.append(Finding("warning", "parent-tolerance",
                         f"parent gains symmetry at larger tolerance ({n} operations at {t:g} A instead of {here}): its "
                         "coordinates may be slightly off-symmetry; consider `symmetrize` or a larger --tol"))
    return f


# ====================================================================== comparison
def check_comparison(res, parent_ops=None, child_sg=None, parent_sg=None):
    """Diagnostics on a ``compare.Comparison`` result."""
    f = []
    ndist = float(np.linalg.norm(res.strain))
    if res.rms_displacement > 0.4:
        f.append(Finding("warning", "large-displacement",
                         f"rms atomic displacement is {res.rms_displacement:.2f} A: the match may be wrong, or the "
                         "structures are not related by a small distortion"))
    if res.max_displacement > 0.8:
        f.append(Finding("warning", "max-displacement",
                         f"largest displacement {res.max_displacement:.2f} A; check the atom matching"))
    if ndist > 0.06:
        f.append(Finding("warning", "large-strain", f"strain magnitude {ndist:.3f} is large (> 6%); linearised "
                                                     "strain/mode analysis loses accuracy"))
    if res.strain_forbidden > max(2e-3, 0.1 * ndist):
        f.append(Finding("warning", "forbidden-strain",
                         f"{res.strain_forbidden:.4f} of the strain is not allowed by the child's own symmetry: "
                         "noise or a wrong alignment"))
    if not res.child_in_parent:
        f.append(Finding("error", "not-subgroup",
                         "the child's point group is not a subgroup of the parent's: the parent is not a valid "
                         "high-symmetry reference for this child"))
    oc = getattr(res, "ops_check", None)
    if oc is not None and not oc.ok:
        kinds = {k for k, _ in oc.extra}
        if "rotation" in kinds:
            f.append(Finding("error", "not-subgroup-ops",
                             f"{oc.n_extra} symmetry operation(s) of the child are not operations of the parent (e.g. "
                             f"{oc.extra[0][1]}): the child is not a subgroup of this parent. The real parent probably has "
                             "more symmetry, or the two structures are siblings; see `relate`"))
        else:
            f.append(Finding("warning", "origin-mismatch",
                             f"{oc.n_extra} symmetry operation(s) of the child differ from the parent's only by a translation "
                             f"({oc.extra[0][1]}): different origin or setting, or a glide/screw the parent lacks"))
    if res.total_displacement > 0 and res.neglected > 0.05 * res.total_displacement:
        f.append(Finding("warning", "neglected-modes",
                         f"{res.neglected:.3f} A of displacement ({100 * res.neglected / res.total_displacement:.0f}%) "
                         "sits in wavevector stars below the threshold and was not decomposed; lower min_star_amp"))
    if res.unexplained > 1e-4:
        f.append(Finding("warning", "unexplained",
                         f"{res.unexplained:.2e} A of displacement is not explained by the decomposed modes"))
    if res.total_displacement < 1e-4 and np.linalg.norm(res.strain) < 1e-4:
        f.append(Finding("info", "no-distortion", "child is identical to the parent within tolerance"))
    if parent_sg is not None and child_sg is not None:
        if child_sg.number == parent_sg.number:
            f.append(Finding("info", "same-space-group", "child and parent have the same space group"))
    for m in res.modes:
        lab = getattr(m, "label", None)
        if lab is not None and m.amplitude > 1e-3 and lab.candidates and not lab.certain:
            f.append(Finding("info", "ambiguous-label",
                             f"k=({','.join(str(x) for x in m.k)}) irrep {m.irrep.index} could be {'/'.join(lab.candidates)} "
                             "(identical isotropy-subgroup fingerprints)"))
        if lab is not None and m.amplitude > 1e-3 and not lab.candidates:
            f.append(Finding("info", "no-label", f"irrep {m.irrep.index} at k=({','.join(str(x) for x in m.k)}) "
                                                 "matches no tabulated irrep"))
    return f


def check_pair(parent: Structure, child: Structure):
    """Cheap pre-flight checks before running the comparison."""
    f = []
    prim, _ = to_primitive(parent, 1e-3)
    n = len(prim)
    if len(child) % n:
        f.append(Finding("error", "atom-count",
                         f"child has {len(child)} atoms but the parent's primitive cell has {n}: not an integer supercell"))
        return f
    if sorted(child.species) != sorted(prim.species * (len(child) // n)):
        f.append(Finding("error", "composition", "child and parent compositions differ"))
    ratio = child.volume / prim.volume
    N = len(child) // n
    if abs(ratio - N) > 0.2 * N:
        f.append(Finding("warning", "volume", f"child volume is {ratio:.2f} parent cells but holds {N} cells' worth of atoms"))
    return f
