"""Parent / child comparison: the equivalent of ISODISTORT's "view distortion" analysis.

Given a high-symmetry parent structure and a low-symmetry child structure this module

1. finds the integer supercell matrix M relating the child cell to the parent primitive cell
   (any setting/orientation; the child lattice may be strained),
2. finds the origin shift and the one-to-one atom correspondence,
3. enumerates the wavevectors present in the child supercell, groups them into stars and
   decomposes the displacements into symmetry-adapted modes star by star,
4. decomposes the lattice strain (rotation removed) into strain irreps,
5. reports how well the modes explain the data and checks the child's symmetry against the
   parent's.

Conventions: displacements are Cartesian (Angstrom) measured on the *undeformed parent* lattice;
amplitudes are root-sum-square over the child's supercell, so the squared amplitudes of all modes
add up to the squared total displacement (Parseval).  Strain is the symmetric part of the
deformation, epsilon = U - I with F = R U, expressed in the parent's Cartesian axes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import isodata
from .irreps import split_real_irreps
from .modes import ModeAnalysis
from .structure import Structure
from .symmetry import find_symmetry, is_pure_translation, point_group_name, to_primitive


# ====================================================================== cells
class CellSet:
    """Integer translations T in Z^3 modulo the lattice spanned by the rows of ``M``."""

    def __init__(self, M):
        self.M = np.asarray(M, dtype=int)
        self.n = abs(int(round(np.linalg.det(self.M))))
        self._inv = np.linalg.inv(self.M)
        corners = np.array([sum(c * self.M[i] for i, c in enumerate(cs)) for cs in np.ndindex(2, 2, 2)])
        self._lo = corners.min(axis=0) - 1
        hi = corners.max(axis=0) + 1
        grid = np.array(list(np.ndindex(*(hi - self._lo + 1)))) + self._lo
        c = grid @ self._inv
        mask = np.all(c > -1e-9, axis=1) & np.all(c < 1 - 1e-9, axis=1)
        self.cells = grid[mask]
        if len(self.cells) != self.n:
            raise RuntimeError("cell enumeration failed")
        self._table = -np.ones(tuple(hi - self._lo + 1), dtype=int)
        for i, T in enumerate(self.cells):
            self._table[tuple(T - self._lo)] = i

    def canon(self, V):
        V = np.asarray(V, dtype=int)
        c = V @ self._inv
        V0 = V - np.rint(np.floor(c + 1e-9) @ self.M).astype(int)
        return self._table[tuple((V0 - self._lo).T)]


# ====================================================================== 1. cell relation
def strain_from_lattices(P, C):
    """Strain of child lattice rows ``C`` relative to reference lattice rows ``P``.

    ``C = P F^T`` (rows are lattice vectors).  Returns ``(eps, Rot)``: symmetric strain
    ``U - I`` in the reference axes and the removed rigid rotation.
    """
    F = np.linalg.solve(P, C).T
    w, V = np.linalg.eigh(F.T @ F)
    U = V @ np.diag(np.sqrt(w)) @ V.T
    return U - np.eye(3), F @ np.linalg.inv(U)


def candidate_cell_relations(A, C, N, strain_tol, max_candidates):
    """Integer matrices M (det = +N) such that ``M A`` approximates the child lattice ``C``.

    Only metric information is used, so the child may be in any orientation.  Candidates are
    sorted by strain size.
    """
    G = A @ A.T
    Ginv = np.linalg.inv(G)
    Gc = C @ C.T
    lc = np.sqrt(np.diag(Gc))
    rmax = lc.max() * (1 + strain_tol)
    bound = [int(np.floor(rmax * np.sqrt(Ginv[j, j]))) + 1 for j in range(3)]
    grid = np.array(list(itertools.product(*[range(-b, b + 1) for b in bound])), dtype=int)
    norms = np.sqrt(np.einsum("ni,ij,nj->n", grid, G, grid))
    S = [grid[np.abs(norms - lc[i]) <= strain_tol * lc[i] + 1e-9] for i in range(3)]
    if any(len(s) == 0 for s in S):
        return []
    cosc = Gc / np.outer(lc, lc)
    ctol = 2 * strain_tol + 0.03
    ns = [np.sqrt(np.einsum("ni,ij,nj->n", s, G, s)) for s in S]

    def cos_ok(i, j):
        cm = (S[i] @ G @ S[j].T) / np.outer(ns[i], ns[j])
        return np.abs(cm - cosc[i, j]) <= ctol

    ok01, ok02, ok12 = cos_ok(0, 1), cos_ok(0, 2), cos_ok(1, 2)
    out = []
    for a, b in zip(*np.where(ok01)):
        ma, mb = S[0][a], S[1][b]
        cand = np.where(ok02[a] & ok12[b])[0]
        if len(cand) == 0:
            continue
        cr = np.cross(ma, mb)
        det = S[2][cand] @ cr
        for c in cand[det == N]:
            M = np.array([ma, mb, S[2][c]])
            eps, _ = strain_from_lattices(M @ A, C)
            if np.abs(eps).max() <= strain_tol:
                out.append((float(np.linalg.norm(eps)), M, eps))
    out.sort(key=lambda t: t[0])
    return [(M, eps) for _, M, eps in out[:max_candidates]]


# ====================================================================== 2. atom matching
def match_atoms(Xs, sp_par, Y, sp_ch, P, match_tol, all_matches=False):
    """Find the origin shift and one-to-one correspondence child atom -> parent atom.

    ``Xs`` / ``Y`` are fractional coordinates in the same (supercell) basis, ``P`` the lattice
    used to convert fractional differences to Angstrom.  Returns ``(shift, perm, d_frac, rms, max)``
    or ``None``.  With ``all_matches`` it returns the list of every distinct valid pairing (one per
    anchor atom and origin), best first.
    """
    n = len(Y)
    sp_par, sp_ch = np.array(sp_par), np.array(sp_ch)
    same = sp_ch[:, None] == sp_par[None, :]
    names, counts = np.unique(sp_ch, return_counts=True)
    rare = names[np.argmin(counts)]
    j0 = int(np.where(sp_ch == rare)[0][0])
    best = None
    found = []
    for a in np.where(sp_par == rare)[0]:
        s = Y[j0] - Xs[a]
        for _ in range(4):
            diff = Y[:, None, :] - (Xs[None, :, :] + s)
            diff -= np.rint(diff)
            dist = np.linalg.norm(diff @ P, axis=2)
            dist[~same] = np.inf
            near = dist.argmin(axis=1)
            d = diff[np.arange(n), near]
            s = s + d.mean(axis=0)
        diff = Y[:, None, :] - (Xs[None, :, :] + s)
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff @ P, axis=2)
        dist[~same] = np.inf
        near = dist.argmin(axis=1)
        if len(set(near.tolist())) != n:
            continue
        d = diff[np.arange(n), near]
        dc = np.linalg.norm(d @ P, axis=1)
        if dc.max() > match_tol:
            continue
        rms = float(np.sqrt(np.mean(dc**2)))
        if all_matches:
            found.append((s % 1.0, near, d, rms, float(dc.max())))
        if best is None or rms < best[3] - 1e-12:
            best = (s % 1.0, near, d, rms, float(dc.max()))
    if all_matches:
        return sorted(found, key=lambda m: m[3])
    return best


class AlignmentError(ValueError):
    """The child cannot be laid onto the parent.  ``stage`` is ``"cell"`` (no integer supercell of the parent
    has the child's lattice) or ``"match"`` (the cell fits but the atoms cannot be paired within ``match_tol``)."""

    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def align_candidates(prim, child, N, strain_tol=0.12, match_tol=1.0, max_candidates=300):
    """Every distinct way of laying ``child`` onto a supercell of ``prim``: a list of
    ``(M, eps, cellset, P, (shift, perm, d_frac, rms, max))`` sorted by rms displacement then strain.
    Symmetry-equivalent overlays (same displacement pattern) appear once.  Raises ``AlignmentError`` if there is none."""
    n_par = len(prim)
    A = prim.lattice
    cands = candidate_cell_relations(A, child.lattice, N, strain_tol, max_candidates)
    if not cands:
        raise AlignmentError("cell", "no integer supercell of the parent matches the child lattice; try a larger strain_tol")
    out, seen = [], set()
    for M, eps in cands:
        cs = CellSet(M)
        Xs = np.array([(prim.frac[a] + T) @ np.linalg.inv(M) for a in range(n_par) for T in cs.cells])
        sp = [prim.species[a] for a in range(n_par) for _ in cs.cells]
        P = M @ A
        for m in match_atoms(Xs, sp, child.frac, child.species, P, match_tol, all_matches=True):
            key = (round(m[3], 5), round(m[4], 4), round(float(np.linalg.norm(eps)), 5))
            if key in seen:
                continue
            seen.add(key)
            out.append((M, eps, cs, P, m))
    if not out:
        raise AlignmentError("match", "could not match the child atoms onto any parent supercell within match_tol; "
                                      "increase match_tol or check that the child is a distortion of this parent")
    out.sort(key=lambda c: (round(c[4][3], 6), float(np.linalg.norm(c[1]))))
    return out


def align(prim, child, N, strain_tol=0.12, match_tol=1.0, max_candidates=300):
    """The overlay of ``child`` (N parent cells, right-handed) on the primitive ``prim`` with the smallest atomic
    displacements: ``(M, eps, cellset, P, (shift, perm, d_frac, rms, max))``; raises ``AlignmentError``."""
    return align_candidates(prim, child, N, strain_tol, match_tol, max_candidates)[0]


def align_checked(prim, parent_ops, child, N, strain_tol=0.12, match_tol=1.0, max_candidates=300, sym_tol=0.02,
                  max_alternatives=30):
    """Like ``align`` but prefers an overlay in which every symmetry operation of the child is an operation of the parent.

    The tightest overlay is checked first; only if it fails are looser ones tried (up to ``max_alternatives``).
    Returns ``(alignment, OpsCheck, note)``: when no overlay satisfies the parent's symmetry the tightest one is returned
    with its (failing) check, and ``note`` is empty; ``note`` says so when a looser overlay was chosen.
    """
    from .subgroup import child_ops_in_parent
    n_par = len(prim)
    cands = align_candidates(prim, child, N, strain_tol, match_tol, max_candidates)
    first = None
    for i, al in enumerate(cands[:1 + max_alternatives]):
        M, eps, cs, P, (shift, perm, d_frac, rms, dmax) = al
        u = np.empty((len(child), 3))
        u[perm] = d_frac @ P
        u -= u.mean(axis=0)
        Xs = np.array([(prim.frac[a] + T) @ np.linalg.inv(M) for a in range(n_par) for T in cs.cells])
        sp = [prim.species[a] for a in range(n_par) for _ in cs.cells]
        chk = child_ops_in_parent(prim, parent_ops, M, Xs, sp, u, P, sym_tol)
        if first is None:
            first = (al, chk)
        if chk.ok:
            note = ""
            if i:
                note = (f"the tightest overlay (rms {first[0][4][3]:.3f} A) does not respect the parent's symmetry; used "
                        f"the tightest one that does (rms {rms:.3f} A, origin/cell choice {i + 1})")
            return al, chk, note
    return first[0], first[1], ""


# ====================================================================== 4. strain irreps
_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _sym_basis():
    E = []
    for i, j in _VOIGT:
        m = np.zeros((3, 3))
        if i == j:
            m[i, i] = 1.0
        else:
            m[i, j] = m[j, i] = 1 / np.sqrt(2)
        E.append(m)
    return E


_E = _sym_basis()


def strain_vector(eps):
    return np.array([np.sum(e * eps) for e in _E])


def strain_matrix(vec):
    return sum(v * e for v, e in zip(vec, _E))


def strain_rep(Q):
    """6x6 orthogonal matrix of the action eps -> Q eps Q^T in the orthonormal basis ``_E``."""
    return np.array([[np.sum(_E[i] * (Q @ _E[m] @ Q.T)) for m in range(6)] for i in range(6)])


def _unique_rotations(Qs, tol=1e-6):
    out = []
    for Q in Qs:
        if not any(np.abs(Q - P).max() < tol for P in out):
            out.append(Q)
    return out


def _orthogonalize(Q):
    u, _, vt = np.linalg.svd(Q)
    return u @ vt


# ====================================================================== results
@dataclass
class ModeResult:
    k: tuple                    # representative wavevector (Fractions, primitive reciprocal basis)
    star_size: int              # arms of the star (before adding -k)
    irrep: object               # modes.Irrep
    species: list
    amplitude: float            # root-sum-square over the child supercell (Angstrom)
    copy_amplitudes: list       # per mode copy
    components: list            # per copy: components along the irrep basis (scaled, Angstrom)
    best_subgroup: object = None
    subgroup_residual: float = None
    canonical_basis: bool = False   # components are in the irrep's canonical (OPD) basis
    id: str = ""                    # short id (m1, m2, ... by decreasing amplitude) used to select the mode later
    label: object = None            # labels.IrrepLabel (standard irrep name) when available
    atom_displacements: np.ndarray = None   # (n_prim, n_cells, 3) Cartesian displacement carried by this mode (A)
    site_amplitudes: dict = None    # {site index: root-sum-square amplitude of this mode on that Wyckoff site (A)}


@dataclass
class StrainIrrep:
    index: int
    dim: int
    multiplicity: int
    fully_symmetric: bool
    amplitude: float


@dataclass
class Comparison:
    parent: Structure                # primitive parent
    child: Structure                 # child as given (right-handed)
    M: np.ndarray                    # child supercell vectors in parent primitive basis (rows)
    n_cells: int
    shift: np.ndarray                # origin shift in supercell fractional coordinates
    rms_displacement: float
    max_displacement: float
    strain: np.ndarray               # 3x3 symmetric strain, parent axes
    rotation_angle: float            # degrees, removed rigid rotation
    strain_irreps: list
    strain_forbidden: float          # part of the strain the child's symmetry does not allow
    child_point_group: str
    child_n_ops: int
    child_in_parent: bool
    modes: list
    total_displacement: float
    unexplained: float
    neglected: float                 # combined amplitude of stars below ``min_star_amp`` (not decomposed)
    n_neglected: int
    star_checks: list = field(default_factory=list)
    displacement_cart: np.ndarray = None   # (n_prim, n_cells, 3) after translation removal
    parent_sg: object = None         # spacegroup.SpaceGroupResult of the parent
    child_sg: object = None          # ... of the child (None if it could not be identified)
    findings: list = field(default_factory=list)   # checks.Finding
    ops_check: object = None         # subgroup.OpsCheck: are the child's operations operations of the parent?
    sites: list = None               # wyckoff.SiteInfo of the parent's atom orbits
    atom_positions: np.ndarray = None   # (n_prim, n_cells, 3) parent fractional positions of the child's atoms
    _cells: np.ndarray = None        # (n_cells, 3) integer translations of the child's cells

    # -------------------------------------------------------------- output
    def report(self, min_amp=1e-3, atoms=False):
        L = []
        s = self.parent
        pg = f", space group {self.parent_sg.label()}" if self.parent_sg else ""
        L.append(f"parent: primitive cell, {len(s)} atoms{pg}; child: {len(self.child)} atoms = {self.n_cells} parent cell(s)")
        L.append("supercell vectors in the parent primitive basis (rows = child a,b,c):")
        for r in self.M:
            L.append("   " + " ".join(f"{int(x):3d}" for x in r))
        L.append(f"origin shift (supercell fractional): {' '.join(f'{x:+.4f}' for x in self.shift)}")
        if self.parent_sg is not None:
            o = -(np.asarray(self.parent_sg.P, float) @ np.asarray(self.parent_sg.p, float))
            near = [s.species[i] for i in range(len(s)) if np.linalg.norm(((s.frac[i] - o) - np.rint(s.frac[i] - o)) @ s.lattice) < 0.05]
            at = f" (on {near[0]})" if near else ""
            L.append(f"irrep labels: standard origin of {self.parent_sg.symbol} at parent coordinates "
                     f"({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}){at}; the +/- in a label depends on this origin choice")
        L.append(f"atom matching: rms displacement {self.rms_displacement:.4f} A, max {self.max_displacement:.4f} A")
        L.append("(origin chosen so that the net displacement is zero: a rigid translation is not a distortion)")
        cs = f"space group {self.child_sg.label()}, " if self.child_sg else ""
        L.append(f"child symmetry: {cs}point group {self.child_point_group} ({self.child_n_ops} operations modulo lattice translations); "
                 f"{'subgroup of the parent point group' if self.child_in_parent else 'NOT a subgroup of the parent point group (check cell/tolerances)'}")
        if self.ops_check is not None:
            L.append("  " + self.ops_check.summary())
            for kind, text in self.ops_check.extra[:6]:
                L.append(f"    not in parent [{kind}]: {text}")
        L.append("")
        e = self.strain
        L.append(f"strain (parent Cartesian axes), rigid rotation {self.rotation_angle:.3f} deg removed:")
        for row in e:
            L.append("   " + " ".join(f"{x:+.5f}" for x in row))
        L.append(f"  strain magnitude {np.linalg.norm(e):.5f}; part not allowed by child symmetry {self.strain_forbidden:.2e}")
        for si in self.strain_irreps:
            if si.amplitude > 1e-6:
                tag = " (fully symmetric)" if si.fully_symmetric else ""
                L.append(f"  strain irrep {si.index}: dim {si.dim}{tag}, amplitude {si.amplitude:.5f}")
        L.append("")
        L.append(f"displacement modes (amplitude = root-sum-square over the {self.n_cells}-cell supercell, A):")
        shown = 0
        for m in sorted(self.modes, key=lambda m: -m.amplitude):
            if m.amplitude < min_amp:
                continue
            shown += 1
            k = ",".join(str(x) for x in m.k)
            name = f"{m.label.text} " if m.label is not None and m.label.candidates else ""
            L.append(f"  [{m.id}] k=({k}) star of {m.star_size}, {name}irrep {m.irrep.index} (dim {m.irrep.dim}, {m.irrep.kind}) "
                     f"on {'/'.join(m.species)}: {m.amplitude:.4f}")
            if m.best_subgroup is not None:
                sg = m.best_subgroup
                cg = getattr(sg, "space_group", None)
                sgt = f" -> {cg.label()}" if cg else ""
                L.append(f"      direction fits isotropy subgroup {sg.point_group} {sg.opd_label}{sgt} "
                         f"(residual {m.subgroup_residual:.1e})")
        if not shown:
            L.append("  (none above threshold)")
        L.append("")
        L += self.site_report(min_amp)
        if atoms:
            L.append("")
            L += self.atom_report(min_amp)
        L.append("")
        if self.n_neglected:
            L.append(f"{self.n_neglected} wavevector star(s) with amplitude below the threshold were not decomposed "
                     f"(combined {self.neglected:.4f} A; typically noise)")
        L.append(f"total displacement {self.total_displacement:.4f} A; unexplained by the decomposed modes {self.unexplained:.2e} A")
        notable = [f for f in self.findings if f.level != "info" or f.code in ("ambiguous-label", "no-label")]
        if notable:
            L.append("")
            L.append("checks:")
            L += [f"  {f}" for f in notable]
        return "\n".join(L)

    # -------------------------------------------------------------- saving for downstream use
    def save(self, path, meta=None):
        """Write everything needed to rebuild structures later (``freeze.Freezer.load``) as JSON."""
        import json
        modes = []
        for m in self.modes:
            if m.atom_displacements is None:
                continue
            sg = getattr(m.best_subgroup, "space_group", None) if m.best_subgroup is not None else None
            modes.append({
                "id": m.id, "name": m.label.text if m.label is not None and m.label.candidates else f"irrep{m.irrep.index}",
                "candidates": list(m.label.candidates) if m.label is not None else [],
                "certain": bool(m.label.certain) if m.label is not None else False,
                "k": [str(x) for x in m.k], "irrep": m.irrep.index, "dim": m.irrep.dim, "amplitude": m.amplitude,
                "species": m.species, "subgroup": None if m.best_subgroup is None else m.best_subgroup.point_group,
                "subgroup_opd": None if m.best_subgroup is None else m.best_subgroup.opd_label,
                "leads_to": None if sg is None else {"number": sg.number, "symbol": sg.symbol},
                "site_amplitudes": {str(k): v for k, v in (m.site_amplitudes or {}).items()},
                "field": m.atom_displacements.tolist(),
            })
        total = self.displacement_cart
        rest = total - sum((np.array(m["field"]) for m in modes), np.zeros_like(total))
        labs = self.atom_labels()
        doc = {
            "format": "distortkit-analysis", "version": 1, "meta": meta or {},
            "parent": {"lattice": self.parent.lattice.tolist(), "frac": self.parent.frac.tolist(),
                       "species": list(self.parent.species)},
            "parent_space_group": None if self.parent_sg is None else {"number": self.parent_sg.number, "symbol": self.parent_sg.symbol},
            "child_space_group": None if self.child_sg is None else {"number": self.child_sg.number, "symbol": self.child_sg.symbol},
            "supercell_matrix": self.M.tolist(), "n_cells": self.n_cells,
            "cells": [[int(x) for x in T] for T in self._cells],
            "atom_labels": [[labs[(a, c)] for c in range(self.n_cells)] for a in range(len(self.parent))],
            "sites": [{"index": si.index, "species": si.species, "wyckoff": si.label, "template": si.template,
                       "site_symmetry": si.site_symmetry, "atoms": si.atoms} for si in (self.sites or [])],
            "strain": self.strain.tolist(), "modes": modes,
            "rest": rest.tolist(), "total": total.tolist(),
            "total_displacement": self.total_displacement, "neglected": self.neglected,
            "rms_displacement": self.rms_displacement,
        }
        with open(path, "w") as fh:
            json.dump(doc, fh)
        return path

    # -------------------------------------------------------------- per-site and per-atom views
    def _shown_modes(self, min_amp):
        return [m for m in sorted(self.modes, key=lambda m: -m.amplitude) if m.amplitude >= min_amp and m.atom_displacements is not None]

    def _mode_name(self, m):
        k = ",".join(str(x) for x in m.k)
        base = m.label.text if m.label is not None and m.label.candidates else f"irrep{m.irrep.index}"
        return base if all(x == 0 for x in m.k) else f"{base}@({k})"

    def atom_labels(self):
        """Labels like Hf1, O3 for the atoms of the child supercell, in (primitive atom, cell) order."""
        count, out = {}, {}
        for a in range(len(self.parent)):
            for c in range(self.n_cells):
                sp = self.parent.species[a]
                count[sp] = count.get(sp, 0) + 1
                out[(a, c)] = f"{sp}{count[sp]}"
        return out

    def site_report(self, min_amp=1e-3):
        """Displacement of each Wyckoff site (atom orbit of the parent) and the share carried by each mode."""
        if not self.sites:
            return []
        u = self.displacement_cart
        modes = self._shown_modes(min_amp)
        sg = f" of {self.parent_sg.symbol} (#{self.parent_sg.number}, standard setting)" if self.parent_sg else ""
        L = [f"displacement by Wyckoff site{sg}: rms |u| per atom, then the root-sum-square amplitude of each mode on the site (A)"]
        labs = self.atom_labels()
        for si in self.sites:
            uu = u[si.atoms]                                           # (atoms, cells, 3)
            rms = float(np.sqrt(np.mean(np.sum(uu**2, axis=2))))
            where = f"{si.label} {si.template}" if si.label != "?" else "(Wyckoff site not identified)"
            first, last = labs[(si.atoms[0], 0)], labs[(si.atoms[-1], self.n_cells - 1)]
            L.append(f"  {si.species:<3s} {where}, site symmetry {si.site_symmetry}, {len(si.atoms) * self.n_cells} atoms "
                     f"({first}-{last}): rms |u| = {rms:.4f}")
            parts = [f"{self._mode_name(m)} {m.site_amplitudes.get(si.index, 0.0):.4f}" for m in modes]
            if parts:
                L.append("        " + "   ".join(parts))
        return L

    def atom_rows(self, min_amp=1e-3):
        labs = self.atom_labels()
        site_of = {a: si for si in (self.sites or []) for a in si.atoms}
        modes = self._shown_modes(min_amp)
        rows = []
        for a in range(len(self.parent)):
            for c in range(self.n_cells):
                u = self.displacement_cart[a, c]
                rest = u - sum((m.atom_displacements[a, c] for m in modes), np.zeros(3))
                si = site_of.get(a)
                rows.append({
                    "atom": labs[(a, c)], "species": self.parent.species[a],
                    "site": f"{si.label} {si.template}" if si and si.label != "?" else "",
                    "parent_frac": [float(x) for x in self.atom_positions[a, c]],
                    "u": [float(x) for x in u], "norm": float(np.linalg.norm(u)),
                    "modes": {self._mode_name(m): [float(x) for x in m.atom_displacements[a, c]] for m in modes},
                    "other": [float(x) for x in rest],
                })
        return rows

    def atom_report(self, min_amp=1e-3, max_modes=4):
        rows = self.atom_rows(min_amp)
        names = list(rows[0]["modes"])[:max_modes] if rows else []
        head = f"{'atom':<6s}{'site':<14s}{'parent fractional position':<30s}{'ux':>8s}{'uy':>8s}{'uz':>8s}{'|u|':>8s}"
        head += "".join(f"{('|u| ' + n)[:14]:>15s}" for n in names)
        L = ["displacement of every atom of the child supercell (Cartesian components along the parent's axes, A; "
             "last columns: size of each mode's share)", head]
        for r in rows:
            f = r["parent_frac"]
            line = (f"{r['atom']:<6s}{r['site'].split(' ')[0]:<14s}({f[0]:+.4f}, {f[1]:+.4f}, {f[2]:+.4f})  "
                    f"{r['u'][0]:+8.4f}{r['u'][1]:+8.4f}{r['u'][2]:+8.4f}{r['norm']:8.4f}")
            line += "".join(f"{np.linalg.norm(r['modes'][n]):15.4f}" for n in names)
            L.append(line)
        return L

    def to_dict(self):
        return {
            "supercell_matrix": self.M.tolist(),
            "origin_shift": self.shift.tolist(),
            "rms_displacement": self.rms_displacement,
            "strain": self.strain.tolist(),
            "child_point_group": self.child_point_group,
            "parent_space_group": None if self.parent_sg is None else {"number": self.parent_sg.number, "symbol": self.parent_sg.symbol},
            "child_space_group": None if self.child_sg is None else {"number": self.child_sg.number, "symbol": self.child_sg.symbol},
            "checks": [{"level": f.level, "code": f.code, "message": f.message} for f in self.findings],
            "modes": [
                {"k": [str(x) for x in m.k], "irrep": m.irrep.index, "dim": m.irrep.dim, "species": m.species,
                 "label": None if m.label is None else {"candidates": m.label.candidates, "certain": m.label.certain},
                 "amplitude": m.amplitude, "components": [c.tolist() for c in m.components],
                 "subgroup": None if m.best_subgroup is None else m.best_subgroup.point_group}
                for m in self.modes
            ],
            "strain_irreps": [{"index": s.index, "dim": s.dim, "amplitude": s.amplitude} for s in self.strain_irreps],
            "total_displacement": self.total_displacement,
            "sites": [{"index": si.index, "species": si.species, "wyckoff": si.label, "template": si.template,
                       "site_symmetry": si.site_symmetry, "atoms": si.atoms,
                       "mode_amplitudes": {self._mode_name(m): m.site_amplitudes.get(si.index, 0.0)
                                           for m in self._shown_modes(1e-3)}} for si in (self.sites or [])],
            "atoms": self.atom_rows(),
            "unexplained": self.unexplained,
            "neglected_below_threshold": self.neglected,
        }


# ====================================================================== main entry point
def _kgroup(M):
    """Wavevectors k (Fractions mod 1) with k.m_i integer for every supercell vector m_i."""
    N = abs(int(round(np.linalg.det(M))))
    adj = np.rint(np.linalg.inv(M) * np.linalg.det(M)).astype(int)
    sign = 1 if np.linalg.det(M) > 0 else -1
    ks = set()
    for z in itertools.product(range(N), repeat=3):
        v = adj @ np.array(z) * sign
        ks.add(tuple(Fraction(int(x), N) % 1 for x in v))
    assert len(ks) == N
    return sorted(ks)


def _star_cells(arms):
    """Number of primitive cells in the supercell commensurate with all the given wavevectors."""
    D = 1
    for a in arms:
        for x in a:
            D = D * x.denominator // np.gcd(D, x.denominator)
    sols = sum(1 for T in itertools.product(range(D), repeat=3)
               if all(sum(a[i] * T[i] for i in range(3)).denominator == 1 for a in arms))
    return D**3 // sols


def _star_arms(k, Rs):
    arms = set()
    for R in Rs:
        Rit = np.rint(np.linalg.inv(R)).astype(int).T
        kp = tuple(sum(int(Rit[i, j]) * k[j] for j in range(3)) % 1 for i in range(3))
        arms.add(kp)
        arms.add(tuple((-x) % 1 for x in kp))
    return arms


def compare(parent: Structure, child: Structure, *, tol=1e-3, strain_tol=0.12, match_tol=1.0,
            sym_tol=0.02, opd_tol=0.05, canonical=True,
            max_candidates=300, min_star_amp=2e-3, fit_min_amp=5e-3,
            max_star_cells=16, big_star_amp=0.02, labels=True, check_ops=True, canonical_star=False):
    """Mode and strain analysis of ``child`` relative to ``parent``.  See the module docstring."""
    prim, _ = to_primitive(parent, tol)
    A = prim.lattice
    n_par = len(prim)
    if len(child) % n_par:
        raise ValueError(f"child has {len(child)} atoms, not a multiple of the parent primitive cell's {n_par}")
    N = len(child) // n_par
    if sorted(child.species) != sorted(prim.species * N):
        raise ValueError("child composition is not N x the parent's primitive cell (vacancies/substitutions are not supported)")
    child = Structure(child.lattice.copy(), child.frac.copy(), list(child.species))
    if np.linalg.det(child.lattice) < 0:                       # right-handed child basis
        child.lattice[2] *= -1
        child.frac[:, 2] *= -1
    vol_ratio = child.volume / prim.volume
    if abs(vol_ratio - N) > 0.35 * N:
        raise ValueError(f"child volume is {vol_ratio:.2f} parent cells but it has {N} cells' worth of atoms")

    # 1+2: cell relation and atom matching, keep the candidate with the smallest displacements
    ops = find_symmetry(prim, tol)
    ops_check, align_note = None, ""
    if check_ops:
        (M, eps, cs, P, (shift, perm, d_frac, rms, dmax)), ops_check, align_note = align_checked(
            prim, ops, child, N, strain_tol, match_tol, max_candidates, sym_tol=sym_tol)
    else:
        M, eps, cs, P, (shift, perm, d_frac, rms, dmax) = align(prim, child, N, strain_tol, match_tol, max_candidates)
    d_cart = d_frac @ P                                        # child order
    u = np.empty((n_par * N, 3))
    u[perm] = d_cart                                           # parent order: index a*N + c
    u = u - u.mean(axis=0)                                     # matching already centres it; guard rounding
    u3 = u.reshape(n_par, N, 3)
    total2 = float(np.sum(u**2))

    # 3: wavevectors, stars, mode decomposition
    Rs = [R for R, _ in ops]
    parent_sg = None
    sites = None
    try:
        from .spacegroup import identify
        parent_sg = identify(ops, prim.lattice)
        from .wyckoff import describe_sites
        sites = describe_sites(prim, ops, parent_sg)
    except ValueError:
        parent_sg = None
    if sites is None:
        from .wyckoff import atom_orbits, SiteInfo
        sites = [SiteInfo(i, prim.species[o[0]], o, "?", 0, "", "?") for i, o in enumerate(atom_orbits(prim, ops, tol))]
    site_of_atom = {a: si.index for si in sites for a in si.atoms}
    K = _kgroup(M)
    Kset = set(K)
    T = cs.cells
    ck = {}
    for k in K:
        ph = np.exp(-2j * np.pi * (T @ np.array([float(x) for x in k])))
        ck[k] = np.einsum("acx,c->ax", u3, ph) / N
    modes, star_checks, covered = [], [], set()
    neglected2, n_neglected = 0.0, 0
    for k in K:
        if k in covered:
            continue
        orbit = _star_arms(k, Rs)
        covered |= orbit
        members = [q for q in orbit if q in Kset]
        power = N * sum(float(np.sum(np.abs(ck[q]) ** 2)) for q in members)
        if power < min_star_amp**2 or (power < big_star_amp**2 and _star_cells(orbit) > max_star_cells):
            star_checks.append((k, power, power))
            neglected2 += power
            n_neglected += power > 1e-12
            continue
        krep = min(orbit) if canonical_star else k      # canonical: irrep indices comparable between analyses
        ma = ModeAnalysis(prim, krep, tol=tol)
        assert {q for q in ma.star} | {tuple((-x) % 1 for x in q) for q in ma.star} == orbit
        Tm = ma.cells
        fld = np.zeros((n_par, ma.n_cells, 3))
        for q in members:
            ph = np.exp(2j * np.pi * (Tm @ np.array([float(x) for x in q])))
            fld += np.real(np.einsum("ax,c->acx", ck[q], ph))
        scale = np.sqrt(N / ma.n_cells)
        rows, _ = ma.decompose(fld.reshape(-1, 3), canonical=canonical, canonical_min=fit_min_amp / scale)
        by_irrep = {}
        for ir, mi, amp, c in rows:
            by_irrep.setdefault(ir.index, (ir, []))[1].append((mi, amp * scale, c * scale, c))
        explained = 0.0
        for idx, (ir, lst) in by_irrep.items():
            tot = float(np.sqrt(sum(a**2 for _, a, _, _ in lst)))
            explained += tot**2
            if tot < 1e-9:
                continue
            r = ModeResult(
                k=tuple(krep), star_size=len(ma.star), irrep=ir,
                species=sorted({ir.modes[mi].species for mi, a, _, _ in lst if a > 1e-9}),
                amplitude=tot, copy_amplitudes=[a for _, a, _, _ in lst], components=[c for _, _, c, _ in lst],
            )
            # displacement field carried by this irrep, evaluated on the child's own cells
            flat = sum(np.tensordot(c0, ir.modes[mi].vectors, axes=(0, 0)) for mi, _, _, c0 in lst)
            fma = flat.reshape(n_par, ma.n_cells, 3)
            r.atom_displacements = fma[:, ma._canon(cs.cells), :]
            r.site_amplitudes = {}
            for si in sites:
                amp2 = float(np.sum(r.atom_displacements[si.atoms] ** 2))
                if amp2 > 1e-24:
                    r.site_amplitudes[si.index] = float(np.sqrt(amp2))
            if canonical and tot >= fit_min_amp:
                r.canonical_basis = True
                _fit_subgroup(ma, ir, r, opd_tol)
                if labels and parent_sg is not None:
                    ma._sg = parent_sg
                    if isodata.labels_available():                       # irrep names need the ISOTROPY label tables
                        r.label = ma.irrep_labels(only={ir.index}).get(ir.index)
                    if r.best_subgroup is not None:
                        ma.child_space_group(r.best_subgroup)
            modes.append(r)
        star_checks.append((k, power, explained))

    # 4: strain
    strain, rot = eps, strain_from_lattices(P, child.lattice)[1]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    Qs = _unique_rotations([A.T @ R @ np.linalg.inv(A.T) for R in {tuple(R.ravel()): R for R in Rs}.values()])
    s_irreps = _strain_irreps(Qs, strain_vector(strain))

    # 5: child symmetry and consistency with the parent
    cprim, _ = to_primitive(child, sym_tol)                    # symmetry must be read off the child's primitive cell
    cops = find_symmetry(cprim, sym_tol)
    Rcs = [R for R, _ in cops]
    Ac = cprim.lattice
    # child operations -> Cartesian rotations in the parent's (undeformed) axes: undo the rigid rotation
    Qc = _unique_rotations([_orthogonalize(rot.T @ (Ac.T @ R @ np.linalg.inv(Ac.T)) @ rot) for R in Rcs], 1e-5)
    in_parent = all(any(np.abs(Qp - q).max() < 0.05 for Qp in Qs) for q in Qc)
    D = np.vstack([strain_rep(q) - np.eye(6) for q in Qc])
    _, sv, vt = np.linalg.svd(D)
    allowed = vt[int((sv > 1e-6).sum()):]
    ev = strain_vector(strain)
    forbidden = float(np.linalg.norm(ev - allowed.T @ (allowed @ ev))) if len(allowed) else float(np.linalg.norm(ev))
    done_power = sum(p for _, p, _ in star_checks) - neglected2
    child_sg = None
    try:
        from .spacegroup import identify
        child_sg = identify(cops, cprim.lattice)
    except ValueError:
        pass
    explained_total = sum(m.amplitude**2 for m in modes)
    modes.sort(key=lambda m: (-m.amplitude, str(m.k)))
    for i, m in enumerate(modes, start=1):
        m.id = f"m{i}"
    result = Comparison(
        parent=prim, child=child, M=M, n_cells=N, shift=shift, rms_displacement=rms, max_displacement=dmax,
        strain=strain, rotation_angle=ang, strain_irreps=s_irreps,
        strain_forbidden=forbidden, child_point_group=point_group_name(Rcs, cprim.lattice),
        child_n_ops=len(cops), child_in_parent=bool(in_parent), modes=modes,
        total_displacement=float(np.sqrt(total2)),
        unexplained=float(np.sqrt(max(done_power - explained_total, 0.0))),
        neglected=float(np.sqrt(neglected2)), n_neglected=int(n_neglected),
        star_checks=star_checks, displacement_cart=u3, parent_sg=parent_sg, child_sg=child_sg,
        ops_check=ops_check, sites=sites, atom_positions=np.array([[prim.frac[a] + T for T in cs.cells] for a in range(n_par)]),
        _cells=cs.cells,
    )
    from .checks import check_comparison
    result.findings = check_comparison(result, parent_sg=parent_sg, child_sg=child_sg)
    if align_note:
        from .checks import Finding
        result.findings.append(Finding("info", "alignment", align_note))
    return result


def _fit_subgroup(ma, ir, result, opd_tol):
    """Which isotropy subgroup's fixed subspace (up to symmetry) contains the measured OPD?"""
    C = np.array(result.components).T                          # (dim, n_copies)
    nrm = np.linalg.norm(C)
    if nrm < 1e-12:
        return
    best = None
    for iso in ma.isotropy_subgroups(ir):
        Qf, _ = np.linalg.qr(iso.fix_basis.T)                   # orthonormal basis (fix_basis is only row-reduced)
        P = Qf @ Qf.T
        imgs = ir.rho @ P @ ir.rho.transpose(0, 2, 1)
        res = np.linalg.norm((np.eye(ir.dim) - imgs) @ C, axis=(1, 2)).min() / nrm
        if res < opd_tol:
            key = (len(iso.op_indices), -iso.fix_dim)
            if best is None or key > best[0]:
                best = (key, iso, float(res))
    if best is not None:
        result.best_subgroup, result.subgroup_residual = best[1], best[2]


def _strain_irreps(Qs, evec):
    rho = np.array([strain_rep(Q) for Q in Qs])
    copies = split_real_irreps(rho, seed=3)
    classes = []
    for W, r, chi, norm in copies:
        for cl in classes:
            if cl[0][1].shape[1] == r.shape[1] and np.abs(cl[0][2] - chi).max() < 1e-6:
                cl.append((W, r, chi))
                break
        else:
            classes.append([(W, r, chi)])
    classes.sort(key=lambda cl: (cl[0][1].shape[1], -float(np.mean(cl[0][2]))))
    out = []
    for i, cl in enumerate(classes, start=1):
        d = cl[0][1].shape[1]
        amp = float(np.sqrt(sum(np.sum((W.T @ evec) ** 2) for W, _, _ in cl)))
        sym = d == 1 and np.allclose(cl[0][2], 1.0)
        out.append(StrainIrrep(i, d, len(cl), bool(sym), amp))
    return out
