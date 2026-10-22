"""Symmetry of phonon modes: which irrep of the parent a phonon (q, band) belongs to, and what it can lead to.

A phonon at wavevector q is a symmetry-adapted displacement pattern: for a given band its eigenvector transforms as one
irrep of the parent space group at the star of q.  Frozen in at small amplitude, it gives a child structure whose space group is
one of that irrep's isotropy subgroups; which one depends on the direction of the pattern inside the irrep (the order
parameter direction).  This module finds, for every band at a q-point,

* the irrep (number, standard label when the tables are available, dimension),
* how much of the mode lies in it (a value near 1 means a clean assignment; less means an accidental degeneracy or a
  structure that does not have the symmetry you think it has),
* the order-parameter direction of the eigenvector and the isotropy subgroup (child space group) it fits, if any, and
* all the isotropy subgroups of the irrep, i.e. everything the mode could lead to.

Method.  The eigenvector is turned into a real displacement field on the supercell that ``ModeAnalysis`` builds for the star of q,
and projected onto its irreps with ``ModeAnalysis.decompose``, exactly as ``compare`` does for a child structure.  An eigenvector
is complex in general; after removing its arbitrary overall phase so that the real part is as large as possible, the real and
imaginary parts are two snapshots of the vibration a quarter of a period apart, and both are analysed.  At a q equal to minus
itself the eigenvector is real and only the first snapshot is non-zero.

Conventions (see viewer/README.md for how they were checked): phonopy's eigenvectors belong to the dynamical matrix whose phase
contains the atomic positions, so the displacement of atom a in cell T is ``e_a / sqrt(m_a) * exp(2 pi i q.(T + x_a))``.  Pass
``basis_phase=False`` for data that uses lattice translations only.

Only q-points whose star fits in a small supercell are analysed (``max_cells``); the others are reported as skipped with the reason.
numpy only, like the rest of the package.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from . import isodata
from .compare import _fit_subgroup, _star_arms, _star_cells
from .modes import ModeAnalysis
from .structure import Structure
from .symmetry import find_symmetry


class Unclassifiable(ValueError):
    """The q-point cannot be analysed (too large a supercell, not a simple fraction, ...)."""


# ======================================================================================================================
# reading phonopy band.yaml
# ======================================================================================================================
_GREEK = {"gamma": "Γ", "delta": "Δ", "lambda": "Λ", "sigma": "Σ", "pi": "Π", "omega": "Ω", "theta": "Θ", "xi": "Ξ"}


def _clean_label(raw):
    s = raw.strip().strip("'\"").replace("$", "")
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1).lower(), m.group(1)), s)
    return "Γ" if s.upper() == "GAMMA" else s.replace("{", "").replace("}", "")


@dataclass
class BandData:
    """What the analysis needs from a phonopy ``band.yaml`` (written with eigenvectors)."""

    lattice: np.ndarray            # rows, Angstrom (the primitive cell phonopy used)
    species: list
    masses: np.ndarray
    frac: np.ndarray
    qpts: np.ndarray               # (nq, 3) reduced coordinates of that cell's reciprocal lattice
    dist: np.ndarray               # (nq,) distance along the path
    freq: np.ndarray               # (nq, nb) THz, as written (negative = imaginary)
    eig: np.ndarray                # (nq, nb, natom, 3) complex dynamical-matrix eigenvectors
    ticks: list = field(default_factory=list)   # [(q index, label)]

    @property
    def structure(self):
        return Structure(self.lattice, self.frac, self.species)


def _floats(line):
    s, e = line.index("["), line.index("]")
    return [float(x) for x in line[s + 1:e].split(",")]


def read_band_yaml(source):
    """Parse phonopy's ``band.yaml`` (a file name, or its text).  Needs eigenvectors (EIGENVECTORS = .TRUE.)."""
    if "\n" in source:
        text = source
    else:
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
    sec, nq, natom = "", 0, 0
    lattice, species, masses, frac, seg_n, pairs = [], [], [], [], [], []
    qpts = dist = freq = eig = None
    qi = bi = ai = ci = -1
    saw_eig = False
    for line in text.replace("\r", "").split("\n"):
        if not line:
            continue
        if line[0] not in " -":
            key, _, rest = line.partition(":")
            sec = key.strip().lstrip("﻿")
            rest = rest.split("#")[0].strip()
            if sec == "nqpoint":
                nq = int(rest)
            elif sec == "natom":
                natom = int(rest)
            elif sec == "phonon":
                nb = 3 * natom
                qpts, dist = np.zeros((nq, 3)), np.zeros(nq)
                freq, eig = np.zeros((nq, nb)), np.zeros((nq, nb, natom, 3), dtype=complex)
            continue
        t = line.strip()
        if sec == "phonon":
            if t.startswith("- ["):
                if ai >= 0 and ci < 3:
                    re_, im_ = _floats(t)
                    eig[qi, bi, ai, ci] = complex(re_, im_)
                    ci += 1
                    saw_eig = True
            elif t.startswith("- q-position:"):
                qi, bi, ai = qi + 1, -1, -1
                qpts[qi] = _floats(t)
            elif t.startswith("- #"):
                if len(line) - len(line.lstrip()) <= 3:
                    bi, ai = bi + 1, -1
                else:
                    ai, ci = ai + 1, 0
            elif t.startswith("frequency:"):
                freq[qi, bi] = float(t[10:])
            elif t.startswith("distance:"):
                dist[qi] = float(t[9:])
        elif sec == "segment_nqpoint":
            seg_n.append(int(t.lstrip("- ")))
        elif sec == "labels":
            pairs.append([_clean_label(x) for x in t[t.index("[") + 1:t.rindex("]")].split(",")])
        elif sec == "lattice" and t.startswith("- ["):
            lattice.append(_floats(t))
        elif sec == "points":
            if t.startswith("- symbol:"):
                species.append(t[9:].split("#")[0].strip())
            elif t.startswith("coordinates:"):
                frac.append(_floats(t))
            elif t.startswith("mass:"):
                masses.append(float(t[5:]))
    if eig is None or not natom:
        raise ValueError("not a phonopy band.yaml (no 'phonon:' section)")
    if qi + 1 != nq:
        raise ValueError(f"band.yaml: expected {nq} q-points, found {qi + 1}")
    if not saw_eig:
        raise ValueError("this band.yaml has no eigenvectors; re-run phonopy with EIGENVECTORS = .TRUE.")
    if len(lattice) != 3 or len(frac) != natom:
        raise ValueError("band.yaml: missing lattice or atomic positions")
    if dist[-1] - dist[0] < 1e-9:
        dist = np.arange(nq, dtype=float)
    ticks = []
    if seg_n and len(pairs) == len(seg_n):
        start = 0
        for n, (a, b) in zip(seg_n, pairs):
            end = start + n - 1
            if ticks and ticks[-1][0] == start - 1:
                if ticks[-1][1] != a:
                    ticks[-1] = (ticks[-1][0], f"{ticks[-1][1]}|{a}")
            else:
                ticks.append((start, a))
            ticks.append((end, b))
            start += n
    return BandData(np.array(lattice), species, np.array(masses), np.array(frac), qpts, dist, freq, eig, ticks)


# ======================================================================================================================
# results
# ======================================================================================================================
@dataclass
class SubgroupInfo:
    """One isotropy subgroup of an irrep: the child symmetry for an order-parameter direction."""

    index: int
    opd: str                       # e.g. "(a,0,0)"
    fix_dim: int                   # free amplitudes of this direction
    point_group: str
    n_conjugates: int              # symmetry-equivalent domains
    space_group: str = None        # "Pbca (#61)"; None when DistortKit could not identify it
    number: int = None

    def to_dict(self):
        return dict(index=self.index, opd=self.opd, fix_dim=self.fix_dim, point_group=self.point_group,
                    n_conjugates=self.n_conjugates, space_group=self.space_group, number=self.number)


@dataclass
class IrrepInfo:
    index: int
    label: str                     # standard label ("X2-") or None
    label_certain: bool
    dim: int
    kind: str
    species: list
    subgroups: list                # [SubgroupInfo]
    note: str = ""

    def to_dict(self):
        return dict(index=self.index, label=self.label, label_certain=self.label_certain, dim=self.dim, kind=self.kind,
                    species=self.species, note=self.note, subgroups=[s.to_dict() for s in self.subgroups])


@dataclass
class StarInfo:
    """Everything that depends on the wavevector star only (shared by all bands and equivalent q-points)."""

    k: tuple                       # representative wavevector (strings, e.g. "1/2")
    star_size: int                 # arms of the star (before adding -k)
    n_cells: int                   # primitive cells in the supercell the analysis used
    parent: str                    # parent space group, "Fm-3m (#225)"
    irreps: list                   # [IrrepInfo]

    def irrep(self, index):
        return next(i for i in self.irreps if i.index == index)

    def to_dict(self):
        return dict(k=list(self.k), star_size=self.star_size, n_cells=self.n_cells, parent=self.parent,
                    irreps=[i.to_dict() for i in self.irreps])


@dataclass
class Snapshot:
    """One of the two real snapshots of a mode (``re`` and ``im``, a quarter period apart)."""

    part: str
    share: float                   # fraction of the mode's displacement power in this snapshot
    opd: list = None               # direction in the irrep's canonical basis (strongest copy, unit norm)
    subgroup: SubgroupInfo = None  # isotropy subgroup this direction fits (None: a generic direction)
    residual: float = None         # misfit of the direction to that subgroup (relative)

    def to_dict(self):
        return dict(part=self.part, share=round(self.share, 6), opd=self.opd, residual=self.residual,
                    subgroup=self.subgroup.to_dict() if self.subgroup else None)


@dataclass
class ModeSymmetry:
    q_index: int
    band: int                      # 0-based, in the order of the file (ascending frequency)
    freq: float                    # THz
    irrep: int                     # DistortKit irrep index within the star
    label: str
    dim: int
    weight: float                  # fraction of the displacement that lies in `irrep`
    outside: float                 # fraction outside every irrep of this star (should be ~0)
    weights: dict                  # {irrep index: fraction} for every irrep above 0.1 %
    snapshots: list
    degenerate_with: list = field(default_factory=list)   # other bands at this q within the numerical tolerance
    rigid: bool = False            # a rigid translation of the whole crystal (acoustic mode at q = 0): the structure does not change
    adapted: bool = False          # eigenvector rotated inside its degenerate level into a symmetry-adapted basis
    eigenvector: np.ndarray = None  # (natom, 3) complex dynamical-matrix eigenvector actually analysed (set when adapted)

    @property
    def clean(self):
        return self.weight > 0.98 and self.outside < 1e-6

    def to_dict(self):
        return dict(band=self.band, freq=round(float(self.freq), 6), irrep=self.irrep, label=self.label, dim=self.dim,
                    weight=round(self.weight, 6), outside=float(f"{self.outside:.3g}"),
                    weights={str(k): round(v, 6) for k, v in self.weights.items()},
                    degenerate_with=self.degenerate_with, adapted=self.adapted, rigid=self.rigid,
                    eigenvector=None if self.eigenvector is None else
                    [[[float(f"{z.real:.9g}"), float(f"{z.imag:.9g}")] for z in row] for row in self.eigenvector],
                    snapshots=[s.to_dict() for s in self.snapshots])


# ======================================================================================================================
# analysis
# ======================================================================================================================
def _frac(x, max_den):
    f = Fraction(float(x)).limit_denominator(max_den)
    if abs(float(f) - float(x)) > 2e-4:
        raise Unclassifiable(f"q component {float(x):.5f} is not a fraction with denominator <= {max_den}")
    return f


def _fstr(f):
    return str(f)


class PhononSymmetry:
    """Irrep and isotropy-subgroup analysis of the phonons of one parent structure.

    ``parent`` must be a primitive cell with the symmetry of the high-symmetry phase (phonopy's primitive cell is).
    Stars are cached, so analysing many q-points of the same file is cheap.
    """

    def __init__(self, parent: Structure, *, tol=1e-3, max_cells=16, opd_tol=0.05, labels=True, max_dim=6, max_den=24,
                 progress=None):
        """``max_dim``: irreps of larger dimension are still assigned to modes, but their isotropy subgroups (and so labels and
        child symmetries) are not enumerated, which can take minutes (L of a cubic cell: ~7 min for two 8-dimensional irreps)."""
        self.parent = parent
        self.progress = progress          # optional callback (stage: str, fraction: float) for long analyses
        self.tol, self.max_cells, self.opd_tol = tol, max_cells, opd_tol
        self.labels, self.max_dim, self.max_den = labels and isodata.labels_available(), max_dim, max_den
        ops = find_symmetry(parent, tol)
        self.Rs = [R for R, _ in ops]
        self.n_ops = len(ops)
        self.problem = self._check_group(ops)   # why the symmetry cannot be used at this tolerance, or None
        # A lattice from a DFT code carries numerical noise (1e-7 relative), which makes the Cartesian rotations orthogonal only to
        # about that much; the irrep splitter needs much better.  Use the nearest lattice that has the symmetry exactly (the metric
        # averaged over the group) and express the displacements in its frame.
        self.cart = np.eye(3)                   # file's Cartesian frame -> the symmetrised one (displacement rows are multiplied by it)
        if not self.problem:
            A = np.asarray(parent.lattice, float)
            G = A @ A.T
            Gs = np.linalg.inv(np.mean([R @ np.linalg.inv(G) @ R.T for R in self.Rs], axis=0))
            dev = float(np.abs(Gs - G).max() / np.abs(G).max())
            if dev > 1e-3:
                self.problem = (f"the lattice does not have the symmetry the atoms have (its metric differs from the symmetric one by "
                                f"{dev:.1e}); the structure is a strained or distorted cell, not the high-symmetry phase")
            else:
                As = np.linalg.cholesky(Gs)
                self.parent = Structure(As, parent.frac, parent.species)
                self.cart = np.linalg.inv(A) @ As
        self._stars = {}            # krep -> (ModeAnalysis, StarInfo)

    def _check_group(self, ops):
        """A structure that is only approximately symmetric can give operations that do not close into a group at a tight tolerance;
        the irrep analysis needs a true group."""
        from .spacegroup import _check_group
        try:
            _check_group(ops)
        except ValueError as exc:
            return (f"the symmetry found at tolerance {self.tol:g} A is not a consistent group ({exc}). The structure is probably only "
                    f"approximately symmetric: try a larger tolerance, such as {max(10 * self.tol, 0.01):g} A")
        return None

    # ------------------------------------------------------------------ stars
    def _krep(self, q):
        if self.problem:
            raise Unclassifiable(self.problem)
        k = tuple(_frac(x, self.max_den) for x in q)
        orbit = _star_arms(k, self.Rs)
        cells = _star_cells(orbit)
        if cells > self.max_cells:
            raise Unclassifiable(f"its star needs a {cells}-cell supercell (limit {self.max_cells})")
        return min(orbit)

    def _say(self, stage, frac):
        if self.progress:
            self.progress(stage, float(frac))

    def star(self, q):
        """(ModeAnalysis, StarInfo) for the star containing ``q``; raises ``Unclassifiable`` if it is too large."""
        krep = self._krep(q)
        if krep not in self._stars:
            self._say("Finding the irreps of this wavevector", 0.05)
            try:
                ma = ModeAnalysis(self.parent, krep, tol=self.tol)
            except (RuntimeError, ValueError) as exc:          # numerical trouble inside the irrep decomposition
                raise Unclassifiable(f"the irreps of this wavevector could not be separated ({exc}). This usually means the structure "
                                     f"is only approximately symmetric: try a larger tolerance than {self.tol:g} A")
            self._stars[krep] = (ma, self._describe(ma, krep))
        return self._stars[krep]

    def _subgroup(self, ma, iso):
        sg = ma.child_space_group(iso)
        return SubgroupInfo(iso.index, iso.opd_label, iso.fix_dim, iso.point_group, iso.n_conjugates,
                            sg.label() if sg else None, sg.number if sg else None)

    def _describe(self, ma, krep):
        labs = {}
        n_ir = len(ma.irreps)
        irreps = []
        for k, ir in enumerate(ma.irreps):
            self._say(f"Listing the child symmetries of irrep {k + 1} of {n_ir}", 0.1 + 0.5 * k / max(n_ir, 1))
            note, subs = "", []
            if ir.dim > self.max_dim:
                note = f"isotropy subgroups not enumerated (dimension {ir.dim} > max_dim={self.max_dim})"
            else:
                try:
                    subs = [self._subgroup(ma, iso) for iso in ma.isotropy_subgroups(ir)]   # also fixes the canonical basis
                except RuntimeError as exc:
                    note = f"isotropy subgroups not enumerated: {exc}"
            irreps.append(IrrepInfo(ir.index, None, False, ir.dim, ir.kind, ir.species, subs, note))
        if self.labels:                       # standard names (X2-): they need every subgroup's space group, so they come last
            self._say("Naming the irreps", 0.7)
            small = {ir.index for ir in ma.irreps if ir.dim <= self.max_dim}
            labs = ma.irrep_labels(only=small) if small else {}
            for info in irreps:
                lab = labs.get(info.index)
                if lab and lab.candidates:
                    info.label, info.label_certain = lab.text, bool(lab.certain)
        sg = ma.space_group()
        return StarInfo(tuple(_fstr(x) for x in krep), len(ma.star), ma.n_cells, sg.label(), irreps)

    # ------------------------------------------------------------------ modes
    def field(self, ma, q, e, *, masses=None, basis_phase=True, gauge=True):
        """Complex displacement field (n_atoms * n_cells, 3) of one eigenvector ``e`` (natom, 3) on the star's supercell."""
        n = len(self.parent)
        if masses is not None:                                       # eigenvector of the dynamical matrix -> displacement
            e = e / np.sqrt(np.asarray(masses, float))[:, None]
        phase = 2 * np.pi * (ma.cells @ np.asarray(q, float))[None, :]
        if basis_phase:
            phase = phase + 2 * np.pi * (self.parent.frac @ np.asarray(q, float))[:, None]
        z = (e @ self.cart)[:, None, :] * np.exp(1j * phase)[:, :, None]          # (n, Nc, 3), in the symmetrised frame
        s = np.sum(z.ravel() ** 2)                                   # overall phase: make the real part as large as possible
        if gauge and abs(s) > 1e-14 * max(1.0, float(np.sum(np.abs(z) ** 2))):
            z = z * np.exp(-0.5j * np.angle(s))
        return z.reshape(n * ma.n_cells, 3)

    def classify_mode(self, q, e, *, q_index=0, band=0, freq=0.0, masses=None, basis_phase=True):
        ma, info = self.star(q)
        z = self.field(ma, q, e, masses=masses, basis_phase=basis_phase)
        # at q = 0 an acoustic mode moves every atom the same way: that only translates the crystal, so its space group is unchanged
        # (the isotropy subgroup of the irrep describes relative displacements and does not apply)
        f0 = z.reshape(-1, 3)
        # (DFT acoustic modes are only nearly uniform: the acoustic sum rule holds to ~1e-3, so allow a few percent)
        rigid = bool(np.allclose(q, 0, atol=1e-9)) and float(np.linalg.norm(f0 - f0.mean(axis=0))) < 0.05 * float(np.linalg.norm(f0))
        parts = {"re": z.real, "im": z.imag}
        power = {p: float(np.sum(f * f)) for p, f in parts.items()}
        total = sum(power.values())
        if total < 1e-20:
            raise Unclassifiable("the eigenvector is zero")
        amp2, outside, rows_by_part = {}, 0.0, {}
        for p, f in parts.items():
            if power[p] < 1e-12 * total:
                continue
            rows, resid = ma.decompose(f, canonical=False)
            rows_by_part[p] = rows
            outside += resid**2
            for ir, mi, amp, c in rows:
                amp2[ir.index] = amp2.get(ir.index, 0.0) + amp**2
        weights = {i: v / total for i, v in amp2.items() if v / total > 1e-3}
        main = max(amp2, key=amp2.get)
        ir = next(x for x in ma.irreps if x.index == main)
        snaps = []
        for p, rows in rows_by_part.items():
            mine = [(mi, amp, c) for i, mi, amp, c in rows if i.index == main]
            if sum(a * a for _, a, _ in mine) < 1e-6 * power[p] or power[p] < 0.02 * total:
                continue
            ns = SimpleNamespace(components=[c for _, _, c in mine], best_subgroup=None, subgroup_residual=None)
            if not info.irrep(main).note and not rigid:
                _fit_subgroup(ma, ir, ns, self.opd_tol)
            strongest = max(mine, key=lambda t: t[1])[2]
            opd = (strongest / np.linalg.norm(strongest)).round(6).tolist() if np.linalg.norm(strongest) > 1e-12 else None
            sub = self._subgroup(ma, ns.best_subgroup) if ns.best_subgroup is not None else None
            snaps.append(Snapshot(p, power[p] / total, opd, sub, ns.subgroup_residual))
        ii = info.irrep(main)
        return ModeSymmetry(q_index, band, float(freq), main, ii.label, ir.dim, amp2[main] / total, outside / total,
                            weights, snaps, rigid=rigid)

    # ------------------------------------------------------------------ degenerate levels
    def _gauge(self, ma, q, e, masses, basis_phase):
        """``e`` times the phase that makes the real part of its field as large as possible (see ``field``)."""
        z = self.field(ma, q, e, masses=masses, basis_phase=basis_phase)
        s = np.sum(z.ravel() ** 2)
        if abs(s) > 1e-14 * max(1.0, float(np.sum(np.abs(z) ** 2))):
            return e * np.exp(-0.5j * np.angle(s))
        return e

    def _adapt_cluster(self, ma, q, E, masses, basis_phase):
        """Rotate the eigenvectors ``E`` (m, natom, 3) of one degenerate level so each lies in a single irrep.

        The level's eigenspace is a sum of pieces, one per irrep; any basis of it mixes them.  The projectors onto the irreps
        (``M_i`` below) commute and add up to the identity on the eigenspace, so diagonalising a generic combination of them
        separates the pieces.  Inside one irrep the basis stays arbitrary here (see ``_canonical_directions``).

        When q equals minus itself (zone centre and the other special points where the standing wave is real) the level has a
        real basis and the rotation is done in it, so the adapted eigenvectors still describe real displacement patterns.
        """
        m = len(E)
        n, Nc = len(self.parent), ma.n_cells
        Z = np.stack([self.field(ma, q, e, masses=masses, basis_phase=basis_phase, gauge=False).ravel() for e in E], axis=1)

        def projectors(Q):
            Ms = []
            for ir in ma.irreps:
                V = np.vstack([mode.vectors.reshape(ir.dim, -1) for mode in ir.modes])
                A = V @ Q
                Ms.append(A.conj().T @ A)
            return Ms

        weights = lambda Ms: sum(np.sqrt(2.0 + 3 * i) * M for i, M in enumerate(Ms))   # distinct, generic weight per irrep
        if np.allclose(2 * np.asarray(q, float), np.round(2 * np.asarray(q, float)), atol=1e-6):
            Zg = Z * np.exp(-0.5j * np.angle(np.sum(Z * Z, axis=0)))[None, :]
            U, s, _ = np.linalg.svd(np.hstack([Zg.real, Zg.imag]), full_matrices=False)
            if (s > 1e-9 * s[0]).sum() == m:
                Q = U[:, :m]                                         # real orthonormal basis of the level
                _, W = np.linalg.eigh(weights(projectors(Q)))
                F = (Q @ W).T.reshape(m, n, Nc, 3)                   # real fields of the adapted modes
                c0 = int(np.flatnonzero(np.all(ma.cells == 0, axis=1))[0])
                phase = np.exp(2j * np.pi * (np.asarray(self.parent.frac @ np.asarray(q, float)) if basis_phase else np.zeros(n)))
                d = 1.0 / np.sqrt(np.asarray(masses, float)) if masses is not None else np.ones(n)
                out = F[:, :, c0, :] / (d * phase)[None, :, None]    # back to eigenvectors: field(e)[a, T=0] = d_a e_a phase_a
                out = out @ np.linalg.inv(self.cart)                  # ... in the file's Cartesian frame again
                return out / np.linalg.norm(out.reshape(m, -1), axis=1)[:, None, None]
        Q, R = np.linalg.qr(Z)
        _, W = np.linalg.eigh((lambda M: (M + M.conj().T) / 2)(weights(projectors(Q))))
        cols = E.reshape(m, -1).T @ np.linalg.inv(R) @ W            # the map Z -> Q acts on eigenvectors the same way
        out = cols.T.reshape(E.shape)
        return out / np.linalg.norm(out.reshape(m, -1), axis=1)[:, None, None]

    def _canonical_directions(self, ma, q, E, masses, basis_phase):
        """Within a multiplet of bands of one irrep, rotate to directions along the irrep's canonical axes.

        The bands' order-parameter vectors span a subspace of the irrep; its row-reduced basis (vectors like (a,0,0), (0,a,0))
        gives the sparsest, most symmetric directions, and a real rotation of the eigenvectors realises it.  Only for real
        eigenvectors (q equal to minus itself); returns ``E`` unchanged otherwise.
        """
        m = len(E)
        E = np.array([self._gauge(ma, q, e, masses, basis_phase) for e in E])
        coeffs, main = [], None
        for e in E:
            z = self.field(ma, q, e, masses=masses, basis_phase=basis_phase)
            if np.sum(z.imag ** 2) > 1e-9 * np.sum(z.real ** 2):
                return E
            rows, _ = ma.decompose(z.real, canonical=False)
            by = {}
            for ir, mi, amp, c in rows:
                by.setdefault(ir.index, []).append(c)
            top = max(by, key=lambda i: sum(float(c @ c) for c in by[i]))
            if main not in (None, top):
                return E
            main = top
            coeffs.append(by[top])
        ncopy = len(coeffs[0])
        best = max(range(ncopy), key=lambda i: sum(float(np.linalg.norm(c[i])) for c in coeffs))
        Cm = np.array([c[best] for c in coeffs])                     # (m, dim)
        if np.linalg.matrix_rank(Cm, tol=1e-6 * np.abs(Cm).max()) < m:
            return E
        Rr, _ = ma._rref(Cm, tol=1e-7 * np.abs(Cm).max())
        if len(Rr) != m:
            return E
        G = []
        for r in Rr:                                                 # Gram-Schmidt keeps the rows orthonormal
            for g in G:
                r = r - (r @ g) * g
            G.append(r / np.linalg.norm(r))
        T = np.array(G) @ np.linalg.pinv(Cm)
        return np.tensordot(T, E, axes=(1, 0))

    def classify(self, q, eigs, freqs=None, *, q_index=0, masses=None, basis_phase=True, adapt=True):
        """All bands at ``q``: ``eigs`` is (nbands, natom, 3) complex.  Returns ``[ModeSymmetry]``.

        With ``adapt`` (default) the eigenvectors of every degenerate level (bands within a numerical tolerance of each other)
        are replaced by an equivalent set adapted to the symmetry: one irrep per vector, and inside an irrep along its
        canonical axes.  The file's own eigenvectors there are an arbitrary basis of the level, so they say nothing about
        symmetry.  Adapted bands carry their eigenvector (``ModeSymmetry.eigenvector``) so a viewer can animate them.
        """
        eigs = np.array(eigs, dtype=complex)
        freqs = np.zeros(len(eigs)) if freqs is None else np.asarray(freqs, float)
        ma, _ = self.star(q)
        tol = max(2e-3, 2e-4 * float(np.max(np.abs(freqs))))
        clusters, cur = [], [0]
        for b in range(1, len(eigs)):
            if abs(freqs[b] - freqs[b - 1]) < tol:
                cur.append(b)
            else:
                clusters.append(cur)
                cur = [b]
        clusters.append(cur)
        adapted = set()
        kw = dict(masses=masses, basis_phase=basis_phase)
        if adapt:
            for cl in clusters:
                if len(cl) > 1:
                    eigs[cl] = self._adapt_cluster(ma, q, eigs[cl], **kw)
                    adapted.update(cl)

        def run(b):
            return self.classify_mode(q, eigs[b], q_index=q_index, band=b, freq=freqs[b], **kw)

        out = [run(b) for b in range(len(eigs))]
        if adapt:                                                    # second stage: canonical directions inside each irrep
            for cl in clusters:
                groups = {}
                for b in cl:
                    if out[b].clean:
                        groups.setdefault(out[b].irrep, []).append(b)
                for irr, bs in groups.items():
                    if len(bs) > 1 and out[bs[0]].dim > 1:
                        eigs[bs] = self._canonical_directions(ma, q, eigs[bs], **kw)
                        for b in bs:
                            out[b] = run(b)
        for a in out:
            a.degenerate_with = [b.band for b in out if b.band != a.band and abs(b.freq - a.freq) < tol]
            if a.band in adapted:
                e = eigs[a.band]
                a.adapted, a.eigenvector = True, e / np.linalg.norm(e)      # unit norm, like phonopy's eigenvectors
        return out


# ======================================================================================================================
# one q-point at a time (the CLI loops over this; the web viewer calls it on request)
# ======================================================================================================================
def analyze_qpoint(ps, q, eig, freq, *, q_index=0, masses=None, basis_phase=True):
    """Analyse all bands at one q-point.  Returns ``{"star": k, "star_info": {...}, "bands": [...]}`` or ``{"skipped": reason}``."""
    try:
        ma, info = ps.star(q)
        ps._say("Classifying the modes", 0.8)
        modes = ps.classify(q, eig, freq, q_index=q_index, masses=masses, basis_phase=basis_phase)
    except Unclassifiable as exc:
        return {"skipped": str(exc)}
    return {"star": list(info.k), "star_info": info.to_dict(), "bands": [m.to_dict() for m in modes]}


# ======================================================================================================================
# whole files
# ======================================================================================================================
def analyze_band_yaml(path_or_data, *, max_cells=16, max_dim=6, labels=True, tol=1e-3, opd_tol=0.05, mass_scale=True,
                      basis_phase=True, progress=None):
    """Analyse every q-point of a phonopy ``band.yaml``; returns the JSON-ready dictionary written by ``distortkit phonons``.

    q-points whose star needs more than ``max_cells`` primitive cells are listed as skipped, with the reason.
    """
    bd = path_or_data if isinstance(path_or_data, BandData) else read_band_yaml(path_or_data)
    ps = PhononSymmetry(bd.structure, tol=tol, max_cells=max_cells, max_dim=max_dim, labels=labels, opd_tol=opd_tol)
    sg = None
    qpoints, done, stars = [], {}, {}
    for qi, q in enumerate(bd.qpts):
        key = tuple(np.round(q, 6))
        entry = {"index": qi, "q": [round(float(x), 8) for x in q]}
        if key not in done:
            res = analyze_qpoint(ps, q, bd.eig[qi], bd.freq[qi], q_index=qi, basis_phase=basis_phase,
                                 masses=bd.masses if mass_scale else None)
            if "star_info" in res:
                info = res.pop("star_info")
                stars.setdefault(tuple(info["k"]), info)
                sg = sg or info["parent"]
            done[key] = res
        entry.update(done[key])
        qpoints.append(entry)
        if progress:
            progress(qi + 1, len(bd.qpts))
    return {
        "format": "distortkit-phonon-symmetry", "version": 1,
        "parent": {"space_group": sg, "n_atoms": len(bd.species), "species": bd.species},
        "settings": {"max_cells": max_cells, "max_dim": max_dim, "tol": tol, "opd_tol": opd_tol, "mass_scale": mass_scale, "basis_phase": basis_phase},
        "ticks": [[i, lab] for i, lab in bd.ticks],
        "stars": list(stars.values()),
        "qpoints": qpoints,
    }


def _qtext(q):
    return "(" + ", ".join(str(Fraction(float(x)).limit_denominator(48)) if abs(float(x)) > 1e-9 else "0" for x in q) + ")"


def report(result, *, subgroups=True, only=None):
    """Plain-text summary of ``analyze_band_yaml`` output: per q-point and band, the irrep and the child symmetry it leads to."""
    st = result["settings"]
    lines = [f"parent {result['parent']['space_group']}, species {', '.join(result['parent']['species'])}; "
             f"q-points needing more than {st['max_cells']} cells are skipped"]
    stars = {tuple(s["k"]): s for s in result["stars"]}
    ticks = {i: lab for i, lab in result["ticks"]}
    seen, shown = set(), set()
    for qp in result["qpoints"]:
        key = tuple(round(x, 5) for x in qp["q"])
        if key in seen or (only is not None and qp["index"] not in only):
            continue
        seen.add(key)
        head = f"\nq-point {qp['index'] + 1}" + (f" ({ticks[qp['index']]})" if qp["index"] in ticks else "") + f": {_qtext(qp['q'])}"
        if "skipped" in qp:
            lines.append(f"{head}: not analysed, {qp['skipped']}")
            continue
        star = stars[tuple(qp["star"])]
        shown.add(tuple(qp["star"]))
        lines.append(f"{head}   star of k=({', '.join(star['k'])}), {star['n_cells']}-cell supercell")
        lines.append("  band   freq (THz)  irrep        dim  child symmetry (direction of this mode)")
        for b in qp["bands"]:
            irr = next(i for i in star["irreps"] if i["index"] == b["irrep"])
            name = irr["label"] or f"irrep{irr['index']}"
            if not irr["label_certain"] and irr["label"]:
                name += "?"
            snap = b["snapshots"][0] if b["snapshots"] else None
            if b.get("rigid"):
                child = "rigid translation: the crystal moves but keeps its symmetry"
            elif snap is None:
                child = "-"
            elif snap["subgroup"]:
                s = snap["subgroup"]
                opd = s["opd"] if len(s["opd"]) <= 30 else f"any direction in a {s['fix_dim']}-parameter subspace"
                child = f"{s['space_group'] or 'unidentified, point group ' + s['point_group']}   {opd}"
            elif irr["note"]:
                child = "not enumerated (irrep too large; see --max-dim)"
            else:
                child = "no isotropy subgroup fits this direction"
            note = ""
            if b["weight"] < 0.98:
                note += f"  [only {100 * b['weight']:.0f}% in this irrep]"
            if b["adapted"]:
                note += "  [adapted]"
            elif b["degenerate_with"]:
                note += "  [degenerate]"
            lines.append(f"  {b['band'] + 1:>4}  {b['freq']:>10.3f}  {name:<11}  {b['dim']:>3}  {child}{note}")
    if subgroups:
        for kk in shown:
            star = stars[kk]
            lines.append(f"\nstar of k=({', '.join(star['k'])}): every irrep and the child symmetries it can lead to")
            for irr in star["irreps"]:
                lines.append(f"  {irr['label'] or 'irrep' + str(irr['index'])} (dim {irr['dim']}, {irr['kind']}, on {', '.join(irr['species'])})"
                             + (f"   {irr['note']}" if irr["note"] else ""))
                for s in irr["subgroups"]:
                    lines.append(f"      {s['opd']:<24} {s['space_group'] or 'unidentified (point group ' + s['point_group'] + ')'}"
                                 f"{'  x' + str(s['n_conjugates']) if s['n_conjugates'] > 1 else ''}")
    return "\n".join(lines)


def save(result, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return os.path.abspath(path)
