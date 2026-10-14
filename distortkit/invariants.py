"""Polynomial invariants of the order parameters: the allowed terms of the Landau free energy and the couplings.

For a saved analysis (``compare`` output) and a choice of its modes this module computes, for the parent's space
group G, the polynomials in the order-parameter components that G leaves unchanged.  Each mode's order-parameter
space is the real irrep space it lives in, found as the orbit of the mode's displacement field under G.  The fields
are extended to a supercell that contains every symmetry image of the child's cell (the intersection of the
parent-rotated child lattices), so the orbit contains all arms of the star and the translations act as phases:
momentum conservation comes out automatically.

Invariants split by *multidegree* (how many components of each mode a term has), and that split does not depend on
the choice of basis, so the results are reported that way:

* ``terms``: for each multidegree the number of independent invariants, and how many are *new* (not products of
  lower ones): e.g. ``X5+ X2- GM4-`` (a trilinear coupling), ``R4+^4``.  These are the allowed Landau terms.
* ``induced``: for each mode s and each multidegree linear in s, the force the coupling exerts on s when the other
  modes sit at their measured amplitudes (the gradient of the invariant with respect to s at s = 0), and how much of
  the measured direction of s lies in the span of those forces.  A high match with a non-zero force means the
  measured mode is what the coupling would induce; no force means the symmetry-allowed coupling vanishes there.

Coefficients of the free energy are free, so nothing here says how *large* a term is, only that it exists.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .compare import CellSet
from .explain import FieldGroup
from .symmetry import find_symmetry, lattice_basis

MAX_BIG_CELLS = 256


# ====================================================================== the supercell that holds every image
def _intersect(B1, B2):
    """Intersection of the integer lattices with basis columns ``B1`` and ``B2`` (through the duals)."""
    D = abs(round(np.linalg.det(B1))) * abs(round(np.linalg.det(B2)))
    gens = [np.rint(D * np.linalg.inv(B).T[:, i]).astype(int) for B in (B1, B2) for i in range(3)]
    C = np.array(lattice_basis(gens), dtype=float)
    out = np.linalg.inv(C.T / D).T
    r = np.rint(out)
    if np.abs(out - r).max() > 1e-6:
        raise ValueError("lattice intersection failed")
    return r.astype(int)


def invariant_supercell(M, ops):
    """Smallest supercell (rows) contained in the child's lattice and mapped onto itself by every operation."""
    B = np.asarray(M, int).T
    cur = B.copy()
    for R in {tuple(np.asarray(R, int).ravel()): np.asarray(R, int) for R, _ in ops}.values():
        cur = _intersect(cur, R @ B)
    big = cur.T
    if np.linalg.det(big) < 0:
        big[2] *= -1
    return big


def extend_field(F, cs_child, cells_big):
    """Periodic extension of ``F[atom, cell, 3]`` (child cells) onto the cells of the larger supercell."""
    idx = cs_child.canon(cells_big)
    return F[:, idx, :]


# ====================================================================== order-parameter spaces
@dataclass
class OrderParameter:
    id: str
    name: str
    amplitude: float
    dim: int
    basis: np.ndarray           # (dof, dim) orthonormal; column 0 is the measured direction
    rho: list                   # per generator: (dim, dim) orthogonal matrix


def generating_set(G: FieldGroup):
    gens, group = [], {G.identity}
    for e in range(G.n_el):
        if e in group:
            continue
        gens.append(e)
        group = G.generated(gens, lambda _: True)
    return gens


def order_parameters(fz, refs, G, F_big, gens):
    out = []
    for ref in refs:
        m = fz.mode_by_ref(ref) if isinstance(ref, str) else ref
        F = F_big[m.id]
        orbit = np.array([G.act(F, e).ravel() for e in range(G.n_el)])
        u, s, vt = np.linalg.svd(orbit, full_matrices=False)
        r = int((s > 1e-8 * s[0]).sum())
        Bm = vt[:r].T
        f = F.ravel() / np.linalg.norm(F)
        fc = Bm.T @ f
        if abs(np.linalg.norm(fc) - 1) > 1e-6:
            raise ValueError(f"mode {m.id} is not in its own orbit span")
        Q, _ = np.linalg.qr(np.column_stack([fc, np.eye(r)]))
        if Q[:, 0] @ fc < 0:
            Q[:, 0] *= -1
        B = Bm @ Q[:, :r]
        rho = []
        for g in gens:
            img = np.array([G.act(B[:, j].reshape(F.shape), g).ravel() for j in range(r)]).T
            R = B.T @ img
            if np.abs(img - B @ R).max() > 1e-6:
                raise ValueError(f"order-parameter space of {m.id} is not invariant (dimension {r})")
            rho.append(R)
        out.append(OrderParameter(m.id, m.text, float(np.linalg.norm(F) * np.sqrt(fz.N / G.N)), r, B, rho))
    return out


# ====================================================================== symmetric powers
class Powers:
    """Monomial bases of one mode's components and the matrices of ``p(x) -> p(rho x)`` on them."""

    def __init__(self, n, dmax):
        self.n = n
        self.monos = [list(itertools.combinations_with_replacement(range(n), e)) for e in range(dmax + 1)]
        self.index = [{m: i for i, m in enumerate(ms)} for ms in self.monos]

    def size(self, e):
        return len(self.monos[e])

    def sym(self, rho, e):
        """S with S[beta, alpha] = coefficient of monomial beta in monomial alpha evaluated at rho x."""
        n = self.n
        prev = np.ones((1, 1))
        for d in range(1, e + 1):
            cur = np.zeros((self.size(d), self.size(d)))
            for a, mono in enumerate(self.monos[d]):
                v = mono[-1]
                base = self.index[d - 1][mono[:-1]]
                vec = prev[:, base]                                 # image of the shorter monomial
                for k in range(n):
                    c = rho[v, k]
                    if abs(c) < 1e-14:
                        continue
                    for b, mb in enumerate(self.monos[d - 1]):
                        if vec[b] != 0:
                            cur[self.index[d][tuple(sorted(mb + (k,)))], a] += vec[b] * c
            prev = cur
        return prev


# ====================================================================== results
@dataclass
class Term:
    degrees: tuple              # exponent per mode
    n_invariants: int
    n_new: int                  # not products of invariants of lower multidegree


@dataclass
class Induced:
    target: str                 # the mode being pushed
    degrees: tuple
    n_invariants: int
    force: float                # size of the strongest force (arbitrary units; coefficients are free)
    match: float                # share of the measured direction inside the span of the forces (0..1)


@dataclass
class Invariants:
    modes: list
    degree: int
    terms: list
    induced: list
    group_elements: int
    big_cells: int
    notes: list = field(default_factory=list)
    _vectors: dict = field(default_factory=dict, repr=False)

    def term_name(self, deg):
        return " ".join(f"{m.name}" + (f"^{e}" if e > 1 else "") for m, e in zip(self.modes, deg) if e)

    def report(self, show_all=False):
        L = [f"order parameters ({self.group_elements} operations on a {self.big_cells}-cell supercell): "
             + ", ".join(f"{m.id} {m.name} (dim {m.dim}, {m.amplitude:.4f} A)" for m in self.modes), "",
             f"1. ALLOWED TERMS up to degree {self.degree}  (independent invariants; 'new' = not a product of lower-degree ones)"]
        shown = [t for t in sorted(self.terms, key=lambda t: (sum(t.degrees), t.degrees)) if show_all or t.n_new]
        if not shown:
            L.append("   none")
        for t in shown:
            L.append(f"   degree {sum(t.degrees)}: {self.term_name(t.degrees):<28s}{t.n_invariants:>3d} invariant(s), {t.n_new} new")
        L.append("")
        L.append("2. COUPLINGS LINEAR IN A MODE: what the other measured modes would induce in it")
        by = {}
        for i in self.induced:
            by.setdefault(i.target, []).append(i)
        for k, m in enumerate(self.modes):
            lst = sorted(by.get(m.name, []), key=lambda i: (sum(i.degrees), i.degrees))
            active = [i for i in lst if i.force >= 1e-6]
            dead = len(lst) - len(active)
            if not lst:
                L.append(f"   {m.name}: no allowed term is linear in it (to degree {self.degree}): nothing else can induce it")
                continue
            if not active:
                L.append(f"   {m.name}: {len(lst)} allowed coupling(s), but all vanish at the measured amplitudes")
                continue
            low = min(sum(i.degrees) for i in active) if not show_all else None
            for i in active:
                if not show_all and sum(i.degrees) != low:
                    continue
                others = self.term_name(tuple(0 if j == k else e for j, e in enumerate(i.degrees)))
                if i.match >= 0.95:
                    verdict = f"induces exactly the measured direction ({100 * i.match:.0f}% match)"
                else:
                    verdict = f"pushes it only {100 * i.match:.0f}% along the measured direction (not fully induced: its own terms set its direction)"
                L.append(f"   {m.name} <- {others} (degree {sum(i.degrees)}, {i.n_invariants} invariant(s)): {verdict}")
            more = len(active) - sum(1 for i in active if show_all or sum(i.degrees) == low)
            extra = []
            if more:
                extra.append(f"{more} higher-order active")
            if dead:
                extra.append(f"{dead} allowed but vanishing")
            if extra:
                L.append(f"      (+ {', '.join(extra)}; --all lists them)")
        L += [""] + [f"note: {n}" for n in self.notes]
        L.append("(coefficients of the free energy are free: these say which terms exist, not how large they are; a trilinear "
                 "term is symmetric in its modes, so it does not say which of them drives: see `explain` for that)")
        return "\n".join(L)

    def to_dict(self):
        return {
            "modes": [{"id": m.id, "name": m.name, "dim": m.dim, "amplitude": m.amplitude} for m in self.modes],
            "degree": self.degree, "group_elements": self.group_elements, "supercell_cells": self.big_cells,
            "terms": [{"degrees": dict(zip([m.name for m in self.modes], t.degrees)), "invariants": t.n_invariants, "new": t.n_new}
                      for t in self.terms],
            "induced": [{"target": i.target, "degrees": dict(zip([m.name for m in self.modes], i.degrees)),
                         "invariants": i.n_invariants, "force": i.force, "match": i.match} for i in self.induced],
            "notes": self.notes,
        }


# ====================================================================== computation
def _compositions(total, n):
    if n == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, n - 1):
            yield (first,) + rest


def _kron(mats):
    out = np.ones((1, 1))
    for m in mats:
        out = np.kron(out, m)
    return out


def _nullspace(mats, tol=1e-8):
    n = mats[0].shape[0]
    A = np.zeros((n, n))
    for K in mats:
        D = K - np.eye(n)
        A += D.T @ D
    w, v = np.linalg.eigh(A)
    return v[:, w < tol * max(1.0, w.max())].T            # rows


def _product_maps(P, e1, e2):
    """Index maps of the monomial products of two bases of the same mode: ``[mode] -> (n1, n2) target index``."""
    t = P.index[e1 + e2]
    return np.array([[t[tuple(sorted(a + b))] for b in P.monos[e2]] for a in P.monos[e1]], dtype=int)


def compute(fz, refs=None, degree=3, min_amp=0.02, max_cells=MAX_BIG_CELLS, max_block=2500):
    """Landau invariants and induced couplings of the given modes of a loaded analysis (``freeze.Freezer``)."""
    prim = fz.parent
    ops = find_symmetry(prim, 1e-3)
    if refs is None:
        mods = [m for m in sorted(fz.modes, key=lambda m: -m.amplitude) if m.amplitude >= min_amp]
    else:
        mods = [fz.mode_by_ref(r) for r in refs]
    if not mods:
        raise ValueError("no modes selected (all are below --min-amp?)")
    Mbig = invariant_supercell(fz.M, ops)
    nbig = abs(round(np.linalg.det(Mbig)))
    if nbig > max_cells:
        raise ValueError(f"the supercell holding every symmetry image of the child has {nbig} cells (limit {max_cells})")
    G = FieldGroup(prim, Mbig, CellSet(Mbig).cells)
    cs_child = CellSet(fz.M)
    F_big = {m.id: extend_field(m.field, cs_child, G.cells) for m in mods}
    gens = generating_set(G)
    ops_ = order_parameters(fz, mods, G, F_big, gens)
    notes = []
    for op in ops_:
        if op.dim > 12:
            notes.append(f"{op.name} has a {op.dim}-dimensional order-parameter space; invariants of high degree may be slow")
    P = [Powers(op.dim, degree) for op in ops_]
    S = [[[P[i].sym(op.rho[g], e) for e in range(degree + 1)] for g in range(len(gens))] for i, op in enumerate(ops_)]
    nm = len(ops_)
    patterns = [d for tot in range(1, degree + 1) for d in _compositions(tot, nm)]
    inv = {}
    for d in sorted(patterns, key=lambda d: (sum(d), d)):
        size = int(np.prod([P[i].size(e) for i, e in enumerate(d)]))
        if size > max_block:
            notes.append(f"multidegree {d} skipped ({size} monomials)")
            continue
        mats = [_kron([S[i][g][e] for i, e in enumerate(d)]) for g in range(len(gens))]
        null = _nullspace(mats)
        if len(null):
            inv[d] = null
    terms = []
    for d, null in inv.items():
        prods = []
        for e1 in inv:
            e2 = tuple(a - b for a, b in zip(d, e1))
            if min(e2) < 0 or not any(e2) or e2 not in inv:
                continue
            maps = [_product_maps(P[i], e1[i], e2[i]) for i in range(nm)]
            shapes = tuple(P[i].size(d[i]) for i in range(nm))
            for u in inv[e1]:
                ut = u.reshape([P[i].size(e1[i]) for i in range(nm)])
                for v in inv[e2]:
                    vt = v.reshape([P[i].size(e2[i]) for i in range(nm)])
                    outer = np.multiply.outer(ut, vt)                       # axes: a1..am, b1..bm
                    mult = 1
                    flat = np.zeros(outer.shape, dtype=np.int64)
                    for i in reversed(range(nm)):
                        shp = [1] * (2 * nm)
                        shp[i] = maps[i].shape[0]
                        shp[nm + i] = maps[i].shape[1]
                        flat = flat + mult * maps[i].reshape(shp)
                        mult *= shapes[i]
                    res = np.zeros(int(np.prod(shapes)))
                    np.add.at(res, flat.ravel(), outer.ravel())
                    prods.append(res)
        rank = 0
        if prods:
            sv = np.linalg.svd(np.array(prods), compute_uv=False)
            rank = int((sv > 1e-8 * max(sv[0], 1e-30)).sum()) if sv.size else 0
        terms.append(Term(d, len(null), max(len(null) - rank, 0)))
    # forces on a mode that appears linearly: gradient at the measured point
    induced = []
    amp = [op.amplitude for op in ops_]
    for d, null in inv.items():
        for s in range(nm):
            if d[s] != 1 or sum(d) < 2:
                continue
            forces = []
            for vec in null:
                t = vec.reshape([P[i].size(d[i]) for i in range(nm)])
                idx = []
                for i in range(nm):
                    if i == s:
                        idx.append(slice(None))
                    else:
                        idx.append(P[i].index[d[i]][(0,) * d[i]])           # monomial x_0^e: the measured point has only x_0
                g = t[tuple(idx)].copy()
                # ordering of the single variables of mode s is that of Powers.monos[1]: (j,) -> j
                for i in range(nm):
                    if i != s:
                        g = g * amp[i] ** d[i]
                forces.append(g)
            F = np.array(forces)
            scale = float(np.prod([amp[i] ** d[i] for i in range(nm) if i != s]))
            fmax = float(np.linalg.norm(F, axis=1).max()) / max(scale, 1e-300)   # in units of the other amplitudes' powers
            if fmax < 1e-6:
                match = 0.0
            else:
                u, sv, vt = np.linalg.svd(F, full_matrices=False)
                r = int((sv > 1e-8 * sv[0]).sum())
                e1 = np.zeros(ops_[s].dim)
                e1[0] = 1.0
                match = float(np.linalg.norm(vt[:r] @ e1))
            induced.append(Induced(ops_[s].name, d, len(null), fmax, match))
    return Invariants(ops_, degree, terms, induced, G.n_el, G.N, notes)
