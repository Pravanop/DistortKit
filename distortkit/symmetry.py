"""Space-group symmetry of a periodic structure, found by brute force (numpy only).

An operation is stored as ``(R, t)`` acting on fractional coordinates (column vectors)
as ``x' = R x + t``; ``R`` is an integer matrix, ``t`` is reduced into [0, 1).
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from math import gcd

import numpy as np

from .structure import Structure


# ---------------------------------------------------------------- integer lattices
def lattice_basis(gens):
    """Triangular basis (list of int rows) of the integer lattice spanned by ``gens``."""
    rows = [[int(v) for v in g] for g in gens if any(int(v) for v in g)]
    basis = []
    for col in range(3):
        while True:
            nz = [r for r in rows if r[col] != 0]
            if len(nz) <= 1:
                break
            nz.sort(key=lambda r: abs(r[col]))
            p = nz[0]
            for r in nz[1:]:
                q = r[col] // p[col]
                for j in range(3):
                    r[j] -= q * p[j]
        nz = [r for r in rows if r[col] != 0]
        if nz:
            p = nz[0]
            rows = [r for r in rows if r is not p and any(r)]
            basis.append(p)
    return basis


def _lcm(a, b):
    return a * b // gcd(a, b)


# ---------------------------------------------------------------- symmetry search
def _candidate_columns(G, Ginv, tol_m):
    cands = []
    for i in range(3):
        bound = [int(np.floor(np.sqrt(G[i, i] * Ginv[j, j]) + 1e-6)) for j in range(3)]
        rng = [range(-b, b + 1) for b in bound]
        grid = np.array(list(itertools.product(*rng)), dtype=int)
        norms = np.einsum("ni,ij,nj->n", grid, G, grid)
        cands.append(grid[np.abs(norms - G[i, i]) < tol_m])
    return cands


def find_symmetry(s: Structure, tol: float = 1e-3):
    """All space-group operations of ``s`` as a list of ``(R, t)``.

    ``tol`` is a Cartesian distance (Angstrom) within which atoms are considered equal.
    """
    A = s.lattice
    G = A @ A.T
    Ginv = np.linalg.inv(G)
    tol_m = 2 * tol * np.sqrt(G.diagonal().max()) + tol**2 + 1e-9
    c0, c1, c2 = _candidate_columns(G, Ginv, tol_m)
    sp = np.array(s.species)
    counts = {u: int(np.sum(sp == u)) for u in set(s.species)}
    ref = min(range(len(s)), key=lambda i: (counts[s.species[i]], i))
    same_as_ref = [j for j in range(len(s)) if s.species[j] == s.species[ref]]
    same_species = sp[:, None] == sp[None, :]
    ops = []
    for a in c0:
        for b in c1:
            if abs(a @ G @ b - G[0, 1]) > tol_m:
                continue
            for c in c2:
                if abs(a @ G @ c - G[0, 2]) > tol_m or abs(b @ G @ c - G[1, 2]) > tol_m:
                    continue
                R = np.stack([a, b, c], axis=1)
                if abs(round(np.linalg.det(R))) != 1:
                    continue
                Rx = s.frac @ R.T
                found = []
                for j in same_as_ref:
                    t = s.frac[j] - Rx[ref]
                    Y = Rx + t
                    diff = Y[:, None, :] - s.frac[None, :, :]
                    diff -= np.rint(diff)
                    dist = np.linalg.norm(diff @ A, axis=2)
                    dist[~same_species] = np.inf
                    if np.all(dist.min(axis=1) < tol):
                        t = t % 1.0
                        t[np.abs(t - 1.0) < 1e-9] = 0.0
                        dt = (t - np.array([f for f in found]).reshape(-1, 3)) if found else None
                        if dt is not None:
                            dt -= np.rint(dt)
                            if np.any(np.linalg.norm(dt @ A, axis=1) < tol):
                                continue
                        found.append(t)
                ops += [(R.copy(), t) for t in found]
    return ops


def is_pure_translation(R):
    return np.array_equal(R, np.eye(3, dtype=int))


# ---------------------------------------------------------------- primitive cell
def to_primitive(s: Structure, tol: float = 1e-3):
    """Reduce ``s`` to a primitive cell.

    Returns ``(primitive_structure, B)`` where the new lattice is ``B @ s.lattice`` and a
    wavevector given in the old reciprocal basis maps to ``B @ k`` in the new one.
    """
    ops = find_symmetry(s, tol)
    ts = [t for R, t in ops if is_pure_translation(R) and np.linalg.norm(t) > 1e-9]
    if not ts:
        return s, np.eye(3)
    fr = [[Fraction(float(x)).limit_denominator(48) for x in t] for t in ts]
    D = 1
    for row in fr:
        for f in row:
            D = _lcm(D, f.denominator)
    gens = [[D * int(i == j) for j in range(3)] for i in range(3)]
    gens += [[int(f * D) for f in row] for row in fr]
    B = np.array(lattice_basis(gens), dtype=float) / D
    # shorten the basis greedily
    A = s.lattice
    improved = True
    while improved:
        improved = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for sgn in (1, -1):
                    cand = B[i] + sgn * B[j]
                    if np.linalg.norm(cand @ A) < np.linalg.norm(B[i] @ A) - 1e-9:
                        B[i] = cand
                        improved = True
    if np.linalg.det(B) < 0:
        B[2] = -B[2]
    newA = B @ A
    x = s.frac @ np.linalg.inv(B.T).T  # x' = B^{-T} x
    x %= 1.0
    keep, kept = [], []
    for i in range(len(s)):
        dup = False
        for j in keep:
            d = x[i] - x[j]
            d -= np.rint(d)
            if s.species[i] == s.species[j] and np.linalg.norm(d @ newA) < tol:
                dup = True
                break
        if not dup:
            keep.append(i)
    return Structure(newA, x[keep], [s.species[i] for i in keep]), B


# ---------------------------------------------------------------- point-group names
_PG = {
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0): "1",
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0): "-1",
    (1, 1, 0, 0, 0, 0, 0, 0, 0, 0): "2",
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0): "m",
    (1, 1, 0, 0, 0, 1, 1, 0, 0, 0): "2/m",
    (1, 3, 0, 0, 0, 0, 0, 0, 0, 0): "222",
    (1, 1, 0, 0, 0, 0, 2, 0, 0, 0): "mm2",
    (1, 3, 0, 0, 0, 1, 3, 0, 0, 0): "mmm",
    (1, 1, 0, 2, 0, 0, 0, 0, 0, 0): "4",
    (1, 1, 0, 0, 0, 0, 0, 0, 2, 0): "-4",
    (1, 1, 0, 2, 0, 1, 1, 0, 2, 0): "4/m",
    (1, 5, 0, 2, 0, 0, 0, 0, 0, 0): "422",
    (1, 1, 0, 2, 0, 0, 4, 0, 0, 0): "4mm",
    (1, 3, 0, 0, 0, 0, 2, 0, 2, 0): "-42m",
    (1, 5, 0, 2, 0, 1, 5, 0, 2, 0): "4/mmm",
    (1, 0, 2, 0, 0, 0, 0, 0, 0, 0): "3",
    (1, 0, 2, 0, 0, 1, 0, 2, 0, 0): "-3",
    (1, 3, 2, 0, 0, 0, 0, 0, 0, 0): "32",
    (1, 0, 2, 0, 0, 0, 3, 0, 0, 0): "3m",
    (1, 3, 2, 0, 0, 1, 3, 2, 0, 0): "-3m",
    (1, 1, 2, 0, 2, 0, 0, 0, 0, 0): "6",
    (1, 0, 2, 0, 0, 0, 1, 0, 0, 2): "-6",
    (1, 1, 2, 0, 2, 1, 1, 2, 0, 2): "6/m",
    (1, 7, 2, 0, 2, 0, 0, 0, 0, 0): "622",
    (1, 1, 2, 0, 2, 0, 6, 0, 0, 0): "6mm",
    (1, 3, 2, 0, 0, 0, 4, 0, 0, 2): "-6m2",
    (1, 7, 2, 0, 2, 1, 7, 2, 0, 2): "6/mmm",
    (1, 3, 8, 0, 0, 0, 0, 0, 0, 0): "23",
    (1, 3, 8, 0, 0, 1, 3, 8, 0, 0): "m-3",
    (1, 9, 8, 6, 0, 0, 0, 0, 0, 0): "432",
    (1, 3, 8, 0, 0, 0, 6, 0, 6, 0): "-43m",
    (1, 9, 8, 6, 0, 1, 9, 8, 6, 0): "m-3m",
}


def point_group_name(Rs, lattice):
    """Hermann-Mauguin point group of the rotation parts ``Rs`` (list of int 3x3)."""
    A = np.asarray(lattice, float)
    sig = [0] * 10  # E C2 C3 C4 C6 i m S6 S4 S3
    for R in {tuple(np.asarray(R, int).ravel()) for R in Rs}:
        R = np.array(R).reshape(3, 3)
        Q = A.T @ R @ np.linalg.inv(A.T)
        det = round(np.linalg.det(Q))
        tr = round(np.trace(Q) * det)  # trace of the proper part
        order = {3: 0, -1: 1, 0: 2, 1: 3, 2: 4}[tr]  # E, C2, C3, C4, C6
        if det > 0:
            sig[order] += 1
        else:
            sig[{0: 5, 1: 6, 2: 7, 3: 8, 4: 9}[order]] += 1
    return _PG.get(tuple(sig), "?")


def op_to_string(R, t, decimals=None):
    """Format an operation as 'x,y,z'-style text (translations as fractions, or decimals if ``decimals`` is set)."""
    names = "xyz"
    parts = []
    for i in range(3):
        terms = ""
        for j in range(3):
            c = int(R[i, j])
            if c:
                terms += ("+" if c > 0 else "-") + ("" if abs(c) == 1 else str(abs(c))) + names[j]
        if decimals is not None:
            v = float(t[i]) % 1.0
            if abs(v) > 10.0 ** (-decimals):
                terms += f"+{v:.{decimals}f}"
        else:
            f = Fraction(float(t[i])).limit_denominator(24)
            if f:
                terms += ("+" if f > 0 else "-") + (str(abs(f)))
        parts.append(terms.lstrip("+") or "0")
    return ",".join(parts)
