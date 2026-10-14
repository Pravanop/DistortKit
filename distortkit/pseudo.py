"""Pseudo-symmetry: the higher symmetry a distorted structure almost has.

A structure distorted from a high-symmetry parent still obeys the parent's operations *approximately*.  This
module finds them for a given tolerance and returns the largest *group* of such operations that contains the
structure's own exact symmetry, then idealises the structure with it: the result is a candidate parent.

Differences from ``symmetry.find_symmetry`` (which is exact-ish and tolerance-limited):

* an operation must pair every atom with a *different* atom of the same species, each within its own tolerance
  (so the check cannot be satisfied by two atoms collapsing onto one);
* the operations found are made into a group (closure under multiplication, every product re-validated against
  the structure), so a tolerance that is too loose yields the largest consistent group and not a heap of unrelated
  operations;
* tolerances can differ by species (``scale={"O": 3}``): in oxides the anions move several times more than the
  cations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .structure import Structure
from .symmetry import _candidate_columns, find_symmetry, to_primitive

DEFAULT_LADDER = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def parse_species_scale(text):
    """'O=3,Hf=1' -> {'O': 3.0, 'Hf': 1.0}."""
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = float(v)
    return out


def _tol_vector(s, tol, scale):
    scale = scale or {}
    return np.array([tol * float(scale.get(sp, 1.0)) for sp in s.species])


class _Matcher:
    """Distance bookkeeping of one structure, shared by all operations tested on it."""

    def __init__(self, s, tolv):
        self.s = s
        self.A = s.lattice
        self.n = len(s)
        sp = np.array(s.species)
        self.same = sp[:, None] == sp[None, :]
        self.tolv = tolv

    def refine(self, R, t, iters=4):
        """Least-squares translation for rotation ``R`` starting from ``t``; ``(t, dist per atom, near)`` or None
        when the atoms cannot be paired one-to-one."""
        f = self.s.frac
        Rx = f @ R.T
        n = self.n
        for _ in range(iters):
            diff = (Rx + t)[:, None, :] - f[None, :, :]
            diff -= np.rint(diff)
            dist = np.linalg.norm(diff @ self.A, axis=2)
            dist[~self.same] = np.inf
            near = dist.argmin(axis=1)
            t = t - np.mean(diff[np.arange(n), near], axis=0)
        diff = (Rx + t)[:, None, :] - f[None, :, :]
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff @ self.A, axis=2)
        dist[~self.same] = np.inf
        near = dist.argmin(axis=1)
        if len(set(near.tolist())) != n:
            return None
        return t, dist[np.arange(n), near], near

    def holds(self, R, t):
        r = self.refine(R, t, iters=2)
        if r is None:
            return None
        t2, d, near = r
        return (t2, d) if np.all(d <= self.tolv[near] + 1e-9) else None


def _candidate_rotations(s, tol):
    A = s.lattice
    G = A @ A.T
    Ginv = np.linalg.inv(G)
    tol_m = 2 * tol * np.sqrt(G.diagonal().max()) + tol**2 + 1e-9
    c0, c1, c2 = _candidate_columns(G, Ginv, tol_m)
    out = []
    for a in c0:
        for b in c1:
            if abs(a @ G @ b - G[0, 1]) > tol_m:
                continue
            for c in c2:
                if abs(a @ G @ c - G[0, 2]) > tol_m or abs(b @ G @ c - G[1, 2]) > tol_m:
                    continue
                R = np.stack([a, b, c], axis=1)
                if abs(round(np.linalg.det(R))) == 1:
                    out.append(R)
    return out


def pseudo_ops(s: Structure, tol: float, scale=None):
    """Every operation ``(R, t, worst)`` that maps ``s`` onto itself within ``tol`` (x species scale), with
    ``worst`` the largest atom mismatch in Angstrom."""
    tolv = _tol_vector(s, tol, scale)
    m = _Matcher(s, tolv)
    sp = np.array(s.species)
    counts = {u: int(np.sum(sp == u)) for u in set(s.species)}
    ref = min(range(len(s)), key=lambda i: (counts[s.species[i]], i))
    refs = [j for j in range(len(s)) if s.species[j] == s.species[ref]]
    out = []
    for R in _candidate_rotations(s, float(tolv.max())):
        Rx = s.frac @ R.T
        got = []
        for j in refs:
            r = m.holds(R, s.frac[j] - Rx[ref])
            if r is None:
                continue
            t, d = r
            t = t % 1.0
            if any(_cart_dist(t - g[0], m.A) < 0.5 * tol for g in got):
                continue
            got.append((t, float(np.max(d))))
        out += [(R, t, dev) for t, dev in got]
    return out


def _cart_dist(dfrac, A):
    d = dfrac - np.rint(dfrac)
    return float(np.linalg.norm(d @ A))


def _mul(a, b):
    return a[0] @ b[0], (a[0] @ b[1] + a[1]) % 1.0


class _Group:
    """Operations (R, t mod 1) kept distinct within ``merge`` Angstrom."""

    def __init__(self, A, merge):
        self.A, self.merge = A, merge
        self.items = []
        self._by_R = {}

    def find(self, op):
        for i in self._by_R.get(tuple(op[0].ravel()), []):
            if _cart_dist(op[1] - self.items[i][1], self.A) < self.merge:
                return i
        return None

    def add(self, op):
        self._by_R.setdefault(tuple(op[0].ravel()), []).append(len(self.items))
        self.items.append(op)


def _close(gens, matcher, merge, cap):
    """Group generated by ``gens`` with every element validated against the structure; ``None`` if some product fails
    (the generators are not mutually consistent at this tolerance) or the group grows past ``cap``."""
    G = _Group(matcher.A, merge)
    G.add((np.eye(3, dtype=int), np.zeros(3)))
    i = 0
    while i < len(G.items):
        e = G.items[i]
        i += 1
        for g in gens:
            p = _mul(e, g)
            if G.find(p) is not None:
                continue
            ok = matcher.holds(p[0], p[1])
            if ok is None:
                return None
            G.add((p[0], ok[0] % 1.0))
            if len(G.items) > cap:
                return None
    return G.items


def pseudo_group(s: Structure, tol: float, scale=None, base_tol=1e-3):
    """Largest group of approximate operations (within ``tol``) containing the exact symmetry of ``s``:
    a list of ``(R, t)`` in the cell of ``s``."""
    tolv = _tol_vector(s, tol, scale)
    m = _Matcher(s, tolv)
    exact = find_symmetry(s, base_tol)
    gens = [(np.asarray(R, int), np.asarray(t, float)) for R, t in exact]
    merge = max(tol * 0.6, 1e-3)
    group = _close(gens, m, merge, 4000) or list(gens)
    cands = sorted(pseudo_ops(s, tol, scale), key=lambda c: c[2])
    gset = _Group(s.lattice, merge)
    for g in group:
        gset.add(g)
    for R, t, _ in cands:
        if gset.find((R, t)) is not None:
            continue
        trial = _close(gens + [(R, t)], m, merge, 4000)
        if trial is None:
            continue
        gens = gens + [(R, t)]
        group = trial
        gset = _Group(s.lattice, merge)
        for g in group:
            gset.add(g)
    return group


@dataclass
class Rung:
    tol: float
    n_ops: int                    # operations in the cell of the input structure (including centring translations)
    ideal: Structure = None       # idealised primitive structure with the pseudo-symmetry made exact
    sg: object = None             # spacegroup.SpaceGroupResult of ``ideal``
    new: bool = False             # symmetry higher than at the previous rung


def idealized_parent(s: Structure, ops, tol):
    """The structure made exactly symmetric under ``ops`` (positions and metric averaged), reduced to its primitive cell."""
    from .symmetrize import idealize
    ideal, _ = idealize(s, tol, ops=ops)
    prim, _ = to_primitive(ideal, 1e-4)
    return prim


def ladder(s: Structure, tols=DEFAULT_LADDER, scale=None, base_tol=1e-3):
    """Pseudo-symmetry of ``s`` at increasing tolerance.  One ``Rung`` per tolerance; ``new`` marks the rungs where the
    group first grows, and those carry the idealised primitive structure and its space group."""
    from .spacegroup import identify
    out = []
    last = len(find_symmetry(s, base_tol))
    for tol in tols:
        ops = pseudo_group(s, tol, scale, base_tol)
        rung = Rung(tol, len(ops))
        if len(ops) > last:
            try:
                prim = idealized_parent(s, ops, tol)
                rung.ideal = prim
                rung.sg = identify(find_symmetry(prim, 1e-4), prim.lattice)
                rung.new = True
                last = len(ops)
            except (ValueError, np.linalg.LinAlgError):
                pass
        out.append(rung)
    return out
