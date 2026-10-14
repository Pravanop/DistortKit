#!/usr/bin/env python3
"""Build the open crystallographic tables shipped with DistortKit.

Writes ``distortkit/tables/space_groups.json`` and ``distortkit/tables/wyckoff.json``.

* space groups: the 230 groups in the International Tables default setting (origin choice 2 for the 24 groups that
  have two origins, hexagonal axes for rhombohedral groups, unique axis b for monoclinic), taken from spglib's
  database (spglib is BSD-3 licensed; this script needs it, the package does not);
* Wyckoff positions: computed here from the space-group operations alone.  For every point of a fine grid the
  site-symmetry group is found; the fixed subspace of that group through the point is a special position (a
  point, line or plane); subspaces equivalent under the group are one Wyckoff position.  The letters are the
  International Tables letters, read from spglib on a structure with an atom at a generic point of each position.

Run:   python tools/build_tables.py [--out DIR] [--only N,M,...]      (python with numpy and spglib)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from fractions import Fraction

import numpy as np

try:                                  # only needed to build the tables, not to read or verify them
    import spglib
except ImportError:                   # pragma: no cover
    spglib = None

GRID = 24


def frac_str(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ====================================================================== space groups
def hall_numbers():
    """ITA number -> Hall number of the default setting used here."""
    by = {}
    for h in range(1, 531):
        t = spglib.get_spacegroup_type(h)
        by.setdefault(t.number, []).append((h, t))
    out = {}
    for n, lst in by.items():
        two = [h for h, t in lst if t.choice == "2"]
        out[n] = two[0] if two else lst[0][0]
    return out


def space_group_records():
    recs = {}
    for n, h in sorted(hall_numbers().items()):
        t = spglib.get_spacegroup_type(h)
        sym = spglib.get_symmetry_from_database(h)
        ops = []
        for R, tr in zip(sym["rotations"], sym["translations"]):
            tt = [Fraction(float(v)).limit_denominator(24) % 1 for v in tr]
            ops.append((tuple(int(x) for x in np.asarray(R).ravel()), tuple(tt)))
        ops.sort()
        recs[n] = {"number": n, "hall": h, "symbol": t.international_short, "full": t.international_full,
                   "schoenflies": t.schoenflies, "point_group": t.pointgroup_international,
                   "ops": [[list(R), [frac_str(v) for v in tt]] for R, tt in ops]}
    return recs


# ====================================================================== exact linear algebra helpers
def rref(rows):
    """Reduced row echelon form of a list of Fraction vectors; returns the non-zero rows and the pivot columns."""
    M = [list(r) for r in rows]
    piv, r = [], 0
    ncol = len(M[0]) if M else 0
    for c in range(ncol):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        M[r] = [v / M[r][c] for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return [tuple(x) for x in M[:r]], piv


def nullspace(rows, n=3):
    """Basis (RREF rows) of {v : rows . v = 0}."""
    if not rows:
        return [tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n)]
    R, piv = rref(rows)
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for r, c in zip(R, piv):
            v[c] = -r[f]
        basis.append(tuple(v))
    return rref(basis)[0] if basis else []


# ====================================================================== special positions
class Subspace:
    """Affine subspace p + span(dirs) of the unit torus.  ``dirs`` are in RREF (pivot entry 1)."""

    def __init__(self, p, dirs):
        self.dirs = tuple(dirs)
        self.piv = [next(i for i, v in enumerate(d) if v != 0) for d in self.dirs]
        p = [Fraction(v) for v in p]
        for d, c in zip(self.dirs, self.piv):                  # zero the pivot coordinates
            f = p[c]
            p = [a - f * b for a, b in zip(p, d)]
        self.p = tuple(v % 1 if i not in self.piv else Fraction(0) for i, v in enumerate(p))

    @property
    def dim(self):
        return len(self.dirs)

    def key(self):
        """Canonical key modulo integer translations."""
        npc = [i for i in range(3) if i not in self.piv]
        if not self.dirs:
            return (self.dirs, tuple(v % 1 for v in self.p))
        shifts = [[d[j] for j in npc] for d in self.dirs]
        q = 1
        for s in shifts:
            for v in s:
                q = q * v.denominator // np.gcd(q, v.denominator)
        best = None
        for m in itertools.product(range(int(q)), repeat=len(self.dirs)):
            r = tuple((self.p[j] - sum(mk * s[a] for mk, s in zip(m, shifts))) % 1 for a, j in enumerate(npc))
            if best is None or r < best:
                best = r
        return (self.dirs, best)

    def image(self, R, t):
        R = np.array(R).reshape(3, 3)
        p = [sum(int(R[i, j]) * self.p[j] for j in range(3)) + t[i] for i in range(3)]
        dirs = [tuple(sum(int(R[i, j]) * d[j] for j in range(3)) for i in range(3)) for d in self.dirs]
        return Subspace(p, rref(dirs)[0] if dirs else [])

    def point(self, rng):
        s = rng.uniform(0.11, 0.89, self.dim)
        y = np.array([float(v) for v in self.p]) + sum(si * np.array([float(v) for v in d]) for si, d in zip(s, self.dirs)) if self.dim else \
            np.array([float(v) for v in self.p])
        return y

    def record(self):
        rows = [[Fraction(0)] * 3 for _ in range(3)]
        for d, c in zip(self.dirs, self.piv):
            rows[c] = list(d)
        return {"const": [frac_str(v) for v in self.p], "rows": [[frac_str(v) for v in r] for r in rows]}


def site_symmetry_masks(ops):
    """For every grid point (a,b,c)/GRID: which operations fix it?  Returns (points int array, bool mask (n_ops, n_points))."""
    g = np.array(list(itertools.product(range(GRID), repeat=3)), dtype=np.int64)
    masks = []
    for R, t in ops:
        R = np.array(R).reshape(3, 3)
        t24 = np.array([int(v * GRID) for v in t])
        assert all(v * GRID == int(v * GRID) for v in t), "translation not on the grid"
        d = g @ R.T + t24 - g
        masks.append(np.all(d % GRID == 0, axis=1))
    return g, np.array(masks)


def special_subspaces(ops, rng):
    pts, masks = site_symmetry_masks(ops)
    Rm = [np.array(R).reshape(3, 3) for R, _ in ops]
    uniq, inverse = np.unique(masks.T, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    found = {}
    for ui, u in enumerate(uniq):
        H = np.flatnonzero(u)
        rows = []
        for h in H:
            A = Rm[h] - np.eye(3, dtype=int)
            rows += [tuple(Fraction(int(v)) for v in r) for r in A if any(r)]
        dirs = nullspace(rows)
        if len(dirs) == 3:                                        # the identity only: the general position, added separately
            continue
        # every grid point with this site symmetry: the fixed subspace has several components (e.g. the 8 inversion centres)
        for i0 in np.flatnonzero(inverse == ui):
            S = Subspace([Fraction(int(v), GRID) for v in pts[i0]], dirs)
            k = S.key()
            if k in found:
                continue
            y = S.point(rng)
            hy = []
            for kk, (R, t) in enumerate(ops):
                d = np.array(R).reshape(3, 3) @ y + np.array([float(v) for v in t]) - y
                if np.abs(d - np.rint(d)).max() < 1e-9:
                    hy.append(kk)
            if set(hy) == set(H.tolist()):                        # a generic point of the subspace keeps the same site symmetry
                found[k] = S
    # the general position
    found[("general",)] = Subspace(
        [Fraction(0)] * 3, [tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)])
    return found


def wyckoff_classes(ops, rng):
    subs = special_subspaces(ops, rng)
    # close under the group and form orbits
    parent = {}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for k in subs:
        parent[k] = k
    queue = list(subs)
    while queue:
        k = queue.pop()
        S = subs[k]
        for R, t in ops:
            I = S.image(R, t)
            ki = I.key()
            if ki not in subs:
                subs[ki] = I
                parent[ki] = ki
                queue.append(ki)
            a, b = find(k), find(ki)
            if a != b:
                parent[a] = b
    classes = {}
    for k in subs:
        classes.setdefault(find(k), []).append(subs[k])
    return list(classes.values())


# ====================================================================== letters from spglib
def lattice_for(n):
    if n <= 2:
        return np.array([[5.0, 0, 0], [0.7, 6.0, 0], [0.9, 0.5, 7.0]])
    if n <= 15:
        b = np.radians(100.0)
        return np.array([[5.0, 0, 0], [0, 6.0, 0], [7.0 * np.cos(b), 0, 7.0 * np.sin(b)]])
    if n <= 74:
        return np.diag([5.0, 6.0, 7.0])
    if n <= 142:
        return np.diag([5.0, 5.0, 7.0])
    if n <= 194:
        g = np.radians(120.0)
        return np.array([[5.0, 0, 0], [5.0 * np.cos(g), 5.0 * np.sin(g), 0], [0, 0, 7.0]])
    return np.diag([5.0, 5.0, 5.0])


def _orbit(ops, y):
    imgs = []
    for R, t in ops:
        z = (np.array(R).reshape(3, 3) @ y + np.array([float(v) for v in t])) % 1.0
        if not any(np.abs((z - w) - np.rint(z - w)).max() < 1e-7 for w in imgs):
            imgs.append(z)
    return imgs


def letter_of(n, ops, S, rng):
    """Wyckoff letter of the position ``S``: put an atom (species 1) on a generic point of it, add a second species on a
    generic orbit so that the structure has exactly the symmetry of the group, and let spglib classify the site."""
    L = lattice_for(n)
    for _ in range(10):
        first = _orbit(ops, S.point(rng))
        extra = _orbit(ops, rng.uniform(0.05, 0.95, 3))
        if len(ops) == 1 or len(extra) == 1:                       # P1: two generic atoms, or inversion about the atom appears
            extra = extra + _orbit(ops, rng.uniform(0.05, 0.95, 3))
        pos = np.array(first + extra)
        nums = [1] * len(first) + [2] * len(extra)
        ds = spglib.get_symmetry_dataset((L, pos, nums), symprec=1e-4)
        if ds is None:
            continue
        num = ds.number if hasattr(ds, "number") else ds["number"]
        wy = ds.wyckoffs if hasattr(ds, "wyckoffs") else ds["wyckoffs"]
        if num == n and len(set(wy[:len(first)])) == 1:
            return wy[0], len(first)
    return None, None


def build_wyckoff(recs, only=None, log=print):
    out = {}
    for n in sorted(recs):
        if only and n not in only:
            continue
        ops = [(tuple(R), tuple(Fraction(v) for v in t)) for R, t in recs[n]["ops"]]
        ops = [(R, tuple(Fraction(v) for v in t)) for R, t in ops]
        rng = np.random.default_rng(n)
        classes = wyckoff_classes(ops, rng)
        items = []
        for cl in classes:
            rep = min(cl, key=lambda S: (sum(1 for d in S.dirs for v in d if v < 0), S.p, S.dirs))      # fewest minus signs, then smallest constants
            letter, mult = letter_of(n, ops, rep, rng)
            if letter is None:
                raise RuntimeError(f"group {n}: spglib could not assign a letter")
            r = rep.record()
            r["letter"], r["multiplicity"] = letter, mult
            items.append(r)
        letters = [i["letter"] for i in items]
        if len(set(letters)) != len(letters):
            raise RuntimeError(f"group {n}: duplicate letters {sorted(letters)}")
        items.sort(key=lambda i: (i["letter"].isupper(), i["letter"]))
        out[str(n)] = items
        log(f"group {n:3d} {recs[n]['symbol']:<10s} {len(items):2d} Wyckoff positions")
    return out


def main():
    if spglib is None:
        sys.exit("building the tables needs spglib (pip install spglib)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distortkit", "tables"))
    ap.add_argument("--only", help="comma-separated group numbers (for testing)")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",")} if a.only else None
    os.makedirs(a.out, exist_ok=True)
    recs = space_group_records()
    wy = build_wyckoff(recs, only)
    meta = {"generator": "tools/build_tables.py", "spglib": spglib.__version__}
    if not only:
        with open(os.path.join(a.out, "space_groups.json"), "w") as fh:
            json.dump({"meta": meta, "groups": {str(n): r for n, r in recs.items()}}, fh, separators=(",", ":"))
        with open(os.path.join(a.out, "wyckoff.json"), "w") as fh:
            json.dump({"meta": meta, "groups": wy}, fh, separators=(",", ":"))
        print("wrote", os.path.abspath(a.out))
    else:
        json.dump(wy, open(os.path.join(a.out, "wyckoff_partial.json"), "w"))


if __name__ == "__main__":
    main()
