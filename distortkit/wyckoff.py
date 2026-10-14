"""Wyckoff sites of the atoms of a structure, from the open Wyckoff table (``tables/wyckoff.json``).

The site of an atom orbit is found in the parent's standard setting (``spacegroup.identify`` gives the
basis change and origin shift): among the tabulated Wyckoff positions of the space group with the right
multiplicity, the one whose coordinate template contains the atom (or a symmetry-equivalent image of it).
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import numpy as np

from . import isodata
from .symmetry import point_group_name


@dataclass
class WyckoffPosition:
    letter: str
    const: tuple              # constant part (Fractions)
    rows: tuple               # 3 rows (x, y, z coefficient vectors), Fractions
    multiplicity: int         # in the conventional cell

    def template(self):
        parts = []
        for k in range(3):
            terms = []
            if self.const[k]:
                terms.append(str(self.const[k]))
            for name, r in zip("xyz", self.rows):
                if r[k]:
                    terms.append(("" if r[k] == 1 else "-" if r[k] == -1 else str(r[k])) + name)
            parts.append("+".join(terms).replace("+-", "-") or "0")
        return "(" + ",".join(parts) + ")"


@dataclass
class SiteInfo:
    index: int
    species: str
    atoms: list                # indices into the primitive cell
    label: str                 # e.g. "4d"
    multiplicity: int          # conventional-cell multiplicity
    template: str              # e.g. "(x,y,1/4)"
    site_symmetry: str         # point group of the site, e.g. "m"
    std_position: tuple = ()   # position of the first atom in the standard setting


@lru_cache(maxsize=None)
def _wyckoff_doc():
    with open(os.path.join(isodata.BUNDLED, "wyckoff.json")) as fh:
        return json.load(fh)["groups"]


@lru_cache(maxsize=None)
def wyckoff_positions(g):
    """Wyckoff positions of space group ``g`` in the conventional cell of its default setting (International Tables
    letters, increasing multiplicity)."""
    out = []
    for item in _wyckoff_doc()[str(g)]:
        const = tuple(Fraction(v) for v in item["const"])
        rows = tuple(tuple(Fraction(v) for v in r) for r in item["rows"])
        out.append(WyckoffPosition(item["letter"], const, rows, int(item["multiplicity"])))
    return out


def _on_template(x, pos, ops, cent, tol=1e-4):
    """Is the point ``x`` (or any symmetry image of it) on the template of ``pos``?"""
    const = np.array([float(c) for c in pos.const])
    M = np.array([[float(pos.rows[i][k]) for i in range(3)] for k in range(3)])       # columns = x, y, z directions
    shifts = [np.zeros(3)] + [np.array([float(v) for v in c]) for c in cent]
    seen = []
    for R, t in ops:
        y = R @ x + np.array([float(v) for v in t])
        if any(np.abs((y - z) - np.rint(y - z)).max() < 1e-9 for z in seen):
            continue
        seen.append(y)
        for n in itertools.product((-1, 0, 1), repeat=3):
            for s in shifts:
                rhs = y - const - np.array(n) - s
                sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                if np.abs(M @ sol - rhs).max() < tol:
                    return True
    return False


def atom_orbits(structure, ops, tol=1e-3):
    """Atom orbits of a structure under ``ops`` (list of (R, t)): list of lists of atom indices."""
    n = len(structure)
    sp = np.array(structure.species)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for R, t in ops:
        Y = structure.frac @ R.T + t
        diff = Y[:, None, :] - structure.frac[None, :, :]
        diff -= np.rint(diff)
        d = np.linalg.norm(diff @ structure.lattice, axis=2)
        d[sp[:, None] != sp[None, :]] = np.inf
        for a, b in enumerate(d.argmin(axis=1)):
            ra, rb = find(a), find(int(b))
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: g[0])


def describe_sites(structure, ops, sg, tol=1e-3):
    """Wyckoff site of every atom orbit of the (primitive) ``structure`` with operations ``ops``.

    ``sg`` is the ``SpaceGroupResult`` of the structure (basis change ``P`` and origin ``p``).
    """
    G = isodata.space_groups()[sg.number]
    P = np.asarray(sg.P, float)
    Pinv = np.linalg.inv(P)
    det = int(round(abs(np.linalg.det(P))))
    poss = wyckoff_positions(sg.number)
    out = []
    for oi, atoms in enumerate(atom_orbits(structure, ops, tol)):
        x = Pinv @ structure.frac[atoms[0]] + np.asarray(sg.p, float)         # standard-setting coordinates
        mult = len(atoms) * det
        label, templ, spg = "?", "", "?"
        for pos in poss:
            if pos.multiplicity == mult and _on_template(x, pos, G.ops, G.centering):
                label, templ = f"{mult}{pos.letter}", pos.template()
                break
        # site symmetry: stabiliser of the point among the conventional-cell operations
        rots = []
        for R, t in G.ops:
            y = R @ x + np.array([float(v) for v in t]) - x
            if np.abs(y - np.rint(y)).max() < 1e-4:
                rots.append(R)
        spg = point_group_name(rots, np.eye(3)) if rots else "?"
        out.append(SiteInfo(oi, structure.species[atoms[0]], atoms, label, mult, templ, spg, tuple(np.round(x, 6))))
    return out
