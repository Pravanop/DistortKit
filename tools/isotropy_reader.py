"""Reader for the ISOTROPY data tables (data_space.txt, data_wyckoff.txt), kept ONLY to verify the open tables
against them (tools/verify_tables.py).  Not part of the package.

Original docstring: reader for the ISOTROPY data tables (data_*.txt, const.dat) distributed with the ISOTROPY suite.

The tables are plain text: a line holding an array name, followed by whitespace-separated integers or
double-quoted strings.  Only data_space.txt, data_isotropy.txt and data_irreps.txt are needed; copies are
bundled in ``isodistort_py/tables``.  To use another copy set the ``ISODATA`` environment variable,
call ``set_data_dir()`` or use the ``--isodata`` CLI option.
"""
from __future__ import annotations

import os
import re
from fractions import Fraction
from functools import lru_cache

import numpy as np

_DIR = None


def set_data_dir(path):
    global _DIR
    _DIR = os.path.abspath(path)
    get_tables.cache_clear()
    space_groups.cache_clear()


BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


def data_dir():
    """Directory with the tables: set_data_dir(), $ISODATA, then the copy bundled in the package."""
    for d in (_DIR, os.environ.get("ISODATA"), BUNDLED):
        if d and os.path.isfile(os.path.join(d, "data_space.txt")):
            return d
    raise FileNotFoundError(
        "ISOTROPY data tables not found. Set the ISODATA environment variable (or use --isodata) to the "
        "directory containing data_space.txt, data_isotropy.txt and data_irreps.txt")


@lru_cache(maxsize=None)
def get_tables(name):
    """Parse ``data_<name>.txt`` into {array name: raw text}."""
    path = os.path.join(data_dir(), f"data_{name}.txt")
    out, cur = {}, None
    with open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*", s):
                cur = s.strip()
                out[cur] = []
            elif cur:
                out[cur].append(s)
    return {k: " ".join(v) for k, v in out.items()}


def ints(tables, name):
    return np.array(tables[name].split(), dtype=np.int64)


def strings(tables, name):
    return re.findall(r'"([^"]*)"', tables[name])


# ---------------------------------------------------------------- space groups
CENTERING = {
    "P": [],
    "A": [(0, 1, 1)],
    "B": [(1, 0, 1)],
    "C": [(1, 1, 0)],
    "I": [(1, 1, 1)],
    "F": [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
    "R": [(2, 1, 1), (1, 2, 2)],        # obverse setting, hexagonal axes (thirds)
}


class SpaceGroupData:
    """One of the 230 space groups in the ITA default setting of the ISOTROPY database."""

    def __init__(self, number, symbol, full, schoenflies, point_group, ops, centering):
        self.number = number
        self.symbol = symbol          # short Hermann-Mauguin symbol, e.g. "Pnma"
        self.full = full
        self.schoenflies = schoenflies
        self.point_group = point_group   # Hermann-Mauguin point group, e.g. "mmm"
        self.ops = ops                # list of (R int 3x3, t tuple of Fractions) in the conventional cell
        self.centering = centering    # list of Fraction triples (nonzero centering translations)

    def __repr__(self):
        return f"<SpaceGroup {self.number} {self.symbol} ({len(self.ops)} ops in the conventional cell)>"


def _translation(rec):
    n1, n2, n3, d = rec[:4]
    return tuple(Fraction(int(n), int(d)) for n in (n1, n2, n3))


def _close(gens, centering):
    """Closure of generators (R, t) plus pure centering translations, modulo integer translations."""
    ident = (tuple(np.eye(3, dtype=int).ravel()), (Fraction(0),) * 3)
    gens = list(gens) + [(tuple(np.eye(3, dtype=int).ravel()), c) for c in centering]

    def mul(a, b):
        Ra = np.array(a[0], dtype=int).reshape(3, 3)
        Rb = np.array(b[0], dtype=int).reshape(3, 3)
        tb = b[1]
        t = tuple((sum(int(Ra[i, j]) * tb[j] for j in range(3)) + a[1][i]) % 1 for i in range(3))
        return (tuple((Ra @ Rb).ravel()), t)

    els = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = mul(g, x)
                y = (y[0], tuple(v % 1 for v in y[1]))
                if y not in els:
                    els.add(y)
                    nxt.append(y)
        frontier = nxt
    return sorted(els)


@lru_cache(maxsize=None)
def space_groups():
    sp = get_tables("space")
    gens = ints(sp, "ispace_generators_conv").reshape(-1, 4, 5)
    gcount = ints(sp, "ispace_generators_conv_count")
    choice = ints(sp, "ispace_inter_choice")
    short = [s.strip() for s in strings(sp, "space_inter_label_short")]
    full = [s.strip() for s in strings(sp, "space_inter_label_full")]
    sch = [s.strip() for s in strings(sp, "space_label")]
    pgroup = ints(sp, "ispace_point_group")
    pg_labels = [s.strip() for s in strings(sp, "point_group_label_inter")]
    pops = ints(sp, "ipoint_op").reshape(-1, 3, 3)
    pg_order = ints(sp, "ipoint_group_order")
    out = {}
    for g in range(1, 231):
        idx = int(choice[g - 1]) - 1
        recs = gens[idx][: int(gcount[idx])]
        # the tables store matrices for row vectors (x' = x M); we use column vectors (x' = R x), so transpose
        gl = [(tuple(pops[int(r[4]) - 1].T.ravel()), _translation(r)) for r in recs]
        letter = short[idx][0]
        cent = [tuple(Fraction(int(n), 3 if letter == "R" else 2) for n in v) for v in CENTERING[letter]]
        els = _close(gl, cent)
        pgi = int(pgroup[g - 1]) - 1
        expect = int(pg_order[pgi]) * (1 + len(cent))
        if len(els) != expect:
            raise RuntimeError(f"space group {g} ({short[idx]}): closure has {len(els)} ops, expected {expect}")
        ops = [(np.array(R, dtype=int).reshape(3, 3), t) for R, t in els]
        out[g] = SpaceGroupData(g, short[idx], full[idx], sch[g - 1], pg_labels[pgi], ops, cent)
    return out
