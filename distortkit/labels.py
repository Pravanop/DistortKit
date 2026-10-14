"""Standard (ISOTROPY / Miller-Love) irrep labels for the modes found by ``ModeAnalysis``.

Method.  The ISOTROPY tables list, for every parent space group and irrep label, the isotropy
subgroups (child space-group numbers) with their number of free order-parameter components.  For each
irrep found by ``ModeAnalysis`` we compute the same fingerprint, identifying each isotropy subgroup's space
group with ``spacegroup.identify``, and look it up in the tables.  A label is reported as *certain*
when exactly one tabulated irrep (of the right dimension) has that fingerprint, and as *ambiguous*
(all candidates are listed) when several do.  Tables are never used to guess: if nothing matches, no
label is given.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import isodata
from .spacegroup import identify
from .symmetry import lattice_basis


@dataclass
class IrrepLabel:
    candidates: list                    # labels whose fingerprint equals ours, e.g. ["R4+"]
    certain: bool                       # exactly one candidate
    fingerprint: tuple                  # sorted tuple of (child space-group number, free parameters)
    child_numbers: list = field(default_factory=list)   # number per isotropy subgroup class (same order)
    note: str = ""

    @property
    def text(self):
        if not self.candidates:
            return "?"
        return self.candidates[0] if self.certain else "/".join(self.candidates)


@lru_cache(maxsize=None)
def _database():
    I = isodata.get_tables("isotropy")
    Ir = isodata.get_tables("irreps")
    labels = [s.strip() for s in isodata.strings(Ir, "irrep_label")]
    parent = isodata.ints(I, "isotropy_parent")
    irr = isodata.ints(I, "isotropy_irrep")
    sub = isodata.ints(I, "isotropy_subgroup")
    free = isodata.ints(I, "isotropy_orderparam_freeparam")
    odim = isodata.ints(I, "isotropy_orderparam_dim")
    by_parent = {}
    for r in range(len(parent)):
        by_parent.setdefault(int(parent[r]), {}).setdefault(int(irr[r]) - 1, []).append((int(sub[r]), int(free[r]), int(odim[r])))
    out = {}
    for g, d in by_parent.items():
        out[g] = [(labels[i], recs[0][2], tuple(sorted((s, f) for s, f, _ in recs))) for i, recs in d.items()]
    return out


def child_space_group(ma, iso):
    """Space group (number) of the isotropy subgroup ``iso`` of a ModeAnalysis ``ma``."""
    ops = [(ma.R[i], ma.t[i]) for i in iso.op_indices]
    gens = [list(map(int, r)) for r in ma.Sb]                                    # supercell lattice
    for R, t in ops:
        if np.array_equal(R, np.eye(3, dtype=int)):                              # pure translation kept
            tt = np.rint(t).astype(int)
            if tt.any():
                gens.append(list(tt))
    Bc = np.array(lattice_basis(gens), dtype=float)                              # rows: child lattice, parent prim basis
    if round(np.linalg.det(Bc)) < 0:
        Bc[2] = -Bc[2]
    BtI = np.linalg.inv(Bc.T)
    seen, cops = set(), []
    for R, t in ops:
        key = tuple(R.ravel())
        if key in seen:
            continue
        seen.add(key)
        Rn = np.rint(BtI @ R @ Bc.T).astype(int)
        cops.append((Rn, BtI @ t))
    lat = Bc @ ma.parent.lattice
    return identify(cops, lat)


def inversion_parity(ma, irrep, P, p):
    """+1 / -1: character of the inversion at the standard origin on ``irrep`` (None if undefined).

    The standard origin is ``p`` in standard-setting coordinates, i.e. the inversion (-I | 0)_std, which in the
    parent's primitive coordinates is (-I | -2 P p).  Only meaningful for irreps at wavevectors with k = -k.
    """
    sg = ma._sg_std
    if sg is None:
        return None
    t_inv = -2.0 * (np.asarray(P, float) @ np.asarray(p, float))
    Sinv = np.linalg.inv(ma.Sb.astype(float))
    for i in range(ma.n_ops):
        if not np.array_equal(ma.R[i], -np.eye(3, dtype=int)):
            continue
        lam = ma.t[i] - t_inv
        c = lam @ Sinv                                      # must be an integer combination of the supercell vectors
        if np.abs(c - np.rint(c)).max() < 1e-6:
            chi = float(np.trace(irrep.rho[i]))
            if abs(abs(chi) - irrep.dim) < 1e-6:
                return 1 if chi > 0 else -1
            return None
    return None


def label_irreps(ma, parent_number, P=None, p=None, only=None):
    """Return {irrep.index: IrrepLabel} for every irrep of ``ma`` (parent space group ``parent_number``).

    ``P``, ``p`` (from ``spacegroup.identify``) let the parity (+/-) of an irrep be computed and used to tell
    apart labels that share an isotropy fingerprint, such as R4+ and R5-.
    """
    db = _database().get(parent_number, [])
    ma._sg_std = True if (P is not None and p is not None) else None
    out = {}
    for ir in ma.irreps:
        if only is not None and ir.index not in only:
            continue
        nums, fp = [], []
        for iso in ma.isotropy_subgroups(ir):
            try:
                n = child_space_group(ma, iso).number
            except ValueError:
                n = 0
            nums.append(n)
            fp.append((n, iso.fix_dim))
        fingerprint = tuple(sorted(fp))
        at_gamma = all(x == 0 for x in ma.k)
        # the label letters name the k-point: only GM labels exist at Gamma, and none elsewhere
        ok_k = (lambda lab: lab.startswith("GM")) if at_gamma else (lambda lab: not lab.startswith("GM"))
        cand = [lab for lab, d, f in db if f == fingerprint and (d == ir.dim or d * 2 == ir.dim) and ok_k(lab)]
        note = ""
        if len(cand) > 1 and P is not None:
            par = inversion_parity(ma, ir, P, p)
            if par is not None:
                keep = [c for c in cand if not c.endswith(("+", "-")) or (c.endswith("+") == (par > 0))]
                if keep:
                    cand = keep
                    note = f"parity {'+' if par > 0 else '-'} used to choose between labels"
        if not cand:
            # fall back: same fingerprint regardless of dimension (flagged)
            cand = [lab for lab, d, f in db if f == fingerprint and ok_k(lab)]
            note = "dimension differs from the tabulated irrep" if cand else "no tabulated irrep has this isotropy fingerprint"
        out[ir.index] = IrrepLabel(sorted(cand), len(cand) == 1, fingerprint, nums, note)
    return out
