"""Is the child's symmetry really a subgroup of the parent's?  Tested operation by operation.

Comparing point groups cannot tell Pbcm from Pbca (both mmm).  Here the child is laid onto the parent's own lattice
(strain and rigid rotation removed), its symmetry operations are found in that frame, converted to the parent's
primitive basis and looked up among the parent's operations (rotation part equal, translation equal modulo lattice
vectors).  Operations of the child that the parent does not have are returned with a classification:

* ``rotation``    the rotation part (in the parent primitive basis) is not one of the parent's, or is not even an
                  integer matrix there: the child is more symmetric than the parent in this alignment, so the parent is
                  not a valid high-symmetry reference (the real parent has more symmetry, or they are siblings);
* ``translation`` the rotation is the parent's but the translation differs by a vector ``v`` (reported): a different
                  origin/setting or a glide/screw the parent lacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .structure import Structure
from .symmetry import find_symmetry, op_to_string


@dataclass
class OpsCheck:
    ok: bool                       # every operation of the child is an operation of the parent
    n_parent: int                  # operations of the parent per primitive cell
    n_child: int                   # distinct operations of the child modulo the parent's lattice
    index: float                   # n_parent / n_child (index of the child's group in the parent's, point-group part x cell part)
    extra: list = field(default_factory=list)   # [(kind, text)] operations of the child that the parent does not have
    n_extra: int = 0

    def summary(self):
        if self.ok:
            return (f"every operation of the child is an operation of the parent ({self.n_child} of {self.n_parent}; "
                    f"index {self.index:g})")
        return f"{self.n_extra} operation(s) of the child are NOT operations of the parent"


def _frac_dist(d, A):
    d = d - np.rint(d)
    return float(np.linalg.norm(d @ A))


def child_ops_in_parent(prim: Structure, parent_ops, M, Xs, species, u_cart, P, sym_tol=0.02, tol_t=0.1):
    """``OpsCheck`` for the child ``parent + u`` (``u_cart`` in parent order, Angstrom) on the supercell with
    basis ``M`` (rows, parent primitive basis), ``Xs`` the parent atoms' supercell fractional positions, ``P = M @ A``."""
    A = prim.lattice
    frac = Xs + u_cart @ np.linalg.inv(P)
    S = Structure(P.copy(), frac % 1.0, list(species))
    Minv_t = np.linalg.inv(M.T)
    cops = find_symmetry(S, sym_tol)
    par = [(np.asarray(R, int), np.asarray(t, float)) for R, t in parent_ops]
    seen, extra, n_child = [], [], 0
    for Rs, ts in cops:
        Rp = M.T @ Rs @ Minv_t
        tp = M.T @ ts
        Ri = np.rint(Rp)
        if np.abs(Rp - Ri).max() > 1e-6:
            key = ("rotation", tuple(np.round(Rp, 3).ravel()), tuple(np.round(tp % 1, 3)))
            if key not in seen:
                seen.append(key)
                extra.append(("rotation", f"rotation {np.round(Rp, 2).tolist()} does not preserve the parent's primitive lattice"))
            continue
        Ri = Ri.astype(int)
        cands = [(R, t) for R, t in par if np.array_equal(R, Ri)]
        key = (tuple(Ri.ravel()), tuple(np.round(tp % 1.0, 2)))
        if key in seen:
            continue
        seen.append(key)
        n_child += 1
        if not cands:
            extra.append(("rotation", f"({op_to_string(Ri, tp % 1.0, decimals=3)}) has a rotation the parent does not have"))
            continue
        dmin = min(cands, key=lambda c: _frac_dist(tp - c[1], A))
        d = _frac_dist(tp - dmin[1], A)
        if d > tol_t:
            v = (tp - dmin[1]) - np.rint(tp - dmin[1])
            extra.append(("translation", f"({op_to_string(Ri, tp % 1.0, decimals=3)}) differs from the parent's operation by the "
                                         f"translation ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}) [{d:.2f} A]"))
    n_par = len({(tuple(R.ravel()), tuple(np.round(t % 1.0, 4))) for R, t in par})
    n_child = max(n_child, 1)
    return OpsCheck(ok=not extra, n_parent=n_par, n_child=n_child, index=n_par / n_child, extra=extra, n_extra=len(extra))
