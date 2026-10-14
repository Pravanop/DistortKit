"""Numerical splitting of a real finite-group representation into irreducible copies."""
from __future__ import annotations

import numpy as np


def split_real_irreps(rho, seed=0, attempts=8):
    """Split the real representation ``rho`` (n_ops, m, m) into real irreducible copies.

    A random symmetric matrix averaged over the group commutes with the action, so each of its
    eigenspaces is exactly one irreducible copy (including separate copies of the same irrep).
    Returns a list of ``(W, r, chi, norm)``: ``W`` (m, d) orthonormal basis of the copy, ``r``
    (n_ops, d, d) its representation matrices, ``chi`` the characters and ``norm`` = mean(chi^2)
    which is 1, 2 or 4 for real, complex or quaternionic type.
    """
    rho = np.asarray(rho)
    m = rho.shape[1]
    for attempt in range(attempts):
        rng = np.random.default_rng(seed + attempt)
        S = rng.normal(size=(m, m))
        S = S + S.T
        C = (rho @ S @ rho.transpose(0, 2, 1)).mean(axis=0)
        C = (C + C.T) / 2
        w, V = np.linalg.eigh(C)
        scale = max(np.abs(w).max(), 1e-12)
        clusters, cur = [], [0]
        for i in range(1, m):
            if w[i] - w[i - 1] < 1e-7 * scale:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        out, ok = [], True
        for idx in clusters:
            W = V[:, idx]
            r = W.T @ rho @ W
            chi = np.trace(r, axis1=1, axis2=2)
            norm = float(np.mean(chi**2))
            if min(abs(norm - 1), abs(norm - 2), abs(norm - 4)) > 1e-6:
                ok = False
                break
            out.append((W, r, chi, norm))
        if ok:
            return out
    raise RuntimeError("could not split the representation into irreps (accidental degeneracy)")
