"""Space-group identification (number, Hermann-Mauguin symbol, setting transformation).

Given the operations of a structure (found by ``symmetry.find_symmetry`` in a primitive cell) this finds
the one of the 230 space groups, in the ISOTROPY database's default (ITA) setting, that is affinely
equivalent to it, together with the basis change ``P`` and origin shift ``p``:

    x_std = P^-1 x + p        (column vectors of fractional coordinates; columns of P are the
                               conventional basis vectors in the primitive basis)

Only the tables of the ISOTROPY suite are used (no spglib).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from . import isodata

DEN = 24                       # denominator limit when rationalising translation parts


@dataclass
class SpaceGroupResult:
    number: int
    symbol: str                # short Hermann-Mauguin symbol (default ITA setting)
    full_symbol: str
    schoenflies: str
    point_group: str
    P: np.ndarray              # conventional basis vectors (columns) in the primitive basis (integers)
    p: np.ndarray              # origin shift (fractions of the conventional cell)
    centering: str
    n_ops_conventional: int

    def label(self):
        return f"{self.symbol} (#{self.number})"


def _order(R):
    M = R.copy()
    for n in range(1, 13):
        if np.array_equal(M, np.eye(3, dtype=int)):
            return n
        M = M @ R
    raise ValueError("rotation matrix of unexpected order")


WD = 24 * 12                    # common denominator for intrinsic translation parts


def _canon_w(R, w, cent):
    """Canonical form of the intrinsic translation ``w`` of rotation ``R``.

    Adding a lattice vector (cell or centering) to an operation's translation changes ``w`` by the
    lattice vector *projected* onto the rotation's fixed space, ``(1/n) sum_k R^k lambda``. So ``w``
    is only defined modulo that projected lattice; reduce it to an echelon-form representative.
    """
    from .symmetry import lattice_basis
    n = _order(R)
    gens = []
    base = [tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)] + [tuple(c) for c in cent]
    for lam in base:
        acc = [Fraction(0)] * 3
        M = np.eye(3, dtype=int)
        for _ in range(n):
            for i in range(3):
                acc[i] += sum(int(M[i, j]) * lam[j] for j in range(3))
            M = R @ M
        gens.append([int(a / n * WD) for a in acc])
    basis = lattice_basis(gens)
    v = [int(Fraction(x) * WD) for x in w]
    for row in basis:
        c = next(i for i in range(3) if row[i])
        b = row[c] if row[c] > 0 else -row[c]
        sgn = 1 if row[c] > 0 else -1
        q = v[c] // b
        v = [v[i] - q * sgn * row[i] for i in range(3)]
    return tuple(Fraction(x, WD) for x in v)


def _intrinsic(R, t):
    n = _order(R)
    acc = [Fraction(0)] * 3
    M = np.eye(3, dtype=int)
    for _ in range(n):
        for i in range(3):
            acc[i] += sum(int(M[i, j]) * t[j] for j in range(3))
        M = R @ M
    return tuple(a / n for a in acc)


@lru_cache(maxsize=None)
def _std_index():
    """Fingerprint tables of the 230 standard groups."""
    G = isodata.space_groups()
    by_key = {}
    info = {}
    for g, sg in G.items():
        cent = sg.centering
        rot_t = {}
        for R, t in sg.ops:
            rot_t.setdefault(tuple(R.ravel()), t)
        key = frozenset((r, _canon_w(np.array(r).reshape(3, 3), _intrinsic(np.array(r).reshape(3, 3), t), cent))
                        for r, t in rot_t.items())
        by_key.setdefault((key, len(cent)), []).append(g)
        info[g] = (rot_t, cent)
    return by_key, info


@lru_cache(maxsize=None)
def _rotsets():
    _, info = _std_index()
    return {frozenset(rt) for rt, _ in info.values()}


def _rationalise(x, tol=0.02):
    f = Fraction(float(x)).limit_denominator(DEN)
    if abs(float(f) - float(x)) > tol:
        raise ValueError("translation is not a crystallographic fraction (tolerance too tight or structure not periodic)")
    return f


def reduce_basis(A, extra=2):
    """Unimodular U (integers) such that U @ A is a short, nearly orthogonal basis."""
    U = np.eye(3, dtype=int)
    B = A.copy()
    improved = True
    coeffs = list(itertools.product(range(-extra, extra + 1), repeat=2))
    while improved:
        improved = False
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            best = (np.linalg.norm(B[i]), 0, 0)
            for cj, ck in coeffs:
                v = B[i] + cj * B[j] + ck * B[k]
                n = np.linalg.norm(v)
                if n < best[0] - 1e-9:
                    best = (n, cj, ck)
            if best[1] or best[2]:
                B[i] = B[i] + best[1] * B[j] + best[2] * B[k]
                U[i] = U[i] + best[1] * U[j] + best[2] * U[k]
                improved = True
    if np.linalg.det(B) < 0:
        B[2] = -B[2]
        U[2] = -U[2]
    return U


def _adj_int(P):
    return np.rint(np.linalg.inv(P) * np.linalg.det(P)).astype(int)


def _candidates(rots, entry_range, dets):
    """Yield integer matrices P (columns = conventional vectors) that make every rotation integral."""
    vals = np.arange(-entry_range, entry_range + 1)
    cols = np.array(list(itertools.product(vals, repeat=3)), dtype=np.int64)       # (n3, 3)
    n3 = len(cols)
    # order candidates by total size so simple settings come first
    for c0 in sorted(range(n3), key=lambda i: np.abs(cols[i]).sum()):
        if not cols[c0].any():
            continue
        c1, c2 = np.meshgrid(np.arange(n3), np.arange(n3), indexing="ij")
        c1, c2 = c1.ravel(), c2.ravel()
        P = np.stack([np.broadcast_to(cols[c0], (len(c1), 3)), cols[c1], cols[c2]], axis=2)      # (m, 3, 3) columns
        det = np.rint(np.linalg.det(P.astype(float))).astype(int)
        ok = np.isin(det, dets)
        P, det = P[ok], det[ok]
        if len(P) == 0:
            continue
        # adj(P) via cofactors, then check adj R P divisible by det for every generator rotation
        adj = np.rint(np.linalg.inv(P.astype(float)) * det[:, None, None]).astype(np.int64)
        good = np.ones(len(P), dtype=bool)
        for R in rots:
            M = adj @ R @ P
            good &= np.all(M % det[:, None, None] == 0, axis=(1, 2))
        for Pg, dg in zip(P[good], det[good]):
            yield Pg, int(dg)


def _check_group(ops, tol=0.02):
    """Raise ValueError unless ``ops`` (one op per rotation, primitive cell) are closed under multiplication."""
    Rs = np.array([R for R, _ in ops])
    ts = np.array([t for _, t in ops])
    keys = {tuple(R.ravel()): i for i, (R, _) in enumerate(ops)}
    if len(keys) != len(ops):
        raise ValueError("operations are not one per rotation: not a primitive cell")
    for i in range(len(ops)):
        prods = Rs[i] @ Rs                                           # R_i R_j for all j
        tnew = ts[i] + ts @ Rs[i].T                                  # R_i t_j + t_i
        for j in range(len(ops)):
            k = keys.get(tuple(prods[j].ravel()))
            if k is None:
                raise ValueError("symmetry operations do not form a group at this tolerance (rotations not closed)")
            d = tnew[j] - ts[k]
            d -= np.rint(d)
            if np.abs(d).max() > tol:
                raise ValueError("symmetry operations do not form a group at this tolerance "
                                 "(translations inconsistent): the structure is only approximately symmetric")


def identify(ops, lattice=None, max_entry=(1, 3), dets=(1, 2, 3, 4)):
    """Identify the space group of ``ops`` = list of (R int 3x3, t float 3) in a *primitive* basis.

    ``lattice`` (3x3 rows) is only used to pick a reduced basis first. Returns SpaceGroupResult.

    The conventional axes are searched among integer combinations of the primitive vectors with entries up to ``max_entry``
    (tried in turn: first up to 1, then up to 3). Entries up to 3 are needed when a high-order axis is not along a short lattice
    vector of the reduced basis: the trigonal (a,a,-a) subgroup of a cubic F lattice has its 3-fold axis along (-1,-1,3). A range
    of exactly 2 was dropped: it never found anything the range of 3 does not, and its failing search cost seconds.
    """
    by_key, info = _std_index()
    rotsets = _rotsets()
    ops = [(np.array(R, dtype=int), np.array(t, dtype=float)) for R, t in ops]
    Ured = np.eye(3, dtype=int)
    if lattice is not None:
        Ured = reduce_basis(np.asarray(lattice, float))
        Ui = np.linalg.inv(Ured.astype(float))                    # x' = U^-T x
        ops = [(np.rint(Ui.T @ R @ Ured.T).astype(int), Ui.T @ t) for R, t in ops]
    _check_group(ops)
    rots = [R for R, _ in ops]
    uniq = {tuple(R.ravel()): R for R in rots}
    gens = list(uniq.values())
    # a few generators are enough to prune; all rotations are checked afterwards
    for entry in max_entry:
        for P, det in _candidates(gens, entry, dets):
            adj = _adj_int(P)
            try:
                conv = {}
                for R, t in ops:
                    Rc = (adj @ R @ P) // det
                    conv[tuple(Rc.ravel())] = np.linalg.solve(P.astype(float), t)
            except np.linalg.LinAlgError:
                continue
            if len(conv) != len(ops):
                continue
            if frozenset(conv) not in rotsets:
                continue
            # centering translations = primitive lattice points inside the conventional cell
            Pinv = np.linalg.inv(P.astype(float))
            cent = set()
            for v in itertools.product(range(det), repeat=3):
                c = Pinv @ np.array(v, dtype=float)
                cf = tuple(_rationalise(x, 1e-6) % 1 for x in c)
                if any(cf):
                    cent.add(cf)
            if len(cent) + 1 != det:
                continue
            cent = sorted(cent)
            try:
                fp = frozenset(
                    (r, _canon_w(np.array(r).reshape(3, 3), tuple(_rationalise(x) for x in _intrinsic(np.array(r).reshape(3, 3), t)), cent))
                    for r, t in conv.items()
                )
            except ValueError:
                continue
            for g in by_key.get((fp, len(cent)), []):
                rot_t, scent = info[g]
                if set(scent) != set(cent):
                    continue
                p = _solve_origin(conv, rot_t, scent)
                if p is not None:
                    sg = isodata.space_groups()[g]
                    return SpaceGroupResult(g, sg.symbol, sg.full, sg.schoenflies, sg.point_group,
                                            Ured.T @ P, np.asarray(p, dtype=float), _centering_letter(sg), len(sg.ops))
    raise ValueError("could not identify the space group (symmetry operations inconsistent with any of the 230 groups)")


def _centering_letter(sg):
    return sg.symbol[0]


def smith_normal_form(M):
    """Integer Smith normal form: returns (U, A, V) with U @ M @ V = A diagonal (all int, U and V unimodular)."""
    M = [list(map(int, r)) for r in M]
    m, n = len(M), len(M[0])
    A = [r[:] for r in M]
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def row_swap(i, j):
        A[i], A[j] = A[j], A[i]
        U[i], U[j] = U[j], U[i]

    def col_swap(i, j):
        for r in A:
            r[i], r[j] = r[j], r[i]
        for r in V:
            r[i], r[j] = r[j], r[i]

    def row_add(i, j, q):               # row_i -= q * row_j
        A[i] = [x - q * y for x, y in zip(A[i], A[j])]
        U[i] = [x - q * y for x, y in zip(U[i], U[j])]

    def col_add(i, j, q):               # col_i -= q * col_j
        for r in A:
            r[i] -= q * r[j]
        for r in V:
            r[i] -= q * r[j]

    for t in range(min(m, n)):
        while True:
            best = None
            for i in range(t, m):
                for j in range(t, n):
                    if A[i][j] and (best is None or abs(A[i][j]) < abs(A[best[0]][best[1]])):
                        best = (i, j)
            if best is None:
                break
            row_swap(t, best[0])
            col_swap(t, best[1])
            done = True
            for i in range(t + 1, m):
                if A[i][t]:
                    row_add(i, t, A[i][t] // A[t][t])
                    if A[i][t]:
                        done = False
            for j in range(t + 1, n):
                if A[t][j]:
                    col_add(j, t, A[t][j] // A[t][t])
                    if A[t][j]:
                        done = False
            if not done:
                continue
            # divisibility: A[t][t] must divide everything in the remaining block
            bad = next(((i, j) for i in range(t + 1, m) for j in range(t + 1, n) if A[i][j] % A[t][t]), None)
            if bad is None:
                break
            row_add(t, bad[0], -1)      # add the offending row to row t and repeat
        if best is None:
            break
    return U, A, V


def _lattice_matrix(cent):
    """Columns = basis of Z^3 + centering vectors (rational, as floats)."""
    from .symmetry import lattice_basis
    D = 6
    gens = [[D * int(i == j) for j in range(3)] for i in range(3)]
    gens += [[int(c * D) for c in v] for v in cent]
    rows = lattice_basis(gens)
    return np.array(rows, dtype=float).T / D


def _solve_origin(conv, rot_t, cent, tol=0.03):
    """Origin shift p (conventional-cell fractions) with  tau_std(R) = t(R) + (I - R) p  (mod lattice).

    Solved exactly: in the basis of the full lattice (cell + centering) the conditions become
    M q = D (mod 1) with M integer, which a Smith normal form solves, leaving the free
    (non-polar) directions at zero.  Returns None if no origin exists.
    """
    B = _lattice_matrix(cent)
    Binv = np.linalg.inv(B)
    rows, rhs = [], []
    for r, t in conv.items():
        R = np.array(r).reshape(3, 3)
        Rb = Binv @ R @ B
        Ri = np.rint(Rb).astype(int)
        if np.abs(Rb - Ri).max() > 1e-8:
            return None
        d = Binv @ (np.array([float(x) for x in rot_t[r]]) - np.asarray(t, dtype=float))
        rows += (np.eye(3, dtype=int) - Ri).tolist()
        rhs += list(d)
    U, A, V = smith_normal_form(rows)
    e = np.array(U, dtype=float) @ np.array(rhs)
    qp = np.zeros(3)
    for j in range(len(rows)):
        sj = A[j][j] if j < 3 else 0
        if sj:
            qp[j] = e[j] / sj
        else:
            dev = abs(e[j] - np.rint(e[j]))
            if dev > tol:
                return None
    q = np.array(V, dtype=float) @ qp
    p = B @ q
    # verify against every operation
    cvecs = [np.zeros(3)] + [np.array([float(x) for x in c]) for c in cent]
    for r, t in conv.items():
        R = np.array(r).reshape(3, 3)
        new = np.asarray(t, dtype=float) + (np.eye(3) - R) @ p
        diff = (np.array([float(x) for x in rot_t[r]]) - new) % 1.0
        if not any(np.all(np.minimum(np.abs(diff - c), 1 - np.abs(diff - c)) < tol) for c in cvecs):
            # also allow differences by any lattice vector combination (cvecs reduced mod 1 already)
            return None
    return _nearest_origin(conv, cent, p)


def _generators(rot_keys):
    """A small generating set of the rotation group given by flat 3x3 integer tuples."""
    mats = [np.array(k).reshape(3, 3) for k in rot_keys]
    gens, group = [], {tuple(np.eye(3, dtype=int).ravel())}
    for k, M in zip(rot_keys, mats):
        if k in group:
            continue
        gens.append(k)
        frontier = list(group)
        group = set(group)
        while True:
            new = set()
            for a in list(group):
                for gk in gens:
                    prod = tuple((np.array(gk).reshape(3, 3) @ np.array(a).reshape(3, 3)).ravel())
                    if prod not in group:
                        new.add(prod)
            if not new:
                break
            group |= new
    return gens


def _nearest_origin(conv, cent, p):
    """Among the equivalent origins p + s (s in the group's Euclidean normaliser) take the one closest to the
    user's own origin, so that origin-dependent labels (parity +/-) follow the coordinates that were given."""
    g = np.array(list(itertools.product(range(DEN), repeat=3)), dtype=np.int64)
    cvecs = np.array([[0, 0, 0]] + [[float(x) for x in c] for c in cent], dtype=float)
    ok = np.ones(len(g), dtype=bool)
    for r in _generators(list(conv)):
        R = np.array(r).reshape(3, 3)
        lhs = (((np.eye(3, dtype=np.int64) - R) @ g.T).T / DEN) % 1.0
        hit = np.zeros(len(g), dtype=bool)
        for c in cvecs:
            e = np.abs(lhs - c[None, :])
            hit |= np.all(np.minimum(e, 1 - e) < 1e-9, axis=1)
        ok &= hit
    cand = p[None, :] + g[ok] / DEN
    cand -= np.floor(cand + 0.5)
    best = cand[np.argmin(np.linalg.norm(cand, axis=1) + 1e-9 * cand.sum(axis=1))]
    return best
