"""Symmetry-adapted distortion modes, isotropy subgroups and child structures.

Method (all numerical, numpy only)
----------------------------------
* The parent space group G is found by brute force (``symmetry.find_symmetry``).
* For a wavevector k the star of k fixes a supercell lattice L_s = {T : k_j.T integer}.
  Real displacement fields on that supercell with Fourier components in the star
  (plus -star) form a G-invariant space; G (mod L_s) acts on it by real orthogonal matrices.
* For every atomic orbit of the parent, that space is split into real irreducible
  representations by diagonalising a random symmetric matrix averaged over the group
  (it lies in the commutant, so its eigenspaces are single irreducible copies).
* Copies from different orbits that are equivalent are aligned with an intertwiner, so a
  single order-parameter direction (OPD) means the same thing on every atom.
* Isotropy subgroups of an irrep are stabilisers of vectors in its representation space,
  found by enumerating the subgroups of the (finite) image group.

Irreps are identified by number (dimension, type, characters), not by the Miller-Love /
Stokes-Hatch labels used by ISOTROPY/ISODISTORT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np

from .irreps import split_real_irreps
from .structure import Structure
from .symmetry import (
    find_symmetry,
    is_pure_translation,
    lattice_basis,
    op_to_string,
    point_group_name,
)


def _frac(x):
    return Fraction(x) if isinstance(x, (str, Fraction, int)) else Fraction(float(x)).limit_denominator(1000)


def _lcm(a, b):
    return a * b // gcd(a, b)


@dataclass
class Mode:
    """One irreducible copy: ``vectors`` has shape (dim, n_atoms, 3), orthonormal, Cartesian."""

    orbit: int
    atoms: list  # primitive-cell atom indices of the orbit
    species: str
    vectors: np.ndarray


@dataclass
class IsotropySubgroup:
    index: int
    opd: np.ndarray            # representative generic direction in the irrep basis
    opd_label: str             # parametric form, e.g. "(a,a,0)"
    fix_basis: np.ndarray = field(repr=False, default=None)  # row-reduced basis of the fixed subspace
    fix_dim: int = 1           # dimension of the fixed subspace (free amplitudes)
    point_group: str = ""
    n_point_ops: int = 0
    n_translations: int = 0    # pure translations of the primitive cell kept (mod supercell)
    n_conjugates: int = 1      # symmetry-equivalent domains (conjugate subgroups)
    op_indices: list = field(repr=False, default_factory=list)
    space_group: object = field(repr=False, default=None)


@dataclass
class Irrep:
    index: int
    dim: int
    kind: str                  # real / complex / quaternionic (Frobenius-Schur type)
    modes: list
    rho: np.ndarray = field(repr=False, default=None)  # (n_ops, dim, dim)
    _iso: list = field(repr=False, default=None)

    @property
    def label(self):
        return f"irrep{self.index}"

    @property
    def species(self):
        return sorted({m.species for m in self.modes})

    def __repr__(self):
        where = ", ".join(f"{m.species}(orbit {m.orbit})" for m in self.modes)
        return f"<Irrep {self.index}: dim {self.dim} ({self.kind}), modes on {where}>"


class ModeAnalysis:
    """Symmetry-adapted distortion modes of ``structure`` at wavevector ``k``.

    ``k`` is given in fractional coordinates of the *reciprocal basis of ``structure``'s
    lattice*; ``structure`` must be a primitive cell (use ``symmetry.to_primitive``).
    """

    max_subspaces = 20000

    def __init__(self, structure: Structure, k, tol: float = 1e-3, seed: int = 0):
        self.parent = structure
        self.tol = tol
        self.seed = seed
        self.k = tuple(_frac(x) for x in k)
        self._setup_symmetry()
        self._setup_star()
        self._setup_supercell()
        self._setup_group()
        self._decompose()

    # ------------------------------------------------------------------ symmetry
    def _setup_symmetry(self):
        s = self.parent
        ops = find_symmetry(s, self.tol)
        if sum(is_pure_translation(R) and np.linalg.norm(t) > 1e-9 for R, t in ops):
            raise ValueError("structure is not a primitive cell; call symmetry.to_primitive() first")
        self.prim_ops = ops
        n = len(s)
        sp = np.array(s.species)
        same = sp[:, None] == sp[None, :]
        self._amap = []
        for R, t in ops:
            Y = s.frac @ R.T + t
            diff = Y[:, None, :] - s.frac[None, :, :]
            shift = np.rint(diff)
            dist = np.linalg.norm((diff - shift) @ s.lattice, axis=2)
            dist[~same] = np.inf
            b = dist.argmin(axis=1)
            if np.any(dist[np.arange(n), b] > self.tol):
                raise RuntimeError("symmetry operation does not map the atoms onto themselves")
            S = shift[np.arange(n), b].astype(int)
            self._amap.append((b, S))
        # atomic orbits (union-find over the atom permutations)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for b, _ in self._amap:
            for a in range(n):
                ra, rb = find(a), find(int(b[a]))
                if ra != rb:
                    parent[ra] = rb
        groups = {}
        for a in range(n):
            groups.setdefault(find(a), []).append(a)
        self.orbits = list(groups.values())

    # ------------------------------------------------------------------ star
    def _setup_star(self):
        arms = []
        for R, _ in self.prim_ops:
            Rit = np.rint(np.linalg.inv(R)).astype(int).T
            kp = tuple(sum(int(Rit[i, j]) * self.k[j] for j in range(3)) % 1 for i in range(3))
            if kp not in arms:
                arms.append(kp)
        self.star = arms
        classes = []
        for a in arms:
            neg = tuple((-x) % 1 for x in a)
            if a not in classes and neg not in classes:
                classes.append(a)
        self.star_classes = classes  # one per +/- pair

    # ------------------------------------------------------------------ supercell
    def _setup_supercell(self):
        D = 1
        for a in self.star:
            for x in a:
                D = _lcm(D, x.denominator)
        gens = [[D * int(i == j) for j in range(3)] for i in range(3)]
        for T in np.ndindex(D, D, D):
            if all(sum(a[i] * T[i] for i in range(3)).denominator == 1 for a in self.star):
                gens.append(list(T))
        Sb = np.array(lattice_basis(gens), dtype=int)
        if round(np.linalg.det(Sb)) < 0:
            Sb[2] = -Sb[2]
        self.Sb = Sb
        self.n_cells = abs(int(round(np.linalg.det(Sb))))
        self._Sbinv = np.linalg.inv(Sb)
        corners = np.array([sum(c * Sb[i] for i, c in enumerate(cs)) for cs in np.ndindex(2, 2, 2)])
        self._lo = corners.min(axis=0) - 1
        hi = corners.max(axis=0) + 1
        grid = np.array(list(np.ndindex(*(hi - self._lo + 1)))) + self._lo
        c = grid @ self._Sbinv
        mask = np.all(c > -1e-9, axis=1) & np.all(c < 1 - 1e-9, axis=1)
        self.cells = grid[mask]
        if len(self.cells) != self.n_cells:
            raise RuntimeError("supercell enumeration failed")
        self._table = -np.ones(tuple(hi - self._lo + 1), dtype=int)
        for i, T in enumerate(self.cells):
            self._table[tuple(T - self._lo)] = i
        add = np.empty((self.n_cells, self.n_cells), dtype=int)
        for j, T in enumerate(self.cells):
            add[:, j] = self._canon(self.cells + T)
        self._add = add

    def _canon(self, V):
        V = np.asarray(V, dtype=int)
        c = V @ self._Sbinv
        V0 = V - np.rint(np.floor(c + 1e-9) @ self.Sb).astype(int)
        return self._table[tuple((V0 - self._lo).T)]

    # ------------------------------------------------------------------ group on supercell
    def _setup_group(self):
        s = self.parent
        n, Nc = len(s), self.n_cells
        self.n_atoms = n * Nc
        A = s.lattice
        R_all, t_all, Q_all, perm_all = [], [], [], []
        for (R, t), (b, S) in zip(self.prim_ops, self._amap):
            Q = A.T @ R @ np.linalg.inv(A.T)
            base = np.empty(n * Nc, dtype=int)
            cellimg = []
            for a in range(n):
                ci = self._canon(self.cells @ R.T + S[a])
                if np.any(ci < 0):
                    raise RuntimeError("operation does not preserve the supercell lattice")
                cellimg.append(ci)
            for j, T in enumerate(self.cells):
                perm = np.empty(n * Nc, dtype=int)
                for a in range(n):
                    perm[a * Nc + np.arange(Nc)] = b[a] * Nc + self._add[cellimg[a], j]
                R_all.append(R)
                t_all.append(t + T)
                Q_all.append(Q)
                perm_all.append(perm)
        self.R = np.array(R_all)
        self.t = np.array(t_all)
        self.Q = np.array(Q_all)
        self.perm = np.array(perm_all)
        self.n_ops = len(self.R)
        self.is_translation = np.array([is_pure_translation(R) for R in self.R])

    def apply_op(self, i, B):
        """Apply operation ``i`` to displacement field(s) ``B`` of shape (n_atoms, 3, m)."""
        out = np.empty_like(B)
        out[self.perm[i]] = np.einsum("ij,njm->nim", self.Q[i], B)
        return out

    # ------------------------------------------------------------------ mode decomposition
    def _star_basis(self, atoms):
        Nc = self.n_cells
        funcs = []
        for a in self.star_classes:
            phase = 2 * np.pi * (self.cells @ np.array([float(x) for x in a]))
            funcs.append(np.cos(phase))
            if tuple((-x) % 1 for x in a) != a:
                funcs.append(np.sin(phase))
        cols = []
        for at in atoms:
            for d in range(3):
                for f in funcs:
                    v = np.zeros((self.n_atoms, 3))
                    v[at * Nc:(at + 1) * Nc, d] = f
                    cols.append(v.ravel())
        Q, _ = np.linalg.qr(np.array(cols).T)
        return Q  # (3 n_atoms, m)

    def _orbit_irreps(self, atoms, seed):
        B = self._star_basis(atoms)
        m = B.shape[1]
        B3 = B.reshape(self.n_atoms, 3, m)
        rho = np.empty((self.n_ops, m, m))
        for i in range(self.n_ops):
            new = self.apply_op(i, B3).reshape(-1, m)
            rho[i] = B.T @ new
            if np.abs(new - B @ rho[i]).max() > 1e-7:
                raise RuntimeError("star subspace is not invariant: tolerance or lattice problem")
        out = []
        for W, r, chi, norm in split_real_irreps(rho, seed):
            out.append((B @ W, r, chi, norm))
        return out

    def _decompose(self):
        copies = []
        for oi, atoms in enumerate(self.orbits):
            res = self._orbit_irreps(atoms, self.seed)
            for vecs, r, chi, norm in res:
                copies.append((oi, atoms, vecs, r, chi, norm))
        classes = []  # each: list of copy tuples
        for cp in copies:
            for cl in classes:
                ref = cl[0]
                if ref[3].shape[1] == cp[3].shape[1] and np.abs(ref[4] - cp[4]).max() < 1e-6:
                    cl.append(cp)
                    break
            else:
                classes.append([cp])
        classes.sort(key=lambda cl: cl[0][3].shape[1])
        rng = np.random.default_rng(self.seed + 1000)
        self.irreps = []
        for ci, cl in enumerate(classes, start=1):
            _, _, vref, rref, _, norm = cl[0]
            d = rref.shape[1]
            modes = []
            for oi, atoms, vecs, r, chi, _ in cl:
                if r is not rref:
                    X = rng.normal(size=(d, d))
                    T = (r @ X @ rref.transpose(0, 2, 1)).mean(axis=0)
                    u, sv, vt = np.linalg.svd(T)
                    if sv.min() < 1e-6 * sv.max():
                        raise RuntimeError("equivalent copies could not be aligned")
                    U = u @ vt
                    vecs = vecs @ U
                    if np.abs(U.T @ r @ U - rref).max() > 1e-6:
                        raise RuntimeError("alignment of equivalent irreps failed")
                modes.append(
                    Mode(
                        orbit=oi,
                        atoms=list(atoms),
                        species=self.parent.species[atoms[0]],
                        vectors=vecs.T.reshape(d, self.n_atoms, 3),
                    )
                )
            kind = {1: "real", 2: "complex", 4: "quaternionic"}[int(round(norm))]
            self.irreps.append(Irrep(ci, d, kind, modes, rref))

    # ------------------------------------------------------------------ space group and labels
    def space_group(self):
        """Space group of the parent (``spacegroup.SpaceGroupResult``): number, symbol, setting transformation."""
        if getattr(self, "_sg", None) is None:
            from .spacegroup import identify
            self._sg = identify(self.prim_ops, self.parent.lattice)
        return self._sg

    def irrep_labels(self, only=None):
        """Standard irrep labels, ``{irrep.index: labels.IrrepLabel}`` (see ``labels.py`` for the method)."""
        from . import labels as _labels
        sg = self.space_group()
        cache = getattr(self, "_labels", None)
        if cache is None:
            cache = self._labels = {}
        need = [ir.index for ir in self.irreps if (only is None or ir.index in only) and ir.index not in cache]
        if need:
            cache.update(_labels.label_irreps(self, sg.number, sg.P, sg.p, only=set(need)))
        return {k: v for k, v in cache.items() if only is None or k in only}

    def child_space_group(self, iso):
        """Space group (``SpaceGroupResult``) of an isotropy subgroup; cached on the subgroup object."""
        if getattr(iso, "space_group", None) is None:
            from . import labels as _labels
            try:
                iso.space_group = _labels.child_space_group(self, iso)
            except ValueError:
                iso.space_group = False
        return iso.space_group or None

    # ------------------------------------------------------------------ isotropy subgroups
    def isotropy_subgroups(self, irrep: Irrep):
        """Isotropy subgroups of ``irrep``, one representative per conjugacy class.

        An isotropy subgroup is the stabiliser of a generic vector of some subspace Fix(H).
        The relevant subspaces are exactly the intersections of the Fix(g), so we close
        {Fix(g)} under intersection instead of enumerating the whole subgroup lattice.
        The irrep basis is first rotated so that the highest-symmetry direction is a
        coordinate axis, which makes OPDs read like (a,0,0), (a,a,0), ...
        """
        if irrep._iso is None:
            first = self._isotropy_in_basis(irrep)
            O = self._nice_basis(irrep, first)
            if O is not None:
                irrep.rho = O.T @ irrep.rho @ O
                for m in irrep.modes:
                    m.vectors = np.einsum("ba,bnx->anx", O, m.vectors)
            irrep._iso = self._isotropy_in_basis(irrep)
        return irrep._iso

    @staticmethod
    def _key(M, nd=6):
        return (np.round(M, nd) + 0.0).tobytes()

    def _isotropy_in_basis(self, irrep):
        rho, d, n = irrep.rho, irrep.dim, self.n_ops
        eye = np.eye(d)

        def projector(Bs):  # Bs: rows orthonormal
            return Bs.T @ Bs

        def null_rows(M):
            u, sv, vt = np.linalg.svd(M)
            return vt[int((sv > 1e-8).sum()):]

        # subspaces Fix(g) for every operation (batched SVD, deduplicated by projector)
        fixes = {}
        U, sv, Vt = np.linalg.svd(rho - eye)
        rank = (sv > 1e-8).sum(axis=1)
        seen_img = set()
        for g in range(n):
            k0 = self._key(rho[g])
            if k0 in seen_img:
                continue
            seen_img.add(k0)
            Bs = Vt[g][rank[g]:]
            if len(Bs):
                fixes.setdefault(self._key(projector(Bs)), Bs)
        base = list(fixes.values())
        Pb = np.array([projector(B) for B in base]) if base else np.zeros((0, d, d))
        Qb = eye - Pb
        spaces = {self._key(eye): eye}            # whole space = Fix(kernel)
        spaces.update(fixes)
        frontier = list(spaces.values())
        while frontier:
            nxt = []
            for W in frontier:
                # W ∩ Fix(g) for all g at once: null space of (I - P_g) W^T, batched
                M = Qb @ W.T                                   # (nb, d, a)
                _, sv, vt = np.linalg.svd(M)
                rank = (sv > 1e-8).sum(axis=1)
                nullity = W.shape[0] - rank
                for i in np.where((nullity > 0) & (nullity < W.shape[0]))[0]:
                    X = vt[i][rank[i]:] @ W
                    X, _ = np.linalg.qr(X.T)
                    X = X.T
                    k = self._key(projector(X))
                    if k not in spaces:
                        spaces[k] = X
                        nxt.append(X)
                        if len(spaces) > self.max_subspaces:
                            raise RuntimeError(
                                f"isotropy enumeration exceeded {self.max_subspaces} subspaces "
                                f"(irrep dim {d}, {n} operations); use a k-point with a smaller supercell "
                                "or raise ModeAnalysis.max_subspaces")
            frontier = nxt

        rng = np.random.default_rng(self.seed + 7)
        entries = []
        for key, Bs in spaces.items():
            v = rng.normal(size=len(Bs)) @ Bs
            v /= np.linalg.norm(v)
            ops = np.where(np.linalg.norm(rho @ v - v, axis=1) < 1e-7)[0]
            entries.append((key, Bs, ops))
        # one representative per orbit of subspaces under the group
        seen, reps = set(), []
        for key, Bs, ops in sorted(entries, key=lambda e: (-len(e[2]), len(e[1]))):
            if key in seen:
                continue
            P = projector(Bs)
            imgs = rho @ P @ rho.transpose(0, 2, 1)
            orbit = {self._key(M) for M in imgs}
            seen |= orbit
            reps.append((Bs, ops, len(orbit)))
        out = []
        letters = [chr(ord("a") + i) if i < 26 else f"p{i}" for i in range(64)]
        for Bs, ops, nconj in reps:
            R, piv = self._rref(Bs)
            coeff = np.random.default_rng(12345 + len(R)).uniform(0.4, 1.0, len(R))  # generic, deterministic
            opd = coeff @ R
            opd /= np.linalg.norm(opd)
            Rs = [self.R[g] for g in ops]
            comps = []
            for j in range(d):
                terms = [(letters[r], R[r, j]) for r in range(len(R)) if abs(R[r, j]) > 1e-8]
                comps.append(self._fmt(terms))
            out.append(
                IsotropySubgroup(
                    index=len(out) + 1,
                    opd=opd,
                    opd_label="(" + ",".join(comps) + ")",
                    fix_basis=R,
                    fix_dim=len(R),
                    point_group=point_group_name(Rs, self.parent.lattice),
                    n_point_ops=len({tuple(Rm.ravel()) for Rm in Rs}),
                    n_translations=int(sum(self.is_translation[g] for g in ops)),
                    n_conjugates=nconj,
                    op_indices=[int(g) for g in ops],
                )
            )
        return out

    @staticmethod
    def _rref(B, tol=1e-9):
        B = B.copy()
        r, piv = 0, []
        for c in range(B.shape[1]):
            if r >= B.shape[0]:
                break
            p = r + int(np.argmax(np.abs(B[r:, c])))
            if abs(B[p, c]) < tol:
                continue
            B[[r, p]] = B[[p, r]]
            B[r] /= B[r, c]
            for i in range(B.shape[0]):
                if i != r:
                    B[i] -= B[i, c] * B[r]
            piv.append(c)
            r += 1
        return B[:r], piv

    @staticmethod
    def _fmt(terms):
        if not terms:
            return "0"
        parts = []
        for name, c in terms:
            f = Fraction(float(c)).limit_denominator(12)
            if abs(float(f) - c) < 1e-6:
                txt = name if f == 1 else ("-" + name if f == -1 else f"{f}{name}")
            else:
                txt = f"{c:.3f}{name}"
            parts.append(txt)
        return "+".join(parts).replace("+-", "-")

    def _nice_basis(self, irrep, subgroups):
        """Orthogonal matrix O taking the irrep basis to one where the highest-symmetry
        one-parameter direction (and its symmetry-equivalent copies) are coordinate axes."""
        d = irrep.dim
        if d == 1:
            return None
        cand = [s for s in subgroups if s.fix_dim == 1]
        if not cand:
            return None
        v = cand[0].opd
        orbit = []
        for g in range(self.n_ops):
            w = irrep.rho[g] @ v
            if not any(abs(abs(w @ x) - 1) < 1e-6 for x in orbit):
                orbit.append(w)
        basis = []
        for w in orbit:
            if all(abs(w @ x) < 1e-6 for x in basis):
                basis.append(w)
            if len(basis) == d:
                break
        if len(basis) < d:
            return None
        return np.array(basis).T

    # ------------------------------------------------------------------ distortions
    def displacement(self, irrep: Irrep, opd, weights=None):
        """Unit-norm Cartesian displacement field (n_atoms, 3) for ``opd`` in ``irrep``.

        ``weights`` mixes the irrep's copies (one entry per mode); default is equal weights.
        """
        self.isotropy_subgroups(irrep)           # make sure the canonical basis is in force
        opd = np.asarray(opd, float)
        opd = opd / np.linalg.norm(opd)
        w = np.ones(len(irrep.modes)) if weights is None else np.asarray(weights, float)
        w = w / np.linalg.norm(w)
        u = np.zeros((self.n_atoms, 3))
        for wi, m in zip(w, irrep.modes):
            u += wi * np.tensordot(opd, m.vectors, axes=(0, 0))
        return u

    def supercell_parent(self):
        """The undistorted parent in the supercell (atom order: primitive atom, then cell)."""
        s = self.parent
        sc_frac, sp = [], []
        Tinv = np.linalg.inv(self.Sb.T)
        for a in range(len(s)):
            for T in self.cells:
                sc_frac.append(Tinv @ (s.frac[a] + T))
                sp.append(s.species[a])
        return Structure(self.Sb @ s.lattice, np.array(sc_frac), sp)

    def child_structure(self, u, amplitude):
        """Parent supercell displaced by ``amplitude * u`` (u unit norm; amplitude in Angstrom,
        root-sum-square over the whole supercell)."""
        par = self.supercell_parent()
        disp_frac_prim = (amplitude * u) @ np.linalg.inv(self.parent.lattice)
        Tinv = np.linalg.inv(self.Sb.T)
        return Structure(par.lattice, par.frac + disp_frac_prim @ Tinv.T, par.species)

    def displacements_from(self, child: Structure):
        """Cartesian displacements of ``child`` relative to ``supercell_parent()`` (same atom order,
        same lattice basis)."""
        par = self.supercell_parent()
        d = child.frac - par.frac
        d -= np.rint(d)
        return d @ par.lattice

    def symmetry_of(self, u, tol=1e-6):
        """Indices of group operations that leave the displacement field ``u`` unchanged."""
        keep = []
        for i in range(self.n_ops):
            new = np.empty_like(u)
            new[self.perm[i]] = u @ self.Q[i].T
            if np.abs(new - u).max() < tol * max(1.0, np.abs(u).max()):
                keep.append(i)
        return keep

    def describe_ops(self, op_indices):
        """Operations of a (sub)group as 'x,y,z' strings in the supercell basis."""
        Sit = np.linalg.inv(self.Sb.T)
        out = []
        for i in op_indices:
            Rs = np.rint(Sit @ self.R[i] @ self.Sb.T).astype(int)
            out.append(op_to_string(Rs, Sit @ self.t[i]))
        return out

    # ------------------------------------------------------------------ amplitudes
    def decompose(self, u, canonical=True, canonical_min=1e-9):
        """Project a displacement field onto the modes.

        Returns ``(rows, residual)``; each row is ``(irrep, mode_index, amplitude, coefficients)``
        with amplitudes in Angstrom (root-sum-square over the supercell). ``residual`` is the
        norm of the part of ``u`` outside this wavevector's mode space.
        """
        flat = u.ravel()
        rows, rest = [], flat.copy()
        for ir in self.irreps:
            if canonical and any(np.linalg.norm(m.vectors.reshape(ir.dim, -1) @ flat) > canonical_min for m in ir.modes):
                self.isotropy_subgroups(ir)       # fixes the canonical basis so components match the OPDs
            for mi, m in enumerate(ir.modes):
                V = m.vectors.reshape(ir.dim, -1)
                c = V @ flat
                rest -= V.T @ c
                rows.append((ir, mi, float(np.linalg.norm(c)), c))
        return rows, float(np.linalg.norm(rest))
