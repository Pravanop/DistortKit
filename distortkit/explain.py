"""How much of a distortion does a symmetry explain, and which modes drive which?

Works from a saved analysis (``compare`` output, loaded with ``freeze.Freezer.load``).  The parent's space group
acts on the child's displacement field: an operation moves each atom to an equivalent parent site and rotates its
displacement vector.  Averaging the field over a subgroup H keeps exactly the part of the distortion that H
allows (the projection onto H-invariant displacements); what is left breaks H.

1. **Symmetry ladder** (``ladder``): for increasing tolerance, the largest group H of parent operations under which
   the whole displacement pattern is invariant to within that tolerance, the space group of H, the share of the
   squared displacement H explains, and the residual that breaks it.  A clean child shows its own group at the
   smallest tolerance (100% explained); a relaxed or noisy one shows a symmetric core plus a small remainder, and
   the remainder is attributed to the modes that carry it.

2. **Mode roles** (``cascade``): the leading mode is the *primary* one.  The operations that leave it unchanged form
   its isotropy subgroup H1; every mode whose displacement is H1-invariant is *allowed* by the primary (symmetry
   permits it to appear and to couple to it).  A mode that H1 does not allow is *independent*: it needs its own order
   parameter, which cuts the symmetry further (H2 = H1 and its isotropy subgroup), and so on.  The sequence ends
   at the group of the child.  Allowed is not forced: symmetry permits a coupling, energetics decide its size.

Operations act on the child's own supercell, so only parent operations that map the supercell lattice onto itself
(and the supercell's translations) take part: other operations would carry the child into a different domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compare import CellSet
from .structure import Structure
from .symmetry import find_symmetry, to_primitive

DEFAULT_TOLS = (1e-4, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.4)


# ====================================================================== the group acting on supercell fields
class FieldGroup:
    """Parent operations x supercell translations, acting on displacement arrays ``F[atom, cell, 3]`` (Angstrom)."""

    def __init__(self, prim: Structure, M, cells, tol=1e-3):
        self.prim = prim
        A = prim.lattice
        n = len(prim)
        self.n_par = n
        self.cs = CellSet(M)
        if not np.array_equal(self.cs.cells, np.asarray(cells)):
            raise ValueError("the saved cell order does not match the supercell matrix")
        self.cells = self.cs.cells
        self.N = len(self.cells)
        Mt = np.asarray(M, float).T
        self.ops, self.Q = [], []
        for R, t in find_symmetry(prim, tol):
            Rm = np.linalg.inv(Mt) @ R @ Mt                 # does the operation map the supercell lattice onto itself?
            if np.abs(Rm - np.rint(Rm)).max() < 1e-6:
                self.ops.append((np.asarray(R, int), np.asarray(t, float)))
                self.Q.append(A.T @ R @ np.linalg.inv(A.T))
        no = len(self.ops)
        self.n_el = no * self.N
        sp = np.array(prim.species)
        self.atom_b = np.zeros((no, n), dtype=int)
        base = np.zeros((no, n, self.N), dtype=int)
        for oi, (R, t) in enumerate(self.ops):
            for a in range(n):
                y = R @ prim.frac[a] + t
                d = y[None, :] - prim.frac
                d -= np.rint(d)
                dist = np.linalg.norm(d @ A, axis=1)
                dist[sp != sp[a]] = np.inf
                b = int(dist.argmin())
                if dist[b] > 5e-3:
                    raise ValueError("the parent's operations do not map its atoms onto each other (symmetry detection tolerance)")
                w = np.rint(y - prim.frac[b]).astype(int)
                self.atom_b[oi, a] = b
                base[oi, a] = self.cs.canon(self.cells @ R.T + w)
        self.cell_base = base
        self.ADD = np.array([self.cs.canon(self.cells + self.cells[c0]) for c0 in range(self.N)]).T   # ADD[c, c0]
        self._by_R = {}
        for oi, (R, t) in enumerate(self.ops):
            self._by_R.setdefault(tuple(R.ravel()), []).append(oi)
        zero = self.cs.canon(np.zeros((1, 3), dtype=int))[0]
        self.identity = next(oi for oi, (R, t) in enumerate(self.ops)
                             if np.array_equal(R, np.eye(3, dtype=int)) and np.abs(t).max() < 1e-9) * self.N + int(zero)
        self._mult = {}

    # -------------------------------------------------------------- elements
    def split(self, e):
        return divmod(int(e), self.N)

    def act(self, F, e):
        oi, c0 = self.split(e)
        out = np.empty_like(F)
        cidx = self.ADD[self.cell_base[oi], c0]
        out[self.atom_b[oi][:, None], cidx] = F @ self.Q[oi].T
        return out

    def mult(self, e1, e2):
        key = (e1, e2)
        if key in self._mult:
            return self._mult[key]
        o1, c1 = self.split(e1)
        o2, c2 = self.split(e2)
        R1, t1 = self.ops[o1]
        R2, t2 = self.ops[o2]
        R = R1 @ R2
        tfull = R1 @ t2 + t1
        for oi in self._by_R[tuple(R.ravel())]:
            d = tfull - self.ops[oi][1]
            if np.abs(d - np.rint(d)).max() < 1e-5:
                w = np.rint(d).astype(int)
                break
        else:
            raise ValueError("operation product not found in the group (non-closed operation set)")
        Tvec = R1 @ self.cells[c2] + self.cells[c1] + w
        out = oi * self.N + int(self.cs.canon(Tvec[None, :])[0])
        self._mult[key] = out
        return out

    # -------------------------------------------------------------- fields
    def deviations(self, F):
        """For every element the largest atom mismatch |e.F - F| (Angstrom)."""
        return np.array([np.linalg.norm(self.act(F, e) - F, axis=2).max() for e in range(self.n_el)])

    def project(self, F, elements):
        acc = np.zeros_like(F)
        for e in elements:
            acc += self.act(F, e)
        return acc / len(elements)

    def generated(self, gens, ok):
        """Group generated by ``gens`` if every element satisfies ``ok(e)``, else None."""
        seen = {self.identity}
        todo = [self.identity]
        while todo:
            e = todo.pop()
            for g in gens:
                p = self.mult(e, g)
                if p in seen:
                    continue
                if not ok(p):
                    return None
                seen.add(p)
                todo.append(p)
        return seen

    def largest_group(self, dev, tol, within=None):
        """Largest group of elements whose mismatch ``dev`` is at most ``tol`` (and which lie in ``within``), built
        greedily from the best-fitting operations."""
        ok = lambda e: dev[e] <= tol + 1e-12 and (within is None or e in within)
        gens, group = [], {self.identity}
        for e in np.argsort(dev, kind="stable"):
            e = int(e)
            if dev[e] > tol + 1e-12:
                break
            if e in group or not ok(e):
                continue
            trial = self.generated(gens + [e], ok)
            if trial is not None:
                gens.append(e)
                group = trial
        return group

    # -------------------------------------------------------------- symmetry of a group
    def space_group(self, elements, M, seed=0):
        """Space group of the displacement patterns that ``elements`` allow: identified from a generic invariant field."""
        from .spacegroup import identify
        rng = np.random.default_rng(seed)
        F = self.project(rng.normal(size=(self.n_par, self.N, 3)), sorted(elements))
        top = np.linalg.norm(F, axis=2).max()
        F = F * (0.2 / top) if top > 1e-9 else np.zeros_like(F)         # no invariant displacement: the parent itself
        A = self.prim.lattice
        P = np.asarray(M, float) @ A
        Xs = np.array([(self.prim.frac[a] + T) @ np.linalg.inv(np.asarray(M, float)) for a in range(self.n_par) for T in self.cells])
        sp = [self.prim.species[a] for a in range(self.n_par) for _ in self.cells]
        S = Structure(P, (Xs + F.reshape(-1, 3) @ np.linalg.inv(P)) % 1.0, sp)
        try:
            prim, _ = to_primitive(S, 1e-5)
            return identify(find_symmetry(prim, 1e-5), prim.lattice)
        except ValueError:
            return None


def _norm2(F):
    return float(np.sum(F ** 2))


# ====================================================================== 1. ladder
@dataclass
class LadderRung:
    tol: float
    n_el: int                  # operations of the group, counting supercell translations
    sg: object                 # spacegroup.SpaceGroupResult of the group (None if not identified)
    explained: float           # share of the squared displacement the group allows (0..1)
    resid_rss: float           # displacement that breaks the group, root-sum-square (A)
    resid_rms: float           # per atom (A)
    breaks: list = field(default_factory=list)   # [(name, amplitude of the mode, part that breaks the group)]
    group: frozenset = None


def ladder(fz, tols=DEFAULT_TOLS, G=None, min_amp=1e-3):
    """The symmetry of the displacement pattern as the tolerance grows; one entry per distinct group."""
    G = G or FieldGroup(fz.parent, fz.M, fz.cells)
    u = fz.total
    dev = G.deviations(u)
    total2 = max(_norm2(u), 1e-30)
    out, seen = [], set()
    for tol in tols:
        grp = frozenset(G.largest_group(dev, tol))
        if grp in seen:
            continue
        seen.add(grp)
        els = sorted(grp)
        pu = G.project(u, els)
        r2 = _norm2(u - pu)
        sg = G.space_group(grp, fz.M)
        breaks = []
        for m in fz.modes:
            b = _norm2(m.field - G.project(m.field, els))
            if b >= min_amp ** 2:
                breaks.append((m.id, m.text, float(np.sqrt(_norm2(m.field))), float(np.sqrt(b))))
        rb = _norm2(fz.rest - G.project(fz.rest, els))
        if rb >= min_amp ** 2:
            breaks.append(("rest", "rest", float(np.sqrt(_norm2(fz.rest))), float(np.sqrt(rb))))
        breaks.sort(key=lambda x: -x[3])
        out.append(LadderRung(tol, len(grp), sg, 1.0 - r2 / total2, float(np.sqrt(r2)),
                              float(np.sqrt(r2 / (G.n_par * G.N))), breaks, grp))
    return out


# ====================================================================== 2. cascade
@dataclass
class Role:
    id: str
    name: str
    amplitude: float
    role: str                  # fully-symmetric | primary | allowed | independent
    level: int                 # 0 for fully symmetric modes; k for the level where the role is decided
    invariant: float           # share of the mode's squared amplitude the previous group allows (the group it was judged against)
    by: list = field(default_factory=list)      # ids of the drivers that allow it


@dataclass
class Level:
    index: int
    driver: str
    driver_name: str
    n_el: int
    sg: object
    allowed: list = field(default_factory=list)   # Role of the modes that become allowed at this level


@dataclass
class Cascade:
    roles: dict
    levels: list
    child_n_el: int
    child_sg: object
    final_n_el: int
    notes: list = field(default_factory=list)


def invariant_share(G, F, els):
    n = _norm2(F)
    return 1.0 if n < 1e-24 else _norm2(G.project(F, els)) / n


def cascade(fz, primary=None, allowed_share=0.95, opd_tol=0.05, min_amp=1e-3, G=None):
    """Roles of the modes: primary, allowed by it, or independent; see the module docstring.

    ``primary``: ids/labels of modes to use as drivers, in order (default: the largest mode that the current group does
    not allow, repeatedly).  ``allowed_share``: share of a mode's squared amplitude that a group must allow for the
    mode to count as allowed by it.  ``opd_tol``: how far (fraction of the largest atomic displacement of the mode)
    an operation may be from leaving the mode unchanged to count as an isotropy operation.
    """
    G = G or FieldGroup(fz.parent, fz.M, fz.cells)
    modes = sorted([m for m in fz.modes if m.amplitude >= min_amp], key=lambda m: -m.amplitude)
    allg = set(range(G.n_el))
    roles, levels = {}, []
    H = allg
    remaining = []
    for m in modes:
        if invariant_share(G, m.field, sorted(allg)) >= allowed_share and _norm2(G.project(m.field, sorted(allg))) > 0:
            roles[m.id] = Role(m.id, m.text, m.amplitude, "fully-symmetric", 0, 1.0)
        else:
            remaining.append(m)
    queue = [fz.mode_by_ref(p) for p in (primary or [])]
    queue = [m for m in queue if m in remaining]
    drivers = []
    while remaining:
        before = sorted(H)
        if queue:
            d = queue.pop(0)
        else:
            d = next((m for m in remaining if invariant_share(G, m.field, before) < allowed_share), None)
            if d is None:
                break                                              # everything left is allowed by the current group
        remaining.remove(d)
        dev = G.deviations(d.field)
        stab = G.largest_group(dev, opd_tol * float(np.linalg.norm(d.field, axis=2).max()), within=H)
        H = set(H) & set(stab)
        els = sorted(H)
        lvl = Level(len(levels) + 1, d.id, d.text, len(H), G.space_group(H, fz.M))
        roles[d.id] = Role(d.id, d.text, d.amplitude, "primary" if lvl.index == 1 else "independent", lvl.index,
                           invariant_share(G, d.field, before), by=[x.id for x in drivers])
        drivers.append(d)
        for m in list(remaining):
            inv = invariant_share(G, m.field, els)
            if inv >= allowed_share:
                r = Role(m.id, m.text, m.amplitude, "allowed", lvl.index, inv, by=[x.id for x in drivers])
                roles[m.id] = r
                lvl.allowed.append(r)
                remaining.remove(m)
        levels.append(lvl)
    # what the final group does not allow: the unlisted remainder and the child's own group
    notes = []
    dev_u = G.deviations(fz.total)
    K = G.largest_group(dev_u, 1e-3 * max(float(np.linalg.norm(fz.total, axis=2).max()), 1e-6))
    ksg = G.space_group(K, fz.M)
    rb = _norm2(fz.rest - G.project(fz.rest, sorted(H))) if len(fz.rest) else 0.0
    if rb > 1e-8:
        notes.append(f"the unlisted remainder (rest) carries {np.sqrt(rb):.4f} A that even the final group does not allow")
    if len(H) > len(K):
        notes.append(f"the listed modes leave {len(H)} operations but the child keeps only {len(K)}: the rest/strain/small "
                     "modes lower the symmetry further")
    return Cascade(roles, levels, len(K), ksg, len(H), notes)


# ====================================================================== text
def sg_text(sg):
    return sg.label() if sg is not None else "(space group not identified)"


def report(fz, tols=DEFAULT_TOLS, primary=None, allowed_share=0.95, opd_tol=0.05, min_amp=0.02):
    G = FieldGroup(fz.parent, fz.M, fz.cells)
    pg = fz.doc.get("parent_space_group")
    L = [f"parent {pg['symbol']} (#{pg['number']}); the child's {fz.N}-cell supercell is invariant under {len(G.ops)} of its "
         f"operations (x {G.N} translations = {G.n_el} elements acting on the displacement field)", ""]
    rungs = ladder(fz, tols, G, min_amp)
    L.append("1. HOW MUCH DOES EACH SYMMETRY EXPLAIN?  (largest group the whole displacement pattern obeys to within a tolerance)")
    L.append(f"   {'tolerance':>10s}  {'group of the pattern':<24s}{'elements':>9s}{'explains':>10s}{'breaks it (rms/atom)':>22s}")
    for r in rungs:
        L.append(f"   {r.tol:>8g} A  {sg_text(r.sg):<24s}{r.n_el:>9d}{100 * r.explained:>9.2f}%{r.resid_rms:>14.4f} A  ({r.resid_rss:.4f} A total)")
    for r in rungs:
        if r.resid_rss > 1e-4 and r.breaks:
            top = "; ".join(f"{b[1]} ({b[0]}): {b[3]:.4f} of {b[2]:.4f} A" for b in r.breaks[:4])
            L.append(f"   breaking {sg_text(r.sg)} at {r.tol:g} A: {top}")
    if len(rungs) == 1:
        L.append(f"   the pattern has {sg_text(rungs[0].sg)} symmetry at every tolerance: nothing further breaks it")
    L.append("")
    c = cascade(fz, primary, allowed_share, opd_tol, min_amp, G)
    L.append(f"2. WHICH MODES DRIVE WHICH?  (isotropy-subgroup cascade of the {sum(1 for m in fz.modes if m.amplitude >= min_amp)} mode(s) "
             f"above {min_amp:g} A; 'allowed' = permitted by the symmetry kept so far; an operation counts as keeping a mode "
             f"if it moves no atom of it by more than {100 * opd_tol:g}% of its largest displacement)")
    fs = [r for r in c.roles.values() if r.role == "fully-symmetric"]
    for r in fs:
        L.append(f"   {r.id} {r.name}: {r.amplitude:.4f} A  fully symmetric (allowed at any symmetry, e.g. breathing/isotropic strain-like)")
    for lv in c.levels:
        r = c.roles[lv.driver]
        head = "PRIMARY" if lv.index == 1 else f"INDEPENDENT (not allowed by {' + '.join(c.roles[x].name for x in r.by)} alone)"
        L.append(f"   level {lv.index}: {lv.driver} {lv.driver_name} ({r.amplitude:.4f} A) is {head}")
        L.append(f"      symmetry kept: {sg_text(lv.sg)}  ({lv.n_el} elements)")
        if lv.allowed:
            who = " + ".join(f"{c.roles[x].name}" for x in lv.allowed[0].by)
            L.append(f"      allowed once {who} is frozen: " + ", ".join(f"{x.id} {x.name} ({x.amplitude:.4f} A)" for x in lv.allowed))
        else:
            L.append("      no further listed mode becomes allowed at this level")
    L.append(f"   child: {sg_text(c.child_sg)} ({c.child_n_el} elements)")
    for n in c.notes:
        L.append(f"   note: {n}")
    L.append("   (allowed is not forced: the symmetry permits a coupling, the energetics decide how large it is)")
    return "\n".join(L), rungs, c


def to_dict(fz, rungs, casc):
    """JSON-friendly summary of ``report``'s two tables."""
    return {
        "ladder": [{"tolerance": r.tol, "elements": r.n_el, "space_group": None if r.sg is None else {"number": r.sg.number, "symbol": r.sg.symbol},
                    "explained": r.explained, "residual_rss": r.resid_rss, "residual_rms_per_atom": r.resid_rms,
                    "breaking_modes": [{"id": b[0], "name": b[1], "amplitude": b[2], "breaking_part": b[3]} for b in r.breaks]}
                   for r in rungs],
        "roles": [{"id": r.id, "name": r.name, "amplitude": r.amplitude, "role": r.role, "level": r.level,
                   "allowed_share_before": r.invariant, "allowed_by": r.by} for r in casc.roles.values()],
        "cascade": [{"level": lv.index, "driver": lv.driver, "elements": lv.n_el,
                     "space_group": None if lv.sg is None else {"number": lv.sg.number, "symbol": lv.sg.symbol},
                     "allowed": [x.id for x in lv.allowed]} for lv in casc.levels],
        "child": {"elements": casc.child_n_el, "space_group": None if casc.child_sg is None else
                  {"number": casc.child_sg.number, "symbol": casc.child_sg.symbol}},
        "notes": casc.notes,
    }
