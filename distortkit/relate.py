"""How are two structures related?  Diagnose it, and say what to do next.

``relate(a, b)`` answers, for two structures in any setting:

* are they the same composition and is one a supercell-multiple of the other (atom counts)?
* could A be the parent of B, or B of A?  Each direction is tried with progressively looser tolerances; the
  first that lays the child onto the parent says how much strain and displacement that needed, and the child's
  symmetry operations are then checked one by one against the parent's (``subgroup.child_ops_in_parent``);
* if neither is the parent of the other: why not (composition, cell, atoms too far apart, extra symmetry), and the
  next step (a common parent; see ``common_parent`` further down).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .compare import AlignmentError, align_checked
from .structure import Structure
from .subgroup import OpsCheck, child_ops_in_parent
from .symmetry import find_symmetry, to_primitive

# (strain_tol, match_tol): tried in order; the first level that works is reported
LEVELS = ((0.12, 1.0), (0.25, 1.6), (0.45, 2.4))


# ====================================================================== one structure
def reduced_formula(species):
    cnt = Counter(species)
    g = 0
    for v in cnt.values():
        g = math.gcd(g, v)
    return {k: v // g for k, v in sorted(cnt.items())}, g


def formula_text(species):
    red, g = reduced_formula(species)
    return "".join(f"{k}{v if v > 1 else ''}" for k, v in red.items()) + (f" (x{g})" if g > 1 else "")


@dataclass
class Side:
    name: str
    structure: Structure
    prim: Structure
    ops: list
    sg: object = None               # spacegroup.SpaceGroupResult or None

    @property
    def n_prim(self):
        return len(self.prim)

    @property
    def sg_text(self):
        return self.sg.label() if self.sg is not None else f"{len(self.ops)} operations (space group not identified)"


def make_side(name, structure, tol):
    prim, _ = to_primitive(structure, tol)
    ops = find_symmetry(prim, tol)
    sg = None
    try:
        from .spacegroup import identify
        sg = identify(ops, prim.lattice)
    except ValueError:
        pass
    return Side(name, structure, prim, ops, sg)


# ====================================================================== one direction
@dataclass
class Direction:
    parent: str
    child: str
    status: str                     # subgroup | not-subgroup | count | composition | volume | no-cell | no-match
    message: str = ""
    n_cells: int = None             # child = n_cells parent primitive cells
    level: tuple = None             # (strain_tol, match_tol) at which the alignment first worked
    rms: float = None
    max: float = None
    strain: float = None
    M: np.ndarray = None
    ops: OpsCheck = None
    child_sg: object = None

    @property
    def ok(self):
        return self.status == "subgroup"


def test_direction(par: Side, ch: Side, tol=1e-3, sym_tol=0.02, levels=LEVELS):
    """Can ``par`` be the parent of ``ch``?  (cheap: no mode decomposition)"""
    d = Direction(par.name, ch.name, "count")
    n = par.n_prim
    if ch.n_prim % n:
        d.message = (f"{ch.name}'s primitive cell has {ch.n_prim} atoms, {par.name}'s has {n}: "
                     f"not an integer multiple, so {ch.name} cannot be a supercell distortion of {par.name}")
        return d
    N = ch.n_prim // n
    d.n_cells = N
    if sorted(ch.prim.species) != sorted(par.prim.species * N):
        d.status = "composition"
        d.message = f"compositions differ ({formula_text(par.prim.species)} vs {formula_text(ch.prim.species)}); " \
                    "vacancies and substitutions are not supported"
        return d
    child = Structure(ch.prim.lattice.copy(), ch.prim.frac.copy(), list(ch.prim.species))
    if np.linalg.det(child.lattice) < 0:
        child.lattice[2] *= -1
        child.frac[:, 2] *= -1
    ratio = child.volume / par.prim.volume
    if abs(ratio - N) > 0.6 * N:
        d.status = "volume"
        d.message = f"{ch.name}'s cell is {ratio:.2f} x {par.name}'s but holds {N} cells' worth of atoms"
        return d
    last, fallback = None, None
    for lvl in levels:
        try:
            al, chk, note = align_checked(par.prim, par.ops, child, N, strain_tol=lvl[0], match_tol=lvl[1], sym_tol=sym_tol)
        except AlignmentError as e:
            last = e
            continue
        M, eps, cs, P, (shift, perm, d_frac, rms, dmax) = al
        cur = Direction(par.name, ch.name, "subgroup" if chk.ok else "not-subgroup", n_cells=N, level=lvl, rms=rms, max=dmax,
                        strain=float(np.linalg.norm(eps)), M=M, ops=chk, message=note)
        if chk.ok:
            return cur
        fallback = fallback or cur          # keep the first (tightest) failure for the report, look for a looser overlay that works
    if fallback is not None:
        return fallback
    d.status = "no-cell" if last.stage == "cell" else "no-match"
    big = levels[-1]
    d.message = (f"no integer supercell of {par.name} has {ch.name}'s lattice within {100 * big[0]:.0f}% strain" if last.stage == "cell"
                 else f"the atoms of {ch.name} cannot be paired with {par.name}'s within {big[1]:.1f} A "
                      f"(the cell fits, so the atomic arrangements differ)")
    return d


# ====================================================================== result
@dataclass
class Relation:
    a: Side
    b: Side
    ab: Direction                  # a as the parent of b
    ba: Direction                  # b as the parent of a
    verdict: str = ""              # a>b | b>a | isotypic | same | related-not-nested | unrelated
    notes: list = field(default_factory=list)
    common: object = None          # CommonParent search result (see ``common_parent``), when run

    def report(self):
        L = []
        for s in (self.a, self.b):
            L.append(f"{s.name}: {formula_text(s.structure.species)}, {len(s.structure)} atoms in the given cell, "
                     f"{s.n_prim} in the primitive cell; {s.sg_text}")
        L.append("")
        for d in (self.ab, self.ba):
            L.append(f"{d.parent} as parent of {d.child}: " + _dir_text(d))
            if d.ops is not None:
                for kind, text in d.ops.extra[:4]:
                    L.append(f"    not in the parent [{kind}]: {text}")
        L.append("")
        L.append("verdict: " + VERDICTS[self.verdict].format(a=self.a.name, b=self.b.name))
        for n in self.notes:
            L.append("  " + n)
        if self.common is not None:
            L.append("")
            L += self.common.report_lines()
        return "\n".join(L)


VERDICTS = {
    "a>b": "{b} is a distortion of {a}: {a} can be the parent (a group-subgroup relation); run `compare {a} {b}` for the modes.",
    "b>a": "{a} is a distortion of {b}: {b} can be the parent (a group-subgroup relation); run `compare {b} {a}` for the modes.",
    "isotypic": "{a} and {b} have the same space group and the same atoms per cell: they differ only by strain and/or "
                "displacements that keep the symmetry (no symmetry-lowering modes).",
    "same": "{a} and {b} are the same structure within tolerance.",
    "unrelated": "no group-subgroup relation either way.",
    "siblings": "no group-subgroup relation between {a} and {b}, but both are distortions of a common parent (below).",
}


def _dir_text(d):
    if d.status == "subgroup":
        s = (f"yes. {d.n_cells} parent cell(s); rms displacement {d.rms:.3f} A, max {d.max:.3f} A, strain {d.strain:.3f}; "
             + d.ops.summary())
        if d.level != LEVELS[0]:
            s += f"  [needed strain up to {100 * d.level[0]:.0f}% / displacement up to {d.level[1]:.1f} A]"
        return s
    if d.status == "not-subgroup":
        return (f"no. The structures overlay ({d.n_cells} parent cell(s); rms {d.rms:.3f} A, max {d.max:.3f} A, "
                f"strain {d.strain:.3f}) but {d.ops.summary()}")
    return f"no. {d.message}"


def relate(a: Structure, b: Structure, names=("A", "B"), tol=1e-3, sym_tol=0.02, levels=LEVELS, common=False,
           **common_kwargs) -> Relation:
    """Diagnose how ``a`` and ``b`` are related.  With ``common=True`` and no nesting, also look for a common parent
    (see ``common_parent``; its keyword arguments are accepted here)."""
    rel = _relate(a, b, names, tol, sym_tol, levels)
    if common:
        if rel.verdict in ("a>b", "b>a", "same", "isotypic"):
            rel.notes.append("already nested (one is a distortion of the other): no separate common parent is needed; "
                             "the parent in the verdict is the reference")
        else:
            cp = common_parent(a, b, names, tol=tol, sym_tol=sym_tol, levels=levels, **common_kwargs)
            rel.common = cp
            if cp.chosen is not None:
                rel.verdict = "siblings"
                rel.notes = [n for n in rel.notes if not n.startswith("next:")]
    return rel


def _relate(a, b, names, tol, sym_tol, levels):
    sa, sb = make_side(names[0], a, tol), make_side(names[1], b, tol)
    ab = test_direction(sa, sb, tol, sym_tol, levels)
    ba = test_direction(sb, sa, tol, sym_tol, levels)
    rel = Relation(sa, sb, ab, ba)
    # composition first: nothing else makes sense if the formulas differ
    if reduced_formula(sa.prim.species)[0] != reduced_formula(sb.prim.species)[0]:
        rel.verdict = "unrelated"
        rel.notes.append(f"different compositions ({formula_text(sa.structure.species)} vs {formula_text(sb.structure.species)}): "
                         "this tool compares structures with the same atoms (no vacancies or substitutions)")
        return rel
    equal_cells = sa.n_prim == sb.n_prim
    if ab.ok and ba.ok and equal_cells:
        rel.verdict = "same" if max(ab.max, ba.max) < 1e-3 and max(ab.strain, ba.strain) < 1e-3 else "isotypic"
        return rel
    if ab.ok and not (ba.ok and equal_cells):
        rel.verdict = "a>b"
    elif ba.ok:
        rel.verdict = "b>a"
    else:
        rel.verdict = "unrelated"
        rel.notes += _why_not(rel)
    return rel


def _why_not(rel):
    n = []
    sa, sb = rel.a, rel.b
    ds = {d.status for d in (rel.ab, rel.ba)}
    if ds <= {"count"}:
        g = math.gcd(sa.n_prim, sb.n_prim)
        n.append(f"the primitive cells hold {sa.n_prim} and {sb.n_prim} atoms: a common parent would hold a divisor of both "
                 f"(at most {g} atoms)")
    if "not-subgroup" in ds:
        for d in (rel.ab, rel.ba):
            if d.status == "not-subgroup" and any(k == "rotation" for k, _ in d.ops.extra):
                n.append(f"{d.child} has symmetry that {d.parent} lacks: the true parent has more symmetry than {d.parent}; "
                         "look for a common parent")
                break
        else:
            n.append("the operations agree up to a translation: check that both are in the same origin/setting, or that the "
                     "glides and screws really differ")
    if ds & {"no-cell", "no-match"}:
        n.append("the cells or atomic positions cannot be overlaid even with large strain and displacement: if both "
                 "are distortions of some higher-symmetry structure, that structure is the starting point (common parent)")
    n.append("next: `relate --common` searches for a high-symmetry structure that both are distortions of, "
             "or pass one with --parent")
    return n


# ====================================================================== common parent (siblings)
@dataclass
class Candidate:
    source: str                     # where this parent came from ("file", "pseudo-symmetry of A at 0.30 A", ...)
    parent: Side
    da: Direction = None            # parent as the parent of A
    db: Direction = None
    tol: float = None

    @property
    def ok(self):
        return self.da is not None and self.db is not None and self.da.ok and self.db.ok

    @property
    def cost(self):
        return max(self.da.rms, self.db.rms) if self.ok else float("inf")


@dataclass
class ModeRow:
    k: tuple
    label: str
    dim: int
    amp_a: float
    amp_b: float

    @property
    def where(self):
        if self.amp_a > 1e-9 and self.amp_b > 1e-9:
            return "both"
        return "only A" if self.amp_a > 1e-9 else "only B"


@dataclass
class CommonParent:
    names: tuple
    candidates: list                # all candidates tried, valid ones first (cheapest first)
    ladders: dict = field(default_factory=dict)   # name -> [pseudo.Rung]
    chosen: Candidate = None
    comp_a: object = None           # compare.Comparison of A relative to the chosen parent
    comp_b: object = None
    rows: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def report_lines(self, min_amp=1e-3):
        a, b = self.names
        L = ["common parent search"]
        for nm, rungs in self.ladders.items():
            steps = [f"{r.tol:g} A: {r.sg.label()}" for r in rungs if r.new]
            L.append(f"  pseudo-symmetry of {nm}: " + ("; ".join(steps) if steps else "no higher symmetry up to "
                                                       f"{rungs[-1].tol:g} A" if rungs else "not searched"))
        valid = [c for c in self.candidates if c.ok]
        for c in self.candidates:
            tag = f"{c.parent.sg_text}, {c.parent.n_prim} atoms/cell ({c.source})"
            if c.ok:
                L.append(f"  candidate {tag}: both fit; rms displacement {a} {c.da.rms:.3f} A (x{c.da.n_cells} cells), "
                         f"{b} {c.db.rms:.3f} A (x{c.db.n_cells} cells)")
            else:
                why = []
                for nm, d in ((a, c.da), (b, c.db)):
                    if d is not None and not d.ok:
                        why.append(f"{nm}: " + ("its operations are not the parent's" if d.status == "not-subgroup" else d.message or d.status))
                L.append(f"  candidate {tag}: rejected ({'; '.join(why) or 'not tested'})")
        if not self.candidates:
            L.append("  no candidate parents (nothing found by the pseudo-symmetry search; pass --parent FILE with a "
                     "higher-symmetry structure, e.g. the aristotype of the family)")
        for n in self.notes:
            L.append("  " + n)
        if self.chosen is None:
            return L
        L.append("")
        L.append(f"parent used: {self.chosen.parent.sg_text}, {self.chosen.parent.n_prim} atoms per primitive cell "
                 f"({self.chosen.source}); {len(valid)} valid candidate(s)")
        if self.comp_a is not None:
            L.append("")
            L += self.mode_table(min_amp)
        return L

    def mode_table(self, min_amp=1e-3):
        a, b = self.names
        ca, cb = self.comp_a, self.comp_b
        L = [f"modes relative to the common parent (amplitude = root-sum-square over each structure's own supercell, A):",
             f"  {'wavevector':<16s}{'irrep':<14s}{'dim':>4s}{a[:14]:>16s}{b[:14]:>16s}   present in"]
        L.append(f"  {'cell (parent cells)':<34s}{ca.n_cells:>16d}{cb.n_cells:>16d}")
        L.append(f"  {'space group':<34s}{(ca.child_sg.label() if ca.child_sg else '?')[:15]:>16s}{(cb.child_sg.label() if cb.child_sg else '?')[:15]:>16s}")
        L.append(f"  {'strain magnitude':<34s}{float(np.linalg.norm(ca.strain)):>16.4f}{float(np.linalg.norm(cb.strain)):>16.4f}")
        for r in self.rows:
            if max(r.amp_a, r.amp_b) < min_amp:
                continue
            k = "(" + ",".join(str(x) for x in r.k) + ")"
            where = {"both": "both", "only A": f"only {a}", "only B": f"only {b}"}[r.where]
            L.append(f"  {k:<16s}{r.label:<14s}{r.dim:>4d}{r.amp_a:>16.4f}{r.amp_b:>16.4f}   {where}")
        both = sum(1 for r in self.rows if r.where == "both" and max(r.amp_a, r.amp_b) >= min_amp)
        only_a = sum(1 for r in self.rows if r.where == "only A" and r.amp_a >= min_amp)
        only_b = sum(1 for r in self.rows if r.where == "only B" and r.amp_b >= min_amp)
        L.append(f"  {both} irrep(s) carry displacement in both, {only_a} only in {a}, {only_b} only in {b}")
        L.append(f"  total displacement from the parent: {a} {ca.total_displacement:.4f} A, {b} {cb.total_displacement:.4f} A")
        L.append("  (the same irrep present in both can still point in different directions, i.e. different domains or "
                 "isotropy subgroups; the +/- and numbering of labels depend on the origin of the parent)")
        return L


def _dedupe(cands):
    best = {}
    for c in cands:
        key = (c.parent.sg.number if c.parent.sg else id(c), c.parent.n_prim)
        if key not in best or c.cost < best[key].cost:
            best[key] = c
    return list(best.values())


def _rank(valid):
    """Best first: among candidates needing about the same displacement (within 25% + 0.02 A of the cheapest) the one
    with the most symmetry (the aristotype) comes first; then the rest by displacement."""
    if not valid:
        return valid
    lo = min(c.cost for c in valid)
    near = [c for c in valid if c.cost <= 1.25 * lo + 0.02]
    near.sort(key=lambda c: (-len(c.parent.ops), c.cost))
    return near + sorted((c for c in valid if c not in near), key=lambda c: c.cost)


def _label_of(m):
    return m.label.text if m.label is not None and m.label.candidates else f"irrep{m.irrep.index}"


def mode_rows(ca, cb):
    """Side-by-side table of the modes of two comparisons made against the same parent (``canonical_star=True``)."""
    table = {}
    for slot, comp in enumerate((ca, cb)):
        for m in comp.modes:
            key = (m.k, m.irrep.index)
            row = table.setdefault(key, [m.k, _label_of(m), m.irrep.dim, 0.0, 0.0])
            row[3 + slot] += m.amplitude
            if m.label is not None and m.label.candidates and row[1].startswith("irrep"):
                row[1] = _label_of(m)
    rows = [ModeRow(r[0], r[1], r[2], r[3], r[4]) for r in table.values()]
    rows.sort(key=lambda r: -max(r.amp_a, r.amp_b))
    return rows


def common_parent(a: Structure, b: Structure, names=("A", "B"), *, parents=(), search=True, ladder_tols=None, scale=None,
                  tol=1e-3, sym_tol=0.02, modes=True, pick=None, compare_kwargs=None, levels=LEVELS) -> CommonParent:
    """Find a structure that both ``a`` and ``b`` are group-subgroup distortions of, and compare their modes.

    Candidate parents are ``parents`` (list of ``(name, Structure)``) plus, when ``search`` is on, the idealised
    pseudo-symmetric structures of ``a`` and of ``b`` (``pseudo.ladder``).  A candidate is valid when each of the two
    structures is a subgroup-distortion of it (every one of its symmetry operations is the candidate's, see
    ``subgroup``).  The cheapest valid candidate (smallest displacement) is used unless ``pick`` (1-based) selects
    another; with ``modes`` both structures are decomposed relative to it and tabulated side by side.
    """
    from . import pseudo
    from .compare import compare
    sa, sb = make_side(names[0], a, tol), make_side(names[1], b, tol)
    cands, ladders = [], {}
    for nm, st in parents:
        cands.append(Candidate(f"file {nm}", make_side(nm, st, tol)))
    if search:
        tols = tuple(ladder_tols) if ladder_tols else pseudo.DEFAULT_LADDER
        for side in (sa, sb):
            rungs = pseudo.ladder(side.prim, tols, scale=scale)
            ladders[side.name] = rungs
            for r in rungs:
                if r.new:
                    cands.append(Candidate(f"pseudo-symmetry of {side.name} at {r.tol:g} A",
                                           make_side(f"{r.sg.symbol} from {side.name}", r.ideal, 1e-4), tol=r.tol))
    for c in cands:
        c.da = test_direction(c.parent, sa, tol, sym_tol, levels)
        c.db = test_direction(c.parent, sb, tol, sym_tol, levels) if c.da.ok else None
    valid = _rank(_dedupe([c for c in cands if c.ok]))
    others = [c for c in cands if not c.ok]
    cp = CommonParent(tuple(names), valid + others, ladders)
    if not valid:
        return cp
    if pick is not None:
        if not 1 <= pick <= len(valid):
            raise ValueError(f"--pick {pick}: there are {len(valid)} valid candidate(s)")
        cp.chosen = valid[pick - 1]
    else:
        cp.chosen = valid[0]
        if len(valid) > 1:
            cp.notes.append(f"{len(valid)} valid parents: using the highest-symmetry one among those needing about the "
                            "smallest displacement (--pick N chooses another, in the order listed)")
    if modes:
        kw = dict(compare_kwargs or {})
        for slot, (st, d) in enumerate(((a, cp.chosen.da), (b, cp.chosen.db))):
            k2 = dict(kw)
            k2.setdefault("strain_tol", d.level[0])
            k2.setdefault("match_tol", d.level[1])
            res = compare(cp.chosen.parent.structure, st, tol=tol, sym_tol=sym_tol, canonical_star=True, **k2)
            if slot == 0:
                cp.comp_a = res
            else:
                cp.comp_b = res
        cp.rows = mode_rows(cp.comp_a, cp.comp_b)
    return cp
