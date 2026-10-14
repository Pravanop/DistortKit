"""Build structures along the distortion by freezing selected modes and/or atoms.

``compare`` saves an analysis file; this module reads it and builds structures from selections written in the
report's own vocabulary:

* **modes**: a label (``GM2-``), an id (``m1``), ``all`` (every listed mode), ``rest`` (displacement not in any listed
  mode) or ``child`` (all + rest = the child itself);
* **atoms**: a species (``Hf``), an atom label (``Hf1``), a range (``O1-O4``), a Wyckoff site (``O@4c``, ``@4d``) or a
  site index (``s2``), comma separated;
* **scale**: ``*0.5`` (half of the amplitude found in the child, negative reverses it) or ``=0.8`` (set the
  root-sum-square amplitude of the selected part to 0.8 A).

A *part* is ``MODE[:ATOMS][*SCALE|=AMPLITUDE]``; the structure is the parent plus the sum of its parts.

    fz = Freezer.load("analysis.json")
    s, info = fz.build(["GM2-"])                   # freeze the polar mode only
    s, info = fz.build(["GM2-:Hf", "GM1+:O*0.5"])  # GM2- on Hf, half of GM1+ on O
    s, info = fz.build(["all"], atoms="Hf")        # certain atoms, all modes
"""
from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .structure import Structure


@dataclass
class ModeEntry:
    id: str
    name: str                     # label text such as "GM2-" (or "irrep8" when no label is known)
    candidates: list
    k: tuple
    irrep: int
    dim: int
    amplitude: float
    species: list
    subgroup: str
    leads_to: str
    field: np.ndarray             # (n_prim, n_cells, 3) displacement carried by the mode (A)

    @property
    def text(self):
        if self.candidates and len(self.candidates) > 1:
            return "/".join(self.candidates)
        return self.name


@dataclass
class AtomEntry:
    label: str
    species: str
    a: int
    c: int
    site: int
    wyckoff: str


class SelectionError(ValueError):
    pass


_PART = re.compile(r"^(?P<mode>[^:*=\[]+?)\s*(?::\s*(?P<atoms>[^*=]*?))?\s*(?:(?P<op>[*=])\s*(?P<num>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?\s*$")


class Freezer:
    def __init__(self, doc):
        self.doc = doc
        p = doc["parent"]
        self.parent = Structure(p["lattice"], p["frac"], p["species"])
        self.M = np.array(doc["supercell_matrix"], dtype=int)
        self.N = int(doc["n_cells"])
        self.cells = np.array(doc["cells"], dtype=int)
        self.strain = np.array(doc["strain"])
        n_prim = len(self.parent)
        self.sites = doc["sites"]
        site_of = {a: s for s in self.sites for a in s["atoms"]}
        self.atoms = []
        for a in range(n_prim):
            for c in range(self.N):
                s = site_of.get(a)
                self.atoms.append(AtomEntry(doc["atom_labels"][a][c], self.parent.species[a], a, c,
                                            s["index"] if s else -1, s["wyckoff"] if s else "?"))
        self.modes = []
        for m in doc["modes"]:
            leads = m.get("leads_to")
            self.modes.append(ModeEntry(
                m["id"], m["name"], m.get("candidates", []), tuple(m["k"]), m["irrep"], m["dim"], m["amplitude"],
                m["species"], f"{m['subgroup']} {m['subgroup_opd']}" if m.get("subgroup") else "",
                f"{leads['symbol']} (#{leads['number']})" if leads else "", np.array(m["field"])))
        self.rest = np.array(doc["rest"])
        self.total = np.array(doc["total"])

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path):
        with open(path) as fh:
            doc = json.load(fh)
        if doc.get("format") not in ("distortkit-analysis", "isodistort_py-analysis"):
            raise ValueError(f"{path} is not an distortkit analysis file (produced by `compare`)")
        return cls(doc)

    # ------------------------------------------------------------------ legend
    def legend(self):
        d = self.doc
        pg = d.get("parent_space_group")
        cg = d.get("child_space_group")
        L = [f"analysis of {d['meta'].get('parent', 'parent')} -> {d['meta'].get('child', 'child')}"
             f"   (parent {pg['symbol'] + ' #' + str(pg['number']) if pg else '?'}, "
             f"child {cg['symbol'] + ' #' + str(cg['number']) if cg else '?'}, {self.N} parent cell(s))", "",
             "MODES   (select with the label or the id)"]
        for m in self.modes:
            k = ",".join(m.k)
            L.append(f"  {m.id:<5s}{m.text:<12s} k=({k})  {m.amplitude:8.4f} A   on {'/'.join(m.species):<8s}"
                     + (f"  -> {m.leads_to}" if m.leads_to else ""))
        L.append(f"  {'rest':<5s}{'':<12s} displacement not in any listed mode ({np.sqrt(np.sum(self.rest ** 2)):.4f} A)")
        L.append(f"  {'all':<5s}{'':<12s} every listed mode")
        L.append(f"  {'child':<5s}{'':<12s} all + rest = the child itself")
        L.append("")
        L.append("SITES   (Wyckoff sites of the parent)")
        for s in self.sites:
            labs = [x.label for x in self.atoms if x.site == s["index"]]
            span = f"atoms {labs[0]}-{labs[-1]}" if len(labs) > 1 else f"atom {labs[0]}"
            L.append(f"  s{s['index'] + 1:<3d}{s['species']:<3s} {s['wyckoff']:<4s}{s['template']:<14s} "
                     f"site symmetry {s['site_symmetry']:<5s} {span}")
        L.append("")
        L.append("ATOMS   (label: species, site, parent fractional position)")
        for x in self.atoms:
            pos = self.parent.frac[x.a] + self.cells[x.c]
            L.append(f"  {x.label:<6s}{x.species:<3s} s{x.site + 1:<2d} {x.wyckoff:<4s} ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
        ex_m = self.modes[0].id if self.modes else "m1"
        sp = self.parent.species[0]
        L += ["", "SELECTING   MODE[:ATOMS][*SCALE | =AMPLITUDE]",
              "  atoms   species (Hf)  label (Hf1)  range (O1-O4)  site (O@4c, @4d, s2)  all    comma separated",
              "  scale   *0.5 = half the amplitude found in the child (negative reverses);  =0.8 sets the part's RSS amplitude to 0.8 A",
              "", "EXAMPLES",
              f"  freeze <analysis> --modes {ex_m}                       one mode",
              f"  freeze <analysis> --modes all --atoms {sp}                  certain atoms, all modes",
              f"  freeze <analysis> --part '{ex_m}:{sp}' --part '{self.modes[-1].id if self.modes else 'm2'}:all*0.5'   per-mode atoms and scale",
              "  freeze <analysis> --each mode | --pairs | --cumulative | --combinations | --each site | --each species | --each atom",
              "  freeze <analysis> --modes all --amplitude 1.0 --n 5   5 structures from the parent (0) up to an amplitude of 1.0 A"]
        return "\n".join(L)

    # ------------------------------------------------------------------ resolving selections
    def mode_by_ref(self, ref):
        r = ref.strip()
        low = r.lower()
        hits = [m for m in self.modes if m.id.lower() == low]
        if not hits:
            hits = [m for m in self.modes if m.name.lower() == low or m.text.lower() == low]
        if not hits:
            hits = [m for m in self.modes if low in [c.lower() for c in m.candidates]]
        if not hits:
            raise SelectionError(f"unknown mode '{ref}'; available: " + ", ".join(f"{m.id} ({m.text})" for m in self.modes)
                                 + ", all, rest, child")
        if len(hits) > 1:
            raise SelectionError(f"mode '{ref}' is ambiguous ({', '.join(m.id for m in hits)}); use the id")
        return hits[0]

    def mode_field(self, ref):
        low = ref.strip().lower()
        if low == "all":
            return sum((m.field for m in self.modes), np.zeros_like(self.total))
        if low == "rest":
            return self.rest
        if low == "child":
            return self.total
        return self.mode_by_ref(ref).field

    def atom_mask(self, selector):
        """Boolean mask (n_prim, n_cells) of the atoms picked by a comma-separated selector."""
        mask = np.zeros((len(self.parent), self.N), dtype=bool)
        if selector is None or not selector.strip():
            mask[:] = True
            return mask
        by_label = {x.label: x for x in self.atoms}
        species = {x.species for x in self.atoms}
        for tok in (t.strip() for t in selector.split(",") if t.strip()):
            if tok.lower() == "all":
                mask[:] = True
            elif re.fullmatch(r"[sS]\d+", tok):
                idx = int(tok[1:]) - 1
                if idx not in {s["index"] for s in self.sites}:
                    raise SelectionError(f"no site {tok}")
                for x in self.atoms:
                    if x.site == idx:
                        mask[x.a, x.c] = True
            elif "@" in tok:
                sp, _, w = tok.partition("@")
                hit = [x for x in self.atoms if (not sp or x.species == sp) and x.wyckoff.lower() == w.lower()]
                if not hit:
                    raise SelectionError(f"no atoms on site '{tok}'; sites: " + ", ".join(sorted({f'{x.species}@{x.wyckoff}' for x in self.atoms})))
                for x in hit:
                    mask[x.a, x.c] = True
            elif tok in by_label:
                x = by_label[tok]
                mask[x.a, x.c] = True
            elif re.fullmatch(r"([A-Z][a-z]?)(\d+)-(?:[A-Z][a-z]?)?(\d+)", tok):
                sp, lo, hi = re.fullmatch(r"([A-Z][a-z]?)(\d+)-(?:[A-Z][a-z]?)?(\d+)", tok).groups()
                hit = [by_label[f"{sp}{i}"] for i in range(int(lo), int(hi) + 1) if f"{sp}{i}" in by_label]
                if not hit:
                    raise SelectionError(f"no atoms in range '{tok}'")
                for x in hit:
                    mask[x.a, x.c] = True
            elif tok in species:
                for x in self.atoms:
                    if x.species == tok:
                        mask[x.a, x.c] = True
            else:
                raise SelectionError(f"cannot understand atom selector '{tok}'; use a species ({', '.join(sorted(species))}), "
                                     f"a label (e.g. {self.atoms[0].label}), a range, a site (e.g. @{self.atoms[0].wyckoff}) or sN")
        return mask

    # ------------------------------------------------------------------ building
    def part_field(self, part, atoms=None, exclude=None):
        """Displacement field (n_prim, n_cells, 3) for one part ``MODE[:ATOMS][*S|=A]``."""
        m = _PART.match(part)
        if not m:
            raise SelectionError(f"cannot parse part '{part}' (expected MODE[:ATOMS][*SCALE|=AMPLITUDE])")
        base = self.mode_field(m.group("mode"))
        mask = self.atom_mask(m.group("atoms"))
        if atoms is not None:
            mask &= self.atom_mask(atoms)
        if exclude:
            mask &= ~self.atom_mask(exclude)
        f = base * mask[..., None]
        if m.group("op") == "*":
            f = f * float(m.group("num"))
        elif m.group("op") == "=":
            rss = float(np.sqrt(np.sum(f ** 2)))
            if rss < 1e-12:
                raise SelectionError(f"part '{part}' has no displacement to scale to an amplitude")
            f = f * (float(m.group("num")) / rss)
        return f

    def field(self, parts, atoms=None, exclude=None, scale=1.0):
        f = np.zeros_like(self.total)
        for p in parts:
            f = f + self.part_field(p, atoms, exclude)
        return f * scale

    def rss(self, parts, atoms=None, exclude=None):
        """Root-sum-square amplitude (A) of the selected displacement as found in the child (scale 1)."""
        return float(np.sqrt(np.sum(self.field(parts, atoms, exclude, 1.0) ** 2)))

    def structure(self, field, strain=0.0, title=None):
        """Parent supercell displaced by ``field``; the lattice is strained by ``strain`` times the child's strain."""
        A = self.parent.lattice
        Minv = np.linalg.inv(self.M)
        pos, sp = [], []
        for a in range(len(self.parent)):
            for c in range(self.N):
                x = self.parent.frac[a] + self.cells[c] + field[a, c] @ np.linalg.inv(A)
                pos.append(x @ Minv)
                sp.append(self.parent.species[a])
        lattice = (self.M @ A) @ (np.eye(3) + strain * self.strain)
        return Structure(lattice, np.array(pos), sp)

    def build(self, parts, atoms=None, exclude=None, scale=1.0, cell="parent", strain=None):
        """Structure with ``parts`` frozen in; returns ``(Structure, info)``.

        ``cell="parent"`` keeps the undistorted supercell lattice, ``"child"`` applies the child's strain scaled by
        ``scale`` (or by ``strain`` if given).
        """
        f = self.field(parts, atoms, exclude, scale)
        s_ = (scale if strain is None else strain) if cell == "child" else 0.0
        n = f.shape[0] * f.shape[1]
        info = {"rss": float(np.sqrt(np.sum(f ** 2))), "rms": float(np.sqrt(np.sum(f ** 2) / n)),
                "max": float(np.max(np.linalg.norm(f, axis=2))), "strain_fraction": float(s_)}
        return self.structure(f, s_), info

    # ------------------------------------------------------------------ enumerations
    def mode_refs(self, modes=None):
        if modes is None or modes == ["all"]:
            return [m.id for m in self.modes]
        out = []
        for r in modes:
            out.append(r if r.lower() in ("all", "rest", "child") else self.mode_by_ref(r).id)
        return out

    def display_name(self, ref):
        low = ref.lower()
        if low in ("all", "rest", "child"):
            return low
        return self.mode_by_ref(ref).name


# ====================================================================== writing files
def slug(text):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_")


def symmetry_of(structure, tol=1e-3):
    """(space group label, point group) of a generated structure, or ('?', '?')."""
    from .spacegroup import identify
    from .symmetry import find_symmetry, point_group_name, to_primitive
    try:
        prim, _ = to_primitive(structure, tol)
        ops = find_symmetry(prim, tol)
        sg = identify(ops, prim.lattice)
        return sg.label(), point_group_name([R for R, _ in ops], prim.lattice), sg
    except (ValueError, np.linalg.LinAlgError):
        return "P1?", "?", None


def write_structure(structure, path, fmt, title, tol=1e-3):
    """Write POSCAR (``fmt='poscar'``) or CIF with detected symmetry operations (``fmt='cif'``)."""
    if fmt == "poscar":
        with open(path, "w") as fh:
            fh.write(structure.to_poscar(title))
        return
    from .symmetrize import cif_with_symmetry
    from .symmetry import find_symmetry
    try:
        ops = find_symmetry(structure, tol)
        _, _, sg = symmetry_of(structure, tol)
        text = cif_with_symmetry(structure, ops, tol, sg.number if sg else None, sg.symbol if sg else None,
                                 title=slug(title)[:40] or "distortkit")
    except Exception:                                              # noqa: BLE001 - fall back to a plain P1 CIF
        text = structure.to_cif(slug(title)[:40] or "distortkit")
    with open(path, "w") as fh:
        fh.write(text)


@dataclass
class Recipe:
    name: str                     # short readable name used in file names
    parts: list
    atoms: str = None
    exclude: str = None
    description: str = ""


def enumerate_recipes(fz, *, modes=None, atoms=None, exclude=None, parts=None, each=None, pairs=False,
                      cumulative=False, combinations=False, max_size=None):
    """Turn the CLI-style options into a list of recipes (one structure each)."""
    refs = fz.mode_refs(modes)
    plain = [r for r in refs if r.lower() not in ("all", "rest", "child")]
    out = []
    atom_note = f" [{atoms}]" if atoms else ""

    def names(rs):
        return " + ".join(fz.display_name(r) for r in rs)

    def fname(rs):
        return "__".join(slug(fz.display_name(r)) for r in rs)

    def part_name(p):
        m = _PART.match(p)
        if not m:
            return slug(p)
        out = slug(fz.display_name(m.group("mode").strip()))
        if m.group("atoms") and m.group("atoms").strip():
            out += "_on_" + slug(m.group("atoms").strip())
        if m.group("op") == "*":
            out += "_x" + m.group("num").replace("-", "m").replace("+", "")
        elif m.group("op") == "=":
            out += "_eq" + m.group("num")
        return out

    if each:
        if each == "mode":
            for r in refs:
                out.append(Recipe(fname([r]) + (f"_on_{slug(atoms)}" if atoms else ""), [r], atoms, exclude,
                                  f"mode {fz.display_name(r)}{atom_note}"))
        else:
            if each == "site":
                groups = [(f"s{t['index'] + 1}_{t['species']}{t['wyckoff']}", f"site s{t['index'] + 1} ({t['species']} {t['wyckoff']})",
                           f"s{t['index'] + 1}") for t in fz.sites]
            elif each == "species":
                groups = [(sp, sp, sp) for sp in dict.fromkeys(x.species for x in fz.atoms)]
            elif each == "atom":
                groups = [(x.label, x.label, x.label) for x in fz.atoms]
            else:
                raise SelectionError(f"--each must be mode, site, species or atom, not '{each}'")
            sel = refs if modes else ["all"]
            for key, desc, selector in groups:                      # each group replaces --atoms in this enumeration
                out.append(Recipe(slug(key) if not modes else f"{fname(sel)}_on_{slug(key)}", list(sel), selector, exclude, f"{names(sel)} on {desc}"))
    elif pairs:
        for a, b in itertools.combinations(plain, 2):
            out.append(Recipe(fname([a, b]) + (f"_on_{slug(atoms)}" if atoms else ""), [a, b], atoms, exclude,
                              f"modes {names([a, b])}{atom_note}"))
    elif cumulative:
        for i in range(1, len(plain) + 1):
            out.append(Recipe(f"cum{i}_" + fname(plain[:i]) + (f"_on_{slug(atoms)}" if atoms else ""), plain[:i], atoms, exclude,
                              f"modes {names(plain[:i])}{atom_note}"))
    elif combinations:
        top = max_size or len(plain)
        for size in range(1, top + 1):
            for combo in itertools.combinations(plain, size):
                out.append(Recipe(fname(list(combo)) + (f"_on_{slug(atoms)}" if atoms else ""), list(combo),
                                  atoms, exclude, f"modes {names(list(combo))}{atom_note}"))
    elif parts:
        out.append(Recipe("__".join(part_name(p) for p in parts), list(parts), atoms, exclude, " + ".join(parts)))
    elif modes:
        out.append(Recipe(fname(refs) + (f"_on_{slug(atoms)}" if atoms else ""), refs, atoms, exclude,
                          f"modes {names(refs)}{atom_note}"))
    else:
        raise SelectionError("nothing selected: give --modes, --part, or one of --each/--pairs/--cumulative/--combinations "
                             "(run `show <analysis>` to see the available modes, atoms and sites)")
    return out


def amplitude_series(final, n=1, start=0.0):
    """The amplitudes (A) to build: just ``final`` for n = 1, else ``n`` evenly spaced values from ``start`` to ``final``."""
    if n < 1:
        raise SelectionError("the number of structures must be at least 1")
    if n == 1:
        return [float(final)]
    return [float(x) for x in np.linspace(start, final, n)]


def _amp_tag(a):
    return "A" + ("m" if a < 0 else "") + f"{abs(a):.3f}"


def run(fz, recipes, *, amplitude="found", n=1, start=0.0, scales=None, cell="parent", fmt="poscar", outdir=".",
        prefix=None, symmetry=True, sym_tol=1e-3, log=print):
    """Build and write every recipe, ``n`` structures each, from ``start`` to the final ``amplitude`` (A).

    ``amplitude`` is the root-sum-square size of the selected displacement (the same quantity the report calls
    amplitude), or ``"found"`` for the size it has in the child.  ``scales`` (fractions of the child's amplitude)
    can be given instead.  Returns the manifest rows, also written to manifest.json / manifest.txt.
    """
    os.makedirs(outdir, exist_ok=True)
    prefix = prefix or slug(os.path.splitext(str(fz.doc["meta"].get("child", "child")))[0].split("/")[-1]) or "frozen"
    formats = ["poscar", "cif"] if fmt == "both" else [fmt]
    rows = []
    explicit = scales is not None or amplitude != "found" or n > 1
    for i, rc in enumerate(recipes, start=1):
        rss1 = fz.rss(rc.parts, rc.atoms, rc.exclude)
        if rss1 < 1e-12:
            raise SelectionError(f"'{rc.description}' selects no displacement (all selected atoms stay at their parent positions)")
        if scales is not None:
            scs = [float(x) for x in scales]
            amps = [sc * rss1 for sc in scs]
        else:
            final = rss1 if amplitude == "found" else float(amplitude)
            amps = amplitude_series(final, n, start)
            scs = [a_ / rss1 for a_ in amps]
        log(f"[{i}] {rc.description}: amplitude found in the child {rss1:.4f} A -> "
            + (f"{len(amps)} structure(s) from {amps[0]:.4f} to {amps[-1]:.4f} A" if len(amps) > 1 else f"1 structure at {amps[0]:.4f} A"))
        for sc, amp in zip(scs, amps):
            s, info = fz.build(rc.parts, rc.atoms, rc.exclude, sc, cell)
            tag = f"_{_amp_tag(amp)}" if explicit else ""
            base = f"{prefix}_{i:02d}_{rc.name}{tag}"
            title = f"{rc.description} (amplitude {amp:.4f} A, {cell} cell)"
            files = []
            for f_ in formats:
                path = os.path.join(outdir, base + (".vasp" if f_ == "poscar" else ".cif"))
                write_structure(s, path, f_, title, sym_tol)
                files.append(path)
            sg, pg, _ = symmetry_of(s, sym_tol) if symmetry else ("", "", None)
            rows.append({"files": files, "recipe": rc.description, "parts": rc.parts, "atoms": rc.atoms, "exclude": rc.exclude,
                         "amplitude": amp, "scale": sc, "found_amplitude": rss1, "cell": cell, "space_group": sg,
                         "point_group": pg, **info})
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    lines = [f"{'file':<46s} {'recipe':<36s} {'amp(A)':>8s} {'x found':>8s} {'rms':>7s} {'max':>7s}  space group"]
    for r in rows:
        lines.append(f"{os.path.basename(r['files'][0]):<46s} {r['recipe'][:36]:<36s} {r['amplitude']:+8.4f} {r['scale']:8.3f} "
                     f"{r['rms']:7.4f} {r['max']:7.4f}  {r['space_group']}")
    with open(os.path.join(outdir, "manifest.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    log(f"\nwrote {len(rows)} structure(s) to {outdir} (manifest.txt / manifest.json list them)")
    return rows


# ====================================================================== arrows for VESTA (mCIF)
def mcif_text(fz, field, title, arrow_scale=1.0, min_disp=1e-6):
    """mCIF text: the parent supercell (P1, every atom listed) with each atom's displacement as a "magnetic moment".

    VESTA draws moments as arrows, so opening the file shows how the atoms move in the selected mode(s).  The
    vectors are in Angstrom (times ``arrow_scale``) and are written as components along the unit vectors of the
    supercell axes (``_atom_site_moment.crystalaxis_x/y/z``).  Atoms with |u| < ``min_disp`` get no vector.
    """
    A = fz.parent.lattice
    P = fz.M @ A                                                   # supercell lattice (rows), parent frame
    a, b, c, al, be, ga = Structure(P, np.zeros((0, 3)), []).lattice_parameters
    unit = P / np.linalg.norm(P, axis=1)[:, None]                  # unit vectors along the supercell axes
    Uinv = np.linalg.inv(unit)
    Minv = np.linalg.inv(fz.M)
    L = [f"# arrows = atomic displacements ({title})",
         f"# vectors are in Angstrom x {arrow_scale:g}; in VESTA use Properties > Vectors to change the arrow scale",
         "# atoms sit at the PARENT positions; moments are used only as a carrier for the displacement vectors",
         "data_distortkit_arrows", "_audit_creation_method 'distortkit (displacement vectors stored as moments)'",
         "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1",
         "_space_group_magn.number_BNS 1.1", "_space_group_magn.name_BNS 'P 1'",
         f"_cell_length_a {a:.8f}", f"_cell_length_b {b:.8f}", f"_cell_length_c {c:.8f}",
         f"_cell_angle_alpha {al:.6f}", f"_cell_angle_beta {be:.6f}", f"_cell_angle_gamma {ga:.6f}",
         "loop_", "_space_group_symop_magn_operation.id", "_space_group_symop_magn_operation.xyz", "1 x,y,z,+1",
         "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
         "_atom_site_fract_z", "_atom_site_occupancy"]
    moments, count = [], 0
    for x in fz.atoms:
        frac = (fz.parent.frac[x.a] + fz.cells[x.c]) @ Minv
        frac = np.round(frac - np.floor(frac + 1e-9), 8) + 0.0                  # wrapped into the cell
        L.append(f"{x.label} {x.species} {frac[0]:.8f} {frac[1]:.8f} {frac[2]:.8f} 1.0")
        d = field[x.a, x.c]
        if np.linalg.norm(d) >= min_disp:
            m = np.round((d @ Uinv) * arrow_scale, 8) + 0.0                     # + 0.0 turns -0.0 into 0.0
            moments.append(f"{x.label} {m[0]:.8f} {m[1]:.8f} {m[2]:.8f}")
            count += 1
    L += ["loop_", "_atom_site_moment.label", "_atom_site_moment.crystalaxis_x", "_atom_site_moment.crystalaxis_y",
          "_atom_site_moment.crystalaxis_z"] + moments
    return "\n".join(L) + "\n", count


def run_mcif(fz, recipes, *, amplitude="found", arrow_scale=1.0, min_disp=1e-6, outdir=".", prefix=None, log=print):
    """Write one mCIF per recipe (selected modes/atoms scaled to ``amplitude`` A, or as found); returns the rows."""
    os.makedirs(outdir, exist_ok=True)
    prefix = prefix or slug(os.path.splitext(str(fz.doc["meta"].get("child", "child")))[0].split("/")[-1]) or "arrows"
    rows = []
    for i, rc in enumerate(recipes, start=1):
        rss1 = fz.rss(rc.parts, rc.atoms, rc.exclude)
        if rss1 < 1e-12:
            raise SelectionError(f"'{rc.description}' selects no displacement (all selected atoms stay at their parent positions)")
        sc = 1.0 if amplitude == "found" else float(amplitude) / rss1
        field = fz.field(rc.parts, rc.atoms, rc.exclude, sc)
        text, count = mcif_text(fz, field, f"{rc.description}, amplitude {sc * rss1:.4f} A", arrow_scale, min_disp)
        path = os.path.abspath(os.path.join(outdir, f"{prefix}_{i:02d}_{rc.name}.mcif"))
        with open(path, "w") as fh:
            fh.write(text)
        rows.append({"file": path, "recipe": rc.description, "amplitude": sc * rss1, "arrows": count,
                     "atoms": len(fz.atoms), "max_displacement": float(np.max(np.linalg.norm(field, axis=2))),
                     "arrow_scale": arrow_scale})
    # the manifest accumulates over runs into the same folder (later runs replace entries for the same file)
    mpath = os.path.join(outdir, "arrows_manifest.json")
    allrows = {}
    if os.path.exists(mpath):
        try:
            with open(mpath) as fh:
                allrows = {os.path.basename(r["file"]): r for r in json.load(fh)}
        except (ValueError, KeyError):
            allrows = {}
    for r in rows:
        allrows[os.path.basename(r["file"])] = r
    allrows = {k: v for k, v in allrows.items() if os.path.exists(v["file"])}
    with open(mpath, "w") as fh:
        json.dump(list(allrows.values()), fh, indent=1)
    lines = [f"{'file':<44s} {'selection':<34s} {'amp(A)':>8s} {'arrows':>7s} {'longest(A)':>11s}"]
    for r in allrows.values():
        lines.append(f"{os.path.basename(r['file']):<44s} {r['recipe'][:34]:<34s} {r['amplitude']:+8.4f} {r['arrows']:>3d}/{r['atoms']:<3d} "
                     f"{r['max_displacement'] * r['arrow_scale']:11.4f}")
    with open(os.path.join(outdir, "arrows_manifest.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    lines = [lines[0]] + [l for l in lines[1:] if any(os.path.basename(r["file"]) in l for r in rows)]
    log("\n".join(lines))
    return rows
