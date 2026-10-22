"""Command line interface: ``python -m distortkit <command> ...``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np

from .modes import ModeAnalysis
from . import checks
from . import animate as _animate
from . import freeze as _freeze
from .compare import compare
from .pipeline import analyze
from . import relate as _relate
from . import explain as _explain
from . import invariants as _invariants
from .io import read_structure
from .structure import Structure
from .symmetrize import symmetrize, tolerance_table
from .symmetry import find_symmetry, point_group_name, to_primitive


_VALUE_OPTS = {"--scales", "--start", "--amplitude", "--part", "--modes", "--atoms", "--not-atoms", "--elev", "--azim", "--view"}


def _glue_negative_values(argv):
    """Let option values start with '-' (e.g. `--scales -1,0,1`, `--modes GM2-`) by turning them into `--opt=value`."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in _VALUE_OPTS and i + 1 < len(argv) and argv[i + 1].startswith("-") and not argv[i + 1].startswith("--"):
            out.append(f"{a}={argv[i + 1]}")
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def _symprec(v):
    return v if v == "auto" else float(v)


def _load(args):
    s = read_structure(args.structure, args.tol)
    k = tuple(Fraction(x) for x in args.k)
    if args.primitive:
        s, B = to_primitive(s, args.tol)
        kk = B @ np.array([float(x) for x in k])
        k = tuple(Fraction(float(x)).limit_denominator(1000) for x in kk)
        print(f"# reduced to primitive cell ({len(s)} atoms); k in the new basis = {tuple(str(x) for x in k)}")
    return ModeAnalysis(s, k, tol=args.tol), s


def cmd_info(args):
    s = read_structure(args.structure, args.tol)
    ops = find_symmetry(s, args.tol)
    ps, _ = to_primitive(s, args.tol)
    print(f"atoms: {len(s)}   operations in this cell: {len(ops)}   point group: "
          f"{point_group_name([R for R, _ in ops], s.lattice)}")
    print(f"primitive cell has {len(ps)} atoms")


def cmd_modes(args):
    ma, _ = _load(args)
    print(f"k = {tuple(str(x) for x in ma.k)}; star has {len(ma.star)} arm(s); supercell = {ma.n_cells} primitive cell(s)")
    labs = None
    try:
        print(f"parent space group: {ma.space_group().label()}")
        if not args.no_labels:
            labs = ma.irrep_labels()
    except (ValueError, FileNotFoundError) as e:
        print(f"(space group / labels unavailable: {e})")
    for ir in ma.irreps:
        cnt = {}
        for m in ir.modes:
            cnt[(m.species, m.orbit)] = cnt.get((m.species, m.orbit), 0) + 1
        parts = [f"{sp}(orbit {o})" + (f" x{c}" if c > 1 else "") for (sp, o), c in cnt.items()]
        name = ""
        if labs is not None:
            lb = labs.get(ir.index)
            name = f"{lb.text if lb and lb.candidates else '?':<9}" if lb else ""
        print(f"  irrep {ir.index}: {name}dim {ir.dim:<2} {ir.kind:<12} modes on {', '.join(parts)}")


def cmd_isotropy(args):
    ma, _ = _load(args)
    ir = ma.irreps[args.irrep - 1]
    print(f"irrep {ir.index} (dim {ir.dim}, {ir.kind}), modes on {', '.join(m.species for m in ir.modes)}")
    try:
        lb = ma.irrep_labels(only={ir.index})[ir.index]
        print(f"parent {ma.space_group().label()}; standard label: {lb.text}" + (f"  ({lb.note})" if lb.note else ""))
    except (ValueError, FileNotFoundError):
        pass
    print(" idx  point group  |PG|  free params  domains  child space group      OPD (in this irrep's basis)")
    for iso in ma.isotropy_subgroups(ir):
        sg = ma.child_space_group(iso)
        print(f" {iso.index:>3}  {iso.point_group:<11} {iso.n_point_ops:>4}  {iso.fix_dim:>11}  {iso.n_conjugates:>7}  "
              f"{sg.label() if sg else '?':<21}  {iso.opd_label}")


def cmd_distort(args):
    ma, _ = _load(args)
    ir = ma.irreps[args.irrep - 1]
    iso = ma.isotropy_subgroups(ir)[args.subgroup - 1]
    u = ma.displacement(ir, iso.opd, args.weights)
    child = ma.child_structure(u, args.amplitude)
    held = ma.symmetry_of(u)
    print(f"# child: irrep {ir.index}, isotropy subgroup {iso.index} ({iso.point_group}), amplitude {args.amplitude} A",
          file=sys.stderr)
    print(f"# {len(child)} atoms in a {ma.n_cells}-cell supercell; {len(held)} operations kept "
          f"(point group {point_group_name([ma.R[i] for i in held], ma.parent.lattice)})", file=sys.stderr)
    text = {"poscar": child.to_poscar, "cif": child.to_cif, "json": lambda: json.dumps(child.to_dict(), indent=1)}[args.format]()
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    else:
        print(text)


def cmd_decompose(args):
    ma, _ = _load(args)
    child = read_structure(args.child, args.tol)
    u = ma.displacements_from(child)
    rows, resid = ma.decompose(u)
    print(f"total displacement (root-sum-square): {np.linalg.norm(u):.4f} A")
    for ir, mi, amp, c in rows:
        if amp > args.min_amp:
            m = ir.modes[mi]
            print(f"  irrep {ir.index} on {m.species} (dim {ir.dim}): amplitude {amp:.4f} A   components "
                  + " ".join(f"{x:+.4f}" for x in c))
    print(f"outside this k-point's mode space: {resid:.4f} A")


def cmd_compare(args):
    res = analyze(args.parent, args.child, symmetrize_parent=args.symmetrize, symprec=args.symprec,
                  workdir=args.workdir, tol=args.tol, strain_tol=args.strain_tol, match_tol=args.match_tol,
                  sym_tol=args.sym_tol, canonical=not args.fast, labels=not args.no_labels)
    saved = None
    if not args.no_save:
        saved = args.save or os.path.join(os.path.dirname(os.path.abspath(args.child)),
                                          os.path.splitext(os.path.basename(args.child))[0] + "_analysis.json")
        res.comparison.save(saved, meta={"parent": os.path.basename(args.parent), "child": os.path.basename(args.child)})
    if args.json:
        d = res.comparison.to_dict()
        d["input_checks"] = [{"level": f.level, "code": f.code, "message": f.message} for f in res.findings]
        if res.symmetrization:
            d["parent_symmetrization"] = {"method": res.symmetrization.method, "number": res.symmetrization.number,
                                          "symbol": res.symmetrization.symbol, "cif": res.parent_cif}
        print(json.dumps(d, indent=1))
    else:
        print(res.report(min_amp=args.min_amp, min_level="info" if args.verbose else "warning", atoms=args.atoms))
        if saved:
            print(f"\nanalysis saved to {saved}\n  next:  python3 -m distortkit show {saved}      (modes, atoms, sites you can select)"
                  f"\n         python3 -m distortkit freeze {saved} --each mode      (one structure per mode)"
                  f"\n         python3 -m distortkit explain {saved}         (what each symmetry explains; which modes drive which)"
                  f"\n         python3 -m distortkit invariants {saved}      (allowed Landau terms and couplings)")
    if saved and args.json:
        print(f"analysis saved to {saved}", file=sys.stderr)
    if args.strict and (not res.ok or any(f.level == "warning" for f in res.findings + res.comparison.findings)):
        sys.exit(2)
    if not res.ok:
        sys.exit(1)


def cmd_relate(args):
    a = read_structure(args.a, args.tol)
    b = read_structure(args.b, args.tol)
    names = (os.path.splitext(os.path.basename(args.a))[0], os.path.splitext(os.path.basename(args.b))[0])
    if names[0] == names[1]:
        names = ("A", "B")
    from . import pseudo as _pseudo
    parents = [(os.path.splitext(os.path.basename(f))[0], read_structure(f, args.tol)) for f in (args.parent or [])]
    common = bool(args.common or parents)
    kw = {}
    if common:
        kw = dict(parents=parents, search=not args.no_search, scale=_pseudo.parse_species_scale(args.species_tol),
                  modes=not args.no_modes, pick=args.pick)
        if args.max_tol is not None:
            kw["ladder_tols"] = [t for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0) if t <= args.max_tol + 1e-9]
    rel = _relate.relate(a, b, names=names, tol=args.tol, sym_tol=args.sym_tol, common=common, **kw)
    print(rel.report())


def cmd_explain(args):
    fz = _freeze.Freezer.load(args.analysis)
    tols = tuple(float(x) for x in args.tols.split(",")) if args.tols else _explain.DEFAULT_TOLS
    primary = [x.strip() for x in args.primary.split(",")] if args.primary else None
    text, rungs, casc = _explain.report(fz, tols=tols, primary=primary, allowed_share=args.allowed_share,
                                        opd_tol=args.opd_tol, min_amp=args.min_amp)
    if args.json:
        print(json.dumps(_explain.to_dict(fz, rungs, casc), indent=1))
    else:
        print(text)


def cmd_invariants(args):
    fz = _freeze.Freezer.load(args.analysis)
    refs = [x.strip() for x in args.modes.split(",")] if args.modes else None
    res = _invariants.compute(fz, refs, degree=args.degree, min_amp=args.min_amp)
    if args.json:
        print(json.dumps(res.to_dict(), indent=1))
    else:
        print(res.report(show_all=args.all))


def cmd_show(args):
    fz = _freeze.Freezer.load(args.analysis)
    print(fz.legend())
    if not (args.mcif or args.gif):
        return
    recipes = _freeze.enumerate_recipes(
        fz, modes=_csv(args.modes), atoms=args.atoms, exclude=args.not_atoms, parts=args.part or None,
        each=args.each or (None if (args.modes or args.part or args.pairs or args.cumulative or args.combinations) else "mode"),
        pairs=args.pairs, cumulative=args.cumulative, combinations=args.combinations, max_size=args.max_size)
    if args.mcif:
        outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.analysis)), "arrows")
        print("\nARROW FILES (mCIF: displacement vectors on the parent positions; open in VESTA)")
        rows = _freeze.run_mcif(fz, recipes, amplitude="found" if args.amplitude is None else args.amplitude,
                                arrow_scale=args.arrow_scale, min_disp=args.min_disp, outdir=outdir, prefix=args.prefix)
        print(f"\nwrote {len(rows)} mCIF file(s) to {os.path.abspath(outdir)}")
        print("  in VESTA: File > Open, then Properties > Vectors to set the arrow scale/colour (values are in Angstrom"
              + (f" x {args.arrow_scale:g}" if args.arrow_scale != 1 else "") + ")")
    if args.gif:
        fraction = None
        if args.fraction is not None:
            fraction = _animate.parse_fraction(args.fraction)
        elif args.amplitude is None and sys.stdin.isatty():
            found = [fz.rss(r.parts, r.atoms, r.exclude) for r in recipes]
            shown = f"{found[0]:.4f} A" if len(found) == 1 else "each selection's own"
            fraction = _animate.parse_fraction(_ask(f"Largest displacement as a percentage (or fraction) of the amplitude found "
                                                    f"in the child ({shown}); 100% = the child itself", "100%", str))
        outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.analysis)), "animations")
        print("\nANIMATIONS (GIF: atoms oscillating about the parent positions)")
        rows = _animate.run_gif(
            fz, recipes, fraction=1.0 if fraction is None else fraction, amplitude=args.amplitude,
            n_frames=args.frames, fps=args.fps, oneway=args.oneway, view=args.view, elev=args.elev, azim=args.azim,
            size=args.size, loops=args.loops, arrows=args.arrows, arrow_scale=args.arrow_scale if args.arrow_scale != 1.0 else 3.0,
            bonds=not args.no_bonds, outdir=outdir, prefix=args.prefix)
        print(f"\nwrote {len(rows)} GIF file(s) to {os.path.abspath(outdir)}")


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _ask(prompt, default, cast):
    """Prompt on a terminal; Enter keeps ``default``."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  please enter a number")


def _amount(v):
    return v if v == "found" else float(v)


def cmd_freeze(args):
    fz = _freeze.Freezer.load(args.analysis)
    recipes = _freeze.enumerate_recipes(
        fz, modes=_csv(args.modes), atoms=args.atoms, exclude=args.not_atoms, parts=args.part or None, each=args.each,
        pairs=args.pairs, cumulative=args.cumulative, combinations=args.combinations, max_size=args.max_size)
    scales = [float(x) for x in args.scales.split(",")] if args.scales else None
    amplitude, n = args.amplitude, args.n
    if scales is None and sys.stdin.isatty():
        found = [fz.rss(r.parts, r.atoms, r.exclude) for r in recipes]
        shown = f"{found[0]:.4f}" if len(found) == 1 else "each selection's own, " + ", ".join(f"{x:.3f}" for x in found[:6]) + (" ..." if len(found) > 6 else "")
        if amplitude is None:
            amplitude = _ask(f"Final amplitude in A (root-sum-square of the selection; found in the child: {shown}; "
                             f"type 'found' to use it)", "found", _amount)
        if n is None:
            n = _ask(f"How many structures per selection (evenly spaced from {args.start:g} A to the final amplitude)", 1, int)
    amplitude = "found" if amplitude is None else amplitude
    n = 1 if n is None else n
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.analysis)), "frozen")
    _freeze.run(fz, recipes, amplitude=amplitude, n=n, start=args.start, scales=scales, cell=args.cell, fmt=args.format,
                outdir=outdir, prefix=args.prefix, symmetry=not args.no_symmetry, sym_tol=args.sym_tol)


def cmd_symmetrize(args):
    s = read_structure(args.structure, args.tol)
    out = args.out or os.path.splitext(args.structure)[0] + "_symmetrized.cif"
    res = symmetrize(s, symprec=args.symprec, angle_tolerance=args.angle_tolerance, method=args.method, cif_path=out)
    print(f"{res.symbol} (#{res.number}), point group {res.point_group}  [{res.method}]")
    print("space group vs tolerance: " + ", ".join(f"{t:g}: {sym}" + (f" (#{n})" if n else "") for t, n, sym in res.table))
    for n in res.notes:
        print("note:", n)
    print(f"wrote {out}")


def cmd_check(args):
    findings = checks.check_file(args.structure)
    if not checks.has_errors(findings):
        s = read_structure(args.structure, args.tol)
        findings += checks.check_structure(s, os.path.basename(args.structure))
        if not checks.has_errors(findings):
            findings += checks.check_parent_symmetry(s, args.tol)
            print("space group vs tolerance (A): " + ", ".join(
                f"{t:g}: {sym}" + (f" (#{n})" if n else "") for t, n, sym in tolerance_table(s, method=args.method)))
    print(checks.format_findings(findings))
    sys.exit(1 if checks.has_errors(findings) else 0)


def cmd_phonons(args):
    from . import phonons as _ph
    bd = _ph.read_band_yaml(args.bandfile)
    only = set(int(x) - 1 for x in args.qpoints.split(",")) if args.qpoints else None
    if only is not None:  # analyse just the q-points asked for (1-based, as printed)
        keep = sorted(only)
        bd = _ph.BandData(bd.lattice, bd.species, bd.masses, bd.frac, bd.qpts[keep], bd.dist[keep], bd.freq[keep], bd.eig[keep],
                          [(keep.index(i), lab) for i, lab in bd.ticks if i in keep])
    res = _ph.analyze_band_yaml(bd, max_cells=args.max_cells, max_dim=args.max_dim, labels=not args.no_labels, tol=args.tol, opd_tol=args.opd_tol,
                                mass_scale=not args.no_mass_scale, basis_phase=not args.lattice_phase,
                                progress=lambda i, n: print(f"# analysed q-point {i}/{n}", file=sys.stderr, end="\r"))
    print(file=sys.stderr)
    out = args.out or os.path.splitext(args.bandfile)[0] + "_symmetry.json"
    _ph.save(res, out)
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        print(_ph.report(res, subgroups=not args.no_subgroups))
        print(f"\nfull results written to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="distortkit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, k=True):
        sp.add_argument("structure", help="parent structure JSON (see examples/)")
        sp.add_argument("--tol", type=float, default=1e-3, help="atomic matching tolerance in Angstrom")
        sp.add_argument("--no-labels", action="store_true", help="skip irrep labels (faster)")
        if k:
            sp.add_argument("--k", nargs=3, required=True, metavar=("KX", "KY", "KZ"),
                            help="wavevector in the reciprocal basis of the cell (e.g. 1/2 1/2 1/2)")
            sp.add_argument("--primitive", action="store_true", help="reduce the structure to a primitive cell first (k is converted)")

    sp = sub.add_parser("info", help="symmetry of the parent"); common(sp, k=False); sp.set_defaults(fn=cmd_info)
    sp = sub.add_parser("modes", help="list irreps at k"); common(sp); sp.set_defaults(fn=cmd_modes)
    sp = sub.add_parser("isotropy", help="isotropy subgroups of an irrep"); common(sp)
    sp.add_argument("--irrep", type=int, required=True); sp.set_defaults(fn=cmd_isotropy)
    sp = sub.add_parser("distort", help="write a distorted child structure"); common(sp)
    sp.add_argument("--irrep", type=int, required=True)
    sp.add_argument("--subgroup", type=int, required=True)
    sp.add_argument("--amplitude", type=float, default=0.3, help="root-sum-square displacement over the supercell (A)")
    sp.add_argument("--weights", type=float, nargs="*", default=None, help="relative weight of each mode copy")
    sp.add_argument("--format", choices=["poscar", "cif", "json"], default="poscar")
    sp.add_argument("--out"); sp.set_defaults(fn=cmd_distort)
    sp = sub.add_parser("decompose", help="mode amplitudes of a child structure (JSON from `distort --format json`)")
    common(sp); sp.add_argument("child"); sp.add_argument("--min-amp", type=float, default=1e-6)
    sp.set_defaults(fn=cmd_decompose)
    sp = sub.add_parser("compare", help="mode + strain analysis of a child structure relative to its parent")
    sp.add_argument("parent", help="high-symmetry parent (.json, POSCAR, or .cif); without symmetry information it is symmetrised first")
    sp.add_argument("child", help="low-symmetry child (same formats); any setting, orientation or atom order")
    sp.add_argument("--tol", type=float, default=1e-3, help="parent symmetry tolerance (A)")
    sp.add_argument("--symmetrize", choices=["auto", "pymatgen", "builtin", "off"], default="auto",
                    help="how to give a symmetry-less parent explicit symmetry (default: pymatgen if available)")
    sp.add_argument("--symprec", type=_symprec, default="auto",
                    help="symmetry precision (A) for symmetrising the parent, or 'auto' (default): highest symmetry up to 0.1 A")
    sp.add_argument("--workdir", help="where to write <parent>_symmetrized.cif (default: next to the parent file)")
    sp.add_argument("--sym-tol", type=float, default=0.02, help="tolerance for detecting the child's symmetry (A)")
    sp.add_argument("--strain-tol", type=float, default=0.12, help="largest lattice strain allowed when relating the cells")
    sp.add_argument("--match-tol", type=float, default=1.0, help="largest atomic displacement allowed in matching (A)")
    sp.add_argument("--min-amp", type=float, default=1e-3, help="hide modes with amplitude below this (A)")
    sp.add_argument("--fast", action="store_true", help="skip canonical OPD bases/subgroup fitting (faster for big stars)")
    sp.add_argument("--no-labels", action="store_true", help="skip irrep labels and child space groups")
    sp.add_argument("--verbose", action="store_true", help="show informational checks too")
    sp.add_argument("--atoms", action="store_true", help="also print the displacement of every atom and each mode's share of it")
    sp.add_argument("--strict", action="store_true", help="exit with status 2 if any warning is raised")
    sp.add_argument("--save", help="where to write the analysis file used by `show` and `freeze` (default: <child>_analysis.json next to the child)")
    sp.add_argument("--no-save", action="store_true", help="do not write the analysis file")
    sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("relate", help="how are two structures related? group-subgroup either way, or why not")
    sp.add_argument("a"); sp.add_argument("b")
    sp.add_argument("--tol", type=float, default=1e-3, help="symmetry tolerance (A) for both structures")
    sp.add_argument("--sym-tol", type=float, default=0.02, help="tolerance for the child's symmetry (A)")
    sp.add_argument("--common", action="store_true",
                    help="if neither is the parent of the other, look for a common parent (pseudo-symmetry search) "
                         "and tabulate both structures' modes relative to it")
    sp.add_argument("--parent", action="append",
                    help="a candidate common parent (structure file, repeatable); implies --common. Needed when the "
                         "distortions are too large for the pseudo-symmetry search to find it")
    sp.add_argument("--max-tol", type=float, help="largest tolerance (A) of the pseudo-symmetry search (default 0.5)")
    sp.add_argument("--species-tol", default="", help="per-species tolerance multipliers, e.g. O=3 (anions often move more); "
                                                       "use with care: loose tolerances can invent symmetry")
    sp.add_argument("--no-search", action="store_true", help="only use the --parent file(s); skip the pseudo-symmetry search")
    sp.add_argument("--no-modes", action="store_true", help="find the common parent but skip the mode decomposition")
    sp.add_argument("--pick", type=int, help="use the N-th valid common parent of the list instead of the closest one")
    sp.set_defaults(fn=cmd_relate)
    sp = sub.add_parser("explain", help="how much of the distortion does each symmetry explain, and which modes drive which "
                                        "(from a saved analysis)")
    sp.add_argument("analysis", help="the analysis file written by `compare`")
    sp.add_argument("--tols", help="comma-separated tolerances (A) for the symmetry ladder (default 1e-4,3e-3,0.01,0.03,0.1,0.2,0.4)")
    sp.add_argument("--primary", help="mode(s) to treat as the drivers, in order (labels or ids); default: the largest mode "
                                      "that the symmetry kept so far does not allow, repeatedly")
    sp.add_argument("--min-amp", type=float, default=0.02, help="ignore modes below this amplitude (A); default 0.02")
    sp.add_argument("--allowed-share", type=float, default=0.95,
                    help="share of a mode's squared amplitude a symmetry must permit for the mode to count as allowed (default 0.95)")
    sp.add_argument("--opd-tol", type=float, default=0.05,
                    help="an operation keeps a mode if it moves no atom more than this fraction of the mode's largest displacement")
    sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_explain)
    sp = sub.add_parser("invariants", help="allowed Landau terms and couplings of the modes of a saved analysis (polynomial invariants)")
    sp.add_argument("analysis", help="the analysis file written by `compare`")
    sp.add_argument("--modes", help="modes to include (labels or ids, comma separated); default: every mode above --min-amp")
    sp.add_argument("--degree", type=int, default=3, help="highest polynomial degree (default 3: quadratic, cubic and trilinear terms; 4 adds quartic)")
    sp.add_argument("--min-amp", type=float, default=0.02, help="ignore modes below this amplitude (A); default 0.02")
    sp.add_argument("--all", action="store_true", help="list every invariant term and every coupling, not just the new/lowest-order ones")
    sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_invariants)
    sp = sub.add_parser("show", help="list the modes, atoms and Wyckoff sites of a saved analysis (what you can freeze)")
    sp.add_argument("analysis")
    sp.add_argument("--gif", action="store_true",
                    help="also write animated GIFs of the atoms moving in the selected modes/atoms (needs matplotlib + Pillow)")
    sp.add_argument("--fraction", help="largest displacement as a fraction or percentage of the amplitude found in the child "
                                       "(0.5 or 50%%; 100%% = the child itself); asked for at a terminal if omitted")
    sp.add_argument("--frames", type=int, default=24, help="frames per cycle (default 24)")
    sp.add_argument("--fps", type=int, default=12, help="frames per second (default 12)")
    sp.add_argument("--oneway", action="store_true", help="animate parent -> maximum -> parent instead of +max and -max")
    sp.add_argument("--view", default="iso", help="iso (default), or look down a cell axis: a, b, c, -a, -b, -c")
    sp.add_argument("--elev", type=float, help="camera elevation in degrees (with --view iso)")
    sp.add_argument("--azim", type=float, help="camera azimuth in degrees from the a axis towards b (with --view iso)")
    sp.add_argument("--size", type=int, default=640, help="image size in pixels (square)")
    sp.add_argument("--loops", type=int, default=0, help="times the GIF repeats (0 = forever)")
    sp.add_argument("--arrows", action="store_true", help="draw the displacement arrows (at the largest excursion) too")
    sp.add_argument("--no-bonds", action="store_true", help="do not draw bonds")
    sp.add_argument("--mcif", action="store_true",
                    help="also write mCIF files that VESTA shows as arrows (atomic displacements of the selected modes/atoms)")
    sp.add_argument("--modes"); sp.add_argument("--atoms"); sp.add_argument("--not-atoms"); sp.add_argument("--part", action="append")
    sp.add_argument("--each", choices=["mode", "site", "species", "atom"])
    sp.add_argument("--pairs", action="store_true"); sp.add_argument("--cumulative", action="store_true")
    sp.add_argument("--combinations", action="store_true"); sp.add_argument("--max-size", type=int)
    sp.add_argument("--amplitude", type=_amount, help="scale the selection to this root-sum-square amplitude in A "
                                                       "(default: as found in the child; negative reverses the arrows; for --gif, "
                                                       "the largest excursion in A instead of --fraction)")
    sp.add_argument("--arrow-scale", type=float, default=1.0, help="multiply the vectors (they are in A); VESTA can also rescale")
    sp.add_argument("--min-disp", type=float, default=1e-6, help="atoms displaced less than this (A) get no arrow")
    sp.add_argument("--outdir", help="default: an 'arrows' folder next to the analysis file"); sp.add_argument("--prefix")
    sp.set_defaults(fn=cmd_show)
    sp = sub.add_parser("freeze", help="write POSCAR/CIF files along the distortion: freeze chosen modes and/or atoms",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="""what to freeze (combine freely):
  --modes m1,GM1+            these modes together (labels or ids from `show`; all = every listed mode)
  --atoms Hf,O1-O4,@4c       restrict every selected mode to these atoms (species, labels, ranges, sites)
  --not-atoms O5             leave these atoms out
  --part 'GM2-:Hf*0.5'       MODE[:ATOMS][*SCALE|=AMPLITUDE], repeatable: different atoms/scale per mode
how far:   --amplitude 1.0 --n 5   final amplitude in A (root-sum-square of what you selected) and how many structures,
           evenly spaced from --start (default 0 = the parent). In a terminal, leaving them out makes freeze ask;
           in scripts: the amplitude found in the child, one structure. --start -1.0 reaches the opposite domain.
           advanced: --scales -1,0,0.5,1  (fractions of the child's amplitude)
sets:      --each mode|site|species|atom   --pairs   --cumulative   --combinations [--max-size 2]
cell:      --cell parent (default, undistorted lattice) | child (strain scaled with the displacement)""")
    sp.add_argument("analysis")
    sp.add_argument("--modes"); sp.add_argument("--atoms"); sp.add_argument("--not-atoms"); sp.add_argument("--part", action="append")
    sp.add_argument("--amplitude", type=_amount, help="final amplitude in A (root-sum-square of the selection), or 'found'")
    sp.add_argument("-n", "--n", type=int, help="number of structures, evenly spaced from --start to the final amplitude")
    sp.add_argument("--start", type=float, default=0.0, help="first amplitude in A (default 0 = the parent; negative = opposite domain)")
    sp.add_argument("--scales", help="advanced: comma-separated fractions of the amplitude found in the child, instead of --amplitude/--n")
    sp.add_argument("--cell", choices=["parent", "child"], default="parent")
    sp.add_argument("--each", choices=["mode", "site", "species", "atom"])
    sp.add_argument("--pairs", action="store_true"); sp.add_argument("--cumulative", action="store_true")
    sp.add_argument("--combinations", action="store_true"); sp.add_argument("--max-size", type=int)
    sp.add_argument("--format", choices=["poscar", "cif", "both"], default="poscar")
    sp.add_argument("--outdir", help="default: a 'frozen' folder next to the analysis file"); sp.add_argument("--prefix")
    sp.add_argument("--no-symmetry", action="store_true", help="skip the space-group detection of each output")
    sp.add_argument("--sym-tol", type=float, default=1e-3); sp.set_defaults(fn=cmd_freeze)
    sp = sub.add_parser("phonons", help="irrep and child symmetry of every phonon of a phonopy band.yaml (written with eigenvectors)")
    sp.add_argument("bandfile", help="phonopy band.yaml")
    sp.add_argument("--qpoints", help="only these q-points (1-based numbers as printed, comma separated), e.g. 5,15")
    sp.add_argument("--max-cells", type=int, default=16, help="skip q-points whose star needs a larger supercell (default 16)")
    sp.add_argument("--tol", type=float, default=1e-3, help="atomic matching tolerance in Angstrom")
    sp.add_argument("--opd-tol", type=float, default=0.05, help="how closely a mode's direction must match an isotropy subgroup (default 0.05)")
    sp.add_argument("--max-dim", type=int, default=6, help="do not enumerate isotropy subgroups of irreps larger than this (they can take minutes); default 6")
    sp.add_argument("--no-labels", action="store_true", help="skip standard irrep labels (faster)")
    sp.add_argument("--no-mass-scale", action="store_true", help="the eigenvectors are already displacements (do not divide by sqrt(mass))")
    sp.add_argument("--lattice-phase", action="store_true", help="eigenvectors use lattice-translation phases only (not phonopy's convention)")
    sp.add_argument("--no-subgroups", action="store_true", help="do not list every irrep's isotropy subgroups")
    sp.add_argument("--out", help="JSON file to write (default: <bandfile>_symmetry.json)")
    sp.add_argument("--json", action="store_true", help="print the JSON instead of the text report")
    sp.set_defaults(fn=cmd_phonons)
    sp = sub.add_parser("symmetrize", help="find the space group of a structure and write an idealised CIF with symmetry operations")
    sp.add_argument("structure"); sp.add_argument("--out"); sp.add_argument("--tol", type=float, default=1e-3)
    sp.add_argument("--symprec", type=_symprec, default="auto"); sp.add_argument("--angle-tolerance", type=float, default=5.0)
    sp.add_argument("--method", choices=["auto", "pymatgen", "builtin"], default="auto"); sp.set_defaults(fn=cmd_symmetrize)
    sp = sub.add_parser("check", help="sanity-check a structure file")
    sp.add_argument("structure"); sp.add_argument("--tol", type=float, default=1e-3)
    sp.add_argument("--method", choices=["auto", "pymatgen", "builtin"], default="auto"); sp.set_defaults(fn=cmd_check)
    argv = _glue_negative_values(sys.argv[1:] if argv is None else list(argv))
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
