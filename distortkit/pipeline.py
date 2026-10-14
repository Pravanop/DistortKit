"""One-call pipeline: files in, full analysis out.

    result = analyze("parent.vasp", "child.vasp")
    print(result.report())

Steps: check both files, give the parent explicit symmetry (pymatgen SpacegroupAnalyzer, falling back to the
built-in detector) and save it as a CIF when it has none, check both structures and the pair, run the
mode/strain comparison, add diagnostics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import checks
from .compare import compare
from .io import read_structure
from .symmetrize import symmetrize


@dataclass
class Analysis:
    comparison: object
    findings: list = field(default_factory=list)
    symmetrization: object = None        # symmetrize.SymmetryResult when the parent was symmetrised
    parent_cif: str = None

    def report(self, min_amp=1e-3, min_level="warning", atoms=False):
        L = []
        s = self.symmetrization
        if s is not None:
            L.append(f"parent symmetrised with {s.method}: {s.symbol} (#{s.number}), point group {s.point_group}"
                     + (f"; idealised structure written to {self.parent_cif}" if self.parent_cif else ""))
            if s.table:
                L.append("  space group vs tolerance (A): " + ", ".join(
                    f"{t:g}: {sym}" + (f" (#{n})" if n else "") for t, n, sym in s.table))
            L += [f"  note: {n}" for n in s.notes]
            L.append("")
        L.append(self.comparison.report(min_amp=min_amp, atoms=atoms))
        pre = [f for f in self.findings if f not in self.comparison.findings]
        order = {"info": 0, "warning": 1, "error": 2}
        pre = [f for f in pre if order[f.level] >= order[min_level]]
        if pre:
            L.append("")
            L.append("input checks:")
            L += [f"  {f}" for f in pre]
        return "\n".join(L)

    @property
    def ok(self):
        return not checks.has_errors(self.findings + self.comparison.findings)


def analyze(parent_path, child_path, *, symmetrize_parent="auto", symprec="auto", angle_tolerance=5.0,
            workdir=None, tol=1e-3, **compare_kwargs):
    """Run the full analysis from two structure files.

    ``symmetrize_parent``: ``"auto"`` (symmetrise only if the parent file has no symmetry operations; pymatgen if
    available, else built-in), ``"pymatgen"``, ``"builtin"`` or ``"off"``.  The idealised parent is written to
    ``<workdir>/<name>_symmetrized.cif`` (``workdir`` defaults to the parent file's directory).
    """
    findings = checks.check_file(parent_path) + checks.check_file(child_path)
    if checks.has_errors(findings):
        raise ValueError("; ".join(f.message for f in findings if f.level == "error"))
    parent = read_structure(parent_path, tol)
    child = read_structure(child_path, tol)
    findings += checks.check_structure(parent, "parent") + checks.check_structure(child, "child")
    if checks.has_errors(findings):
        raise ValueError("; ".join(f.message for f in findings if f.level == "error"))
    sym = None
    cif_path = None
    have_sym = checks.symmetry_info_in_file(parent_path)
    if symmetrize_parent != "off" and not have_sym:
        base = os.path.splitext(os.path.basename(parent_path))[0] or "parent"
        d = workdir or os.path.dirname(os.path.abspath(parent_path))
        cif_path = os.path.join(d, f"{base}_symmetrized.cif")
        sym = symmetrize(parent, symprec=symprec, angle_tolerance=angle_tolerance,
                         method="auto" if symmetrize_parent == "auto" else symmetrize_parent, cif_path=cif_path)
        parent = sym.structure
        findings.append(checks.Finding("info", "symmetrized",
                                       f"parent had no symmetry information: found {sym.symbol} (#{sym.number}) with "
                                       f"{sym.method} and wrote {cif_path}" + ("; " + "; ".join(sym.notes) if sym.notes else "")))
        if sym.number == 1:
            findings.append(checks.Finding("warning", "parent-p1", "the parent is P1 at this symprec; "
                                           "try a larger --symprec if you expect a higher-symmetry reference"))
    findings += checks.check_parent_symmetry(parent, tol)
    findings += checks.check_pair(parent, child)
    if checks.has_errors(findings):
        raise ValueError("; ".join(f.message for f in findings if f.level == "error"))
    try:
        comp = compare(parent, child, tol=tol, **compare_kwargs)
    except ValueError as e:
        raise ValueError(f"{e}. To see why the two structures do not line up (and whether another parent would), "
                         f"run: python3 -m distortkit relate {parent_path} {child_path}") from e
    return Analysis(comp, findings, sym, cif_path)
