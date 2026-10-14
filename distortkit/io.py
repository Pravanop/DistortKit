"""Structure readers: JSON, VASP POSCAR/CONTCAR, and CIF (with explicit symmetry operations)."""
from __future__ import annotations

import os
import re
from fractions import Fraction

import numpy as np

from .structure import Structure


def read_structure(path, tol=1e-3):
    base = os.path.basename(path).lower()
    if base.endswith(".json"):
        return Structure.from_json(path)
    if base.endswith(".cif"):
        return read_cif(path, tol)
    return read_poscar(path)


# ---------------------------------------------------------------- POSCAR
def read_poscar(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:
        scale = (-scale / abs(np.linalg.det(lat))) ** (1 / 3)
    lat = lat * scale
    tokens = lines[5].split()
    if all(re.fullmatch(r"[A-Za-z]{1,3}\d*", t) for t in tokens) and not tokens[0].isdigit():
        names, counts, i = tokens, [int(x) for x in lines[6].split()], 7
    else:
        raise ValueError("POSCAR without element names (VASP4 format) is not supported")
    if lines[i].strip().lower().startswith("s"):
        i += 1
    cartesian = lines[i].strip().lower()[0] in "ck"
    i += 1
    pos, sp = [], []
    for name, c in zip(names, counts):
        for _ in range(c):
            pos.append([float(x) for x in lines[i].split()[:3]])
            sp.append(name)
            i += 1
    pos = np.array(pos)
    if cartesian:
        pos = pos * scale @ np.linalg.inv(lat)
    return Structure(lat, pos, sp)


# ---------------------------------------------------------------- CIF
def _num(s):
    return float(re.sub(r"\(.*?\)", "", s))


def _tokens(line):
    return re.findall(r"'[^']*'|\"[^\"]*\"|\S+", line)


def _parse_cif(text):
    """Return (tags dict, loops list of (header list, rows list))."""
    tags, loops = {}, []
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln or ln.startswith("#"):
            i += 1
        elif ln.lower() == "loop_":
            i += 1
            header = []
            while i < len(lines) and lines[i].startswith("_"):
                header.append(lines[i].lower())
                i += 1
            vals = []
            while i < len(lines) and lines[i] and not lines[i].startswith(("_", "#")) and lines[i].lower() != "loop_" \
                    and not lines[i].lower().startswith("data_"):
                if lines[i].startswith(";"):
                    i += 1
                    continue
                vals += [t.strip("'\"") for t in _tokens(lines[i])]
                i += 1
            n = len(header)
            loops.append((header, [vals[j:j + n] for j in range(0, len(vals) - n + 1, n)]))
        elif ln.startswith("_"):
            parts = _tokens(ln)
            if len(parts) >= 2:
                tags[parts[0].lower()] = parts[1].strip("'\"")
            elif i + 1 < len(lines):
                tags[parts[0].lower()] = lines[i + 1].strip("'\"")
                i += 1
            i += 1
        else:
            i += 1
    return tags, loops


def _parse_op(expr):
    """'-x+1/2, y, z' -> (R int 3x3, t Fraction 3)."""
    R = np.zeros((3, 3), dtype=int)
    t = [Fraction(0)] * 3
    for i, comp in enumerate(expr.lower().replace(" ", "").split(",")):
        for sign, body in re.findall(r"([+-]?)([^+-]+)", comp):
            sg = -1 if sign == "-" else 1
            if body in "xyz" and len(body) == 1:
                R[i, "xyz".index(body)] += sg
            elif re.fullmatch(r"\d+(/\d+)?", body):
                t[i] += sg * Fraction(body)
            elif re.fullmatch(r"\d*\.\d+", body):
                t[i] += sg * Fraction(body)                       # exact: the origin may be arbitrary
            elif re.fullmatch(r"(\d+(/\d+)?)\*?[xyz]", body):
                c = re.match(r"(\d+(/\d+)?)", body).group(1)
                R[i, "xyz".index(body[-1])] += sg * int(Fraction(c))
            else:
                raise ValueError(f"cannot parse symmetry operation '{expr}'")
    return R, np.array([float(f) for f in t])


def read_cif(path, tol=1e-3):
    with open(path) as fh:
        return structure_from_cif_text(fh.read(), tol)


def structure_from_cif_text(text, tol=1e-3):
    tags, loops = _parse_cif(text)
    a, b, c = (_num(tags[f"_cell_length_{x}"]) for x in "abc")
    al, be, ga = (_num(tags[f"_cell_angle_{x}"]) for x in ("alpha", "beta", "gamma"))
    ops = None
    sites = None
    for header, rows in loops:
        for key in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
            if key in header:
                ops = [_parse_op(r[header.index(key)]) for r in rows]
        if "_atom_site_fract_x" in header:
            sites = (header, rows)
    if sites is None:
        raise ValueError("no _atom_site_fract_x loop found in CIF")
    if ops is None:
        sg = tags.get("_symmetry_space_group_name_h-m", tags.get("_space_group_name_h-m_alt", "P 1")).replace(" ", "")
        if sg not in ("P1", "P-1") or sg == "P-1":
            if sg != "P1":
                raise ValueError(f"CIF has no explicit symmetry operations (space group {sg}); "
                                 "expand it to P1 first (e.g. with another tool) or provide JSON/POSCAR")
        ops = [(np.eye(3, dtype=int), np.zeros(3))]
    header, rows = sites
    ix = [header.index(f"_atom_site_fract_{x}") for x in "xyz"]
    notes = []
    if "_atom_site_occupancy" in header:
        occ = [_num(r[header.index("_atom_site_occupancy")]) for r in rows]
        if any(o < 0.999 for o in occ):
            notes.append("CIF has partial occupancies; they are ignored (all sites treated as fully occupied)")
    isym = header.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in header else None
    ilab = header.index("_atom_site_label") if "_atom_site_label" in header else None
    structure0 = Structure.from_parameters(a, b, c, al, be, ga, np.zeros((0, 3)), [])
    A = structure0.lattice
    frac, species = [], []
    for r in rows:
        x = np.array([_num(r[i]) for i in ix])
        name = r[isym] if isym is not None else re.match(r"[A-Za-z]+", r[ilab]).group(0)
        name = re.match(r"[A-Za-z]+", name).group(0)
        for R, t in ops:
            y = (R @ x + t) % 1.0
            y[np.abs(y - 1.0) < 1e-8] = 0.0
            dup = False
            for z, sp in zip(frac, species):
                d = y - z
                d -= np.rint(d)
                if sp == name and np.linalg.norm(d @ A) < max(tol, 1e-3):
                    dup = True
                    break
            if not dup:
                frac.append(y)
                species.append(name)
    st = Structure(A, np.array(frac), species)
    st.notes = notes
    return st
