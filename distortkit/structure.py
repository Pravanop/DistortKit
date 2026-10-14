"""Crystal structure container and simple file writers."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Structure:
    """lattice: rows are lattice vectors (Angstrom); frac: (n, 3) fractional coordinates."""

    lattice: np.ndarray
    frac: np.ndarray
    species: list
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []
        self.lattice = np.asarray(self.lattice, dtype=float).reshape(3, 3)
        self.frac = np.asarray(self.frac, dtype=float).reshape(-1, 3)
        self.species = [str(s) for s in self.species]
        if len(self.species) != len(self.frac):
            raise ValueError("species and frac must have the same length")

    def __len__(self):
        return len(self.frac)

    @property
    def cart(self):
        return self.frac @ self.lattice

    @property
    def volume(self):
        return abs(np.linalg.det(self.lattice))

    @property
    def lattice_parameters(self):
        a, b, c = self.lattice
        la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
        ang = lambda u, v: np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        return la, lb, lc, ang(b, c), ang(a, c), ang(a, b)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_parameters(cls, a, b, c, alpha, beta, gamma, frac, species):
        """Build from conventional lattice parameters (a along x, b in the xy plane)."""
        al, be, ga = np.radians([alpha, beta, gamma])
        v1 = [a, 0.0, 0.0]
        v2 = [b * np.cos(ga), b * np.sin(ga), 0.0]
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        return cls([v1, v2, [cx, cy, cz]], frac, species)

    @classmethod
    def from_dict(cls, d):
        atoms = d["atoms"]
        species = [a[0] for a in atoms]
        frac = [a[1:4] for a in atoms]
        if "lattice" in d:
            return cls(d["lattice"], frac, species)
        return cls.from_parameters(*d["parameters"], frac, species)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self):
        return {
            "lattice": self.lattice.tolist(),
            "atoms": [[s, *map(float, x)] for s, x in zip(self.species, self.frac)],
        }

    # ---- writers ------------------------------------------------------
    def to_poscar(self, title="distortkit"):
        order = list(dict.fromkeys(self.species))
        lines = [title, "1.0"]
        lines += ["  %.10f %.10f %.10f" % tuple(v) for v in self.lattice]
        lines.append(" ".join(order))
        lines.append(" ".join(str(self.species.count(s)) for s in order))
        lines.append("Direct")
        for s in order:
            for sp, x in zip(self.species, self.frac):
                if sp == s:
                    lines.append("  %.10f %.10f %.10f" % tuple(x % 1.0 if abs(x % 1.0 - 1) > 1e-12 else 0.0 for x in x))
        return "\n".join(lines) + "\n"

    def to_cif(self, title="distortkit"):
        a, b, c, al, be, ga = self.lattice_parameters
        out = [
            f"data_{title}",
            "_symmetry_space_group_name_H-M 'P 1'",
            "_symmetry_Int_Tables_number 1",
            f"_cell_length_a {a:.6f}",
            f"_cell_length_b {b:.6f}",
            f"_cell_length_c {c:.6f}",
            f"_cell_angle_alpha {al:.4f}",
            f"_cell_angle_beta {be:.4f}",
            f"_cell_angle_gamma {ga:.4f}",
            "loop_",
            "_symmetry_equiv_pos_as_xyz",
            "'x, y, z'",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]
        counts = {}
        for s, x in zip(self.species, self.frac):
            counts[s] = counts.get(s, 0) + 1
            out.append(f"{s}{counts[s]} {s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}")
        return "\n".join(out) + "\n"
