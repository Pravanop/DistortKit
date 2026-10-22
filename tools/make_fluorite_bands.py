"""Write examples/hfo2/fluorite_springs_band.yaml: a phonopy band.yaml (with eigenvectors) for fluorite HfO2.

The force constants are central springs between Hf-O, O-O and Hf-Hf neighbours.  They give the right geometry and symmetry
(Fm-3m, 3 atoms in the primitive cell) but not the frequencies of real HfO2.  The second-neighbour O-O spring is slightly negative,
which makes the X2- mode (the O sublattice moving antiphase, the soft mode of cubic hafnia on the way to P4_2/nmc and the
monoclinic ground state) imaginary, about -3 THz, as in the real material; every other mode at X is stable.  The file exists to
test the symmetry analysis of phonons (``distortkit phonons``) and to demo it without a DFT calculation.  Needs phonopy (not a
dependency of distortkit).

    python3 tools/make_fluorite_bands.py
"""
import os
import warnings

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "hfo2", "fluorite_springs_band.yaml")
A = 5.08  # cubic lattice parameter, Angstrom


def spring_fc(sc, shells):
    """Force constants -k u u^T for neighbour pairs at the listed distances (minimum image), acoustic sum rule on the diagonal."""
    lat, pos, sym = sc.cell, sc.positions, sc.symbols
    n = len(pos)
    fc = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            f = np.linalg.solve(lat.T, pos[j] - pos[i])
            v = lat.T @ (f - np.round(f))
            d = np.linalg.norm(v)
            for r0, k in shells.get((sym[i], sym[j]), []):
                if abs(d - r0) < 0.05:
                    fc[i, j] += -k * np.outer(v / d, v / d)
    for i in range(n):
        fc[i, i] = -fc[i].sum(axis=0)
    return fc


def main():
    cell = np.array([[0, A / 2, A / 2], [A / 2, 0, A / 2], [A / 2, A / 2, 0]])
    unit = PhonopyAtoms(symbols=["Hf", "O", "O"], cell=cell, scaled_positions=[[0, 0, 0], [.25, .25, .25], [.75, .75, .75]])
    ph = Phonopy(unit, supercell_matrix=np.diag([3, 3, 3]))
    d_hfo, d_oo, d_oo2, d_hfhf = A * np.sqrt(3) / 4, A / 2, A / np.sqrt(2), A / np.sqrt(2)
    shells = {("Hf", "O"): [(d_hfo, 6.0)], ("O", "Hf"): [(d_hfo, 6.0)],
              ("O", "O"): [(d_oo, 1.0), (d_oo2, -1.075)],      # the negative second-neighbour spring softens X2-
              ("Hf", "Hf"): [(d_hfhf, 0.8)]}
    ph.force_constants = spring_fc(ph.supercell, shells)
    G, X, W, L = [0, 0, 0], [.5, 0, .5], [.5, .25, .75], [.5, .5, .5]
    path = [G, X, W, L, G]
    segs = [np.array([np.linspace(x, y, 5) for x, y in zip(a, b)]).T for a, b in zip(path[:-1], path[1:])]
    ph.run_band_structure(segs, with_eigenvectors=True, labels=["$\\Gamma$", "X", "W", "L", "$\\Gamma$"])
    ph.band_structure.write_yaml(filename=OUT)
    print("wrote", os.path.abspath(OUT), os.path.getsize(OUT) // 1024, "kB")


if __name__ == "__main__":
    main()
