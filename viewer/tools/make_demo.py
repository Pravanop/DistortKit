"""Write demo phonon data (phonopy band.yaml, with eigenvectors) for the viewer from simple spring-model force constants.

The force constants are central springs between neighbour shells, so the bands are not those of a real material; they exist
to give the viewer genuine phonopy output (labels, segments, eigenvectors, masses) without needing a DFT calculation.

    python3 viewer/tools/make_demo.py            # writes viewer/examples/*.yaml
"""
from __future__ import annotations

import os
import sys

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "examples")


def spring_force_constants(supercell, shells):
    """FC[i,j] (3x3) for a central spring k_s between atoms whose distance matches shell s (minimum image); ASR on the diagonal."""
    lat = supercell.cell
    pos = supercell.positions
    n = len(pos)
    fc = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # minimum-image vector from i to j
            f = np.linalg.solve(lat.T, pos[j] - pos[i])
            f -= np.round(f)
            v = lat.T @ f
            d = np.linalg.norm(v)
            for shell in shells.get((supercell.symbols[i], supercell.symbols[j]), shells.get(None, [])):
                r0, k, kz = (*shell, 0.0)[:3]
                if abs(d - r0) < 0.05:
                    u = v / d
                    fc[i, j] += -k * np.outer(u, u)
                    fc[i, j] += -kz * np.outer([0, 0, 1.0], [0, 0, 1.0])  # out-of-plane stiffness (keeps flexural modes real)
    for i in range(n):
        fc[i, i] = -fc[i].sum(axis=0)
    return fc


def build(name, cell, symbols, frac, supercell_matrix, shells, path, labels, nsample=40):
    unit = PhonopyAtoms(symbols=symbols, cell=cell, scaled_positions=frac)
    ph = Phonopy(unit, supercell_matrix=supercell_matrix)
    sc = ph.supercell
    ph.force_constants = spring_force_constants(sc, shells)
    ph.run_band_structure(path_to_qpoints(path, nsample), with_eigenvectors=True, labels=labels)
    out = os.path.join(OUT, f"{name}.yaml")
    ph.band_structure.write_yaml(filename=out)
    print("wrote", out, os.path.getsize(out) // 1024, "kB")


def path_to_qpoints(path, n):
    segs = []
    for a, b in zip(path[:-1], path[1:]):
        segs.append(np.array([np.linspace(x, y, n) for x, y in zip(a, b)]).T)
    return segs


def graphene():
    a = 2.46
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 15.0]]
    frac = [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0]]
    cc = a / np.sqrt(3)
    shells = {None: [(cc, 25.0, 3.0), (a, 4.0, 0.6)]}
    G, M, K = [0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0]
    build("graphene_springs", cell, ["C", "C"], frac, np.diag([3, 3, 1]), shells,
          [G, M, K, G], ["$\\Gamma$", "M", "K", "$\\Gamma$"])


def nacl():
    a = 5.64
    cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    frac = [[0, 0, 0], [0.5, 0.5, 0.5]]
    nn, nnn = a / 2, a / np.sqrt(2)
    shells = {("Na", "Cl"): [(nn, 7.0)], ("Cl", "Na"): [(nn, 7.0)], ("Na", "Na"): [(nnn, 1.2)], ("Cl", "Cl"): [(nnn, 1.8)]}
    # fcc primitive reciprocal points in reduced coordinates
    G, X, W, K, L = [0, 0, 0], [0.5, 0, 0.5], [0.5, 0.25, 0.75], [0.375, 0.375, 0.75], [0.5, 0.5, 0.5]
    build("nacl_springs", cell, ["Na", "Cl"], frac, np.diag([3, 3, 3]), shells,
          [G, X, G, L], ["$\\Gamma$", "X", "$\\Gamma$", "L"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["graphene", "nacl"]
    for w in which:
        globals()[w]()
