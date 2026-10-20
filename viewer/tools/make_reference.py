"""Independent check of the viewer's displacement convention, and the reference data its test compares against.

Reads one mode (q-point index 26, band 4) of examples/nacl_springs.yaml with PyYAML, builds the displacement field on the 3x3x3
supercell of the model that generated the file as u = Re[e/sqrt(m) exp(2 pi i q.r)] (r = full reduced position), and asserts that
it satisfies the equation of motion F = -m w^2 u with the model's force constants (a wrong phase convention fails this at q != 0).
The complex field is written to tests/ref_nacl.json for tests/viewer.test.mjs.

    python3 viewer/tools/make_reference.py
"""
import json
import os
import sys

import numpy as np
import yaml
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_demo as md  # noqa: E402

QI, BAND = 26, 4
doc = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "nacl_springs.yaml")))
lat = np.array(doc["lattice"])
q = np.array(doc["phonon"][QI]["q-position"])
band = doc["phonon"][QI]["band"][BAND]
freq = band["frequency"]
ev = np.array([[complex(*c) for c in atom] for atom in band["eigenvector"]])
frac = np.array([p["coordinates"] for p in doc["points"]])
mass = np.array([p["mass"] for p in doc["points"]])

unit = PhonopyAtoms(symbols=[p["symbol"] for p in doc["points"]], cell=lat, scaled_positions=frac)
ph = Phonopy(unit, supercell_matrix=np.diag([3, 3, 3]))
sc = ph.supercell
a = 5.64
nn, nnn = a / 2, a / np.sqrt(2)
shells = {("Na", "Cl"): [(nn, 7.0)], ("Cl", "Na"): [(nn, 7.0)], ("Na", "Na"): [(nnn, 1.2)], ("Cl", "Cl"): [(nnn, 1.8)]}
fc = md.spring_force_constants(sc, shells)
Phi = fc.transpose(0, 2, 1, 3).reshape(len(sc) * 3, len(sc) * 3)

red = np.linalg.solve(lat.T, sc.positions.T).T            # reduced coordinates in the primitive cell basis, [0, 3)
red = np.mod(np.round(red, 8), 3.0)
kappa = np.array([0 if s == "Na" else 1 for s in sc.symbols])
field = np.array([ev[k] / np.sqrt(mass[k]) * np.exp(2j * np.pi * (q @ r)) for k, r in zip(kappa, red)])
u = field.real
F = -Phi @ u.reshape(-1)
target = (sc.masses[:, None] * (freq / 15.633302) ** 2 * u).reshape(-1)
resid = np.linalg.norm(F + target) / np.linalg.norm(target)
print(f"q = {q}, band {BAND + 1}, {freq:.4f} THz: equation-of-motion residual {resid:.2e}")
assert resid < 1e-5, "displacement convention does not satisfy F = -m w^2 u"

out = {"qi": QI, "band": BAND, "freq": freq, "atoms": [
    {"kappa": int(k), "red": [round(float(x), 6) for x in r], "re": [float(x) for x in f.real], "im": [float(x) for x in f.imag]}
    for k, r, f in zip(kappa, red, field)]}
json.dump(out, open(os.path.join(HERE, "..", "tests", "ref_nacl.json"), "w"))
print("wrote tests/ref_nacl.json")
