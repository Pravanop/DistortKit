import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import invariants as I
from distortkit.compare import CellSet
from distortkit.explain import FieldGroup
from distortkit.freeze import Freezer
from distortkit.symmetry import find_symmetry

import test_compare as T

HF = os.path.join(HERE, "..", "examples", "hfo2")


def counts(res):
    names = [m.name for m in res.modes]
    return {frozenset((names[i], e) for i, e in enumerate(t.degrees) if e): (t.n_invariants, t.n_new) for t in res.terms}


def K(*pairs):
    return frozenset(pairs)


class TestTiltInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        f, _ = T.tilt_field(0.2)
        res = T.compare(T.perovskite(), T.make_child(T.perovskite(), T.M0, f, np.zeros((3, 3)), rng))
        cls.fz = Freezer.load(res.save(os.path.join(cls.tmp.name, "a.json")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_supercell_holds_all_images(self):
        ops = find_symmetry(self.fz.parent, 1e-3)
        big = I.invariant_supercell(self.fz.M, ops)
        self.assertEqual(round(abs(np.linalg.det(self.fz.M))), 4)           # the I4/mcm cell
        self.assertEqual(round(abs(np.linalg.det(big))), 8)                  # 2x2x2: invariant under all 48 operations
        self.assertEqual(FieldGroup(self.fz.parent, big, CellSet(big).cells).n_el, 48 * 8)

    def test_r_point_tilt_has_the_known_invariants(self):
        res = I.compute(self.fz, degree=6)
        c = counts(res)
        name = res.modes[0].name
        self.assertEqual(res.modes[0].dim, 3)
        # eta^2; eta^4 (2, one new); eta^6 (3, one new): generators of degrees 2, 4, 6 of the cubic group acting on a 3-vector
        self.assertEqual(c, {K((name, 2)): (1, 1), K((name, 4)): (2, 1), K((name, 6)): (3, 1)})

    def test_counts_agree_with_the_molien_average(self):
        ops = find_symmetry(self.fz.parent, 1e-3)
        big = I.invariant_supercell(self.fz.M, ops)
        G = FieldGroup(self.fz.parent, big, CellSet(big).cells)
        F = {m.id: I.extend_field(m.field, CellSet(self.fz.M), G.cells) for m in self.fz.modes}
        op = I.order_parameters(self.fz, list(self.fz.modes), G, F, list(range(G.n_el)))[0]   # rho of every element
        P = I.Powers(op.dim, 6)
        for d in (1, 2, 3, 4, 5, 6):
            molien = sum(np.trace(P.sym(r, d)) for r in op.rho) / G.n_el
            res = I.compute(self.fz, degree=d)
            have = sum(t.n_invariants for t in res.terms if sum(t.degrees) == d)
            self.assertAlmostEqual(molien, have, places=6, msg=f"degree {d}")


class TestHfO2Couplings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fz = Freezer.load(os.path.join(HF, "fluorite_to_Pca21_analysis.json"))
        cls.res = I.compute(cls.fz, degree=3)
        cls.c = counts(cls.res)

    def test_trilinear_improper_coupling_exists(self):
        self.assertEqual(self.c[K(("X5+", 1), ("X2-", 1), ("GM4-", 1))], (1, 1))
        self.assertEqual(self.c[K(("X5+", 3))], (1, 1))
        for absent in (K(("X2-", 3)), K(("GM4-", 3)), K(("X5+", 2), ("X2-", 1)), K(("X2-", 2), ("GM4-", 1))):
            self.assertNotIn(absent, self.c)

    def test_every_secondary_mode_is_induced_in_the_measured_direction(self):
        by = {}
        for i in self.res.induced:
            if i.force >= 1e-6:
                by.setdefault(i.target, []).append(i.match)
        for name in ("X2-", "GM4-", "X5-", "X3-"):
            self.assertTrue(by[name] and min(by[name]) > 0.99, (name, by.get(name)))
        self.assertLess(max(by["X5+"]), 0.9)                       # the primary is not simply induced by the others

    def test_agrees_with_the_cascade(self):
        from distortkit import explain as E
        casc = E.cascade(self.fz)
        names = {m.id: m.text for m in self.fz.modes}
        for mid, role in casc.roles.items():
            if role.role == "allowed":
                self.assertTrue(any(i.target == names[mid] and i.force >= 1e-6 and i.match > 0.99 for i in self.res.induced), mid)

    def test_selection_and_order_do_not_change_the_answer(self):
        a = I.compute(self.fz, ["X5+", "X2-", "GM4-"], degree=3)
        b = I.compute(self.fz, ["GM4-", "X5+", "X2-"], degree=3)
        self.assertEqual(counts(a), counts(b))
        self.assertEqual(counts(a)[K(("X5+", 1), ("X2-", 1), ("GM4-", 1))], (1, 1))
        # without X5+ nothing can induce GM4- at this order
        c = I.compute(self.fz, ["X2-", "GM4-"], degree=3)
        self.assertEqual([i for i in c.induced if i.target == "GM4-"], [])
        self.assertIn("nothing else can induce it", c.report())

    def test_report_and_cli(self):
        txt = self.res.report()
        self.assertIn("X5+ X2- GM4-", txt)
        self.assertIn("induces exactly the measured direction", txt)
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        a = os.path.join(HF, "fluorite_to_Pca21_analysis.json")
        r = subprocess.run([sys.executable, "-m", "distortkit", "invariants", a, "--modes", "X5+,X2-,GM4-", "--json"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertTrue(any(t["degrees"] == {"X5+": 1, "X2-": 1, "GM4-": 1} and t["new"] == 1 for t in d["terms"]))
        r = subprocess.run([sys.executable, "-m", "distortkit", "invariants", a, "--modes", "nonsense"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
