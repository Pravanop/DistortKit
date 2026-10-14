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
from distortkit import explain as E
from distortkit.freeze import Freezer, SelectionError

import test_compare as T

HF = os.path.join(HERE, "..", "examples", "hfo2")


def freezer_for(child, tmp, **kw):
    res = T.compare(T.perovskite(), child, **kw)
    return Freezer.load(res.save(os.path.join(tmp, "a.json")))


class TestFieldGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        f, _ = T.tilt_field(0.2)
        cls.fz = freezer_for(T.make_child(T.perovskite(), T.M0, f, np.zeros((3, 3)), rng), cls.tmp.name)
        cls.G = E.FieldGroup(cls.fz.parent, cls.fz.M, cls.fz.cells)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_only_operations_that_preserve_the_supercell_act(self):
        # Pm-3m has 48 operations; the I4/mcm cell (a-b, a+b, 2c) is kept by the 16 of 4/mmm
        self.assertEqual(len(self.G.ops), 16)
        self.assertEqual(self.G.n_el, 16 * 4)

    def test_identity_and_composition(self):
        rng = np.random.default_rng(0)
        F = rng.normal(size=self.fz.total.shape)
        self.assertTrue(np.allclose(self.G.act(F, self.G.identity), F))
        for _ in range(25):
            e1, e2 = (int(x) for x in rng.integers(0, self.G.n_el, 2))
            lhs = self.G.act(self.G.act(F, e2), e1)
            rhs = self.G.act(F, self.G.mult(e1, e2))
            self.assertTrue(np.allclose(lhs, rhs), (e1, e2))

    def test_projection_is_a_projector(self):
        rng = np.random.default_rng(2)
        F = rng.normal(size=self.fz.total.shape)
        els = sorted(self.G.largest_group(self.G.deviations(self.fz.total), 1e-6))
        P = self.G.project(F, els)
        self.assertTrue(np.allclose(self.G.project(P, els), P))
        self.assertLessEqual(np.sum(P ** 2), np.sum(F ** 2) + 1e-12)
        # the child itself is already invariant under its own group
        self.assertTrue(np.allclose(self.G.project(self.fz.total, els), self.fz.total, atol=1e-9))


class TestLadder(unittest.TestCase):
    def test_clean_child_has_one_rung(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(1)
            f, _ = T.tilt_field(0.2)
            fz = freezer_for(T.make_child(T.perovskite(), T.M0, f, np.zeros((3, 3)), rng), d)
            rungs = E.ladder(fz)
            self.assertEqual([r.sg.number for r in rungs if r.sg][:1], [140])
            self.assertAlmostEqual(rungs[0].explained, 1.0, places=9)

    def test_noisy_child_shows_core_plus_remainder(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(3)
            f, _ = T.tilt_field(0.20)
            child = T.make_child(T.perovskite(), T.M0, f, np.zeros((3, 3)), rng, noise=0.004)
            fz = freezer_for(child, d, min_star_amp=1e-3)
            rungs = E.ladder(fz)
            self.assertEqual(rungs[0].sg.number, 1)                 # exact symmetry destroyed by the noise
            tet = [r for r in rungs if r.sg is not None and r.sg.number == 140]
            self.assertTrue(tet, [r.sg.label() if r.sg else None for r in rungs])
            self.assertGreater(tet[0].explained, 0.98)              # the I4/mcm core carries nearly all of it
            self.assertGreater(tet[0].resid_rss, 0.0)               # ... and the noise is the remainder
            # a larger group can only explain less
            for a, b in zip(rungs, rungs[1:]):
                self.assertGreaterEqual(b.n_el, a.n_el)
                self.assertLessEqual(b.explained, a.explained + 1e-9)
            # at a tolerance above the tilt the pattern is "the parent" and explains nothing
            self.assertEqual(rungs[-1].sg.number, 221)
            self.assertAlmostEqual(rungs[-1].explained, 0.0, places=6)


class TestCascadeHfO2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fz = Freezer.load(os.path.join(HF, "fluorite_to_Pca21_analysis.json"))

    def test_fluorite_to_pca21(self):
        c = E.cascade(self.fz)
        self.assertEqual([lv.driver_name for lv in c.levels], ["X5+", "X2-"])
        self.assertEqual([lv.sg.number for lv in c.levels], [61, 29])
        self.assertEqual(c.child_sg.number, 29)
        self.assertEqual(c.roles["m1"].role, "primary")
        self.assertEqual(c.roles["m2"].role, "independent")
        self.assertEqual(c.roles["m2"].by, ["m1"])
        for mid in ("m3", "m4", "m5"):                                # GM4-, X5-, X3-: only allowed once both are frozen
            self.assertEqual(c.roles[mid].role, "allowed")
            self.assertEqual(c.roles[mid].level, 2)
            self.assertEqual(sorted(c.roles[mid].by), ["m1", "m2"])
        self.assertEqual(c.final_n_el, c.child_n_el)

    def test_choosing_the_primary_mode(self):
        c = E.cascade(self.fz, primary=["X2-"])
        self.assertEqual([lv.driver_name for lv in c.levels], ["X2-", "X5+"])
        self.assertEqual([lv.sg.number for lv in c.levels], [137, 29])        # P4_2/nmc, the tetragonal phase, then Pca2_1
        self.assertEqual(c.roles["m2"].role, "primary")
        with self.assertRaises(SelectionError):
            E.cascade(self.fz, primary=["nonsense"])

    def test_ladder_of_a_clean_structure(self):
        rungs = E.ladder(self.fz)
        self.assertEqual(rungs[0].sg.number, 29)
        self.assertAlmostEqual(rungs[0].explained, 1.0, places=9)
        self.assertEqual(len(rungs), 1)                               # the largest displacement is 0.78 A: Pca2_1 up to 0.4 A
        far = E.ladder(self.fz, tols=(1e-4, 0.4, 2.0))
        self.assertEqual([r.sg.number for r in far], [29, 225])       # a tolerance above the displacements gives back the parent

    def test_fully_symmetric_mode_is_recognised(self):
        fz = Freezer.load(os.path.join(HF, "Pca21_analysis.json"))   # Pbcm -> Pca2_1
        c = E.cascade(fz)
        self.assertEqual(c.roles["m2"].role, "fully-symmetric")        # GM1+
        self.assertEqual(c.roles["m1"].role, "primary")                # GM2-
        self.assertEqual(c.levels[0].sg.number, 29)


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        return subprocess.run([sys.executable, "-m", "distortkit", "explain", *args], capture_output=True, text=True, env=env)

    def test_text_and_json(self):
        a = os.path.join(HF, "fluorite_to_Pca21_analysis.json")
        r = self.run_cli(a)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PRIMARY", r.stdout)
        self.assertIn("INDEPENDENT", r.stdout)
        self.assertIn("allowed once X5+ + X2- is frozen", r.stdout)
        r = self.run_cli(a, "--json", "--primary", "X2-")
        d = json.loads(r.stdout)
        self.assertEqual([lv["space_group"]["number"] for lv in d["cascade"]], [137, 29])
        self.assertEqual(d["child"]["space_group"]["number"], 29)
        self.assertEqual(d["roles"][0]["role"] in ("primary", "independent", "allowed", "fully-symmetric"), True)
        bad = self.run_cli(a, "--primary", "nonsense")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("unknown mode", bad.stderr)


if __name__ == "__main__":
    unittest.main()
