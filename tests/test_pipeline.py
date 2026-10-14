import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import Structure, checks
from distortkit.pipeline import analyze
from distortkit.symmetrize import find_pymatgen_python, symmetrize, tolerance_table

import test_compare as T

PMG = find_pymatgen_python()


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def read(path):
    with open(path) as fh:
        return fh.read()


def noisy_parent(shift=0.1, noise=3e-4, seed=3):
    p = T.perovskite()
    rng = np.random.default_rng(seed)
    return Structure(p.lattice, (p.frac + shift + rng.normal(scale=noise, size=p.frac.shape)) % 1.0, p.species)


def tilted_child(seed=4, amp=0.2, eps=(-0.002, -0.002, 0.005)):
    rng = np.random.default_rng(seed)
    f, Nc = T.tilt_field(amp)
    return T.make_child(T.perovskite(), T.M0, f, np.diag(eps), rng, noise=0.0005), amp * np.sqrt(4 / Nc)


class TestSymmetrize(unittest.TestCase):
    def test_builtin_assigns_symmetry_and_writes_cif(self):
        with tempfile.TemporaryDirectory() as d:
            r = symmetrize(noisy_parent(), symprec="auto", method="builtin", cif_path=os.path.join(d, "p.cif"))
            self.assertEqual((r.number, r.symbol), (221, "Pm-3m"))
            self.assertEqual(len(r.structure), 5)
            self.assertIn("_symmetry_equiv_pos_as_xyz", read(os.path.join(d, "p.cif")))
            self.assertTrue(checks.symmetry_info_in_file(os.path.join(d, "p.cif")))

    @unittest.skipUnless(PMG, "pymatgen not available (see DISTORTKIT_PMG_PYTHON)")
    def test_pymatgen_assigns_symmetry_in_the_input_setting(self):
        s = noisy_parent(shift=0.1)
        r = symmetrize(s, symprec="auto", method="pymatgen")
        self.assertEqual((r.number, r.method), (221, "pymatgen"))
        self.assertEqual(len(r.structure), 5)
        # the origin is not moved: Ti is still near (0.1, 0.1, 0.1)
        ti = r.structure.frac[[sp == "Ti" for sp in r.structure.species]][0]
        self.assertTrue(np.allclose(ti, 0.1, atol=2e-3))
        # the tolerance table shows the noise: P1 at tight tolerance, Pm-3m once the tolerance covers it
        tab = {t: n for t, n, _ in r.table}
        self.assertEqual(tab[1e-4], 1)
        self.assertEqual(tab[0.1], 221)

    @unittest.skipUnless(PMG, "pymatgen not available")
    def test_standardize_moves_the_origin_and_says_so(self):
        r = symmetrize(noisy_parent(), symprec="auto", method="pymatgen", standardize=True)
        self.assertTrue(any("origin" in n for n in r.notes))

    def test_tolerance_table_builtin(self):
        tab = tolerance_table(noisy_parent(), (1e-4, 1e-2, 1e-1), method="builtin")
        self.assertEqual(tab[0][1], 1)
        self.assertEqual(tab[-1][1], 221)


class TestChecks(unittest.TestCase):
    def test_clean_structure(self):
        self.assertEqual([f for f in checks.check_structure(T.perovskite()) if f.level != "info"], [])

    def test_overlapping_atoms_are_an_error(self):
        p = T.perovskite()
        bad = Structure(p.lattice, np.vstack([p.frac, p.frac[0] + 1e-3]), p.species + ["Sr"])
        f = checks.check_structure(bad)
        self.assertTrue(any(x.code == "overlap" and x.level == "error" for x in f))

    def test_non_element_species_warns(self):
        p = T.perovskite()
        f = checks.check_structure(Structure(p.lattice, p.frac, ["Q"] * 5))
        self.assertTrue(any(x.code == "species-name" for x in f))

    def test_parent_tolerance_warning(self):
        f = checks.check_parent_symmetry(noisy_parent(), 1e-3)
        self.assertTrue(any(x.code == "parent-tolerance" for x in f))

    def test_pair_checks(self):
        p = T.perovskite()
        half = Structure(p.lattice, p.frac[:4], p.species[:4])
        self.assertTrue(any(x.code == "atom-count" for x in checks.check_pair(p, half)))
        wrong = Structure(p.lattice, p.frac, ["Sr", "Ti", "O", "O", "N"])
        self.assertTrue(any(x.code == "composition" for x in checks.check_pair(p, wrong)))

    def test_symmetry_info_detected_with_and_without_id_column(self):
        cif = ("data_t\n_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
               "_cell_angle_gamma 90\nloop_\n_space_group_symop_id\n_space_group_symop_operation_xyz\n1 x,y,z\n2 -x,-y,-z\n"
               "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
               "_atom_site_fract_z\nNa1 Na 0.1 0.2 0.3\n")
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "a.cif"), cif)
            write(os.path.join(d, "b.cif"), cif.replace("_space_group_symop_id\n", "").replace("1 x", "x").replace("2 -x", "-x"))
            write(os.path.join(d, "c.cif"), cif.split("loop_")[0] + "loop_" + cif.split("loop_")[2])   # no operations
            self.assertTrue(checks.symmetry_info_in_file(os.path.join(d, "a.cif")))
            self.assertTrue(checks.symmetry_info_in_file(os.path.join(d, "b.cif")))
            self.assertFalse(checks.symmetry_info_in_file(os.path.join(d, "c.cif")))

    def test_file_checks(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(checks.check_file(os.path.join(d, "nope"))[0].code, "file-missing")
            open(os.path.join(d, "e.vasp"), "w").close()
            self.assertEqual(checks.check_file(os.path.join(d, "e.vasp"))[0].code, "file-empty")


class TestPipeline(unittest.TestCase):
    def run_pipeline(self, method):
        child, expected = tilted_child()
        with tempfile.TemporaryDirectory() as d:
            pp, cp = os.path.join(d, "parent.vasp"), os.path.join(d, "child.vasp")
            write(pp, noisy_parent().to_poscar("noisy parent"))
            write(cp, child.to_poscar("child"))
            res = analyze(pp, cp, symmetrize_parent=method)
            self.assertTrue(os.path.exists(os.path.join(d, "parent_symmetrized.cif")))
        return res, expected

    def check_result(self, res, expected, method):
        self.assertTrue(res.ok)
        self.assertEqual(res.symmetrization.method, method)
        self.assertEqual(res.symmetrization.number, 221)
        c = res.comparison
        self.assertEqual((c.parent_sg.number, c.child_sg.number), (221, 140))
        main = max(c.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(main.amplitude, expected, delta=0.01)
        self.assertEqual(main.label.text, "R4+")
        self.assertEqual(main.best_subgroup.point_group, "4/mmm")
        self.assertLess(c.unexplained, 1e-6)
        text = res.report()
        self.assertIn("R4+", text)
        self.assertIn("I4/mcm (#140)", text)

    def test_poscar_without_symmetry_builtin(self):
        res, exp = self.run_pipeline("builtin")
        self.check_result(res, exp, "builtin")

    @unittest.skipUnless(PMG, "pymatgen not available")
    def test_poscar_without_symmetry_pymatgen(self):
        res, exp = self.run_pipeline("pymatgen")
        self.check_result(res, exp, "pymatgen")

    def test_errors_are_raised_with_a_reason(self):
        with tempfile.TemporaryDirectory() as d:
            pp, cp = os.path.join(d, "p.vasp"), os.path.join(d, "c.vasp")
            p = T.perovskite()
            write(pp, p.to_poscar())
            write(cp, Structure(p.lattice, p.frac[:4], p.species[:4]).to_poscar())
            with self.assertRaises(ValueError) as cm:
                analyze(pp, cp)
            self.assertIn("atoms", str(cm.exception))


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        return subprocess.run([sys.executable, "-m", "distortkit", *args], capture_output=True, text=True, env=env)

    def test_symmetrize_check_compare(self):
        child, expected = tilted_child()
        with tempfile.TemporaryDirectory() as d:
            pp, cp = os.path.join(d, "parent.vasp"), os.path.join(d, "child.vasp")
            write(pp, noisy_parent().to_poscar("noisy parent"))
            write(cp, child.to_poscar("child"))
            r = self.run_cli("check", pp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("space group vs tolerance", r.stdout)
            r = self.run_cli("symmetrize", pp, "--method", "builtin")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Pm-3m (#221)", r.stdout)
            self.assertTrue(os.path.exists(os.path.join(d, "parent_symmetrized.cif")))
            r = self.run_cli("compare", pp, cp, "--symmetrize", "builtin")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("R4+", r.stdout)
            self.assertIn("I4/mcm (#140)", r.stdout)
            r = self.run_cli("compare", pp, os.path.join(d, "missing.vasp"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("error:", r.stderr)


if __name__ == "__main__":
    unittest.main()
