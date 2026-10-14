import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import ModeAnalysis, Structure, compare, read_structure
from distortkit import freeze as F
from distortkit.io import _parse_cif, _num

import test_compare as T

HF = os.path.join(HERE, "..", "examples", "hfo2")


def hf_freezer(tmp):
    res = compare(read_structure(os.path.join(HF, "Pbcm.cif")), read_structure(os.path.join(HF, "Pca21.cif")))
    return res, F.Freezer.load(res.save(os.path.join(tmp, "a.json"), meta={"parent": "Pbcm.cif", "child": "Pca21.cif"}))


def read(path):
    with open(path) as fh:
        return fh.read()


def parse(text):
    tags, loops = _parse_cif(text)
    atoms = moments = None
    for header, rows in loops:
        if "_atom_site_fract_x" in header:
            atoms = (header, rows)
        if "_atom_site_moment.crystalaxis_x" in header:
            moments = (header, rows)
    return tags, atoms, moments


class TestMcif(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res, cls.fz = hf_freezer(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_structure_and_vectors_of_a_single_mode(self):
        f = self.fz.field(["GM2-"])
        text, count = F.mcif_text(self.fz, f, "test")
        tags, atoms, moments = parse(text)
        self.assertEqual((count, len(atoms[1]), len(moments[1])), (12, 12, 12))
        self.assertEqual(tags["_symmetry_space_group_name_h-m"], "P 1")
        self.assertEqual(tags["_space_group_magn.number_bns"], "1.1")
        # the cell is the child's supercell of the parent
        self.assertAlmostEqual(_num(tags["_cell_length_a"]), 5.2832064, places=5)
        # atoms at the PARENT positions (wrapped): compare with the parent's own coordinates
        par = self.fz.parent
        got = {r[0]: [float(x) for x in r[2:5]] for r in atoms[1]}
        self.assertTrue(all(0 <= v < 1 for pos in got.values() for v in pos))
        # orthogonal cell: components along unit axes are the Cartesian components (up to the axis permutation)
        mags = sorted(np.linalg.norm([float(x) for x in r[1:4]]) for r in moments[1])
        self.assertAlmostEqual(mags[-1], 0.4762, places=3)               # the 4d oxygens
        self.assertAlmostEqual(mags[0], 0.2390, places=3)                # Hf
        self.assertNotIn("-0.00000000", text)

    def test_zero_displacements_are_left_out(self):
        f = self.fz.field(["GM2-"], atoms="Hf")
        text, count = F.mcif_text(self.fz, f, "Hf only")
        tags, atoms, moments = parse(text)
        self.assertEqual((count, len(atoms[1]), len(moments[1])), (4, 12, 4))       # every atom listed, 4 arrows
        self.assertEqual({r[0] for r in moments[1]}, {"Hf1", "Hf2", "Hf3", "Hf4"})

    def test_amplitude_sign_and_arrow_scale(self):
        rc = F.enumerate_recipes(self.fz, modes=["GM2-"])
        with tempfile.TemporaryDirectory() as d:
            r1 = F.run_mcif(self.fz, rc, amplitude=0.5, outdir=d, log=lambda *_: None)[0]
            r2 = F.run_mcif(self.fz, rc, amplitude=-0.5, outdir=os.path.join(d, "n"), log=lambda *_: None)[0]
            r3 = F.run_mcif(self.fz, rc, amplitude=0.5, arrow_scale=4.0, outdir=os.path.join(d, "s"), log=lambda *_: None)[0]
            vec = lambda row: np.array([[float(x) for x in r[1:4]] for r in parse(read(row["file"]))[2][1]])
            v1, v2, v3 = vec(r1), vec(r2), vec(r3)
            self.assertAlmostEqual(float(np.sqrt(np.sum(v1**2))), 0.5, places=6)    # RSS amplitude, orthogonal cell
            self.assertLess(np.abs(v1 + v2).max(), 1e-7)                            # negative amplitude reverses the arrows
            self.assertLess(np.abs(v3 - 4 * v1).max(), 1e-7)

    def test_vectors_are_right_in_a_non_orthogonal_cell(self):
        """Hexagonal supercell (gamma = 120): the stored axis components must rebuild the Cartesian displacements."""
        prim = Structure.from_parameters(3.2, 3.2, 5.2, 90, 90, 120,
                                         [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.38], [2 / 3, 1 / 3, 0.88]],
                                         ["Zn", "Zn", "S", "S"])
        ma = ModeAnalysis(prim, (0.5, 0, 0))
        ir = [i for i in ma.irreps if i.dim == ma.irreps[-1].dim][0]
        u = 0.1 * ma.displacement(ir, ma.isotropy_subgroups(ir)[0].opd)
        Nc = ma.n_cells
        child = T.make_child(prim, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
                             lambda a, t: u[a * Nc + ma._canon(t[None, :])[0]], np.zeros((3, 3)), np.random.default_rng(7))
        res = compare(prim, child, labels=False)
        with tempfile.TemporaryDirectory() as d:
            fz = F.Freezer.load(res.save(os.path.join(d, "a.json")))
        field = fz.field(["all"])
        text, _ = F.mcif_text(fz, field, "hex")
        tags, atoms, moments = parse(text)
        a, b, c, al, be, ga = (_num(tags[k]) for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                                                        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"))
        self.assertAlmostEqual(ga, 120.0, places=3)
        unit = Structure.from_parameters(a, b, c, al, be, ga, np.zeros((0, 3)), []).lattice
        unit = unit / np.linalg.norm(unit, axis=1)[:, None]
        got = {r[0]: np.array([float(x) for x in r[1:4]]) @ unit for r in moments[1]}      # rebuilt Cartesian vectors
        want = {fz.atoms[i].label: field[fz.atoms[i].a, fz.atoms[i].c] for i in range(len(fz.atoms))}
        labels = sorted(got)
        G1 = np.array([[got[x] @ got[y] for y in labels] for x in labels])
        G2 = np.array([[want[x] @ want[y] for y in labels] for x in labels])
        self.assertLess(np.abs(G1 - G2).max(), 1e-7)       # same lengths and angles between all arrows (up to a rotation)


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        return subprocess.run([sys.executable, "-m", "distortkit", *args], capture_output=True, text=True, env=env)

    def test_show_without_mcif_writes_nothing_and_with_mcif_writes_files(self):
        an = os.path.join(HF, "Pca21_analysis.json")
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("compare", os.path.join(HF, "Pbcm.cif"), os.path.join(HF, "Pca21.cif"), "--save", os.path.join(d, "an.json"))
            self.assertEqual(r.returncode, 0, r.stderr)
            an = os.path.join(d, "an.json")
            r = self.run_cli("show", an)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "arrows")))
            r = self.run_cli("show", an, "--mcif")                     # default: one file per mode
            self.assertEqual(r.returncode, 0, r.stderr)
            files = sorted(x for x in os.listdir(os.path.join(d, "arrows")) if x.endswith(".mcif"))
            self.assertEqual(files, ["Pca21_01_GM2-.mcif", "Pca21_02_GM1+.mcif"])
            self.assertIn("wrote 2 mCIF file(s)", r.stdout)
            r = self.run_cli("show", an, "--mcif", "--modes", "GM2-,GM1+", "--outdir", os.path.join(d, "c"))   # coupled modes
            self.assertEqual(len(os.listdir(os.path.join(d, "c"))) - 2, 1)       # one mcif + the two manifests
            r = self.run_cli("show", an, "--mcif", "--part", "GM2-:Hf", "--outdir", os.path.join(d, "h"))
            f = [x for x in os.listdir(os.path.join(d, "h")) if x.endswith(".mcif")][0]
            self.assertEqual(len(parse(read(os.path.join(d, "h", f)))[2][1]), 4)
            r = self.run_cli("show", an, "--mcif", "--part", "GM1+", "--outdir", os.path.join(d, "h"))     # second run, same folder
            with open(os.path.join(d, "h", "arrows_manifest.txt")) as fh:
                self.assertEqual(len(fh.read().strip().splitlines()), 3)             # header + both runs
            r = self.run_cli("show", an, "--mcif", "--modes", "nonsense", "--outdir", os.path.join(d, "x"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("unknown mode", r.stderr)


if __name__ == "__main__":
    unittest.main()
