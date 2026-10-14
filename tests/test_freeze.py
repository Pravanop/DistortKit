import itertools
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
from distortkit import compare, read_structure
from distortkit import freeze as F
from distortkit.checks import symmetry_info_in_file

import test_compare as T

HF = os.path.join(HERE, "..", "examples", "hfo2")


def hf_freezer(tmp):
    res = compare(read_structure(os.path.join(HF, "Pbcm.cif")), read_structure(os.path.join(HF, "Pca21.cif")))
    path = res.save(os.path.join(tmp, "a.json"), meta={"parent": "Pbcm.cif", "child": "Pca21.cif"})
    return res, F.Freezer.load(path)


class TestSelections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res, cls.fz = hf_freezer(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ids_labels_and_special_names(self):
        fz = self.fz
        self.assertEqual([m.id for m in fz.modes], ["m1", "m2"])
        self.assertIs(fz.mode_by_ref("GM2-"), fz.modes[0])
        self.assertIs(fz.mode_by_ref("gm1+"), fz.modes[1])
        self.assertIs(fz.mode_by_ref("m2"), fz.modes[1])
        self.assertLess(np.abs(fz.mode_field("all") + fz.mode_field("rest") - fz.mode_field("child")).max(), 1e-12)

    def test_atom_selectors(self):
        fz = self.fz
        n = lambda sel: int(fz.atom_mask(sel).sum())
        self.assertEqual(n("Hf"), 4)
        self.assertEqual(n("O"), 8)
        self.assertEqual(n("Hf1"), 1)
        self.assertEqual(n("O1-O4"), 4)
        self.assertEqual(n("O@4c"), 4)
        self.assertEqual(n("@4d"), 8)          # Hf and O both sit on a 4d site
        self.assertEqual(n("s3"), 4)
        self.assertEqual(n("Hf,O5"), 5)
        self.assertEqual(n("all"), 12)
        self.assertEqual(n(None), 12)

    def test_part_scale_and_amplitude(self):
        fz = self.fz
        full = fz.part_field("GM2-")
        self.assertLess(np.abs(fz.part_field("GM2-*0.5") - 0.5 * full).max(), 1e-12)
        self.assertLess(np.abs(fz.part_field("GM2-*-1") + full).max(), 1e-12)
        self.assertAlmostEqual(float(np.sqrt(np.sum(fz.part_field("GM2-=0.3") ** 2))), 0.3, places=10)
        hf_only = fz.part_field("GM2-:Hf")
        self.assertAlmostEqual(float(np.sqrt(np.sum(hf_only ** 2))), 0.4779, places=3)
        self.assertEqual(float(np.abs(hf_only[4:]).max()), 0.0)             # only the four Hf atoms moved
        # the same through the global --atoms filter, and with an exclusion
        self.assertLess(np.abs(fz.field(["GM2-"], atoms="Hf") - hf_only).max(), 1e-12)
        self.assertLess(np.abs(fz.field(["GM2-"], exclude="O") - hf_only).max(), 1e-12)
        # parts add up
        both = fz.field(["GM2-", "GM1+"])
        self.assertLess(np.abs(both - fz.field(["all"])).max(), 1e-12)

    def test_errors_are_helpful(self):
        fz = self.fz
        with self.assertRaises(F.SelectionError) as cm:
            fz.part_field("XY9")
        self.assertIn("GM2-", str(cm.exception))
        with self.assertRaises(F.SelectionError) as cm:
            fz.atom_mask("Zr")
        self.assertIn("species", str(cm.exception))
        with self.assertRaises(F.SelectionError):
            fz.atom_mask("O@9z")
        with self.assertRaises(F.SelectionError):
            fz.part_field("GM2-:Hf*abc")

    def test_symmetry_of_frozen_modes_follows_the_isotropy_subgroups(self):
        fz = self.fz
        for ref, expect in (("GM2-", 29), ("GM1+", 57), ("all", 29)):
            s, info = fz.build([ref])
            self.assertEqual(F.symmetry_of(s)[2].number, expect, ref)
        s, _ = fz.build(["GM2-"], scale=0.0)
        self.assertEqual(F.symmetry_of(s)[2].number, 57)                    # nothing frozen: still the parent


class TestRoundTrip(unittest.TestCase):
    def test_child_recipe_reproduces_the_analysis(self):
        """freeze 'child' with the child's cell, re-analyse it against the parent: same modes, same strain."""
        rng = np.random.default_rng(21)
        f, Nc = T.tilt_field(0.2)
        child = T.make_child(T.perovskite(), T.M0, f, np.diag([-0.002, -0.002, 0.005]), rng, noise=0.002)
        res = compare(T.perovskite(), child, labels=False)
        with tempfile.TemporaryDirectory() as d:
            fz = F.Freezer.load(res.save(os.path.join(d, "a.json")))
            rebuilt, info = fz.build(["child"], cell="child")
            self.assertAlmostEqual(info["rss"], res.total_displacement, places=9)
            res2 = compare(T.perovskite(), rebuilt, labels=False)
        self.assertAlmostEqual(res2.total_displacement, res.total_displacement, places=7)
        self.assertLess(np.abs(np.sort(np.linalg.eigvalsh(res2.strain)) - np.sort(np.linalg.eigvalsh(res.strain))).max(), 1e-7)
        a1 = sorted(round(m.amplitude, 6) for m in res.modes)
        a2 = sorted(round(m.amplitude, 6) for m in res2.modes)
        self.assertEqual(a1, a2)

    def test_scaled_structure_has_scaled_amplitude(self):
        with tempfile.TemporaryDirectory() as d:
            res, fz = hf_freezer(d)
            s, _ = fz.build(["GM2-"], scale=0.5)
            res2 = compare(read_structure(os.path.join(HF, "Pbcm.cif")), s, labels=False)
        top = max(res2.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(top.amplitude, 0.5 * 1.1844, places=3)
        self.assertEqual(len([m for m in res2.modes if m.amplitude > 1e-6]), 1)   # nothing else sneaked in


class TestEnumeration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res, cls.fz = hf_freezer(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts(self):
        fz = self.fz
        n = len(fz.modes)
        self.assertEqual(len(F.enumerate_recipes(fz, each="mode")), n)
        self.assertEqual(len(F.enumerate_recipes(fz, pairs=True)), n * (n - 1) // 2)
        self.assertEqual(len(F.enumerate_recipes(fz, cumulative=True)), n)
        self.assertEqual(len(F.enumerate_recipes(fz, combinations=True)), 2**n - 1)
        self.assertEqual(len(F.enumerate_recipes(fz, each="site")), 3)
        self.assertEqual(len(F.enumerate_recipes(fz, each="species")), 2)
        self.assertEqual(len(F.enumerate_recipes(fz, each="atom")), 12)
        with self.assertRaises(F.SelectionError):
            F.enumerate_recipes(fz)

    def test_each_site_displaces_only_that_site(self):
        for rc in F.enumerate_recipes(self.fz, each="site"):
            f = self.fz.field(rc.parts, rc.atoms)
            moved = {self.fz.atoms[i].site for i in range(12) if np.abs(f[self.fz.atoms[i].a, self.fz.atoms[i].c]).max() > 1e-12}
            self.assertEqual(len(moved), 1)

    def test_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            rows = F.run(self.fz, F.enumerate_recipes(self.fz, each="mode"), scales=(0.5, 1.0), fmt="both", outdir=d, log=lambda *_: None)
            self.assertEqual(len(rows), 4)
            names = sorted(os.listdir(d))
            self.assertIn("manifest.json", names)
            self.assertEqual(len([x for x in names if x.endswith(".vasp")]), 4)
            cifs = [os.path.join(d, x) for x in names if x.endswith(".cif")]
            self.assertEqual(len(cifs), 4)
            # CIFs carry their own symmetry operations and read back to the same atoms
            polar = [c for c in cifs if "GM2-" in c and "A1.184" in c][0]
            self.assertTrue(symmetry_info_in_file(polar))
            self.assertEqual(len(read_structure(polar)), 12)
            with open(os.path.join(d, "manifest.json")) as fh:
                man = json.load(fh)
            self.assertEqual({r["space_group"] for r in man}, {"Pca2_1 (#29)", "Pbcm (#57)"})
            # POSCAR atom order = species-grouped order of the legend labels
            pos = read_structure([os.path.join(d, x) for x in names if x.endswith(".vasp")][0])
            self.assertEqual(pos.species, ["Hf"] * 4 + ["O"] * 8)


class TestAmplitudeSeries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res, cls.fz = hf_freezer(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_series(self, d, **kw):
        rc = F.enumerate_recipes(self.fz, modes=["GM2-"])
        return F.run(self.fz, rc, outdir=d, log=lambda *_: None, **kw)

    def test_final_amplitude_and_number_of_structures(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.run_series(d, amplitude=1.0, n=5)
            self.assertEqual([round(r["amplitude"], 6) for r in rows], [0.0, 0.25, 0.5, 0.75, 1.0])
            for r in rows:                       # the written structure really has that amplitude
                self.assertAlmostEqual(r["rss"], abs(r["amplitude"]), places=9)
            self.assertEqual([r["space_group"] for r in rows][0], "Pbcm (#57)")
            self.assertTrue(all(r["space_group"] == "Pca2_1 (#29)" for r in rows[1:]))
            self.assertEqual(len([x for x in os.listdir(d) if x.endswith(".vasp")]), 5)

    def test_start_below_zero_gives_the_opposite_domain(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.run_series(d, amplitude=0.8, n=5, start=-0.8)
            self.assertEqual([round(r["amplitude"], 6) for r in rows], [-0.8, -0.4, 0.0, 0.4, 0.8])
            self.assertTrue(any("Am0.800" in os.path.basename(r["files"][0]) for r in rows))

    def test_defaults_use_the_amplitude_found_in_the_child(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.run_series(d)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["amplitude"], 1.1844, places=3)
            self.assertAlmostEqual(rows[0]["scale"], 1.0, places=9)
            self.assertNotIn("_A", os.path.basename(rows[0]["files"][0]))

    def test_amplitude_applies_to_the_selected_atoms_and_parts(self):
        with tempfile.TemporaryDirectory() as d:
            rc = F.enumerate_recipes(self.fz, modes=["GM2-"], atoms="Hf")
            rows = F.run(self.fz, rc, amplitude=0.3, n=1, outdir=d, log=lambda *_: None)
            self.assertAlmostEqual(rows[0]["rss"], 0.3, places=9)          # RSS of the Hf displacement alone
            rc = F.enumerate_recipes(self.fz, parts=["GM2-:Hf", "GM1+:O*0.5"])
            rows = F.run(self.fz, rc, amplitude=0.6, n=2, outdir=d, log=lambda *_: None)
            self.assertAlmostEqual(rows[1]["rss"], 0.6, places=9)
            self.assertAlmostEqual(rows[0]["rss"], 0.0, places=12)

    def test_empty_selection_and_bad_counts_are_errors(self):
        with tempfile.TemporaryDirectory() as d:
            rc = F.enumerate_recipes(self.fz, modes=["GM2-"], atoms="Hf", exclude="Hf")
            with self.assertRaises(F.SelectionError):
                F.run(self.fz, rc, outdir=d, log=lambda *_: None)
            with self.assertRaises(F.SelectionError):
                self.run_series(d, amplitude=1.0, n=0)

    def test_prompts_on_a_terminal(self):
        try:
            import pty
            import select
        except ImportError:
            self.skipTest("no pty")
        with tempfile.TemporaryDirectory() as d:
            path = self.res.save(os.path.join(d, "a.json"), meta={"parent": "p", "child": "Pca21.cif"})
            out = os.path.join(d, "o")
            pid, fd = pty.fork()
            if pid == 0:
                os.environ["PYTHONPATH"] = os.path.join(HERE, "..")
                os.execvp(sys.executable, [sys.executable, "-m", "distortkit", "freeze", path, "--modes", "GM2-", "--outdir", out])

            def read_until(token, timeout=60):
                import time
                buf, t0 = b"", time.time()
                while time.time() - t0 < timeout and token.encode() not in buf:
                    if select.select([fd], [], [], 0.5)[0]:
                        try:
                            chunk = os.read(fd, 4096)
                        except OSError:
                            break
                        if not chunk:
                            break
                        buf += chunk
                return buf.decode(errors="replace")

            self.assertIn("Final amplitude", read_until("to use it)"))
            os.write(fd, b"0.9\n")
            self.assertIn("How many structures", read_until("final amplitude)"))
            os.write(fd, b"3\n")
            self.assertIn("wrote 3 structure", read_until("wrote 3 structure"))
            os.waitpid(pid, 0)
            self.assertEqual(len([x for x in os.listdir(out) if x.endswith(".vasp")]), 3)


class TestCLI(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        return subprocess.run([sys.executable, "-m", "distortkit", *args], capture_output=True, text=True, env=env, cwd=cwd)

    def test_compare_show_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("compare", os.path.join(HF, "Pbcm.cif"), os.path.join(HF, "Pca21.cif"), "--save", os.path.join(d, "an.json"))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("[m1]", r.stdout)
            self.assertIn("analysis saved to", r.stdout)
            r = self.run_cli("show", os.path.join(d, "an.json"))
            self.assertEqual(r.returncode, 0, r.stderr)
            for word in ("MODES", "GM2-", "SITES", "4c", "ATOMS", "Hf1", "EXAMPLES"):
                self.assertIn(word, r.stdout)
            out = os.path.join(d, "o")
            r = self.run_cli("freeze", os.path.join(d, "an.json"), "--modes", "GM2-", "--scales", "-1,0,1", "--outdir", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len([x for x in os.listdir(out) if x.endswith(".vasp")]), 3)
            out2 = os.path.join(d, "o2")
            r = self.run_cli("freeze", os.path.join(d, "an.json"), "--modes", "GM2-", "--start", "-0.5", "--amplitude", "0.5", "-n", "3", "--outdir", out2)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(x for x in os.listdir(out2) if x.endswith(".vasp")),
                             ["Pca21_01_GM2-_A0.000.vasp", "Pca21_01_GM2-_A0.500.vasp", "Pca21_01_GM2-_Am0.500.vasp"])
            r = self.run_cli("freeze", os.path.join(d, "an.json"), "--part", "GM2-:Hf", "--part", "GM1+:O*0.5", "--outdir", os.path.join(d, "p"))
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_cli("freeze", os.path.join(d, "an.json"), "--modes", "nonsense", "--outdir", os.path.join(d, "q"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("unknown mode", r.stderr)
            r = self.run_cli("freeze", os.path.join(d, "an.json"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("nothing selected", r.stderr)


if __name__ == "__main__":
    unittest.main()
