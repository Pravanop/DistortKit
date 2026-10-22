"""Symmetry of phonons: irreps, child symmetries, and an independent check of the predictions with spglib.

The data is examples/hfo2/fluorite_springs_band.yaml: a phonopy band.yaml for fluorite HfO2 (Fm-3m) from a spring model
(tools/make_fluorite_bands.py).  Only the X point is analysed here (index 4); Gamma and L work but take much longer.
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from distortkit import isodata  # noqa: E402
from distortkit import phonons as ph  # noqa: E402

BAND = os.path.join(ROOT, "examples", "hfo2", "fluorite_springs_band.yaml")
X, W = 4, 9                      # q-point indices of X = (1/2, 0, 1/2) and W = (1/2, 1/4, 3/4) in the file


def subset(bd, idx):
    """A BandData holding only the given q-points (keeps the tests fast)."""
    return ph.BandData(bd.lattice, bd.species, bd.masses, bd.frac, bd.qpts[idx], bd.dist[idx], bd.freq[idx], bd.eig[idx], [])


class TestBandYaml(unittest.TestCase):
    def test_reads_phonopy_output(self):
        bd = ph.read_band_yaml(BAND)
        self.assertEqual(bd.species, ["Hf", "O", "O"])
        self.assertEqual(bd.eig.shape, (20, 9, 3, 3))
        self.assertEqual([lab for _, lab in bd.ticks], ["Γ", "X", "W", "L", "Γ"])
        np.testing.assert_allclose(bd.qpts[X], [0.5, 0, 0.5], atol=1e-6)
        # eigenvectors are normalised over the cell
        np.testing.assert_allclose((np.abs(bd.eig[X]) ** 2).sum(axis=(1, 2)), 1, atol=1e-6)

    def test_rejects_files_without_eigenvectors(self):
        with open(BAND) as fh:
            text = fh.read()
        with self.assertRaises(ValueError):
            ph.read_band_yaml(text.replace("      - [", "      - ("))     # the eigenvector lines


class TestXPoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bd = ph.read_band_yaml(BAND)
        cls.ps = ph.PhononSymmetry(cls.bd.structure, labels=isodata.labels_available())
        cls.q = cls.bd.qpts[X]
        cls.ma, cls.info = cls.ps.star(cls.q)
        cls.modes = cls.ps.classify(cls.q, cls.bd.eig[X], cls.bd.freq[X], q_index=X, masses=cls.bd.masses)

    def test_parent_and_star(self):
        self.assertEqual(self.info.parent, "Fm-3m (#225)")
        self.assertEqual((self.info.star_size, self.info.n_cells), (3, 4))
        self.assertEqual(sorted((i.dim for i in self.info.irreps)), [3, 3, 3, 6, 6])

    def test_every_band_is_one_irrep(self):
        for m in self.modes:
            self.assertGreater(m.weight, 0.999, f"band {m.band + 1}")
            self.assertLess(m.outside, 1e-9, f"band {m.band + 1}")
            self.assertTrue(m.clean)

    def test_bands_per_irrep_match_the_decomposition(self):
        # an irrep of dimension d with c copies (one per atomic orbit) appears c*d/star_size times at one arm of the star
        for ir in self.ma.irreps:
            n = sum(m.irrep == ir.index for m in self.modes)
            self.assertEqual(n, len(ir.modes) * ir.dim // self.info.star_size, f"irrep {ir.index}")
        self.assertEqual(len(self.modes), 9)

    @unittest.skipUnless(isodata.labels_available(), "needs the ISOTROPY label tables")
    def test_standard_labels(self):
        labels = sorted(i.label for i in self.info.irreps)
        self.assertEqual(labels, ["X1+", "X2-", "X3-", "X5+", "X5-"])

    def test_the_soft_mode_is_x2_and_leads_to_p42nmc(self):
        # fluorite HfO2 is unstable: its soft mode is X2- (imaginary), which condenses to the tetragonal P4_2/nmc phase
        m = self.modes[0]
        self.assertLess(m.freq, -1.0)
        self.assertEqual(sum(x.freq < -0.05 for x in self.modes), 1)                   # the only unstable mode at X
        self.assertEqual(m.snapshots[0].subgroup.number, 137)
        if isodata.labels_available():
            self.assertEqual(self.info.irrep(m.irrep).label, "X2-")

    def test_adaptation_separates_a_mixed_level(self):
        """A phonon code may return any mixture of a degenerate level.  Mix eigenvectors of three different irreps on purpose
        (a real rotation, then a complex one) and check that adaptation recovers one irrep per vector."""
        E = self.bd.eig[X][[0, 3, 4]]                      # X2-, X3-, X1+
        want = {m.irrep for m in (self.modes[0], self.modes[3], self.modes[4])}
        self.assertEqual(len(want), 3)
        rng = np.random.default_rng(1)
        real_mix = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        cplx_mix = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
        for name, U in (("real", real_mix), ("complex", cplx_mix)):
            mixed = np.tensordot(U, E, axes=(1, 0))
            before = [self.ps.classify_mode(self.q, e, masses=self.bd.masses).weight for e in mixed]
            self.assertLess(min(before), 0.9, f"{name}: the mixture should not be a clean irrep")
            fixed = self.ps._adapt_cluster(self.ma, self.q, mixed, self.bd.masses, True)
            np.testing.assert_allclose(np.linalg.norm(fixed.reshape(3, -1), axis=1), 1, atol=1e-9)   # unit-norm, like phonopy's
            res = [self.ps.classify_mode(self.q, e, masses=self.bd.masses) for e in fixed]
            self.assertTrue(all(r.weight > 0.999 for r in res), f"{name}: {[r.weight for r in res]}")
            self.assertEqual({r.irrep for r in res}, want, name)

    def test_doublets_are_adapted_to_special_directions(self):
        # the X5- doublets (bands 2-3 and 8-9) are degenerate; adapted, each member lies along a special direction (Pmmn)
        for b in (1, 2, 7, 8):
            m = self.modes[b]
            self.assertTrue(m.adapted and m.clean, f"band {b + 1}")
            self.assertEqual(m.snapshots[0].subgroup.number, 59)

    def test_adapted_modes_are_real_standing_waves(self):
        for m in self.modes:
            self.assertEqual([s.part for s in m.snapshots], ["re"], f"band {m.band + 1}")

    def test_adapted_eigenvectors_are_unit_norm(self):
        adapted = [m for m in self.modes if m.adapted]
        self.assertEqual(len(adapted), 6)
        for m in adapted:
            self.assertAlmostEqual(float(np.linalg.norm(m.eigenvector)), 1.0, places=9, msg=f"band {m.band + 1}")

    def test_nondegenerate_modes_keep_the_files_eigenvector(self):
        for m in self.modes:
            if not m.degenerate_with:
                self.assertFalse(m.adapted)
                self.assertIsNone(m.eigenvector)

    def test_possible_child_symmetries(self):
        x2 = next(i for i in self.info.irreps if i.label == "X2-") if isodata.labels_available() else None
        if x2 is not None:
            numbers = {s.number for s in x2.subgroups}
            self.assertIn(137, numbers)                             # (a,0,0): P4_2/nmc
            self.assertIn(16, numbers)                              # general direction: P222
        for ir in self.info.irreps:
            self.assertTrue(ir.subgroups)
            self.assertEqual(ir.subgroups[-1].fix_dim, ir.dim)       # the last one is the general direction

    @unittest.skipUnless(__import__("importlib").util.find_spec("spglib"), "needs spglib")
    def test_predictions_agree_with_spglib(self):
        """Freeze each mode into the structure and ask spglib (independent of distortkit) for its space group."""
        import spglib
        numbers = {s: i + 1 for i, s in enumerate(dict.fromkeys(self.bd.species))}
        checked = 0
        for m in self.modes:
            e = m.eigenvector if m.eigenvector is not None else self.bd.eig[X][m.band]
            z = self.ps.field(self.ma, self.q, e, masses=self.bd.masses)
            for sn in m.snapshots:
                if sn.subgroup is None:
                    continue
                f = z.real if sn.part == "re" else z.imag
                child = self.ma.child_structure(f / np.linalg.norm(f), 0.05 * np.sqrt(len(f)))
                cell = (child.lattice, child.frac, [numbers[s] for s in child.species])
                got = spglib.get_symmetry_dataset(cell, symprec=1e-3).number
                self.assertEqual(got, sn.subgroup.number, f"band {m.band + 1}: predicted {sn.subgroup.space_group}")
                checked += 1
        self.assertEqual(checked, 9)


class TestRigidTranslation(unittest.TestCase):
    def test_acoustic_modes_at_gamma_do_not_change_the_structure(self):
        # a uniform displacement only moves the crystal; spglib sees Fm-3m again, so no child symmetry may be predicted.
        # max_dim=0 skips the (slow) subgroup enumeration, which this test does not need
        bd = ph.read_band_yaml(BAND)
        ps = ph.PhononSymmetry(bd.structure, labels=False, max_dim=0)
        modes = ps.classify(bd.qpts[0], bd.eig[0], bd.freq[0], masses=bd.masses)
        self.assertEqual([m.rigid for m in modes], [True] * 3 + [False] * 6)
        for m in modes[:3]:
            self.assertTrue(all(s.subgroup is None for s in m.snapshots))
        # away from q = 0 nothing is a rigid translation
        ps2 = ph.PhononSymmetry(bd.structure, labels=False, max_dim=0)
        self.assertFalse(any(m.rigid for m in ps2.classify(bd.qpts[X], bd.eig[X], bd.freq[X], masses=bd.masses)))


class TestWebEntryPoint(unittest.TestCase):
    """distortkit.web is what the browser calls (JSON in, JSON out): it must agree with the command line."""

    def test_matches_the_file_analysis(self):
        from distortkit import web
        bd = ph.read_band_yaml(BAND)
        cli = ph.analyze_band_yaml(subset(bd, [X]), labels=False, max_dim=0)["qpoints"][0]
        z = bd.eig[X]
        req = {"lattice": bd.lattice.tolist(), "frac": bd.frac.tolist(), "species": bd.species, "masses": bd.masses.tolist(),
               "q": bd.qpts[X].tolist(), "freq": bd.freq[X].tolist(), "qi": 0,
               "eig": np.stack([z.real, z.imag], axis=-1).ravel().tolist(),            # what the page sends: re, im pairs
               "settings": {"max_cells": 16, "max_dim": 0, "labels": False, "mass_scale": True, "basis_phase": True}}
        stages = []
        out = json.loads(web.analyze(json.dumps(req), progress=lambda s, f: stages.append(s)))
        self.assertEqual(out["star"], cli["star"])

        def strip(bands):          # eigenvectors are compared numerically below: two runs differ at the 1e-33 level
            return [{k: v for k, v in b.items() if k != "eigenvector"} for b in bands]
        self.assertEqual(strip(out["bands"]), strip(cli["bands"]))
        for a, b in zip(out["bands"], cli["bands"]):
            self.assertEqual(a["eigenvector"] is None, b["eigenvector"] is None)
            if a["eigenvector"] is not None:
                np.testing.assert_allclose(a["eigenvector"], b["eigenvector"], atol=1e-9)
        self.assertEqual(out["star_info"]["parent"], "Fm-3m (#225)")
        self.assertIn("Classifying the modes", stages)
        # a q-point with a too-large star comes back as a plain reason, not an exception
        req["q"] = [0.5, 0.25, 0.75]
        self.assertIn("32-cell", json.loads(web.analyze(json.dumps(req)))["skipped"])

    def test_bundle_contains_what_the_browser_needs(self):
        import importlib.util
        import zipfile
        spec = importlib.util.spec_from_file_location("bundle", os.path.join(ROOT, "viewer", "tools", "build_py_bundle.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import io
        names = set(zipfile.ZipFile(io.BytesIO(mod.build())).namelist())
        for need in ("distortkit/__init__.py", "distortkit/web.py", "distortkit/phonons.py", "distortkit/tables/space_groups.json",
                     "distortkit/tables/data_isotropy.txt"):
            self.assertIn(need, names)
        self.assertFalse([n for n in names if "__pycache__" in n])


class TestLatticeFromADftCode(unittest.TestCase):
    """phonopy files carry lattices with numerical noise (~1e-7 relative) and arbitrary orientation; neither may change the answer."""

    def test_noise_and_orientation_do_not_matter(self):
        bd = ph.read_band_yaml(BAND)
        s = bd.structure
        exact = ph.PhononSymmetry(s, labels=False)
        ref = exact.classify(bd.qpts[X], bd.eig[X], bd.freq[X], masses=bd.masses)
        rng = np.random.default_rng(5)
        rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rot *= np.sign(np.linalg.det(rot))                                       # a proper rotation
        lat = (s.lattice + 3e-7 * np.abs(s.lattice).max() * rng.normal(size=(3, 3))) @ rot.T   # noisy and rotated
        ps = ph.PhononSymmetry(ph.Structure(lat, s.frac, s.species), labels=False)
        self.assertIsNone(ps.problem)
        modes = ps.classify(bd.qpts[X], bd.eig[X] @ rot.T, bd.freq[X], masses=bd.masses)      # eigenvectors turn with the crystal
        for a, b in zip(ref, modes):
            self.assertGreater(b.weight, 0.999)
            self.assertEqual(a.dim, b.dim)
            self.assertEqual([sn.subgroup.number if sn.subgroup else None for sn in a.snapshots],
                             [sn.subgroup.number if sn.subgroup else None for sn in b.snapshots], f"band {a.band + 1}")
        # adapted eigenvectors come back in the file's frame: analysing them again gives a clean assignment
        for m in modes:
            if m.adapted:
                again = ps.classify_mode(bd.qpts[X], m.eigenvector, masses=bd.masses)
                self.assertGreater(again.weight, 0.999)
                self.assertEqual(again.irrep, m.irrep)


class TestApproximateSymmetry(unittest.TestCase):
    def test_a_slightly_distorted_structure_is_reported_not_crashed(self):
        """Real structures are often only approximately symmetric.  At a tight tolerance the symmetry found may not be a group;
        the analysis must say so (and what to try), not die inside the irrep splitter."""
        bd = ph.read_band_yaml(BAND)
        s = bd.structure
        rng = np.random.default_rng(3)
        noisy = ph.Structure(s.lattice, s.frac + rng.normal(scale=4e-4, size=s.frac.shape), s.species)   # ~2e-3 A of noise
        for tol in (1e-3, 4e-3):
            ps = ph.PhononSymmetry(noisy, tol=tol, labels=False, max_dim=0)
            try:
                ps.star(bd.qpts[X])
            except ph.Unclassifiable as exc:
                self.assertRegex(str(exc), r"tolerance|approximately")      # a reason the user can act on
        # exactly symmetric structures are unaffected
        ps = ph.PhononSymmetry(s, labels=False, max_dim=0)
        self.assertIsNone(ps.problem)
        ps.star(bd.qpts[X])


class TestWholeFile(unittest.TestCase):
    def test_skips_large_stars_and_serialises(self):
        bd = subset(ph.read_band_yaml(BAND), [X, W])
        res = ph.analyze_band_yaml(bd, max_cells=16, labels=False)
        json.dumps(res)                                              # plain data, ready for the viewer
        self.assertEqual(res["format"], "distortkit-phonon-symmetry")
        x, w = res["qpoints"]
        self.assertEqual(len(x["bands"]), 9)
        self.assertIn("32-cell", w["skipped"])
        self.assertEqual(len(res["stars"]), 1)
        text = ph.report(res, subgroups=False)
        self.assertIn("Pmmn (#59)", text)
        self.assertIn("not analysed", text)

    def test_q_that_is_not_a_simple_fraction_is_skipped(self):
        bd = subset(ph.read_band_yaml(BAND), [X])
        bd.qpts[0] = [0.4999, 0.0123, 0.5]
        res = ph.analyze_band_yaml(bd, labels=False)
        self.assertIn("not a fraction", res["qpoints"][0]["skipped"])


if __name__ == "__main__":
    unittest.main()
