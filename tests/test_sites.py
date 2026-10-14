import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import compare, find_symmetry, identify_space_group, read_structure, to_primitive
from distortkit.wyckoff import describe_sites

HF = os.path.join(HERE, "..", "examples", "hfo2")
import test_compare as T


def sites(path):
    s = read_structure(path)
    prim, _ = to_primitive(s)
    ops = find_symmetry(prim)
    sg = identify_space_group(ops, prim.lattice)
    return sg, describe_sites(prim, ops, sg)


class TestWyckoff(unittest.TestCase):
    def test_labels_match_the_findsym_cifs(self):
        """Ground truth: the Wyckoff letters FINDSYM wrote into these CIFs."""
        sg, st = sites(os.path.join(HF, "Pbcm.cif"))
        self.assertEqual([(x.species, x.label, x.template) for x in st],
                         [("Hf", "4d", "(x,y,1/4)"), ("O", "4d", "(x,y,1/4)"), ("O", "4c", "(x,1/4,0)")])
        self.assertEqual([x.site_symmetry for x in st], ["m", "m", "2"])
        self.assertEqual([x.label for x in sites(os.path.join(HF, "Pca21.cif"))[1]], ["4a", "4a", "4a"])
        self.assertEqual([x.label for x in sites(os.path.join(HF, "Pbca.cif"))[1]], ["8c", "8c", "8c"])

    def test_perovskite(self):
        s = T.perovskite()
        ops = find_symmetry(s)
        sg = identify_space_group(ops, s.lattice)
        got = {x.species: (x.label, x.site_symmetry) for x in describe_sites(s, ops, sg)}
        self.assertEqual(got, {"Sr": ("1b", "m-3m"), "Ti": ("1a", "m-3m"), "O": ("3d", "4/mmm")})


class TestPathwayOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = compare(read_structure(os.path.join(HF, "Pbcm.cif")), read_structure(os.path.join(HF, "Pca21.cif")))

    def test_modes_and_space_groups(self):
        r = self.res
        self.assertEqual((r.parent_sg.number, r.child_sg.number), (57, 29))
        top = sorted(r.modes, key=lambda m: -m.amplitude)
        self.assertEqual([m.label.text for m in top], ["GM2-", "GM1+"])
        self.assertAlmostEqual(top[0].amplitude, 1.1844, places=3)
        self.assertAlmostEqual(top[1].amplitude, 0.3815, places=3)
        self.assertEqual(top[0].best_subgroup.space_group.number, 29)      # the polar mode leads to Pca2_1

    def test_mode_fields_add_up_to_the_atomic_displacements(self):
        r = self.res
        total = sum(m.atom_displacements for m in r.modes)
        self.assertLess(np.abs(total - r.displacement_cart).max(), 1e-9)

    def test_site_amplitudes_are_consistent(self):
        r = self.res
        for m in r.modes:       # a mode's amplitude is the root-sum-square of its amplitudes over the sites
            self.assertAlmostEqual(np.sqrt(sum(a**2 for a in m.site_amplitudes.values())), m.amplitude, places=8)
        for si in r.sites:      # and a site's rms displacement is built from its modes
            amp2 = sum(m.site_amplitudes.get(si.index, 0.0) ** 2 for m in r.modes)
            direct = float(np.sum(r.displacement_cart[si.atoms] ** 2))
            self.assertAlmostEqual(amp2, direct, places=8)

    def test_polar_displacement_pattern(self):
        r = self.res
        polar = max(r.modes, key=lambda m: m.amplitude)
        u = polar.atom_displacements[:, 0, :]
        sp = r.parent.species
        # no net translation, and the polar mode moves Hf along one axis only (they sit on a mirror plane)
        self.assertLess(np.abs(r.displacement_cart.sum(axis=0)).max(), 1e-9)
        hf = u[[i for i, s in enumerate(sp) if s == "Hf"]]
        self.assertLess(np.abs(hf - hf[0]).max(), 1e-9)                    # same displacement for all four Hf
        self.assertEqual(int(np.argmax(np.abs(hf[0]))), 2)                  # along parent c
        # report and json contain the tables
        text = r.report(atoms=True)
        self.assertIn("displacement by Wyckoff site of Pbcm", text)
        self.assertIn("4d (x,y,1/4)", text)
        self.assertIn("displacement of every atom", text)
        d = r.to_dict()
        self.assertEqual(len(d["atoms"]), 12)
        self.assertEqual([s["wyckoff"] for s in d["sites"]], ["4d", "4d", "4c"])
        json.dumps(d)


class TestTiltBySite(unittest.TestCase):
    def test_only_oxygen_moves(self):
        rng = np.random.default_rng(11)
        f, Nc = T.tilt_field(0.2)
        child = T.make_child(T.perovskite(), T.M0, f, np.zeros((3, 3)), rng)
        r = compare(T.perovskite(), child, labels=False)
        by = {si.species: si for si in r.sites}
        main = max(r.modes, key=lambda m: m.amplitude)
        self.assertEqual(set(main.site_amplitudes), {by["O"].index})
        self.assertAlmostEqual(main.site_amplitudes[by["O"].index], main.amplitude, places=8)


if __name__ == "__main__":
    unittest.main()
