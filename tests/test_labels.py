import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distortkit import ModeAnalysis, Structure

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def perovskite(shift=0.0):
    s = Structure.from_json(os.path.join(EX, "srtio3_cubic.json"))
    return Structure(s.lattice, (s.frac + shift) % 1.0, s.species)


def labels(structure, k):
    ma = ModeAnalysis(structure, k)
    lab = ma.irrep_labels()
    return {tuple(ir.species) + (ir.dim,): lab[ir.index] for ir in ma.irreps}, ma


class TestLabels(unittest.TestCase):
    def test_R_point_with_B_site_at_origin(self):
        lab, ma = labels(perovskite(), (0.5, 0.5, 0.5))
        self.assertEqual(ma.space_group().number, 221)
        self.assertEqual(lab[("O", 3)].text, "R4+")            # the octahedral tilt, as in the literature
        self.assertEqual(lab[("Ti", 3)].text, "R4-")
        self.assertEqual(lab[("O", "Sr", 3)].text, "R5+")
        self.assertTrue(all(l.certain for l in lab.values()))

    def test_origin_dependence_of_parity(self):
        """Moving the origin to the A site swaps the parity labels: tilt R4+ -> R5-."""
        lab, ma = labels(perovskite(shift=0.5), (0.5, 0.5, 0.5))
        self.assertEqual(lab[("O", 3)].text, "R5-")

    def test_gamma_and_X(self):
        lab, _ = labels(perovskite(), (0, 0, 0))
        self.assertEqual({v.text for v in lab.values()}, {"GM4-", "GM5-"})
        lab, _ = labels(perovskite(), (0, 0, 0.5))
        self.assertEqual({v.text for v in lab.values()}, {"X1+", "X3-", "X4-", "X5+", "X5-"})

    def test_ambiguity_is_reported_not_guessed(self):
        lab, _ = labels(perovskite(), (0.5, 0.5, 0))
        amb = [v for v in lab.values() if not v.certain]
        self.assertTrue(amb)
        for v in amb:
            self.assertGreater(len(v.candidates), 1)
        # the in-phase tilt is one of two indistinguishable labels, listed as such
        self.assertTrue(any(set(v.candidates) == {"M2+", "M3+"} for v in lab.values()))

    def test_gamma_labels_are_never_other_k_points(self):
        ma = ModeAnalysis(perovskite(), (0, 0, 0))
        for lab in ma.irrep_labels().values():
            self.assertTrue(all(c.startswith("GM") for c in lab.candidates))
        ma = ModeAnalysis(perovskite(), (0.5, 0.5, 0.5))
        for lab in ma.irrep_labels().values():
            self.assertFalse(any(c.startswith("GM") for c in lab.candidates))

    def test_child_space_groups_of_the_tilt(self):
        ma = ModeAnalysis(perovskite(), (0.5, 0.5, 0.5))
        ir = [i for i in ma.irreps if i.dim == 3 and i.species == ["O"]][0]
        got = sorted(ma.child_space_group(iso).number for iso in ma.isotropy_subgroups(ir))
        self.assertEqual(got, sorted([140, 167, 74, 12, 15, 2]))      # I4/mcm R-3c Imma C2/m C2/c P-1


if __name__ == "__main__":
    unittest.main()
