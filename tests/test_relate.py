import os
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import ModeAnalysis, Structure, read_structure
from distortkit import relate as R
from distortkit.subgroup import child_ops_in_parent

import test_compare as T

HF = os.path.join(HERE, "..", "examples", "hfo2")


def hf(name):
    return read_structure(os.path.join(HF, name + ".cif"))


def irrep_child(k, species, dim, point_group, amp, M, rng, strain=None, index=None):
    """Perovskite child distorted along the isotropy subgroup ``point_group`` of the irrep at ``k`` (picked by
    species and dimension, or by ``index``), in the supercell ``M``."""
    prim = T.perovskite()
    ma = ModeAnalysis(prim, k)
    ir = [i for i in ma.irreps if (i.index == index if index else (i.dim == dim and i.species == species))][0]
    iso = [i for i in ma.isotropy_subgroups(ir) if i.point_group == point_group][0]
    u = amp * ma.displacement(ir, iso.opd)
    Nc = ma.n_cells
    field = lambda a, Tc: u[a * Nc + ma._canon(Tc[None, :])[0]]
    return T.make_child(prim, M, field, np.zeros((3, 3)) if strain is None else np.diag(strain), rng)


M_R3C = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])


def sibling_pair(seed=5):
    """A cubic-perovskite family: an R-point tilt (I4/mcm) and a polar zone-centre mode (P4mm): neither is a subgroup
    of the other, both are of Pm-3m."""
    rng = np.random.default_rng(seed)
    a = irrep_child((0.5, 0.5, 0.5), ["O"], 3, "4/mmm", 0.25, T.M0, rng)
    b = irrep_child((0, 0, 0), None, 3, "4mm", 0.12, np.eye(3, dtype=int), rng, strain=(-0.003, -0.003, 0.008), index=1)
    return a, b


class TestDirections(unittest.TestCase):
    def test_parent_and_tilted_child(self):
        rng = np.random.default_rng(1)
        f, _ = T.tilt_field(0.2)
        child = T.make_child(T.perovskite(), T.M0, f, np.diag([-0.002, -0.002, 0.005]), rng)
        rel = R.relate(T.perovskite(), child, names=("parent", "child"))
        self.assertEqual(rel.verdict, "a>b")
        self.assertTrue(rel.ab.ok)
        self.assertEqual(rel.ab.n_cells, 2)                     # the child's primitive cell is 2 parent cells
        self.assertEqual(rel.ba.status, "count")
        self.assertIn("index", rel.ab.ops.summary())
        # and the other way round
        rel2 = R.relate(child, T.perovskite(), names=("child", "parent"))
        self.assertEqual(rel2.verdict, "b>a")

    def test_identical_and_isotypic(self):
        p = T.perovskite()
        rel = R.relate(p, p)
        self.assertEqual(rel.verdict, "same")
        rng = np.random.default_rng(2)
        strained = T.make_child(p, np.eye(3, dtype=int), lambda a, Tc: np.zeros(3), np.diag([0.01, 0.01, 0.01]), rng)
        self.assertEqual(R.relate(p, strained).verdict, "isotypic")

    def test_different_composition(self):
        p = T.perovskite()
        q = Structure(p.lattice, p.frac, ["Ba" if s == "Sr" else s for s in p.species])
        # same atoms count, different element
        self.assertEqual(R.relate(p, q).ab.status, "composition")
        self.assertEqual(R.relate(p, q).verdict, "unrelated")

    def test_atom_counts_not_multiples(self):
        p = T.perovskite()
        rel = R.relate(p, hf("Pca21"))
        self.assertEqual(rel.verdict, "unrelated")
        self.assertTrue(any("different compositions" in n for n in rel.notes))

    def test_translation_mismatch_is_classified(self):
        # shift every translation of the parent's operations by a quarter cell along x: rotations still match, translations do not
        p = T.perovskite()
        rel = R.relate(p, p)
        side = rel.a
        n = len(side.prim)
        M = np.eye(3, dtype=int)
        bad = [(Rr, t + np.array([0.25, 0.0, 0.0])) for Rr, t in side.ops]
        chk = child_ops_in_parent(side.prim, bad, M, side.prim.frac, side.prim.species, np.zeros((n, 3)), side.prim.lattice)
        self.assertFalse(chk.ok)
        self.assertEqual({k for k, _ in chk.extra}, {"translation"})
        self.assertTrue(all("translation (" in t for _, t in chk.extra))


class TestHfO2(unittest.TestCase):
    """The three polymorphs of the example set."""

    def test_pbcm_parent_of_pca21_and_pbca(self):
        rel = R.relate(hf("Pbcm"), hf("Pca21"), names=("Pbcm", "Pca21"))
        self.assertEqual(rel.verdict, "a>b")
        self.assertEqual((rel.a.sg.number, rel.b.sg.number), (57, 29))
        self.assertAlmostEqual(rel.ab.ops.index, 2.0)
        self.assertEqual(rel.ba.status, "not-subgroup")           # same cell, but Pca21's group is the smaller one
        self.assertTrue(any(k == "rotation" for k, _ in rel.ba.ops.extra))
        rel = R.relate(hf("Pbcm"), hf("Pbca"), names=("Pbcm", "Pbca"))
        self.assertEqual(rel.verdict, "a>b")
        self.assertEqual(rel.ab.n_cells, 2)

    def test_pca21_and_pbca_are_not_nested(self):
        rel = R.relate(hf("Pca21"), hf("Pbca"), names=("Pca21", "Pbca"))
        self.assertEqual(rel.verdict, "unrelated")
        self.assertEqual(rel.ab.status, "not-subgroup")          # overlays, but Pbca has symmetry Pca21 lacks
        self.assertTrue(any("has symmetry" in n for n in rel.notes))
        self.assertIn("relate --common", " ".join(rel.notes))
        self.assertIn("NOT operations of the parent", rel.report())

    def test_compare_flags_a_wrong_parent(self):
        from distortkit.checks import has_errors
        res = T.compare(hf("Pca21"), hf("Pbcm"), min_star_amp=5e-2, fit_min_amp=5e-2, strain_tol=0.15, match_tol=1.5)
        self.assertFalse(res.ops_check.ok)
        self.assertTrue(has_errors(res.findings))
        self.assertIn("not-subgroup-ops", [f.code for f in res.findings])
        good = T.compare(hf("Pbcm"), hf("Pca21"), min_star_amp=5e-2, fit_min_amp=5e-2)
        self.assertTrue(good.ops_check.ok)
        self.assertNotIn("not-subgroup-ops", [f.code for f in good.findings])


class TestCommonParent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a, b = sibling_pair()
        cls.rel = R.relate(a, b, names=("tilt", "polar"), common=True)

    def test_found_and_is_the_aristotype(self):
        rel = self.rel
        self.assertEqual(rel.verdict, "siblings")
        cp = rel.common
        self.assertEqual(cp.chosen.parent.sg.number, 221)          # Pm-3m, not the P4/mmm that the polar one alone suggests
        self.assertEqual(cp.chosen.parent.n_prim, 5)
        self.assertTrue(all(c.ok for c in cp.candidates if c.parent.sg.number in (221, 123)))
        self.assertLess(cp.chosen.cost, 0.15)

    def test_mode_table(self):
        rows = {r.k: r for r in self.rel.common.rows}
        R_pt, G_pt = (Fraction(1, 2),) * 3, (Fraction(0),) * 3
        self.assertEqual(set(rows), {R_pt, G_pt})
        self.assertEqual(rows[R_pt].where, "only A")
        self.assertEqual(rows[G_pt].where, "only B")
        self.assertAlmostEqual(rows[R_pt].amp_a, 0.25 * np.sqrt(4 / 2), places=3)    # R-point cell holds 2 cells
        self.assertIn(rows[R_pt].label, ("R4+", "R5-"))             # the label depends on the parent's origin (A/B-site)
        self.assertEqual(rows[G_pt].label, "GM4-")
        txt = self.rel.report()
        self.assertIn("only tilt", txt)
        self.assertIn("only polar", txt)
        self.assertIn("depend on the origin", txt)

    def test_no_nesting_without_common_flag(self):
        a, b = sibling_pair()
        rel = R.relate(a, b, names=("tilt", "polar"))
        self.assertEqual(rel.verdict, "unrelated")
        self.assertIsNone(rel.common)
        self.assertIn("relate --common", rel.report())

    def test_pick_and_no_modes(self):
        a, b = sibling_pair()
        rel = R.relate(a, b, names=("tilt", "polar"), common=True, modes=False, pick=2)
        cp = rel.common
        self.assertIsNone(cp.comp_a)
        self.assertEqual(cp.chosen.parent.sg.number, 123)          # second valid candidate: P4/mmm
        with self.assertRaises(ValueError):
            R.relate(a, b, names=("tilt", "polar"), common=True, modes=False, pick=9)

    def test_same_irrep_in_both_is_matched(self):
        # a0a0c- (I4/mcm) and a-a-a- (R-3c) tilts: different isotropy subgroups of the same R4+ irrep
        rng = np.random.default_rng(8)
        a = irrep_child((0.5, 0.5, 0.5), ["O"], 3, "4/mmm", 0.20, T.M0, rng)
        b = irrep_child((0.5, 0.5, 0.5), ["O"], 3, "-3m", 0.14, M_R3C, rng)
        rel = R.relate(a, b, names=("tetragonal", "rhombohedral"), common=True)
        self.assertEqual(rel.verdict, "siblings")
        self.assertEqual(len(rel.common.rows), 1)
        row = rel.common.rows[0]
        self.assertEqual((row.where, row.k), ("both", (Fraction(1, 2),) * 3))
        self.assertAlmostEqual(row.amp_a, 0.20 * np.sqrt(4 / 2), places=3)
        self.assertAlmostEqual(row.amp_b, 0.14 * np.sqrt(2 / 2), places=3)
        self.assertEqual({(rel.common.comp_a.child_sg.number, rel.common.comp_b.child_sg.number)}, {(140, 167)})

    def test_nested_pair_needs_no_common_parent(self):
        rng = np.random.default_rng(1)
        child = irrep_child((0.5, 0.5, 0.5), ["O"], 3, "4/mmm", 0.2, T.M0, rng)
        rel = R.relate(T.perovskite(), child, names=("parent", "child"), common=True)
        self.assertEqual(rel.verdict, "a>b")
        self.assertIsNone(rel.common)
        self.assertIn("already nested", rel.report())


class TestHfO2Common(unittest.TestCase):
    def test_fluorite_is_a_common_parent_of_pca21_and_pbca(self):
        fl = read_structure(os.path.join(HF, "fluorite_Fm-3m.cif"))
        rel = R.relate(hf("Pca21"), hf("Pbca"), names=("Pca21", "Pbca"), common=True, parents=[("fluorite", fl)],
                       search=False, modes=False)
        self.assertEqual(rel.verdict, "siblings")
        c = rel.common.chosen
        self.assertEqual((c.parent.sg.number, c.parent.n_prim), (225, 3))
        self.assertEqual((c.da.n_cells, c.db.n_cells), (4, 8))
        self.assertEqual(len(c.da.ops.extra) + len(c.db.ops.extra), 0)

    def test_pbcm_is_not_a_subgroup_of_fluorite(self):
        """Pbcm and fluorite are both parents of Pca21 but not of each other: siblings, so no common parent from fluorite."""
        fl = read_structure(os.path.join(HF, "fluorite_Fm-3m.cif"))
        rel = R.relate(hf("Pbcm"), hf("Pca21"), common=True, parents=[("fluorite", fl)], search=False, modes=False)
        self.assertEqual(rel.verdict, "a>b")                        # nested: Pbcm > Pca21, no common parent needed
        rel = R.relate(hf("Pbcm"), hf("Pbca"), common=True)
        self.assertEqual(rel.verdict, "a>b")
        cands = R.common_parent(hf("Pbcm"), hf("Pca21"), parents=[("fluorite", fl)], search=False, modes=False)
        self.assertFalse(any(c.ok for c in cands.candidates))      # Pbcm does not fit under fluorite

    def test_no_candidate_is_reported_honestly(self):
        rel = R.relate(hf("Pca21"), hf("Pbca"), names=("Pca21", "Pbca"), common=True, modes=False)
        self.assertEqual(rel.verdict, "unrelated")                  # the HfO2 distortions are too large for the search
        txt = rel.report()
        self.assertIn("pass --parent", txt)

    @unittest.skipUnless(os.environ.get("DISTORTKIT_SLOW"), "set DISTORTKIT_SLOW=1 (about a minute)")
    def test_full_mode_table(self):
        fl = read_structure(os.path.join(HF, "fluorite_Fm-3m.cif"))
        rel = R.relate(hf("Pca21"), hf("Pbca"), names=("Pca21", "Pbca"), common=True, parents=[("fluorite", fl)], search=False)
        rows = {(r.label, r.k): r for r in rel.common.rows}
        x2 = [r for (lab, k), r in rows.items() if lab == "X2-"]
        self.assertTrue(x2 and x2[0].where == "only A")             # the X2- mode of ferroelectric HfO2
        self.assertTrue(any(r.where == "both" for r in rel.common.rows))


if __name__ == "__main__":
    unittest.main()
