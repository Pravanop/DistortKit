import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distortkit import ModeAnalysis, Structure, find_symmetry, point_group_name, to_primitive

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def perovskite():
    return Structure.from_json(os.path.join(EX, "srtio3_cubic.json"))


def pg_orders(ma, irrep):
    return sorted((i.point_group, i.n_point_ops) for i in ma.isotropy_subgroups(irrep))


class TestSymmetry(unittest.TestCase):
    def test_cubic_perovskite(self):
        s = perovskite()
        ops = find_symmetry(s)
        self.assertEqual(len(ops), 48)
        self.assertEqual(point_group_name([R for R, _ in ops], s.lattice), "m-3m")

    def test_primitive_reduction_rocksalt(self):
        a = 5.63
        s = Structure.from_parameters(a, a, a, 90, 90, 90,
                                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                                       [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]],
                                      ["Na"] * 4 + ["Cl"] * 4)
        self.assertEqual(len(find_symmetry(s)), 192)
        p, B = to_primitive(s)
        self.assertEqual(len(p), 2)
        self.assertAlmostEqual(p.volume, s.volume / 4, places=6)
        self.assertEqual(len(find_symmetry(p)), 48)
        # conventional (0,0,2) is a reciprocal-lattice vector of the fcc lattice (Gamma);
        # conventional (0,0,1) is not: it is the X point
        k = B @ np.array([0.0, 0.0, 2.0])
        self.assertTrue(np.allclose(k - np.rint(k), 0))
        k = B @ np.array([0.0, 0.0, 1.0])
        self.assertFalse(np.allclose(k - np.rint(k), 0))


class TestPerovskiteModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = ModeAnalysis(perovskite(), (0.5, 0.5, 0.5))

    def test_irreps_cover_displacement_space(self):
        dim = sum(ir.dim * len(ir.modes) for ir in self.R.irreps)
        self.assertEqual(dim, 3 * 5)  # 5 atoms x 3, one k-point, 2 cells -> real dim 15

    def test_tilt_mode_isotropy_subgroups(self):
        # R4+ octahedral tilts: the only 3-dim irrep living on oxygen alone.
        tilt = [ir for ir in self.R.irreps if ir.dim == 3 and ir.species == ["O"]]
        self.assertEqual(len(tilt), 1)
        got = pg_orders(self.R, tilt[0])
        # I4/mcm, R-3c, Imma, C2/m, C2/c, F-1
        self.assertEqual(sorted(o for _, o in got), [2, 4, 4, 8, 12, 16])
        self.assertEqual(sorted(p for p, _ in got), sorted(["4/mmm", "-3m", "mmm", "2/m", "2/m", "-1"]))
        domains = {i.point_group + str(i.fix_dim): i.n_conjugates for i in self.R.isotropy_subgroups(tilt[0])}
        self.assertEqual(domains["4/mmm1"], 3)   # three tilt axes
        self.assertEqual(domains["-3m1"], 4)     # four body diagonals

    def test_roundtrip_amplitude_and_symmetry(self):
        ir = [i for i in self.R.irreps if i.dim == 3 and i.species == ["O"]][0]
        for iso in self.R.isotropy_subgroups(ir):
            u = self.R.displacement(ir, iso.opd)
            self.assertAlmostEqual(np.linalg.norm(u), 1.0)
            held = self.R.symmetry_of(u)
            self.assertCountEqual(held, iso.op_indices)       # symmetry of the child is the isotropy subgroup
            child = self.R.child_structure(u, 0.4)
            rows, resid = self.R.decompose(self.R.displacements_from(child))
            amps = {(r[0].index, r[1]): r[2] for r in rows if r[2] > 1e-9}
            self.assertEqual(list(amps), [(ir.index, 0)])
            self.assertAlmostEqual(amps[(ir.index, 0)], 0.4, places=6)
            self.assertLess(resid, 1e-9)

    def test_components_use_the_same_basis_as_the_opds(self):
        ir = [i for i in self.R.irreps if i.dim == 3 and i.species == ["O"]][0]
        isos = self.R.isotropy_subgroups(ir)
        fresh = ModeAnalysis(perovskite(), (0.5, 0.5, 0.5))      # never touched isotropy_subgroups
        fir = [i for i in fresh.irreps if i.dim == 3 and i.species == ["O"]][0]
        for iso in isos:
            child = self.R.child_structure(self.R.displacement(ir, iso.opd), 0.4)
            rows, _ = fresh.decompose(fresh.displacements_from(child))
            c = [r[3] for r in rows if r[0] is fir][0]
            self.assertTrue(np.allclose(c, 0.4 * iso.opd, atol=1e-8), (iso.opd_label, c, iso.opd))

    def test_tilt_geometry_I4mcm(self):
        ir = [i for i in self.R.irreps if i.dim == 3 and i.species == ["O"]][0]
        iso = [i for i in self.R.isotropy_subgroups(ir) if i.point_group == "4/mmm"][0]
        child = self.R.child_structure(self.R.displacement(ir, iso.opd), 0.5)
        par = self.R.supercell_parent()
        d = np.linalg.norm(self.R.displacements_from(child), axis=1)
        sp = np.array(child.species)
        self.assertTrue(np.all(d[sp == "Sr"] < 1e-9) and np.all(d[sp == "Ti"] < 1e-9))
        # a rotation about one axis: the 2 oxygens on the axis stay, the other 4 move
        self.assertEqual(int(np.sum(d[sp == "O"] > 1e-3)), 4)
        # ... and each mover is displaced perpendicular to its Ti-O bond (rigid-octahedron rotation)
        disp = self.R.displacements_from(child)
        pos = par.frac @ par.lattice
        for i in np.where((sp == "O") & (d > 1e-3))[0]:
            primfrac = (self.R.Sb.T @ par.frac[i]) % 1.0
            axis = int(np.argmin(np.abs(primfrac - 0.5)))          # the half-integer coordinate = bond direction
            self.assertLess(abs(disp[i, axis]), 1e-9)
        self.assertEqual(len(child), len(par))

    def test_polar_gamma_mode(self):
        G = ModeAnalysis(perovskite(), (0, 0, 0))
        polar = [ir for ir in G.irreps if ir.dim == 3 and set(ir.species) == {"Sr", "Ti", "O"}]
        self.assertEqual(len(polar), 1)   # polar T1u, appears on every orbit (includes the acoustic translation)
        got = sorted((p, o) for p, o in pg_orders(G, polar[0]))
        self.assertEqual(sorted(o for _, o in got), [1, 2, 2, 4, 6, 8])  # P4mm, R3m, Amm2, Pm, Cm, P1
        self.assertEqual(sorted(p for p, _ in got), sorted(["4mm", "3m", "mm2", "m", "m", "1"]))

    def test_m_and_x_points(self):
        M = ModeAnalysis(perovskite(), (0.5, 0.5, 0))
        X = ModeAnalysis(perovskite(), (0, 0, 0.5))
        for ma, narms in ((M, 3), (X, 3)):
            self.assertEqual(len(ma.star), narms)
            dim = sum(ir.dim * len(ir.modes) for ir in ma.irreps)
            self.assertEqual(dim, 15 * narms)   # 3 arms x (5 atoms x 3) real dimensions
        # in-phase tilts M3+ -> P4/mbm (4/mmm) among the isotropy subgroups of an O-only 3-dim... check existence
        found = [ir for ir in M.irreps if ir.species == ["O"] and any(i.point_group == "4/mmm" for i in M.isotropy_subgroups(ir))]
        self.assertTrue(found)

    def test_nonprimitive_input_rejected(self):
        a = 5.63
        s = Structure.from_parameters(a, a, a, 90, 90, 90, [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], ["X"] * 4)
        with self.assertRaises(ValueError):
            ModeAnalysis(s, (0, 0, 0))


class TestConsistency(unittest.TestCase):
    """For every irrep at several k-points: the predicted isotropy subgroup is the true symmetry of the
    generated distortion, and amplitudes round-trip through decompose()."""

    def check(self, structure, k):
        ma = ModeAnalysis(structure, k)
        # the irreps must exactly fill the real displacement space of the star
        real_per_arm = sum(1 if tuple((-x) % 1 for x in a) == a else 2 for a in ma.star_classes)
        total = sum(ir.dim * len(ir.modes) for ir in ma.irreps)
        self.assertEqual(total, 3 * len(structure) * real_per_arm)
        n = 0
        for ir in ma.irreps:
            for iso in ma.isotropy_subgroups(ir):
                u = ma.displacement(ir, iso.opd)
                self.assertCountEqual(ma.symmetry_of(u), iso.op_indices, (k, ir.index, iso.index))
                rows, resid = ma.decompose(ma.displacements_from(ma.child_structure(u, 0.3)))
                amps = {(r[0].index): 0.0 for r in rows}
                for r in rows:
                    amps[r[0].index] += r[2] ** 2
                self.assertAlmostEqual(amps[ir.index], 0.09, places=8)
                self.assertAlmostEqual(sum(amps.values()), 0.09, places=8)
                self.assertLess(resid, 1e-8)
                n += 1
        return n

    def test_cubic_perovskite_special_points(self):
        for k in [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5)]:
            with self.subTest(k=k):
                self.assertGreater(self.check(perovskite(), k), 0)

    @unittest.skipUnless(os.environ.get("DISTORTKIT_SLOW"), "set DISTORTKIT_SLOW=1 (64-cell supercell, minutes)")
    def test_quarter_point_slow(self):
        self.assertGreater(self.check(perovskite(), (0.25, 0, 0)), 0)

    def test_low_symmetry_structure(self):
        s = Structure.from_parameters(4.0, 5.0, 6.0, 90, 100, 90,
                                      [[0, 0, 0], [0.3, 0.5, 0.2], [0.7, 0.5, 0.8]], ["A", "B", "B"])
        self.assertGreater(self.check(s, (0.5, 0, 0.5)), 0)

    def test_hexagonal(self):
        s = Structure.from_parameters(3.2, 3.2, 5.2, 90, 90, 120,
                                      [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.38], [2 / 3, 1 / 3, 0.88]],
                                      ["Zn", "Zn", "S", "S"])
        self.assertEqual(len(find_symmetry(s)), 12)
        self.assertGreater(self.check(s, (0, 0, 0)), 0)
        self.assertGreater(self.check(s, (0.5, 0, 0)), 0)
        self.assertGreater(self.check(s, (1 / 3, 1 / 3, 0)), 0)


class TestRocksaltX(unittest.TestCase):
    def test_rocksalt_primitive_gamma(self):
        a = 5.63
        s = Structure.from_parameters(a, a, a, 90, 90, 90,
                                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                                       [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]],
                                      ["Na"] * 4 + ["Cl"] * 4)
        p, B = to_primitive(s)
        G = ModeAnalysis(p, (0, 0, 0))
        # Gamma: T1u (polar, dim 3) on Na and Cl; no other displacive irrep
        self.assertEqual([ir.dim for ir in G.irreps], [3])
        self.assertEqual(len(G.irreps[0].modes), 2)


if __name__ == "__main__":
    unittest.main()
