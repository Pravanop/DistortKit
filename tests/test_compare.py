import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distortkit import ModeAnalysis, Structure
from distortkit.compare import CellSet, strain_vector
from distortkit.compare import compare as _compare


def compare(*a, **k):
    k.setdefault("labels", False)       # labels have their own tests (they cost ~2 s per call)
    return _compare(*a, **k)

EX = os.path.join(os.path.dirname(__file__), "..", "examples")
M0 = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]])      # I4/mcm conventional cell of the cubic perovskite


def perovskite():
    return Structure.from_json(os.path.join(EX, "srtio3_cubic.json"))


def rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def make_child(prim, M, field, eps, rng, noise=0.0, shuffle=True):
    """Child structure: parent supercell M displaced by field(a, T) (Cartesian), strained by eps,
    rotated arbitrarily, origin shifted and atoms shuffled."""
    A = prim.lattice
    cs = CellSet(M)
    shift = rng.uniform(0, 1, 3)
    fr, sp = [], []
    for a in range(len(prim)):
        for T in cs.cells:
            d = field(a, T) + noise * rng.normal(size=3)
            fr.append(((prim.frac[a] + T + d @ np.linalg.inv(A)) @ np.linalg.inv(M) + shift) % 1.0)
            sp.append(prim.species[a])
    order = rng.permutation(len(fr)) if shuffle else np.arange(len(fr))
    Rr = rotation(rng)
    lattice = (M @ A) @ (np.eye(3) + eps) @ Rr.T
    return Structure(lattice, np.array(fr)[order], [sp[i] for i in order])


def tilt_field(amp):
    prim = perovskite()
    ma = ModeAnalysis(prim, (0.5, 0.5, 0.5))
    ir = [i for i in ma.irreps if i.dim == 3 and i.species == ["O"]][0]
    iso = [i for i in ma.isotropy_subgroups(ir) if i.point_group == "4/mmm"][0]
    u = amp * ma.displacement(ir, iso.opd)
    Nc = ma.n_cells
    return (lambda a, T: u[a * Nc + ma._canon(T[None, :])[0]]), Nc


def gamma_field(amp):
    prim = perovskite()
    ma = ModeAnalysis(prim, (0, 0, 0))
    ir = [i for i in ma.irreps if i.species == ["O"]][0]       # T2u: O-only, orthogonal to translations
    u = amp * ma.displacement(ir, np.array([0.3, 0.5, 0.8]))
    return (lambda a, T: u[a]), 1


class TestCompare(unittest.TestCase):
    def test_tilt_with_strain_other_setting_rotated_shuffled(self):
        rng = np.random.default_rng(1)
        f, Nc = tilt_field(0.20)
        eps = np.diag([-0.003, -0.003, 0.010])
        child = make_child(perovskite(), M0, f, eps, rng)
        r = compare(perovskite(), child)
        self.assertEqual(r.n_cells, 4)
        # tilt amplitude (RSS over the 4-cell child supercell = generator amplitude * sqrt(4/Nc))
        tilt = [m for m in r.modes if m.amplitude > 1e-6]
        self.assertEqual(len(tilt), 1)
        self.assertAlmostEqual(tilt[0].amplitude, 0.20 * np.sqrt(4 / Nc), places=6)
        self.assertEqual(tilt[0].k, tuple(__import__("fractions").Fraction(1, 2) for _ in range(3)))
        self.assertEqual(tilt[0].best_subgroup.point_group, "4/mmm")
        self.assertLess(r.unexplained, 1e-8)
        self.assertLess(r.rms_displacement, 0.2)
        # strain: invariants (axis labelling is symmetry-equivalent)
        self.assertTrue(np.allclose(np.sort(np.linalg.eigvalsh(r.strain)), np.sort(np.diag(eps)), atol=1e-8))
        sym = [s for s in r.strain_irreps if s.fully_symmetric][0]
        self.assertAlmostEqual(sym.amplitude, np.trace(eps) / np.sqrt(3), places=8)
        self.assertAlmostEqual(np.sqrt(sum(s.amplitude**2 for s in r.strain_irreps)), np.linalg.norm(eps), places=8)
        self.assertLess(r.strain_forbidden, 1e-8)
        self.assertEqual(r.child_point_group, "4/mmm")
        self.assertTrue(r.child_in_parent)

    def test_two_stars_gamma_and_R(self):
        rng = np.random.default_rng(2)
        f1, N1 = tilt_field(0.15)
        f2, N2 = gamma_field(0.04)
        child = make_child(perovskite(), M0, lambda a, T: f1(a, T) + f2(a, T), np.zeros((3, 3)), rng)
        r = compare(perovskite(), child)
        amps = sorted([(round(m.amplitude, 6), m.k) for m in r.modes if m.amplitude > 1e-6], reverse=True)
        self.assertEqual(len(amps), 2)
        self.assertAlmostEqual(amps[0][0], 0.15 * np.sqrt(4 / N1), places=5)
        self.assertAlmostEqual(amps[1][0], 0.04 * np.sqrt(4 / N2), places=5)
        self.assertLess(r.unexplained, 1e-8)
        self.assertLess(np.linalg.norm(r.strain), 1e-8)
        # parseval: total displacement squared equals the sum of squared amplitudes
        self.assertAlmostEqual(r.total_displacement**2, sum(m.amplitude**2 for m in r.modes), places=8)

    def test_noise_is_reported_not_hidden(self):
        rng = np.random.default_rng(3)
        f, Nc = tilt_field(0.20)
        child = make_child(perovskite(), M0, f, np.diag([0.002, 0.002, -0.004]), rng, noise=0.003)
        r = compare(perovskite(), child)
        main = max(r.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(main.amplitude, 0.20 * np.sqrt(4 / Nc), delta=0.02)
        self.assertEqual(main.k, tuple(__import__("fractions").Fraction(1, 2) for _ in range(3)))
        # noise spreads over many modes, all small
        others = [m.amplitude for m in r.modes if m is not main]
        self.assertTrue(all(a < 0.05 for a in others))
        self.assertLess(r.unexplained, 1e-8)       # modes still span the whole displacement field

    def test_undistorted_child_in_supercell(self):
        rng = np.random.default_rng(4)
        child = make_child(perovskite(), M0, lambda a, T: np.zeros(3), np.zeros((3, 3)), rng)
        r = compare(perovskite(), child)
        self.assertLess(r.total_displacement, 1e-8)
        self.assertEqual(r.child_point_group, "m-3m")
        self.assertEqual([m for m in r.modes if m.amplitude > 1e-6], [])

    def test_nonprimitive_parent_and_polar_gamma(self):
        a = 5.63
        conv = Structure.from_parameters(a, a, a, 90, 90, 90,
                                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                                          [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]],
                                         ["Na"] * 4 + ["Cl"] * 4)
        delta = 0.05
        fr = conv.frac.copy()
        fr[:4, 2] += delta / a
        fr[4:, 2] -= delta / a
        eps = np.diag([0.004, 0.004, -0.008])
        child = Structure(conv.lattice @ (np.eye(3) + eps), fr, conv.species)
        r = compare(conv, child)
        self.assertEqual(r.n_cells, 4)
        self.assertEqual(r.child_point_group, "4mm")
        main = max(r.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(main.amplitude, delta * np.sqrt(8), places=6)   # RSS over 8 atoms
        self.assertEqual(main.k, (0, 0, 0))
        self.assertAlmostEqual(np.sqrt(sum(s.amplitude**2 for s in r.strain_irreps)), np.linalg.norm(eps), places=8)

    def test_independent_geometry_rigid_octahedron_rotation(self):
        """Pure geometry, no mode code: TiO6 octahedra rotated about c by +/-theta in a checkerboard."""
        theta, a = 0.04, 3.905

        def field(atom, T):
            sgn = (-1) ** int(T.sum())
            if atom == 2:                      # O on the a axis moves along +y
                return np.array([0.0, sgn * theta * a / 2, 0.0])
            if atom == 3:                      # O on the b axis moves along -x
                return np.array([-sgn * theta * a / 2, 0.0, 0.0])
            return np.zeros(3)

        rng = np.random.default_rng(5)
        child = make_child(perovskite(), M0, field, np.zeros((3, 3)), rng)
        r = compare(perovskite(), child)
        main = [m for m in r.modes if m.amplitude > 1e-6]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0].species, ["O"])
        self.assertEqual(main[0].irrep.dim, 3)
        self.assertAlmostEqual(main[0].amplitude, np.sqrt(8) * theta * a / 2, places=8)
        self.assertEqual(main[0].best_subgroup.point_group, "4/mmm")
        self.assertEqual(r.child_point_group, "4/mmm")
        self.assertLess(r.unexplained, 1e-8)

    def test_every_isotropy_subgroup_is_recovered(self):
        """Generate a child for each isotropy subgroup of the tilt irrep (random domain, strain,
        orientation); the analysis must name the same subgroup and see the same point group."""
        prim = perovskite()
        ma = ModeAnalysis(prim, (0.5, 0.5, 0.5))
        ir = [i for i in ma.irreps if i.dim == 3 and i.species == ["O"]][0]
        seen = []
        for k, iso in enumerate(ma.isotropy_subgroups(ir)):
            rng = np.random.default_rng(100 + k)
            # random symmetry-equivalent domain: apply a random group element to the OPD
            g = rng.integers(ma.n_ops)
            opd = ir.rho[g] @ iso.opd
            u = 0.2 * ma.displacement(ir, opd)
            Nc = ma.n_cells
            child = make_child(prim, M0, lambda a, T: u[a * Nc + ma._canon(T[None, :])[0]], np.zeros((3, 3)), rng)
            r = compare(prim, child)
            main = max(r.modes, key=lambda m: m.amplitude)
            self.assertAlmostEqual(main.amplitude, 0.2 * np.sqrt(4 / Nc), places=6, msg=iso.point_group)
            self.assertEqual(main.best_subgroup.point_group, iso.point_group)
            self.assertEqual(main.best_subgroup.fix_dim, iso.fix_dim)
            self.assertEqual(r.child_point_group, iso.point_group)
            seen.append(iso.point_group)
        self.assertEqual(len(seen), 6)

    def test_left_handed_child_basis(self):
        rng = np.random.default_rng(6)
        f, Nc = tilt_field(0.2)
        child = make_child(perovskite(), M0, f, np.zeros((3, 3)), rng)
        child = Structure(child.lattice * np.array([[1], [1], [-1]]), child.frac * np.array([1, 1, -1]), child.species)
        r = compare(perovskite(), child)
        main = max(r.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(main.amplitude, 0.2 * np.sqrt(4 / Nc), places=6)

    def test_hexagonal_parent(self):
        prim = Structure.from_parameters(3.2, 3.2, 5.2, 90, 90, 120,
                                         [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.38], [2 / 3, 1 / 3, 0.88]],
                                         ["Zn", "Zn", "S", "S"])
        ma = ModeAnalysis(prim, (0.5, 0, 0))
        ir = [i for i in ma.irreps if i.dim == ma.irreps[-1].dim][0]
        iso = ma.isotropy_subgroups(ir)[0]
        u = 0.1 * ma.displacement(ir, iso.opd)
        Nc = ma.n_cells
        M = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        rng = np.random.default_rng(7)
        child = make_child(prim, M, lambda a, T: u[a * Nc + ma._canon(T[None, :])[0]], np.diag([0.001, 0.001, -0.002]), rng)
        r = compare(prim, child)
        self.assertEqual(r.n_cells, 4)
        top = max(r.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(top.amplitude, 0.1 * np.sqrt(4 / Nc), places=6)
        from fractions import Fraction
        half, zero = Fraction(1, 2), Fraction(0)
        self.assertIn(top.k, [(half, zero, zero), (zero, half, zero), (half, half, zero)])   # one arm of the M star
        self.assertLess(r.unexplained, 1e-8)

    def test_files_roundtrip_poscar(self):
        import tempfile
        from distortkit.io import read_structure
        rng = np.random.default_rng(8)
        f, Nc = tilt_field(0.2)
        child = make_child(perovskite(), M0, f, np.diag([-0.002, -0.002, 0.004]), rng)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "POSCAR")
            with open(p, "w") as fh:
                fh.write(child.to_poscar())
            r = compare(perovskite(), read_structure(p))
        main = max(r.modes, key=lambda m: m.amplitude)
        self.assertAlmostEqual(main.amplitude, 0.2 * np.sqrt(4 / Nc), places=5)

    def test_errors(self):
        p = perovskite()
        bad = Structure(p.lattice, p.frac[:4], p.species[:4])
        with self.assertRaises(ValueError):
            compare(p, bad)
        far = Structure(p.lattice * 1.5, p.frac, p.species)
        with self.assertRaises(ValueError):
            compare(p, far, strain_tol=0.1)


if __name__ == "__main__":
    unittest.main()
