import json
import os
import random
import subprocess
import sys
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import isodata, wyckoff
from distortkit.external import find_python_with
from distortkit.spacegroup import identify
from distortkit.structure import Structure
from distortkit.symmetry import find_symmetry, to_primitive

TABLES = os.path.join(HERE, "..", "distortkit", "tables")


class TestSpaceGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = isodata.space_groups()

    def test_all_230_are_groups(self):
        rng = random.Random(0)
        self.assertEqual(sorted(self.G), list(range(1, 231)))
        for n, g in self.G.items():
            keys = {(tuple(R.ravel()), t) for R, t in g.ops}
            self.assertEqual(len(keys), len(g.ops), n)
            self.assertIn((tuple(np.eye(3, dtype=int).ravel()), (Fraction(0),) * 3), keys, n)
            # closure on random pairs
            for _ in range(60):
                (Ra, ta), (Rb, tb) = rng.choice(g.ops), rng.choice(g.ops)
                t = tuple((sum(int(Ra[i, j]) * tb[j] for j in range(3)) + ta[i]) % 1 for i in range(3))
                self.assertIn((tuple((Ra @ Rb).ravel()), t), keys, n)
            self.assertEqual(len(g.ops) % (1 + len(g.centering)), 0)

    def test_known_orders_and_names(self):
        self.assertEqual(len(self.G[225].ops), 192)          # Fm-3m in the conventional cell
        self.assertEqual(len(self.G[221].ops), 48)
        self.assertEqual((self.G[62].symbol, self.G[62].point_group), ("Pnma", "mmm"))
        self.assertEqual(self.G[29].symbol, "Pca2_1")
        self.assertEqual(len(self.G[166].centering), 2)       # R-3m, hexagonal axes

    @unittest.skipUnless(find_python_with(("spglib",)), "spglib not available (pip install spglib)")
    def test_matches_spglib(self):
        py = find_python_with(("spglib",))
        code = ("import json,spglib\nout={}\nfor h in range(1,531):\n t=spglib.get_spacegroup_type(h)\n"
                " out.setdefault(t.number,[]).append((h,t.choice))\n"
                "res={}\nfor n,l in out.items():\n two=[h for h,c in l if c=='2']\n h=two[0] if two else l[0][0]\n"
                " s=spglib.get_symmetry_from_database(h)\n res[n]=[[r.ravel().tolist(),t.tolist()] for r,t in zip(s['rotations'],s['translations'])]\n"
                "print(json.dumps(res))")
        res = json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, check=True).stdout)
        for n in range(1, 231):
            a = {(tuple(R.ravel()), tuple(t)) for R, t in self.G[n].ops}
            b = {(tuple(R), tuple(Fraction(x).limit_denominator(24) % 1 for x in t)) for R, t in res[str(n)]}
            self.assertEqual(a, b, n)


class TestWyckoffTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(TABLES, "wyckoff.json")) as fh:
            cls.doc = json.load(fh)["groups"]
        cls.G = isodata.space_groups()

    def test_structure_of_every_group(self):
        total = 0
        for n in range(1, 231):
            items = self.doc[str(n)]
            total += len(items)
            mult = [i["multiplicity"] for i in items]
            order = len(self.G[n].ops)
            self.assertEqual(mult[-1], order, n)                             # the last letter is the general position
            self.assertEqual(mult, sorted(mult), n)                          # multiplicity never decreases with the letter
            self.assertTrue(all(order % m == 0 for m in mult), n)
            self.assertEqual(len({i["letter"] for i in items}), len(items), n)
        self.assertEqual(total, 1731)                                        # number of positions in the 230 groups

    def test_known_groups(self):
        def mults(n):
            return [i["multiplicity"] for i in self.doc[str(n)]]
        self.assertEqual(mults(221), [1, 1, 3, 3, 6, 6, 8, 12, 12, 12, 24, 24, 24, 48])      # Pm-3m: 1a ... 48n
        self.assertEqual(mults(225), [4, 4, 8, 24, 24, 32, 48, 48, 48, 96, 96, 192])         # Fm-3m: 4a ... 192l
        self.assertEqual(mults(62), [4, 4, 4, 8])                                            # Pnma
        self.assertEqual(mults(2), [1, 1, 1, 1, 1, 1, 1, 1, 2])                              # P-1
        self.assertEqual([i["letter"] for i in self.doc["225"]], list("abcdefghijkl"))

    def test_templates_have_the_right_dimension(self):
        for n in (14, 62, 139, 225):
            for pos in wyckoff.wyckoff_positions(n):
                free = sum(1 for r in pos.rows if any(r))
                orbit = pos.multiplicity * 1.0
                self.assertLessEqual(free, 3)
                self.assertTrue(free == 3 if pos.multiplicity == len(self.G[n].ops) else True)


def structure_on(n, point, lattice, generic=((0.137, 0.291, 0.403), (0.611, 0.177, 0.852))):
    """Structure of space group ``n``: species X on the orbit of ``point``, species Y on a generic orbit (so that the
    symmetry is exactly the group)."""
    G = isodata.space_groups()[n]

    def orbit(p):
        out = []
        for R, t in G.ops:
            y = (R @ np.array(p, float) + np.array([float(v) for v in t])) % 1.0
            if not any(np.abs((y - z) - np.rint(y - z)).max() < 1e-7 for z in out):
                out.append(y)
        return out
    xs = orbit(point)
    ys = [y for g in generic for y in orbit(g)]
    return Structure(np.array(lattice, float), np.array(xs + ys), ["X"] * len(xs) + ["Y"] * len(ys))


def sites_of(structure):
    prim, _ = to_primitive(structure, 1e-4)
    ops = find_symmetry(prim, 1e-4)
    sg = identify(ops, prim.lattice)
    return sg, {s.species: s for s in wyckoff.describe_sites(prim, ops, sg)[:1]} | {s.species: s for s in wyckoff.describe_sites(prim, ops, sg)}


class TestSiteAssignment(unittest.TestCase):
    """Regressions for positions that the ISOTROPY tables could not give directly: lines and planes in monoclinic and
    centred groups (ISOTROPY writes them in another unique axis / the primitive basis)."""

    def test_monoclinic_two_fold_axis(self):
        """Points on the four 2-fold axes (0|1/2, y, 0|1/2) of P2 (letters a-d). Which axis is called 'a' depends on the
        origin and on the choice of the a-c basis (the normaliser of P2 contains translations and shears of that plane), so
        test what does not depend on it: the letter is one of a-d and the template is a line along y."""
        mono = [[5, 0, 0], [0, 6, 0], [7 * np.cos(np.radians(100)), 0, 7 * np.sin(np.radians(100))]]
        for point in ((0, 0.31, 0), (0, 0.31, 0.5), (0.5, 0.31, 0), (0.5, 0.31, 0.5)):
            sg, sites = sites_of(structure_on(3, point, mono))
            self.assertEqual(sg.number, 3)
            self.assertIn(sites["X"].label, ("1a", "1b", "1c", "1d"))
            self.assertRegex(sites["X"].template, r"^\([01/2]+,y,[01/2]+\)$")

    def test_centred_monoclinic(self):
        mono = [[5, 0, 0], [0, 6, 0], [7 * np.cos(np.radians(100)), 0, 7 * np.sin(np.radians(100))]]
        sg, sites = sites_of(structure_on(5, (0, 0.29, 0), mono))
        self.assertEqual(sg.number, 5)
        self.assertEqual(sites["X"].label, "2a")
        self.assertEqual(sites["Y"].label, "4c")

    def test_face_centred_cubic_line(self):
        cubic = np.eye(3) * 5
        sg, sites = sites_of(structure_on(225, (0.2, 0.2, 0.2), cubic))
        self.assertEqual(sg.number, 225)
        self.assertEqual(sites["X"].label, "32f")
        self.assertEqual(sites["X"].template, "(x,x,x)")

    def test_rhombohedral_hexagonal_axes(self):
        hexa = [[5, 0, 0], [-2.5, 5 * np.sqrt(3) / 2, 0], [0, 0, 7]]
        sg, sites = sites_of(structure_on(146, (0, 0, 0.21), hexa))
        self.assertEqual(sg.number, 146)
        self.assertEqual(sites["X"].label, "3a")


class TestWithoutLabelTables(unittest.TestCase):
    def test_everything_but_the_names_works(self):
        from distortkit import compare, read_structure
        hf = os.path.join(HERE, "..", "examples", "hfo2")
        with mock.patch.object(isodata, "data_dir", side_effect=FileNotFoundError("no label tables")):
            self.assertFalse(isodata.labels_available())
            res = compare(read_structure(os.path.join(hf, "Pbcm.cif")), read_structure(os.path.join(hf, "Pca21.cif")))
        self.assertEqual(res.parent_sg.number, 57)
        self.assertEqual(res.child_sg.number, 29)                         # space groups need only the open tables
        top = max(res.modes, key=lambda m: m.amplitude)
        self.assertIsNone(top.label)                                      # ... irrep names need the ISOTROPY label tables
        self.assertIn("irrep", res._mode_name(top))
        self.assertIsNotNone(top.best_subgroup)
        self.assertIn("Pca2_1", res.report())


if __name__ == "__main__":
    unittest.main()
