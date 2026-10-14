import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distortkit import Structure, find_symmetry, isodata, to_primitive
from distortkit.spacegroup import identify

G = isodata.space_groups()

SYSTEM = {}
for g, sg in G.items():
    pg = sg.point_group
    if pg in ("1", "-1"):
        SYSTEM[g] = "tri"
    elif pg in ("2", "m", "2/m"):
        SYSTEM[g] = "mono"
    elif pg in ("222", "mm2", "mmm"):
        SYSTEM[g] = "ortho"
    elif pg in ("4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm"):
        SYSTEM[g] = "tet"
    elif pg in ("3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "-62m", "6/mmm"):
        SYSTEM[g] = "hex"
    else:
        SYSTEM[g] = "cub"

PARAMS = {"tri": (5.1, 6.3, 7.9, 83, 97, 105), "mono": (5.1, 6.3, 7.9, 90, 107, 90), "ortho": (5.1, 6.3, 7.9, 90, 90, 90),
          "tet": (5.1, 5.1, 7.9, 90, 90, 90), "hex": (5.1, 5.1, 7.9, 90, 90, 120), "cub": (5.1, 5.1, 5.1, 90, 90, 90)}


def build(g, rng):
    """Structure with space group exactly g: three species on generic orbits, random cell change and origin."""
    sg = G[g]
    A = Structure.from_parameters(*PARAMS[SYSTEM[g]], np.zeros((0, 3)), []).lattice
    nspec = 3 if len(sg.ops) <= 24 else 2
    pos, sp = [], []
    for k in range(nspec):
        while True:
            x0 = rng.uniform(0.05, 0.95, 3)
            orb = []
            for R, t in sg.ops:
                y = (R @ x0 + np.array([float(v) for v in t])) % 1.0
                if not any(np.linalg.norm(((y - z) - np.rint(y - z)) @ A) < 0.05 for z in orb):
                    orb.append(y)
            if len(orb) == len(sg.ops):
                break
        pos += orb
        sp += [f"X{k}"] * len(orb)
    while True:
        W = np.eye(3, dtype=int)
        for _ in range(4):
            i, j = rng.choice(3, 2, replace=False)
            W[i] += rng.integers(-1, 2) * W[j]
        if round(np.linalg.det(W)) == 1:
            break
    pos = (np.linalg.inv(W).T @ np.array(pos).T).T + rng.uniform(0, 1, 3)
    return Structure(W @ A, pos % 1.0, sp)


def roundtrip(g, seed=0):
    st = build(g, np.random.default_rng(1000 * g + seed))
    prim, _ = to_primitive(st, 1e-3)
    return identify(find_symmetry(prim, 1e-3), prim.lattice)


class TestDatabase(unittest.TestCase):
    def test_all_230_groups_close_to_the_right_order(self):
        self.assertEqual(len(G), 230)            # construction itself raises if any group has the wrong order
        self.assertEqual(G[225].symbol, "Fm-3m")
        self.assertEqual(len(G[225].ops), 192)
        self.assertEqual(G[62].symbol, "Pnma")
        self.assertEqual(G[14].symbol, "P2_1/c")


class TestIdentify(unittest.TestCase):
    SAMPLE = [1, 2, 3, 5, 14, 19, 33, 62, 91, 122, 143, 148, 166, 194, 205, 221, 230]

    def test_sample_of_groups_in_random_settings(self):
        for g in self.SAMPLE:
            with self.subTest(g=g, symbol=G[g].symbol):
                self.assertEqual(roundtrip(g).number, g)

    def test_enantiomorphs_are_distinguished(self):
        for g in (76, 78, 144, 145):
            self.assertEqual(roundtrip(g).number, g)

    def test_perovskite(self):
        s = Structure.from_json(os.path.join(os.path.dirname(__file__), "..", "examples", "srtio3_cubic.json"))
        r = identify(find_symmetry(s), s.lattice)
        self.assertEqual((r.number, r.symbol), (221, "Pm-3m"))

    @unittest.skipUnless(os.environ.get("DISTORTKIT_SLOW"), "set DISTORTKIT_SLOW=1 (all 230 groups, ~3 minutes)")
    def test_all_230_groups(self):
        bad = [g for g in range(1, 231) if roundtrip(g).number != g]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
