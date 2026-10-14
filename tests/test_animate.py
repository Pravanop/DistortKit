import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from distortkit import animate as AN
from distortkit import compare, read_structure
from distortkit import freeze as F
from distortkit.external import find_renderer_python

HF = os.path.join(HERE, "..", "examples", "hfo2")
RENDERER = find_renderer_python()


def hf_freezer(tmp):
    res = compare(read_structure(os.path.join(HF, "Pbcm.cif")), read_structure(os.path.join(HF, "Pca21.cif")))
    return res, F.Freezer.load(res.save(os.path.join(tmp, "a.json"), meta={"parent": "Pbcm.cif", "child": "Pca21.cif"}))


def frames_of(path):
    """All frames of a GIF as RGB arrays, read with Pillow in the renderer's interpreter."""
    code = ("import sys, numpy as np; from PIL import Image\n"
            "im = Image.open(sys.argv[1]); out = []\n"
            "for k in range(im.n_frames):\n    im.seek(k); out.append(np.asarray(im.convert('RGB'), dtype=np.int16))\n"
            "np.save(sys.argv[2], np.array(out))")
    with tempfile.TemporaryDirectory() as d:
        npy = os.path.join(d, "f.npy")
        subprocess.run([RENDERER, "-c", code, path, npy], check=True, capture_output=True)
        return np.load(npy)


class TestPieces(unittest.TestCase):
    def test_fraction_parsing(self):
        self.assertEqual(AN.parse_fraction("50%"), 0.5)
        self.assertEqual(AN.parse_fraction("0.25"), 0.25)
        self.assertAlmostEqual(AN.parse_fraction(" 150 % ".replace(" ", "")), 1.5)
        with self.assertRaises(F.SelectionError):
            AN.parse_fraction("lots")

    def test_frame_scales_trace_a_full_oscillation(self):
        s = AN.frame_scales(24)
        self.assertAlmostEqual(s[0], 0.0)
        self.assertAlmostEqual(s[6], 1.0)
        self.assertAlmostEqual(s[12], 0.0, places=12)
        self.assertAlmostEqual(s[18], -1.0)
        self.assertTrue(np.allclose(s[:12] + s[12:], 0.0, atol=1e-12))            # second half is the mirror image
        o = AN.frame_scales(24, oneway=True)
        self.assertAlmostEqual(o.min(), 0.0)
        self.assertAlmostEqual(o.max(), 1.0)
        with self.assertRaises(F.SelectionError):
            AN.frame_scales(1)

    def test_camera_is_orthonormal_and_named_views_look_down_axes(self):
        P = np.array([[5.0, 0, 0], [-1.5, 4.8, 0], [0.3, 0.2, 5.5]])               # a skewed cell
        for view in ("iso", "a", "b", "c", "-c"):
            r, u, n = AN.view_basis(P, view)
            self.assertAlmostEqual(r @ u, 0.0, places=12)
            self.assertAlmostEqual(r @ n, 0.0, places=12)
            self.assertAlmostEqual(u @ n, 0.0, places=12)
            self.assertAlmostEqual(np.linalg.norm(r), 1.0)
            self.assertAlmostEqual(np.dot(np.cross(r, u), n), 1.0)                  # right-handed
        _, _, n = AN.view_basis(P, "c")
        self.assertTrue(np.allclose(n, P[2] / np.linalg.norm(P[2])))
        _, _, n = AN.view_basis(P, "-b")
        self.assertTrue(np.allclose(n, -P[1] / np.linalg.norm(P[1])))
        r, u, n = AN.view_basis(P, "iso", elev=0, azim=0)                            # camera on the a side, level
        self.assertAlmostEqual(abs(n @ (P[2] / np.linalg.norm(P[2]))), 0.0, places=12)


class TestScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res, cls.fz = hf_freezer(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scene_contents(self):
        sc = AN.Scene(self.fz, self.fz.field(["GM2-"]))
        self.assertGreaterEqual(int(sc.shown.sum()), 12)                              # at least the 12 atoms of the cell
        self.assertEqual(len(sc.edges), 12)
        el = [AN.element(s) for s in sc.species]
        self.assertTrue(sc.bonds)
        self.assertTrue(all(el[i] != el[j] for i, j in sc.bonds))                     # Hf-O only, no O-O or Hf-Hf bonds
        self.assertTrue(all(sc.shown[i] or sc.shown[j] for i, j in sc.bonds))
        # every displayed atom carries the displacement of its parent-cell source atom
        field = self.fz.field(["GM2-"])
        for k in range(len(sc.base)):
            x = self.fz.atoms[sc.src[k]]
            self.assertLess(np.abs(sc.disp[k] - field[x.a, x.c]).max(), 1e-12)
        self.assertEqual(AN.Scene(self.fz, self.fz.field(["GM2-"]), bonds=False).bonds, [])

    def test_missing_renderer_gives_a_clear_error(self):
        sc = AN.Scene(self.fz, self.fz.field(["GM2-"]))
        saved = AN.find_renderer_python
        AN.find_renderer_python = lambda: None
        try:
            with self.assertRaises(RuntimeError) as cm:
                AN.render_gif(sc, AN.frame_scales(4), "x.gif", title="t", subtitle="s", fraction=1.0)
        finally:
            AN.find_renderer_python = saved
        self.assertIn("matplotlib", str(cm.exception))

    def test_empty_selection_is_an_error(self):
        rc = F.enumerate_recipes(self.fz, modes=["GM2-"], atoms="Hf", exclude="Hf")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(F.SelectionError):
                AN.run_gif(self.fz, rc, outdir=d, log=lambda *_: None)


@unittest.skipUnless(RENDERER, "matplotlib + Pillow not available (see DISTORTKIT_PYTHON)")
class TestRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res, cls.fz = hf_freezer(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gif_is_valid_loops_and_the_atoms_really_move(self):
        rc = F.enumerate_recipes(self.fz, modes=["GM2-"])
        with tempfile.TemporaryDirectory() as d:
            rows = AN.run_gif(self.fz, rc, fraction=1.0, n_frames=24, fps=10, size=360, outdir=d, log=lambda *_: None)
            path = rows[0]["file"]
            self.assertEqual(AN.gif_info(path), (360, 360, 24, 0))                  # size, frames, loop forever
            fr = frames_of(path)
            changed = lambda a, b: int((np.abs(fr[a] - fr[b]).max(axis=2) > 8).sum())
            self.assertEqual(changed(0, 12), 0)                                      # back at the parent after half a cycle
            self.assertGreater(changed(0, 6), 2000)                                  # +max differs from the parent
            self.assertGreater(changed(6, 18), 2000)                                 # +max differs from -max
            self.assertEqual(changed(3, 9), 0)                                       # same |displacement| -> same picture shape...
            self.assertEqual(set(map(tuple, fr[:, 5, 5])), {(255, 255, 255)})        # white background in every frame, no flicker

    def test_fraction_sets_the_largest_excursion(self):
        rc = F.enumerate_recipes(self.fz, modes=["GM2-"])
        with tempfile.TemporaryDirectory() as d:
            full = AN.run_gif(self.fz, rc, fraction=1.0, n_frames=8, size=300, outdir=os.path.join(d, "a"), log=lambda *_: None)[0]
            half = AN.run_gif(self.fz, rc, fraction=0.5, n_frames=8, size=300, outdir=os.path.join(d, "b"), log=lambda *_: None)[0]
            absolute = AN.run_gif(self.fz, rc, amplitude=0.3, n_frames=8, size=300, outdir=os.path.join(d, "c"), log=lambda *_: None)[0]
            self.assertAlmostEqual(half["max_amplitude"], 0.5 * full["max_amplitude"], places=9)
            self.assertAlmostEqual(full["max_amplitude"], 1.1844, places=3)
            self.assertAlmostEqual(absolute["max_amplitude"], 0.3, places=9)
            self.assertAlmostEqual(absolute["fraction"], 0.3 / full["max_amplitude"], places=9)
            # half the excursion moves fewer pixels than the full one
            f_full, f_half = frames_of(full["file"]), frames_of(half["file"])
            moved = lambda fr: int((np.abs(fr[2] - fr[0]).max(axis=2) > 8).sum())
            self.assertLess(moved(f_half), moved(f_full))

    def test_oneway_never_goes_negative(self):
        rc = F.enumerate_recipes(self.fz, modes=["GM2-"])
        with tempfile.TemporaryDirectory() as d:
            row = AN.run_gif(self.fz, rc, n_frames=12, size=300, oneway=True, outdir=d, log=lambda *_: None)[0]
            fr = frames_of(row["file"])
            self.assertEqual(int((np.abs(fr[0] - fr[1]).max(axis=2) > 8).sum()) > 0, True)
            self.assertEqual(int((np.abs(fr[5] - fr[7]).max(axis=2) > 8).sum()), 0)     # symmetric about the maximum at frame 6

    def test_selection_sets_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            rc = F.enumerate_recipes(self.fz, each="mode")
            AN.run_gif(self.fz, rc, n_frames=6, size=240, outdir=d, log=lambda *_: None)
            AN.run_gif(self.fz, F.enumerate_recipes(self.fz, parts=["GM2-:Hf"]), n_frames=6, size=240, outdir=d,
                       prefix="hf", log=lambda *_: None)
            gifs = sorted(x for x in os.listdir(d) if x.endswith(".gif"))
            self.assertEqual(len(gifs), 3)
            with open(os.path.join(d, "animations_manifest.txt")) as fh:
                self.assertEqual(len(fh.read().strip().splitlines()), 4)               # header + the three files


@unittest.skipUnless(RENDERER, "matplotlib + Pillow not available")
class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        return subprocess.run([sys.executable, "-m", "distortkit", *args], capture_output=True, text=True, env=env)

    def test_show_gif(self):
        with tempfile.TemporaryDirectory() as d:
            an = os.path.join(d, "an.json")
            self.assertEqual(self.run_cli("compare", os.path.join(HF, "Pbcm.cif"), os.path.join(HF, "Pca21.cif"), "--save", an).returncode, 0)
            r = self.run_cli("show", an, "--gif", "--modes", "GM2-,GM1+", "--fraction", "50%", "--frames", "7", "--size", "240")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ANIMATIONS", r.stdout)
            out = os.path.join(d, "animations")
            gifs = [x for x in os.listdir(out) if x.endswith(".gif")]
            self.assertEqual(len(gifs), 1)
            self.assertEqual(AN.gif_info(os.path.join(out, gifs[0]))[2], 7)
            r = self.run_cli("show", an, "--gif", "--fraction", "lots", "--outdir", os.path.join(d, "x"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("fraction or percentage", r.stderr)
            r = self.run_cli("show", an, "--gif", "--view", "-c", "--part", "GM2-:Hf*0.5", "--fraction", "1", "--frames", "4",
                             "--size", "200", "--arrows", "--no-bonds", "--outdir", os.path.join(d, "v"))
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_cli("show", an, "--gif", "--modes", "nonsense", "--outdir", os.path.join(d, "y"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("unknown mode", r.stderr)


if __name__ == "__main__":
    unittest.main()
