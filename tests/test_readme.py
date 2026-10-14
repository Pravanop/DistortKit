import contextlib
import io
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


class TestReadmePython(unittest.TestCase):
    def test_python_example_runs(self):
        """The Python block of the README ('Use it from Python') is executed as written, from a folder holding examples/."""
        with open(os.path.join(ROOT, "README.md")) as fh:
            text = fh.read()
        sec = text[text.index("## Use it from Python"):]
        block = re.search(r"```python\n(.*?)```", sec, re.S).group(1)
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.symlink(os.path.abspath(os.path.join(ROOT, "examples")), os.path.join(d, "examples"))
            os.chdir(d)
            try:
                ns = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(block, "README.md", "exec"), ns)
                self.assertTrue(os.path.getsize("POSCAR_half") > 100)
                self.assertEqual(ns["cmp"].child_sg.number, 29)
                self.assertEqual(ns["rel"].verdict, "a>b")
                self.assertEqual([lv.driver_name for lv in ns["cascade"].levels], ["X5+", "X2-"])
                self.assertEqual(ns["iso"].point_group, "4/mmm")
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
