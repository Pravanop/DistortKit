"""Pack the distortkit package (sources and data tables) into one zip for the viewer's in-browser analysis.

    python3 viewer/tools/build_py_bundle.py <out.zip>     # used by the Pages workflow
    (viewer/tools/serve.py builds the same zip on the fly at /py/distortkit.zip)

The browser unpacks it into Pyodide's file system.  The ISOTROPY label tables are included so irreps get their standard names;
they compress to about a quarter of their size.
"""
import io
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG = os.path.join(ROOT, "distortkit")
SKIP = {"__pycache__"}


def build():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for base, dirs, files in os.walk(PKG):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for f in sorted(files):
                if f.endswith((".pyc", ".DS_Store")):
                    continue
                path = os.path.join(base, f)
                z.write(path, os.path.relpath(path, ROOT))
    return buf.getvalue()


def newest():
    return max(os.path.getmtime(os.path.join(b, f)) for b, _, fs in os.walk(PKG) for f in fs if not f.endswith(".pyc"))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "distortkit.zip"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    data = build()
    with open(out, "wb") as fh:
        fh.write(data)
    print(f"wrote {out}: {len(data) / 1e6:.2f} MB")
