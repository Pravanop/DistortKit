# Phonon Viewer

A static, build-free web viewer for phonon modes: pick a mode on the band structure and watch the atoms move. It is meant as a
faster, better-aligned, exportable replacement for [phononwebsite](https://henriquemiranda.github.io/phononwebsite/), and reads
its data files.

```bash
python3 viewer/tools/serve.py        # then open http://localhost:8123
```

ES modules need a web server (not `file://`), but any static host works, including GitHub Pages (`serve.py` just turns caching
off so edits show on a normal reload). Everything runs in the browser; files you open are never uploaded.

## What it opens

| Input | Notes |
|---|---|
| phonopy `band.yaml` | Needs eigenvectors: `EIGENVECTORS = .TRUE.` in the band config (or `--eigvecs`). Hand-written parser, ~150 MB/s. Also gzipped (`band.yaml.gz`). |
| phononwebsite `.json` | The same files the website uses (the Examples menu loads a few straight from its server). |

Use **Open file…** (top of the left sidebar) or drop a file anywhere on the page. A file that cannot be used gets a dialog saying
why and what to do (HDF5, phonopy input files, a `qpoints.yaml` without a structure, xz/zip archives, files too large for browser
memory). Files with no path labels get ticks named by their q-points, and `mesh.yaml`-style files without path distances are
spaced by index. `?demo=<name>` and `?url=<file>` in the address bar remain as shortcuts for links.

## What is better than phononwebsite

* **Crystal-aware orientation.** Axes gizmo (a b c, a\* b\* c\*, or x y z) that turns with the crystal; one-click *look down a / b / c /
  a\* / b\* / c\* / x / y / z* (click again for the opposite side) with the lattice, not the screen, setting "right" (looking down c
  puts a to the right and b up); any `[uvw]` direction or `(hkl)` plane normal; orthographic by default, perspective optional; roll
  with Alt-drag. Views re-frame themselves tightly and smoothly.
* **Export.** PNG at 1×–4× with a transparent background, copy to clipboard; GIF (one seamless cycle, single global palette so
  colours do not shimmer, endless loop); MP4/WebM video; one cycle as extended XYZ; the band plot as PNG. Output size and aspect are
  independent of the window.
* **Supercells that match q.** *Fit to q* picks the smallest supercell in which the wave is periodic (K point of a hexagonal cell:
  3×3, or 6×6 to keep the view large) and closes the cell faces with periodic images, so there is no visible seam. A warning appears
  if the chosen supercell does not fit q.
* **Band plot.** Click to select a mode; drag a box to zoom (double-click, Esc or the Reset zoom button restores the full view; Shift-drag scrubs along q); arrow keys step q and band; hover shows the frequency and q; imaginary branches
  are drawn below a dashed zero line and flagged; optional colouring by species; THz / cm⁻¹ / meV. A per-mode list for the
  current q-point shows each mode's species composition.
* **Layout.** One responsive page (resizable split, stacked on narrow screens, light and dark themes) instead of a scrolling box
  that covers the crystal.

## Symmetry of each mode (from DistortKit)

Pick a mode and open the **Symmetry** section. Press **Analyse this q-point** and the viewer works out, for that q-point, every
band's irrep (X₂⁻, with its dimension and the atoms it moves), the child space group that freezing it in leads to and the direction
in irrep space, whether the mode is unstable (only unstable modes drive a transition), and the full list of child symmetries that
irrep allows, with the one this mode gives highlighted. The mode list then shows each band's irrep, and small triangles under the
band plot mark the q-points analysed so far.

**It runs in your browser, nothing is uploaded and nothing needs installing.** DistortKit's analysis is plain Python with numpy, so
the page runs it with [Pyodide](https://pyodide.org) (CPython compiled to WebAssembly) in a background thread: the page stays
usable and a **Cancel** button stops it. The first time, the browser downloads the Python runtime and numpy (about 20 MB) from the
jsDelivr CDN and keeps them; DistortKit itself (about 1 MB) comes from this site. The cost is the size of the q-point's symmetry
star: the X point of the HfO₂ example took about 11 s here; points on symmetry lines or in general positions need a supercell too
large for this analysis and are reported as "not analysed" with the reason.

*Use symmetry-adapted modes* (on by default, once a q-point has been analysed) swaps the file's eigenvectors for symmetry-adapted
ones in degenerate levels. A phonon code writes an arbitrary mixture there, so one band can be a blend of two irreps; the adapted modes
are one irrep each, and a rigid translation (an acoustic mode at Γ) is reported as changing nothing.

The same analysis is available from the command line, which suits many files or q-points (`distortkit phonons band.yaml`, see the
main README). Its JSON can be opened with **Open a precomputed symmetry file…** (or together with the `band.yaml`); it is refused if it was
made for a different band structure. The example *HfO₂ fluorite, with symmetry* in the Examples menu loads such a file: a spring
model with the geometry and symmetry of fluorite HfO₂ and a soft X₂⁻ mode (imaginary, about −3 THz, leading to P4₂/nmc), as in the
real material; the frequencies are not DFT.

## Conventions

These are easy to get wrong, so they are explicit and tested.

* **Displacement.** `u(R, κ, t) = A · Re[ e_κ/√m_κ · exp(2πi q·r) · exp(−iωt) ]`, normalised so the atom that moves most has
  amplitude *A* (the Amplitude slider, in Å). Overall eigenvector phase is fixed so t = 0 is the instant of largest displacement.
* **Bloch phase.** phonopy's eigenvectors belong to a dynamical matrix whose phase includes the atomic positions, so `r` is the
  atom's full reduced position. This is checked against the equation of motion `F = −mω²u` using the force constants that
  generated the data (`tools/make_reference.py`; a lattice-only phase fails it at every q ≠ 0). phononwebsite JSON uses
  lattice-only phases unless the file sets `addatomphase`, and the reader follows that flag. Both conventions can be switched in the
  *Mode* panel.
* **Mass.** phonopy eigenvectors are divided by √m; phononwebsite JSON vectors are already displacements (checked on its MoS₂ data,
  where the acoustic Γ mode has equal amplitude on Mo and S) and are not.

## Credits

The Examples menu loads real DFT data live from phononwebsite (it is not copied into this repository): graphene, MoS₂ and MoTe₂ by
H. Miranda; hBN from F. Paleari *et al.*, PRL 122, 187401 (2018); TiSe₂ from M. Calandra and F. Mauri, PRL 106, 196406 (2011). The
`.json` format and the idea of a browser phonon viewer come from [phononwebsite](https://henriquemiranda.github.io/phononwebsite/)
(H. Miranda); this viewer is independent and not affiliated with it. Bundled libraries: three.js and gifenc, both MIT
(licences in `vendor/`).

## Layout of the code

```
index.html, css/app.css      page and styles
js/parsers.js                band.yaml and phononwebsite JSON -> one PhononData object
js/physics.js                lattice math, supercells, mode patterns, bonds, view bases (no rendering; unit tested)
js/scene.js                  three.js scene, own arcball/roll/pan camera, axes gizmo, offscreen capture
js/bandplot.js               canvas band structure
js/export.js                 PNG, GIF (gifenc), video (MediaRecorder), XYZ
js/main.js                   state and UI wiring
js/symmetry.js               symmetry data: parsing, per-mode lookup, live fill-in (no rendering; unit tested)
js/symworker.js              background worker: Pyodide + DistortKit (distortkit/web.py) for the in-browser analysis
vendor/                      three.js (tree-shaken, MIT) and gifenc (MIT); rebuild with tools/vendor.sh
tools/make_demo.py           writes the bundled demo band.yaml with phonopy (spring-model force constants)
tools/build_py_bundle.py     packs distortkit/ into py/distortkit.zip for the worker (serve.py does it on the fly; the Pages workflow at deploy)
tools/make_reference.py      verifies the displacement convention independently and writes tests/ref_nacl.json
tests/viewer.test.mjs        node --test (parsers, supercells, bonds, views, the verified reference field)
```

```bash
node --test viewer/tests/viewer.test.mjs
```

## Known limits

* The symmetry analysis needs internet the first time (to fetch the Python runtime from a CDN), covers only q-points whose star needs a
  small supercell (Γ, X, L of an fcc cell, not points along lines), and can take a minute or more for the larger ones.
* Only phonopy `band.yaml` and phononwebsite JSON are read so far (no `band.hdf5`, Quantum ESPRESSO, ABINIT, VASP outputs yet).
* The whole file is read into memory; a multi-hundred-MB `band.yaml` is parsed in seconds but needs a few GB of browser memory.
* The structure is the cell in the file (a primitive cell for phonopy), so e.g. fcc is shown as a rhombohedral block; there is no
  conventional-cell transformation yet.
* phononwebsite's mpdb and phonondb collections use a different, compressed layout and are not browsable yet.
* No DOS plot, no group-velocity or irrep labelling.
* Video recording runs in real time and needs a browser with `MediaRecorder`.
