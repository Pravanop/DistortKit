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
* **Band plot.** Click or drag to select a mode; arrow keys step q and band; hover shows the frequency and q; imaginary branches
  are drawn below a dashed zero line and flagged; optional colouring by species; THz / cm⁻¹ / meV. A per-mode list for the
  current q-point shows each mode's species composition.
* **Layout.** One responsive page (resizable split, stacked on narrow screens, light and dark themes) instead of a scrolling box
  that covers the crystal.

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
vendor/                      three.js (tree-shaken, MIT) and gifenc (MIT); rebuild with tools/vendor.sh
tools/make_demo.py           writes the bundled demo band.yaml with phonopy (spring-model force constants)
tools/make_reference.py      verifies the displacement convention independently and writes tests/ref_nacl.json
tests/viewer.test.mjs        node --test (parsers, supercells, bonds, views, the verified reference field)
```

```bash
node --test viewer/tests/viewer.test.mjs
```

## Known limits

* Only phonopy `band.yaml` and phononwebsite JSON are read so far (no `band.hdf5`, Quantum ESPRESSO, ABINIT, VASP outputs yet).
* The whole file is read into memory; a multi-hundred-MB `band.yaml` is parsed in seconds but needs a few GB of browser memory.
* The structure is the cell in the file (a primitive cell for phonopy), so e.g. fcc is shown as a rhombohedral block; there is no
  conventional-cell transformation yet.
* phononwebsite's mpdb and phonondb collections use a different, compressed layout and are not browsable yet.
* No DOS plot, no group-velocity or irrep labelling.
* Video recording runs in real time and needs a browser with `MediaRecorder`.
