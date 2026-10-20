// Run with:  node --test viewer/tests/
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseBandYaml, parseAuto, cleanLabel, plainName } from "../js/parsers.js";
import {
  buildStructure, modePattern, displace, findBonds, viewBasis, commensurateSupercell, isCommensurate, reciprocal,
  latticeParams, dot, norm, cross, speciesWeights, formatFreq,
} from "../js/physics.js";

const here = dirname(fileURLToPath(import.meta.url));
const load = (f) => readFileSync(join(here, "..", "examples", f), "utf8");
const graphene = () => parseAuto(load("graphene_springs.yaml"), "graphene_springs.yaml");
const nacl = () => parseAuto(load("nacl_springs.yaml"), "nacl_springs.yaml");
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} !~ ${b}`);

test("labels are cleaned", () => {
  assert.equal(cleanLabel("$\\Gamma$"), "Γ");
  assert.equal(cleanLabel("'GAMMA'"), "Γ");
  assert.equal(cleanLabel("X_1"), "X₁");
  assert.equal(cleanLabel("\\Sigma_{1}"), "Σ₁");
});

test("band.yaml: header, cell and path", () => {
  const d = graphene();
  assert.equal([d.nq, d.nb, d.natom].join(), "120,6,2");
  assert.deepEqual(d.species, ["C", "C"]);
  assert.equal(d.formula, "C₂");
  close(d.lattice[0][0], 2.46);
  close(d.masses[0], 12.0107, 1e-4);
  assert.deepEqual(d.segments.map((s) => [s.start, s.end]), [[0, 39], [40, 79], [80, 119]]);
  // Gamma | M | K | Gamma: the shared end points are merged into one tick each
  assert.deepEqual(d.ticks.map((t) => t.label), ["Γ", "M", "K", "Γ"]);
  assert.deepEqual(d.ticks.map((t) => t.index), [0, 39, 79, 119]);
  close(d.dist[119] > d.dist[0] ? 1 : 0, 1);
});

test("band.yaml: eigenvectors are normalised and frequencies sorted", () => {
  const d = nacl();
  for (let q = 0; q < d.nq; q += 7)
    for (let b = 0; b < d.nb; b++) {
      let s = 0;
      for (let k = 0; k < d.natom * 6; k++) s += d.eig[(q * d.nb + b) * d.natom * 6 + k] ** 2;
      close(s, 1, 1e-4);
      if (b) assert.ok(d.freq[q * d.nb + b] >= d.freq[q * d.nb + b - 1] - 1e-6);
    }
  // acoustic modes vanish at Gamma
  for (let b = 0; b < 3; b++) assert.ok(Math.abs(d.freq[b]) < 1e-3);
});

test("a file without eigenvectors is rejected with advice", () => {
  const text = load("graphene_springs.yaml").split("\n").filter((l) => !/^\s+(eigenvector:|- # atom|      - \[)/.test(l)).join("\n");
  assert.throws(() => parseBandYaml(text), /EIGENVECTORS/);
});

test("a truncated file is reported", () => {
  const text = load("graphene_springs.yaml");
  assert.throws(() => parseBandYaml(text.slice(0, text.length >> 1)), /truncated|expected/);
});

test("phononwebsite JSON round-trips through parseAuto", () => {
  const d = graphene();
  const vectors = [];
  for (let q = 0; q < d.nq; q++) {
    const bands = [];
    for (let b = 0; b < d.nb; b++) {
      const atoms = [];
      for (let a = 0; a < d.natom; a++) {
        const comps = [];
        for (let c = 0; c < 3; c++) {
          const o = ((q * d.nb + b) * d.natom + a) * 6 + c * 2;
          comps.push([d.eig[o], d.eig[o + 1]]);
        }
        atoms.push(comps);
      }
      bands.push(atoms);
    }
    vectors.push(bands);
  }
  const json = {
    name: "g", lattice: d.lattice, atom_types: d.species, atom_pos_red: d.frac,
    qpoints: Array.from({ length: d.nq }, (_, i) => [...d.qpts.slice(3 * i, 3 * i + 3)]),
    distances: [...d.dist],
    eigenvalues: Array.from({ length: d.nq }, (_, q) => [...d.freq.slice(q * d.nb, q * d.nb + d.nb)].map((f) => f * 33.35641)),
    vectors, highsym_qpts: [[0, "\\Gamma"], [39, "M"], [119, "\\Gamma"]],
  };
  const j = parseAuto(JSON.stringify(json), "g.json");
  assert.equal(j.nq, d.nq);
  close(j.freq[50 * 6 + 4], d.freq[50 * 6 + 4], 1e-5);
  close(j.eig[1000], d.eig[1000], 1e-6);
  assert.equal(j.ticks[0].label, "Γ");
  assert.equal(j.eigIsDynmat, false);   // phononwebsite vectors are already displacements
  assert.equal(j.phaseWithBasis, false); // ... and use lattice-translation phases unless `addatomphase` is set
  assert.equal(parseAuto(JSON.stringify({ ...json, addatomphase: true }), "g.json").phaseWithBasis, true);
  assert.equal(d.phaseWithBasis, true);  // phonopy band.yaml: positions are part of the phase
});

test("unknown input is rejected", () => {
  assert.throws(() => parseAuto("hello world"), /not a phonopy band\.yaml/);
  assert.throws(() => parseAuto("{not json"), /not valid JSON/);
});

test("lattice helpers", () => {
  const lat = graphene().lattice;
  const p = latticeParams(lat);
  close(p.gamma, 120, 1e-6);
  const rec = reciprocal(lat);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) close(dot(lat[i], rec[j]), i === j ? 1 : 0, 1e-9);
});

test("viewing down c puts a to the right and b-ish up", () => {
  const lat = graphene().lattice;
  const v = viewBasis(lat[2], lat);
  close(dot(v.right, lat[0]) / norm(lat[0]), 1, 1e-9);
  close(dot(v.up, lat[1]) > 0 ? 1 : 0, 1);
  const r = cross(v.right, v.up);
  close(dot(r, v.back), 1, 1e-9); // right-handed
  // down a: a is parallel to the view direction, so the part of b perpendicular to a becomes the right axis
  const va = viewBasis(lat[0], lat);
  close(dot(va.right, lat[0]), 0, 1e-9);
  assert.ok(dot(va.right, lat[1]) > 0);
});

test("commensurate supercells", () => {
  assert.deepEqual(commensurateSupercell([0.5, 0, 0.5]).reps, [2, 1, 2]);
  assert.deepEqual(commensurateSupercell([1 / 3, 1 / 3, 0]).reps, [3, 3, 1]);
  assert.equal(commensurateSupercell([0.1234567, 0, 0]).exact, false);
  assert.ok(isCommensurate([0.25, 0, 0], [4, 1, 1]));
  assert.ok(!isCommensurate([0.25, 0, 0], [3, 1, 1]));
});

test("supercell with wrap + complete: atoms on faces are repeated", () => {
  const d = graphene();
  const s1 = buildStructure(d, { reps: [1, 1, 1], complete: false });
  assert.equal(s1.n, 2);
  const s2 = buildStructure(d, { reps: [2, 2, 1], complete: false });
  assert.equal(s2.n, 8);
  // graphene sits at z = 0 but c is a vacuum axis, so nothing is repeated across the vacuum
  const s3 = buildStructure(d, { reps: [1, 1, 1], complete: true });
  assert.equal(s3.n, 2);
  // NaCl's Na is on the corner of the primitive cell: 8 corner copies + the Cl in the middle
  assert.equal(buildStructure(nacl(), { reps: [1, 1, 1], complete: true }).n, 9);
  assert.equal(buildStructure(nacl(), { reps: [1, 1, 1], complete: false }).n, 2);
});

test("zone-centre acoustic mode is a rigid translation", () => {
  const d = nacl();
  const s = buildStructure(d, { reps: [2, 2, 2], complete: false });
  const p = modePattern(d, s, 0, 0);
  for (let i = 1; i < s.n; i++)
    for (let c = 0; c < 3; c++) close(p.re[3 * i + c], p.re[c], 1e-5);
});

test("periodic images of a commensurate mode move together", () => {
  const d = nacl();
  // pick the X point (index 39) where q = (1/2, 0, 1/2): periodic in a 2x1x2 cell
  const qi = d.ticks[1].index;
  const q = [...d.qpts.slice(3 * qi, 3 * qi + 3)];
  const { reps, exact } = commensurateSupercell(q);
  assert.ok(exact);
  const s = buildStructure(d, { reps, complete: true });
  const near = new Map(), far = [];
  for (let i = 0; i < s.n; i++) {
    const key = `${s.kappa[i]}|${s.cell.slice(3 * i, 3 * i + 3).map((x, k) => ((x % reps[k]) + reps[k]) % reps[k])}`;
    if (near.has(key)) far.push([near.get(key), i]);
    else near.set(key, i);
  }
  assert.ok(far.length > 0, "expected image atoms");
  const p = modePattern(d, s, qi, 4);
  for (const [i, j] of far)
    for (let c = 0; c < 3; c++) {
      close(p.re[3 * i + c], p.re[3 * j + c], 1e-4);
      close(p.im[3 * i + c], p.im[3 * j + c], 1e-4);
    }
});

test("peak displacement is normalised and displace() moves atoms", () => {
  const d = nacl();
  const s = buildStructure(d, { reps: [2, 2, 2] });
  const p = modePattern(d, s, 20, 4);
  let peak = 0;
  for (let i = 0; i < s.n; i++) peak = Math.max(peak, Math.hypot(p.re[3 * i], p.re[3 * i + 1], p.re[3 * i + 2]));
  assert.ok(peak > 0.3 && peak <= 1.0001);
  const out = new Float32Array(s.base.length);
  displace(s.base, p, 0.5, 0, out);
  let moved = 0;
  for (let i = 0; i < out.length; i++) moved = Math.max(moved, Math.abs(out[i] - s.base[i]));
  assert.ok(moved > 0.01 && moved <= 0.5 + 1e-6);
});

test("bonds: graphene has 3 per atom, NaCl has 6 per atom in the bulk", () => {
  const g = graphene();
  const s = buildStructure(g, { reps: [4, 4, 1], complete: false });
  const b = findBonds(s.base, [...s.kappa].map((k) => g.elements[k]));
  // an infinite sheet has 3 bonds per atom /2 per bond = 1.5 per atom; edges lose some
  assert.ok(b.length / 2 > 16 * 2 * 1.2 && b.length / 2 < 16 * 2 * 1.5 + 1);
  const n = nacl();
  const sn = buildStructure(n, { reps: [3, 3, 3], complete: false });
  const bn = findBonds(sn.base, [...sn.kappa].map((k) => n.elements[k]));
  const deg = new Array(sn.n).fill(0);
  for (let i = 0; i < bn.length; i++) deg[bn[i]]++;
  assert.equal(Math.max(...deg), 6);
});

test("species weights sum to one", () => {
  const d = nacl();
  const { names, w } = speciesWeights(d);
  assert.deepEqual(names, ["Na", "Cl"]);
  for (let i = 0; i < d.nq * d.nb; i++) close(w[2 * i] + w[2 * i + 1], 1, 1e-4);
});

test("frequency units", () => {
  assert.equal(formatFreq(1, "cm-1"), "33.4");
  assert.equal(formatFreq(1, "meV"), "4.14");
});

test("cell outlines", async () => {
  const { cellSegments } = await import("../js/physics.js");
  const lat = graphene().lattice;
  assert.equal(cellSegments(lat, [3, 3, 1], "off").length, 0);
  assert.equal(cellSegments(lat, [3, 3, 1], "unit").length, 12);
  assert.equal(cellSegments(lat, [3, 3, 1], "super").length, 12);
  // 2x1x1 block: edges along a: 2 cells x (2x2 corner lines) = 8; along b: 3x2 = 6; along c: 3x2 = 6
  assert.equal(cellSegments(lat, [2, 1, 1], "all").length, 8 + 6 + 6);
});

test("default and q-fitted supercells", async () => {
  const { defaultSupercell, fitSupercell } = await import("../js/physics.js");
  assert.deepEqual(defaultSupercell(graphene()), [4, 4, 1]); // a slab: one cell along the vacuum direction
  assert.deepEqual(defaultSupercell(nacl()), [2, 2, 2]);
  assert.deepEqual(fitSupercell([1 / 3, 1 / 3, 0], [4, 4, 1]), [6, 6, 1]);
  assert.deepEqual(fitSupercell([0.5, 0, 0], [4, 4, 1]), [4, 4, 1]);
  assert.deepEqual(fitSupercell([0.1234567, 0, 0], [4, 4, 1]), [4, 4, 1]); // not periodic at any small size: keep
});

test("vacuum axes", async () => {
  const { vacuumAxes } = await import("../js/physics.js");
  assert.deepEqual(vacuumAxes(graphene()), [false, false, true]);
  assert.deepEqual(vacuumAxes(nacl()), [false, false, false]);
});

// The reference field was verified in Python (tools/make_reference.py) to satisfy F = -m w^2 u with the force constants that
// generated the file. The viewer must reproduce it up to the arbitrary global complex phase and scale of an eigenvector.
test("displacement field reproduces the equation-of-motion-verified reference", () => {
  const ref = JSON.parse(readFileSync(join(here, "ref_nacl.json"), "utf8"));
  const d = nacl();
  const s = buildStructure(d, { reps: [3, 3, 3], complete: false });
  const p = modePattern(d, s, ref.qi, ref.band);
  const key = (kappa, r) => `${kappa}|${r.map((x) => Math.round((((x % 3) + 3) % 3) * 1e3) % 3000).join(",")}`;
  const lookup = new Map(ref.atoms.map((a) => [key(a.kappa, a.red), a]));
  assert.equal(lookup.size, s.n);
  // <ref, js> over all atoms and components; |<ref,js>| = |ref||js| iff the two differ by one complex constant
  let sr = 0, si = 0, nr = 0, nj = 0;
  for (let i = 0; i < s.n; i++) {
    const a = lookup.get(key(s.kappa[i], [...s.red.slice(3 * i, 3 * i + 3)]));
    assert.ok(a, `atom ${i} not in reference`);
    for (let c = 0; c < 3; c++) {
      const zr = a.re[c], zi = a.im[c], wr = p.re[3 * i + c], wi = p.im[3 * i + c];
      sr += zr * wr + zi * wi; si += zr * wi - zi * wr;
      nr += zr * zr + zi * zi; nj += wr * wr + wi * wi;
    }
  }
  close(Math.hypot(sr, si) / Math.sqrt(nr * nj), 1, 1e-4);
});

test("the cells-only phase convention gives a different field (so the option matters)", () => {
  const d = nacl();
  const s = buildStructure(d, { reps: [3, 3, 3], complete: false });
  const a = modePattern(d, s, 26, 4), b = modePattern(d, s, 26, 4, { basisPhase: false });
  let diff = 0;
  for (let i = 0; i < a.re.length; i++) diff += Math.abs(a.re[i] - b.re[i]) + Math.abs(a.im[i] - b.im[i]);
  assert.ok(diff > 1);
});

test("no negative zero in frequency text", () => {
  assert.equal(formatFreq(-1e-8, "THz"), "0.000");
  assert.equal(formatFreq(-0.2, "THz"), "-0.200");
});

test("HTML in material names is made plain", () => {
  assert.equal(plainName("Bulk MoS<sub>2</sub>"), "Bulk MoS₂");
  assert.equal(plainName("Ag<sub>10</sub>S<sub>8</sub>Sb<sub>2</sub>"), "Ag₁₀S₈Sb₂");
});

// ---- files that are not quite the usual band.yaml ------------------------------------------------------------------------
const naclText = () => load("nacl_springs.yaml");

test("a path with no labels still gets ticks (named by q)", () => {
  const text = naclText().replace(/^labels:\n(- \[.*\]\n)+/m, "");
  const d = parseBandYaml(text);
  assert.ok(d.ticks.length >= 3, "expected ticks at the segment ends");
  assert.equal(d.ticks[0].label, "Γ");
  assert.match(d.ticks[1].label, /^\(1\/2,0,1\/2\)$/);
});

test("a file without path distances (qpoints/mesh style) is spaced by index", () => {
  const d = parseBandYaml(naclText().replace(/^  distance: .*\n/gm, ""));
  assert.equal(d.dist[0], 0);
  assert.equal(d.dist[d.nq - 1], d.nq - 1);
});

test("CRLF line endings and a byte-order mark are accepted", () => {
  const d = parseAuto("\uFEFF" + naclText().replace(/\n/g, "\r\n"), "x.yaml");
  assert.equal(d.nq, 120);
  assert.deepEqual(d.ticks.map((t) => t.label), ["Γ", "X", "Γ", "L"]);
});

test("unusable files are explained, not just rejected", () => {
  // a qpoints.yaml has the phonons but no crystal structure
  const noCell = naclText().replace(/^lattice:\n(- \[.*\]\n){3}/m, "").replace(/^points:\n([- ] .*\n)+/m, "");
  assert.throws(() => parseBandYaml(noCell), /not the crystal structure/);
  // phonopy input files
  assert.throws(() => parseAuto("phonopy:\n  version: 2.0\n  calculator: vasp\nsupercell_matrix:\n- [ 2, 0, 0 ]\n", "phonopy_disp.yaml"), /phonopy input/);
  // HDF5 (binary, so it decodes to junk starting with the HDF signature)
  assert.throws(() => parseAuto("\u0089HDF\r\n\u001a\n", "band.hdf5"), /HDF5/);
  // everything else
  assert.throws(() => parseAuto("hello", "notes.txt"), /not a phonopy band\.yaml/);
  assert.throws(() => parseAuto("{\"a\": 1}", "x.json"), /no 'lattice'/);
  assert.throws(() => parseAuto("{nope", "x.json"), /not valid JSON/);
});
