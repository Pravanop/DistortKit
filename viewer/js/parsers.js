// File readers. Every reader returns the same `PhononData` object (see `finish`):
//
//   name, lattice[3][3] (rows, A), species[], masses[], frac[][3]   -- the cell
//   nq, nb, natom, qpts (Float64 nq*3, reduced), dist (Float64 nq), freq (Float64 nq*nb, THz, ascending per q)
//   eig (Float32, index ((q*nb + b)*natom + a)*6 + c*2 + {0:re,1:im})
//   ticks [{index, x, label}], segments [{start, end}]   -- band-path decoration
//   eigIsDynmat  true when `eig` are eigenvectors of the dynamical matrix (displacement = e / sqrt(m))
//
// The band.yaml reader is a hand-written line scanner rather than a YAML library: phonopy writes a rigid format and files
// with eigenvectors run to hundreds of MB, where a general parser is slow and memory hungry.

import { elementOf, massOf, symbolOfZ } from "./elements.js";

const GREEK = { gamma: "Γ", delta: "Δ", lambda: "Λ", sigma: "Σ", pi: "Π", omega: "Ω", theta: "Θ", xi: "Ξ", phi: "Φ", psi: "Ψ", upsilon: "Υ" };
const SUB = "₀₁₂₃₄₅₆₇₈₉";

/** '$\\Gamma$' -> 'Γ', 'GAMMA' -> 'Γ', 'X_1' -> 'X₁'. */
export function cleanLabel(raw) {
  let s = String(raw ?? "").trim().replace(/^['"]|['"]$/g, "").replace(/\$/g, "").replace(/\\\\/g, "\\");
  s = s.replace(/\\([A-Za-z]+)/g, (m, n) => GREEK[n.toLowerCase()] ?? n);
  if (/^gamma$/i.test(s)) s = "Γ";
  s = s.replace(/_\{?(\d+)\}?/g, (m, d) => [...d].map((c) => SUB[+c]).join("")).replace(/[{}]/g, "");
  return s;
}

function bad(msg) {
  return new Error(msg);
}

function nums(text) {
  return text.split(",").map((t) => parseFloat(t));
}

/** Parse "[ a, b, c ]" appearing anywhere in `line`. */
function bracket(line) {
  const s = line.indexOf("["), e = line.indexOf("]", s);
  if (s < 0 || e < 0) return null;
  return nums(line.slice(s + 1, e));
}

export function parseBandYaml(text, name = "phonopy") {
  const lines = text.split("\n");
  let sec = "";
  let nq = 0, natom = 0, nb = 0;
  const lattice = [];
  const species = [], masses = [], frac = [];
  const segN = [];
  const labelPairs = [];
  let qpts = null, dist = null, freq = null, eig = null;
  let qi = -1, bi = -1, ai = -1, ci = 0, sawEig = false;

  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    if (!line) continue;
    const c0 = line.charCodeAt(0);
    if (c0 !== 32 && c0 !== 45) {
      // a top-level key
      const colon = line.indexOf(":");
      if (colon < 0) continue;
      sec = line.slice(0, colon).trim();
      const rest = line.slice(colon + 1).replace(/#.*/, "").trim();
      if (sec === "nqpoint") nq = parseInt(rest, 10);
      else if (sec === "natom") { natom = parseInt(rest, 10); nb = 3 * natom; }
      else if (sec === "phonon") {
        if (!nq || !natom) throw bad("band.yaml: 'nqpoint'/'natom' must precede 'phonon'");
        qpts = new Float64Array(nq * 3);
        dist = new Float64Array(nq);
        freq = new Float64Array(nq * nb);
        eig = new Float32Array(nq * nb * natom * 6);
      }
      continue;
    }
    if (sec === "phonon") {
      const t = line.trimStart();
      if (t.startsWith("- [")) {
        if (!eig || ai < 0 || ci > 2) continue;
        const s = t.indexOf("["), cm = t.indexOf(",", s), e = t.indexOf("]", cm);
        const o = ((qi * nb + bi) * natom + ai) * 6 + ci * 2;
        eig[o] = parseFloat(t.slice(s + 1, cm));
        eig[o + 1] = parseFloat(t.slice(cm + 1, e));
        ci++;
        sawEig = true;
      } else if (t.startsWith("- q-position:")) {
        qi++; bi = -1; ai = -1;
        if (qi >= nq) throw bad("band.yaml: more q-points than 'nqpoint'");
        const v = bracket(t);
        qpts.set(v, qi * 3);
      } else if (t.startsWith("- #")) {
        if (line.search(/\S/) <= 3) { bi++; ai = -1; } // "  - # 3": a band
        else { ai++; ci = 0; } //                          "    - # atom 2": an atom of that band
      } else if (t.startsWith("frequency:")) {
        freq[qi * nb + bi] = parseFloat(t.slice(10));
      } else if (t.startsWith("distance:")) {
        dist[qi] = parseFloat(t.slice(9));
      }
      continue;
    }
    switch (sec) {
      case "segment_nqpoint": {
        const v = parseInt(line.slice(line.indexOf("-") + 1), 10);
        if (!isNaN(v)) segN.push(v);
        break;
      }
      case "labels": {
        const v = line.slice(line.indexOf("[") + 1, line.lastIndexOf("]")).split(",");
        if (v.length === 2) labelPairs.push(v.map(cleanLabel));
        break;
      }
      case "lattice": {
        const v = bracket(line);
        if (v) lattice.push(v);
        break;
      }
      case "points": {
        const t = line.trimStart();
        if (t.startsWith("- symbol:")) species.push(t.slice(9).replace(/#.*/, "").trim());
        else if (t.startsWith("coordinates:")) frac.push(bracket(t));
        else if (t.startsWith("mass:")) masses.push(parseFloat(t.slice(5)));
        break;
      }
    }
  }
  if (!natom || !nq || !eig) throw bad("this does not look like phonopy output (there is no 'phonon:' section with frequencies)");
  if (qi + 1 !== nq) throw bad(`band.yaml: expected ${nq} q-points, found ${qi + 1} (file truncated?)`);
  if (!sawEig)
    throw bad("this band.yaml has no eigenvectors. Re-run phonopy with EIGENVECTORS = .TRUE. (or --eigvecs) to write them.");
  if (lattice.length !== 3 || frac.length !== natom)
    throw bad("this file has the phonons but not the crystal structure (a qpoints.yaml, for example). The viewer needs band.yaml, which includes the lattice and atomic positions.");

  const segments = [];
  if (segN.length) {
    let s = 0;
    for (const n of segN) { segments.push({ start: s, end: s + n - 1 }); s += n; }
  } else segments.push({ start: 0, end: nq - 1 });
  if (dist[nq - 1] - dist[0] < 1e-9) for (let i = 0; i < nq; i++) dist[i] = i; // qpoints/mesh files carry no path distance
  const ticks = ticksFromSegments(segments, labelPairs, dist, qpts);

  return finish({
    name, lattice, species, masses, frac, nq, nb, natom, qpts, dist, freq, eig,
    ticks, segments, eigIsDynmat: true,
  });
}

/** Reduced coordinate as a short fraction: 0.5 -> '1/2', 0.3333 -> '1/3', else two decimals. */
function fracText(x) {
  if (Math.abs(x) < 1e-6) return "0";
  for (let d = 1; d <= 12; d++) {
    const n = x * d;
    if (Math.abs(n - Math.round(n)) < 2e-3) return d === 1 ? String(Math.round(n)) : `${Math.round(n)}/${d}`;
  }
  return x.toFixed(2);
}

/** Name for a q-point when the file does not label the path: Γ at the origin, otherwise (1/2,0,1/2). */
export function qLabel(q) {
  return q.every((x) => Math.abs(x) < 1e-6) ? "Γ" : `(${q.map(fracText).join(",")})`;
}

/** Turn per-segment end-point labels into ticks, merging 'X' + 'X' and joining 'X' | 'Y' at a jump. */
function ticksFromSegments(segments, pairs, dist, qpts) {
  const ticks = [];
  if (pairs.length !== segments.length) {
    // no (usable) labels in the file: name the segment end points by their q, so the axis is never blank
    if (segments.length < 2) return ticks;
    pairs = segments.map((s) => [qLabel([...qpts.slice(3 * s.start, 3 * s.start + 3)]), qLabel([...qpts.slice(3 * s.end, 3 * s.end + 3)])]);
  }
  segments.forEach((seg, i) => {
    const [a, b] = pairs[i];
    const prev = ticks[ticks.length - 1];
    if (prev && prev.index === seg.start - 1 && prev.label !== a) {
      prev.label = `${prev.label}|${a}`;   // path jumps between two different points: show both
      prev.joint = true;
    } else if (!prev || prev.index !== seg.start - 1) {
      ticks.push({ index: seg.start, x: dist[seg.start], label: a });
    }
    ticks.push({ index: seg.end, x: dist[seg.end], label: b });
  });
  return ticks;
}

// ---- phononwebsite JSON ------------------------------------------------------------------------------------------------
const CM_PER_THZ = 33.35641;

/** 'Bulk MoS<sub>2</sub>' -> 'Bulk MoS₂' (some databases store HTML in the material name). */
export function plainName(s) {
  return String(s ?? "")
    .replace(/<sub>(\d+)<\/sub>/gi, (m, d) => [...d].map((c) => SUB[+c]).join(""))
    .replace(/<[^>]*>/g, "").trim();
}

export function parsePhononJson(obj, name) {
  const need = ["lattice", "qpoints", "eigenvalues", "vectors"];
  for (const k of need) if (!obj[k]) throw bad(`this JSON has no '${k}', so it is not phononwebsite-style phonon data (it needs ${need.join(", ")}, plus atom types and positions).`);
  const nq = obj.qpoints.length, nb = obj.eigenvalues[0].length, natom = nb / 3;
  const species = obj.atom_types ?? (obj.atom_numbers ?? obj.atomic_numbers ?? []).map(symbolOfZ);
  const frac = obj.atom_pos_red;
  if (!frac || !species.length) throw bad("JSON is missing atom types or reduced positions");
  const qpts = new Float64Array(nq * 3);
  obj.qpoints.forEach((q, i) => qpts.set(q, i * 3));
  const dist = Float64Array.from(obj.distances ?? obj.qpoints.map((_, i) => i));
  const freq = new Float64Array(nq * nb);
  obj.eigenvalues.forEach((row, i) => row.forEach((f, b) => (freq[i * nb + b] = f / CM_PER_THZ)));
  const eig = new Float32Array(nq * nb * natom * 6);
  let o = 0;
  for (let q = 0; q < nq; q++)
    for (let b = 0; b < nb; b++)
      for (let a = 0; a < natom; a++)
        for (let c = 0; c < 3; c++) {
          const z = obj.vectors[q][b][a][c];
          eig[o++] = z[0];
          eig[o++] = z[1];
        }
  const ticks = (obj.highsym_qpts ?? []).map(([index, label]) => ({ index, x: dist[index], label: cleanLabel(label) }));
  const segments = [{ start: 0, end: nq - 1 }];
  const masses = species.map((s) => massOf(elementOf(s)));
  const reps = obj.repetitions && obj.repetitions.length === 3 ? obj.repetitions : null;
  return finish({
    name: plainName(obj.name) || name, lattice: obj.lattice, species, masses, frac, nq, nb, natom, qpts, dist, freq, eig,
    ticks, segments, eigIsDynmat: false, suggestedRepetitions: reps,
    // phononwebsite's own rule: atomic positions enter the Bloch phase only when the dataset says so (`addatomphase`)
    phaseWithBasis: !!obj.addatomphase,
  });
}

// ---- common ------------------------------------------------------------------------------------------------------------
function finish(d) {
  const n = d.natom;
  if (d.species.length !== n || d.masses.length !== n) throw bad("inconsistent atom counts");
  d.elements = d.species.map(elementOf);
  d.phaseWithBasis = d.phaseWithBasis ?? true; // phonopy convention: atomic positions are part of the Bloch phase
  d.formula = formulaOf(d.elements);
  // reduced position of every atom shifted into [0,1); keep the integer shift for the Bloch phase (see physics.js)
  d.wrapShift = d.frac.map((f) => f.map((x) => Math.floor(x + 1e-6)));
  return d;
}

export function formulaOf(elements) {
  const count = new Map();
  for (const e of elements) count.set(e, (count.get(e) ?? 0) + 1);
  return [...count].map(([e, n]) => e + (n > 1 ? String(n).replace(/\d/g, (d) => SUB[+d]) : "")).join("");
}

const PHONOPY_HOWTO = "To make one, add EIGENVECTORS = .TRUE. to your band configuration (for example BAND = AUTO) and run: phonopy -p band.conf";

/** Say what a file that is not phonon data actually is, when it is recognisable. */
function explainUnknown(text, filename) {
  const name = filename.toLowerCase();
  const head = text.slice(0, 4000);
  if (text.startsWith("\u0089HDF") || /^\uFFFD?HDF/.test(text.slice(0, 5)) || name.endsWith(".hdf5") || name.endsWith(".h5"))
    return `${filename} is an HDF5 file. HDF5 band files are not supported yet; please upload the band.yaml that phonopy writes. ${PHONOPY_HOWTO}`;
  if (/^(phonopy_disp|phonopy_params|phonopy)\.ya?ml$/.test(name.split("/").pop()) || (/^phonopy:/m.test(head) && /supercell_matrix:/.test(head) && !/^phonon:/m.test(text)))
    return `${filename} is a phonopy input (structure and displacements), not phonon results. The viewer needs the band structure with eigenvectors. ${PHONOPY_HOWTO}`;
  if (/phono3py/.test(head)) return `${filename} comes from phono3py. Please upload a phonopy band.yaml. ${PHONOPY_HOWTO}`;
  if (/^(FORCE_SETS|FORCE_CONSTANTS|BORN)$/i.test(name) || /^(POSCAR|CONTCAR|vasprun)/i.test(name) || /\.(cif|xml)$/.test(name))
    return `${filename} is not a phonon result file. The viewer needs a phonopy band.yaml (with eigenvectors). ${PHONOPY_HOWTO}`;
  return `${filename} is not a phonopy band.yaml or a phononwebsite .json. ${PHONOPY_HOWTO}`;
}

/** Pick a reader from the file name / content. */
export function parseAuto(text, filename = "") {
  const base = filename.replace(/\.[^.]+$/, "") || "phonons";
  text = text.replace(/^\uFEFF/, "");
  const head = text.slice(0, 200).trimStart();
  if (head.startsWith("{")) {
    let obj;
    try { obj = JSON.parse(text); } catch (e) { throw bad(`${filename || "This file"} is not valid JSON (${e.message}).`); }
    return parsePhononJson(obj, base);
  }
  if (/^(nqpoint|npath|natom|segment_nqpoint|reciprocal_lattice|lattice|points|phonon):/m.test(head + "\n") || /^phonon:/m.test(text))
    return parseBandYaml(text, base);
  throw bad(explainUnknown(text, filename || "This file"));
}
