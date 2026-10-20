// Geometry and lattice-dynamics helpers with no rendering dependencies (so they can be unit-tested under node).

import { radiusOf } from "./elements.js";

export const UNITS = {
  THz: { factor: 1, label: "THz", digits: 3 },
  "cm-1": { factor: 33.35641, label: "cm⁻¹", digits: 1 },
  meV: { factor: 4.135667696, label: "meV", digits: 2 },
};
export const toUnit = (thz, unit) => thz * UNITS[unit].factor;
export function formatFreq(thz, unit) {
  const { digits } = UNITS[unit], v = toUnit(thz, unit);
  return (Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v).toFixed(digits); // no "-0.000" for numerical noise
}

// ---- small vector helpers (plain arrays) ---------------------------------------------------------------------------------
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const unit = (a) => scale(a, 1 / (norm(a) || 1));

/** Cartesian position of reduced coordinates `f` for lattice rows `lat`. */
export const fracToCart = (f, lat) => [
  f[0] * lat[0][0] + f[1] * lat[1][0] + f[2] * lat[2][0],
  f[0] * lat[0][1] + f[1] * lat[1][1] + f[2] * lat[2][1],
  f[0] * lat[0][2] + f[1] * lat[1][2] + f[2] * lat[2][2],
];

/** Reciprocal vectors a*, b*, c* (rows) with a_i . b*_j = delta_ij (no 2 pi). */
export function reciprocal(lat) {
  const [a, b, c] = lat;
  const v = dot(a, cross(b, c));
  return [scale(cross(b, c), 1 / v), scale(cross(c, a), 1 / v), scale(cross(a, b), 1 / v)];
}

export function latticeParams(lat) {
  const [a, b, c] = lat;
  const ang = (u, v) => (Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (norm(u) * norm(v))))) * 180) / Math.PI;
  return { a: norm(a), b: norm(b), c: norm(c), alpha: ang(b, c), beta: ang(a, c), gamma: ang(a, b), volume: Math.abs(dot(a, cross(b, c))) };
}

/**
 * Camera basis for looking *down* `dir` (camera on the +dir side, looking toward the origin). The screen's right axis is the
 * first of `refs` that is not parallel to `dir`, projected perpendicular to it: looking down c puts a to the right and b up.
 */
export function viewBasis(dir, refs) {
  const z = unit(dir);
  let x = null;
  for (const r of refs) {
    const p = sub(r, scale(z, dot(r, z)));
    if (norm(p) > 0.1 * norm(r)) { x = unit(p); break; }
  }
  if (!x) x = unit(sub([1, 0, 0], scale(z, z[0])));
  const y = cross(z, x);
  return { right: x, up: y, back: z };
}

/** Smallest supercell (per axis, <= max) in which the q-point is periodic; `exact` is false if none was found. */
export function commensurateSupercell(q, max = 12, tol = 2e-3) {
  const out = [1, 1, 1];
  let exact = true;
  q.forEach((qi, i) => {
    let found = false;
    for (let n = 1; n <= max && !found; n++) {
      const x = qi * n;
      if (Math.abs(x - Math.round(x)) < tol) { out[i] = n; found = true; }
    }
    if (!found) exact = false;
  });
  return { reps: out, exact };
}

/** Whether supercell `reps` is periodic for q (so atoms on opposite faces move identically). */
export function isCommensurate(q, reps, tol = 2e-3) {
  return q.every((qi, i) => Math.abs(qi * reps[i] - Math.round(qi * reps[i])) < tol);
}

/**
 * Which lattice axes are a vacuum direction (a slab or chain): the atoms leave a gap of at least `gap` A along the axis, even
 * allowing for the gap to wrap around the cell boundary.
 */
export function vacuumAxes(data, gap = 6) {
  return [0, 1, 2].map((i) => {
    const len = norm(data.lattice[i]);
    const f = data.frac.map((x) => ((x[i] % 1) + 1) % 1).sort((a, b) => a - b);
    let biggest = 1 - f[f.length - 1] + f[0]; // the wrap-around gap
    for (let k = 1; k < f.length; k++) biggest = Math.max(biggest, f[k] - f[k - 1]);
    return biggest * len >= gap && len > gap;
  });
}

// ---- supercell ------------------------------------------------------------------------------------------------------------
/**
 * The atoms to draw: every atom of the cell in an nx x ny x nz block of cells.  With `wrap`, atoms are brought into the cell
 * [0,1)^3; with `complete`, atoms that sit on the lower face of the block are repeated on the opposite face, so the outline
 * of the block is closed (the image atoms get the phase of the cell they are in, so a non-commensurate q shows its seam).
 */
export function buildStructure(data, { reps = [1, 1, 1], wrap = true, complete = true } = {}) {
  const kappa = [], cell = [], base = [], red = [];
  const lat = data.lattice;
  const tol = 1e-4;
  const vac = vacuumAxes(data);
  for (let k = 0; k < data.natom; k++) {
    const shift = wrap ? data.wrapShift[k] : [0, 0, 0];
    const f = data.frac[k].map((x, i) => x - shift[i]);
    const extra = f.map((x, i) => (wrap && complete && !vac[i] && Math.abs(x) < tol ? 1 : 0)); // 1 => also on the far face along axis i
    for (let i = 0; i < reps[0] + extra[0]; i++)
      for (let j = 0; j < reps[1] + extra[1]; j++)
        for (let l = 0; l < reps[2] + extra[2]; l++) { // cell index N (far face) is visited only by atoms on the near face
          kappa.push(k);
          cell.push(i - shift[0], j - shift[1], l - shift[2]); // lattice translation of the atom as the file defines it
          red.push(f[0] + i, f[1] + j, f[2] + l); // physical reduced position, which sets the Bloch phase
          base.push(...fracToCart([f[0] + i, f[1] + j, f[2] + l], lat));
        }
  }
  return { n: kappa.length, kappa: Uint16Array.from(kappa), cell: Int16Array.from(cell), red: Float64Array.from(red), base: Float32Array.from(base), reps };
}

/**
 * Complex displacement pattern of one mode over the drawn atoms, so that u(t) = amp * (re*cos(wt) + im*sin(wt))
 * = amp * Re[(re + i im) e^{-i w t}], with (re + i im) = e/sqrt(m) * exp(2 pi i q.r).
 * Normalised so the atom that moves most has |u| = 1 at its peak. The overall phase of an eigenvector is arbitrary; it is
 * fixed so the largest component is real, which makes t = 0 the instant of maximum displacement.
 */
export function modePattern(data, struct, qi, bi, { massScale = data.eigIsDynmat, basisPhase = data.phaseWithBasis } = {}) {
  const { natom, nb } = data;
  const q = [data.qpts[qi * 3], data.qpts[qi * 3 + 1], data.qpts[qi * 3 + 2]];
  const eigAt = (a, c) => ((qi * nb + bi) * natom + a) * 6 + c * 2;
  let best = 0, gauge = [1, 0];
  for (let a = 0; a < natom; a++)
    for (let c = 0; c < 3; c++) {
      const o = eigAt(a, c), m = Math.hypot(data.eig[o], data.eig[o + 1]);
      if (m > best + 1e-9) { best = m; gauge = m ? [data.eig[o] / m, -data.eig[o + 1] / m] : [1, 0]; }
    }
  const n = struct.n;
  const re = new Float32Array(n * 3), im = new Float32Array(n * 3);
  let peak = 0;
  for (let i = 0; i < n; i++) {
    const k = struct.kappa[i];
    const w = massScale ? 1 / Math.sqrt(data.masses[k]) : 1;
    // Bloch phase. phonopy's eigenvectors belong to the dynamical matrix with the atomic positions in the phase, so the wave
    // is exp(2 pi i q.r) with r the atom's full reduced position; codes that use lattice translations only need `basisPhase=false`.
    const r = basisPhase ? struct.red : struct.cell;
    const phi = 2 * Math.PI * (q[0] * r[3 * i] + q[1] * r[3 * i + 1] + q[2] * r[3 * i + 2]);
    // total phase factor = e^{i phi} * gauge
    const pr = Math.cos(phi) * gauge[0] - Math.sin(phi) * gauge[1];
    const pi = Math.cos(phi) * gauge[1] + Math.sin(phi) * gauge[0];
    let n2 = 0;
    for (let c = 0; c < 3; c++) {
      const o = eigAt(k, c);
      const er = data.eig[o] * w, ei = data.eig[o + 1] * w;
      re[3 * i + c] = er * pr - ei * pi;
      im[3 * i + c] = er * pi + ei * pr;
      n2 += er * er + ei * ei;
    }
    if (n2 > peak) peak = n2;
  }
  const s = peak > 0 ? 1 / Math.sqrt(peak) : 0;
  for (let i = 0; i < re.length; i++) { re[i] *= s; im[i] *= s; }
  return { re, im };
}

/** pos = base + amp * (re cos(phase) + im sin(phase)); writes into `out` (Float32Array n*3). */
export function displace(base, pattern, amp, phase, out) {
  const c = Math.cos(phase) * amp, s = Math.sin(phase) * amp;
  const { re, im } = pattern;
  for (let i = 0; i < base.length; i++) out[i] = base[i] + re[i] * c + im[i] * s;
  return out;
}

/** Fraction of each mode's |e|^2 carried by each species, for colouring the band plot. */
export function speciesWeights(data) {
  const names = [...new Set(data.elements)];
  const index = data.elements.map((e) => names.indexOf(e));
  const w = new Float32Array(data.nq * data.nb * names.length);
  const { natom, nb } = data;
  for (let q = 0; q < data.nq; q++)
    for (let b = 0; b < nb; b++) {
      let tot = 0;
      const acc = new Array(names.length).fill(0);
      for (let a = 0; a < natom; a++) {
        let s = 0;
        const o = ((q * nb + b) * natom + a) * 6;
        for (let c = 0; c < 6; c++) s += data.eig[o + c] ** 2;
        acc[index[a]] += s;
        tot += s;
      }
      acc.forEach((v, k) => (w[(q * nb + b) * names.length + k] = tot ? v / tot : 0));
    }
  return { names, w };
}

// ---- bonds ----------------------------------------------------------------------------------------------------------------
/** Pairs (i<j) closer than tol * (r_i + r_j) in the equilibrium positions; a spatial hash keeps it linear in atom count. */
export function findBonds(base, elements, tol = 1.15) {
  const n = base.length / 3;
  const rad = elements.map(radiusOf);
  const rmax = Math.max(...rad, 0.5);
  const cs = 2 * rmax * tol;
  const grid = new Map();
  const key = (x, y, z) => `${x},${y},${z}`;
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(base[3 * i] / cs), Math.floor(base[3 * i + 1] / cs), Math.floor(base[3 * i + 2] / cs));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
  const pairs = [];
  for (let i = 0; i < n; i++) {
    const gx = Math.floor(base[3 * i] / cs), gy = Math.floor(base[3 * i + 1] / cs), gz = Math.floor(base[3 * i + 2] / cs);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const cell = grid.get(key(gx + dx, gy + dy, gz + dz));
          if (!cell) continue;
          for (const j of cell) {
            if (j <= i) continue;
            const d = Math.hypot(base[3 * i] - base[3 * j], base[3 * i + 1] - base[3 * j + 1], base[3 * i + 2] - base[3 * j + 2]);
            if (d > 0.3 && d < tol * (rad[i] + rad[j])) pairs.push(i, j);
          }
        }
  }
  return Uint32Array.from(pairs);
}

// ---- cell outlines ----------------------------------------------------------------------------------------------------------
/**
 * Line segments [[p0, p1], ...] (Cartesian) outlining the cell: 'unit' = the one cell at the origin, 'super' = the outer
 * parallelepiped of the nx x ny x nz block, 'all' = every cell edge in the block (shared edges drawn once).
 */
export function cellSegments(lat, reps, mode) {
  if (mode === "off") return [];
  const P = (i, j, k) => fracToCart([i, j, k], lat);
  const out = [];
  const [nx, ny, nz] = mode === "all" ? reps : mode === "super" ? reps : [1, 1, 1];
  const step = mode === "all" ? 1 : null;
  const along = (axis) => {
    const n = [nx, ny, nz];
    const len = step ? 1 : n[axis];
    const others = [0, 1, 2].filter((x) => x !== axis);
    for (let u = 0; u <= n[others[0]]; u += step ?? n[others[0]])
      for (let v = 0; v <= n[others[1]]; v += step ?? n[others[1]])
        for (let s = 0; s < n[axis]; s += len) {
          const a = [0, 0, 0], b = [0, 0, 0];
          a[axis] = s; b[axis] = s + len;
          a[others[0]] = b[others[0]] = u;
          a[others[1]] = b[others[1]] = v;
          out.push([P(...a), P(...b)]);
        }
  };
  [0, 1, 2].forEach(along);
  return out;
}

// ---- default supercells ----------------------------------------------------------------------------------------------------
/** A reasonable starting supercell: roughly 9 A across, and a single cell along a vacuum direction of a slab. */
export function defaultSupercell(data) {
  const lens = data.lattice.map(norm);
  const vac = vacuumAxes(data);
  return lens.map((L, i) => (vac[i] ? 1 : Math.max(1, Math.min(4, Math.round(9 / L)))));
}

/**
 * Supercell for wavevector q: where q is periodic in an n-cell block (n <= max), use the multiple of n closest to
 * (and not below) `prefer`; otherwise keep `prefer`.
 */
export function fitSupercell(q, prefer, max = 12) {
  const { reps: need } = commensurateSupercell(q, max);
  return prefer.map((p, i) => {
    if (Math.abs(q[i] * need[i] - Math.round(q[i] * need[i])) > 2e-3) return p;
    return Math.min(max, need[i] * Math.ceil(p / need[i]));
  });
}
