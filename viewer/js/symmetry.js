// Symmetry data for the modes of a band structure: the JSON written by `distortkit phonons band.yaml`.
//
// For every analysed q-point and band it holds the irrep of the parent space group, the child space group the mode's direction
// fits, and for every irrep all the child symmetries it can lead to. No rendering here, so it can be unit-tested under node.

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const SUP = { "+": "⁺", "-": "⁻" };
export const FORMAT = "distortkit-phonon-symmetry";

/** 'X5+' -> 'X₅⁺', 'GM4-' -> 'Γ₄⁻'; ambiguous labels 'A/B' are done part by part; anything else is returned unchanged. */
export function prettyIrrep(label) {
  if (!label) return label;
  return String(label).split("/").map((part) => {
    const m = /^([A-Za-z]{1,2})(\d+)([+-]?)$/.exec(part.trim());   // k-point labels are 1-2 letters (X, GM, DT)
    if (!m) return part;
    const base = m[1] === "GM" ? "Γ" : m[1];
    return base + [...m[2]].map((d) => SUB[+d]).join("") + (SUP[m[3]] ?? "");
  }).join("/");
}

/** 'P4_2/nmc (#137)' -> 'P4₂/nmc (#137)' (only the subscript screw indices are converted). */
export function prettySG(text) {
  return text ? String(text).replace(/_(\d)/g, (m, d) => SUB[+d]) : text;
}

export function looksLikeSymmetry(head) {
  return head.includes(`"format"`) && head.includes(FORMAT);
}

export function parseSymmetry(obj) {
  if (!obj || obj.format !== FORMAT) throw new Error(`this JSON is not a DistortKit phonon-symmetry file (expected format "${FORMAT}")`);
  if (obj.version !== 1) throw new Error(`unsupported symmetry file version ${obj.version} (this viewer reads version 1)`);
  if (!Array.isArray(obj.qpoints) || !Array.isArray(obj.stars)) throw new Error("the symmetry file has no q-points or stars");
  const stars = new Map(obj.stars.map((s) => [s.k.join(","), s]));
  const marks = obj.qpoints.filter((q) => q.bands).map((q) => q.index);
  return { raw: obj, parent: obj.parent?.space_group ?? null, settings: obj.settings ?? {}, stars, qpoints: obj.qpoints, marks };
}

/** Does this symmetry file belong to the band structure `data`? (same number of q-points, and the same q-points) */
export function matches(sym, data) {
  const n = sym.qpoints.length;
  if (n !== data.nq) return { ok: false, reason: `it covers ${n} q-points but the band structure has ${data.nq}. It was made for a different band.yaml.` };
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 25))) {
    const q = sym.qpoints[i].q;
    for (let c = 0; c < 3; c++)
      if (Math.abs(q[c] - data.qpts[3 * i + c]) > 1e-4)
        return { ok: false, reason: `q-point ${i + 1} differs (${q.map((x) => +x.toFixed(4)).join(", ")} in the symmetry file). It was made for a different band.yaml.` };
  }
  return { ok: true };
}

/** Symmetry data that fills in as q-points are analysed in the browser (nothing known yet). */
export function emptySymmetry(data, settings = {}) {
  const qpoints = Array.from({ length: data.nq }, (_, i) => ({
    index: i, q: [data.qpts[3 * i], data.qpts[3 * i + 1], data.qpts[3 * i + 2]], pending: true,
  }));
  return { raw: null, live: true, parent: null, settings, stars: new Map(), qpoints, marks: [] };
}

const qkey = (q) => q.map((x) => x.toFixed(5)).join(",");

/** Store the analysis of q-point `qi` (the result of distortkit.web.analyze); every q-point with the same q gets it. */
export function addResult(sym, qi, result) {
  if (result.star_info) {
    sym.stars.set(result.star_info.k.join(","), result.star_info);
    sym.parent ??= result.star_info.parent;
  }
  const key = qkey(sym.qpoints[qi].q);
  for (const qp of sym.qpoints) {
    if (qkey(qp.q) !== key) continue;
    delete qp.pending;
    if (result.skipped) { qp.skipped = result.skipped; delete qp.star; delete qp.bands; }
    else { qp.star = result.star; qp.bands = result.bands; delete qp.skipped; }
  }
  sym.marks = sym.qpoints.filter((q) => q.bands).map((q) => q.index);
}

/**
 * What is known about band `bi` at q-point `qi`:
 *   {status: "pending"}        not analysed yet (the analysis can be asked for)
 *   null                       no symmetry data loaded for this q-point
 *   {status: "skipped", reason}
 *   {status: "ok", band, irrep, star, snap, sub, ...}
 */
export function modeInfo(sym, qi, bi) {
  const q = sym?.qpoints[qi];
  if (!q) return null;
  if (q.pending) return { status: "pending", q };
  if (q.skipped) return { status: "skipped", reason: q.skipped, q };
  const band = q.bands?.[bi];
  if (!band) return null;
  const star = sym.stars.get(q.star.join(","));
  const irrep = star.irreps.find((i) => i.index === band.irrep);
  const snap = [...band.snapshots].sort((a, b) => b.share - a.share)[0] ?? null;
  return {
    status: "ok", q, band, star, irrep, snap, sub: snap?.subgroup ?? null,
    enumerated: !irrep.note,                       // false when the irrep was too large for its subgroups to be listed
    clean: band.weight > 0.98 && band.outside < 1e-6,
    imaginary: band.freq < -0.05,
    label: irrep.label ? prettyIrrep(irrep.label) + (irrep.label_certain ? "" : "?") : `irrep ${irrep.index}`,
  };
}

/** The symmetry-adapted eigenvector of a band as a Float32Array(natom*6) of (re, im) pairs, or null if the file's own is fine. */
export function adaptedEig(sym, qi, bi, natom) {
  const band = sym?.qpoints[qi]?.bands?.[bi];
  if (!band?.adapted || !band.eigenvector) return null;
  const out = new Float32Array(natom * 6);
  band.eigenvector.forEach((row, a) => row.forEach(([re, im], c) => { out[a * 6 + 2 * c] = re; out[a * 6 + 2 * c + 1] = im; }));
  return out;
}

/** Plain-language lines describing what freezing the mode in would do. */
export function describe(info) {
  if (!info || info.status === "pending") return [];
  if (info.status === "skipped") return [`Not analysed: ${info.reason}.`];
  if (info.band.rigid) return ["This is a rigid translation of the whole crystal (an acoustic mode at Γ). It moves the crystal without changing the structure, so its symmetry stays the same."];
  const lines = [];
  const { sub } = info;
  if (!info.enumerated) lines.push("The child symmetries of this irrep were not listed (the irrep is too large to enumerate quickly).");
  else if (!sub) lines.push("This mode's direction fits none of the irrep's special directions: it is a generic direction, so the child symmetry is the lowest the irrep allows.");
  else if (!sub.space_group) lines.push(`Freezing it in gives point group ${sub.point_group}; DistortKit could not name that space group.`);
  else lines.push(`Freezing it in gives ${prettySG(sub.space_group)}.`);
  if (sub && sub.fix_dim >= 2 && sub.opd.length > 30)
    lines.push(`Its direction is only fixed to a ${sub.fix_dim}-parameter subspace; special directions inside it have higher symmetry.`);
  lines.push(info.imaginary
    ? "Its frequency is imaginary, so it can drive a transition to this symmetry."
    : "It is stable, so it does not condense on its own; it can appear as a secondary distortion once another mode has.");
  if (info.band.degenerate_with.length)
    lines.push(`It belongs to a ${info.band.degenerate_with.length + 1}-fold degenerate level${info.band.adapted ? ", shown in a symmetry-adapted basis" : ""}.`);
  return lines;
}
