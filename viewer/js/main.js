import { parseAuto } from "./parsers.js";
import {
  buildStructure, modePattern, defaultSupercell, fitSupercell, isCommensurate, latticeParams, formatFreq, UNITS,
  reciprocal, speciesWeights,
} from "./physics.js";
import { CrystalScene } from "./scene.js";
import { BandPlot } from "./bandplot.js";
import { colorOf } from "./elements.js";
import { parseSymmetry, matches, modeInfo, adaptedEig, describe, looksLikeSymmetry, prettySG, emptySymmetry, addResult } from "./symmetry.js";
import { download, exportGIF, exportPNG, exportVideo, exportXYZ, copyPNG, videoSupport } from "./export.js";

const $ = (id) => document.getElementById(id);
const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode: ignore */ } },
};

const S = {
  sym: null, pendingSym: null, token: 0, symTol: 0.001,
  data: null, name: "phonons", qi: 0, bi: 0, baseReps: [2, 2, 2], reps: [2, 2, 2],
  unit: store.get("unit", "THz"), playing: true, speed: 1, amp: 0.4, phase: 0,
  overrides: {}, bgCustom: false, busy: null,
};

// ---- scene and plot -------------------------------------------------------------------------------------------------------
const scene = new CrystalScene($("gl"));
const plot = new BandPlot($("band"), $("tip"), (q, b) => selectMode(q, b, false));

new ResizeObserver(([e]) => scene.resize(e.contentRect.width, e.contentRect.height)).observe($("view3d"));
new ResizeObserver(([e]) => plot.resize(e.contentRect.width, e.contentRect.height)).observe($("band").parentElement);

// stack the 3D view over the plot when the stage is narrow
new ResizeObserver(([e]) => {
  const col = e.contentRect.width < 900;
  $("stage").classList.toggle("col", col);
  $("splitter").setAttribute("aria-orientation", col ? "horizontal" : "vertical");
  if (!col) renderModes();
}).observe($("stage"));

(() => {
  const sp = $("splitter"), stage = $("stage");
  let drag = false;
  sp.addEventListener("pointerdown", (e) => { drag = true; sp.setPointerCapture(e.pointerId); sp.classList.add("drag"); });
  sp.addEventListener("pointerup", () => { drag = false; sp.classList.remove("drag"); });
  sp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = stage.getBoundingClientRect(), col = stage.classList.contains("col");
    const f = col ? (e.clientY - r.top) / r.height : (e.clientX - r.left) / r.width;
    stage.style.setProperty("--split", `${Math.min(80, Math.max(20, f * 100)).toFixed(1)}%`);
  });
})();

// ---- theme -------------------------------------------------------------------------------------------------------------------
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  requestAnimationFrame(() => {
    if (!S.bgCustom) {
      const bg = getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim() || "#ffffff";
      scene.setBackground(bg);
      $("bg").value = bg;
      scene.setGizmo();
    }
    plot.draw();
  });
}
$("theme").addEventListener("click", () => {
  const dark = (document.documentElement.dataset.theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  const next = dark ? "light" : "dark";
  store.set("theme", next);
  S.bgCustom = false;
  applyTheme(next);
});
applyTheme(store.get("theme", null));

// ---- loading ----------------------------------------------------------------------------------------------------------------
function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.id);
  toast.id = setTimeout(() => (t.hidden = true), ms);
}

/** Persistent dialog: a failed load must not be missable, and says what to do next. */
function showError(msg, title = "Could not open the file") {
  $("loading").hidden = true;
  $("errtitle").textContent = title;
  $("errmsg").textContent = msg;
  $("errdlg").hidden = false;
  $("erropen").focus();
}
$("errclose").addEventListener("click", () => ($("errdlg").hidden = true));
$("erropen").addEventListener("click", () => { $("errdlg").hidden = true; $("file").click(); });
addEventListener("keydown", (e) => { if (e.key === "Escape") $("errdlg").hidden = true; });

function showLoading(name, msg = "") {
  $("loadname").textContent = `Opening ${name}`;
  $("loadmsg").textContent = msg;
  $("loading").hidden = false;
  return new Promise((r) => setTimeout(r, 30)); // let the indicator paint before the heavy work blocks the page
}

async function loadText(text, name) {
  try {
    const t0 = performance.now();
    setData(parseAuto(text, name), name);
    console.info(`parsed ${name} in ${(performance.now() - t0).toFixed(0)} ms`);
    return true;
  } catch (e) {
    showError(e.message);
    return false;
  } finally {
    $("loading").hidden = true;
  }
}

/** Open text that is either a band structure or a symmetry file (told apart by content). */
async function loadAny(text, name) {
  if (!looksLikeSymmetry(text.slice(0, 600))) return loadText(text, name);
  try {
    let obj;
    try { obj = JSON.parse(text); } catch (e) { throw new Error(`${name} is not valid JSON (${e.message}).`); }
    attachSymmetry(parseSymmetry(obj), name);
    return true;
  } catch (e) {
    showError(e.message, "Could not use the symmetry file");
    return false;
  } finally {
    $("loading").hidden = true;
  }
}

/** Tie symmetry data to the band structure it was made for (or hold it until that is opened). */
function attachSymmetry(sym, name) {
  if (!S.data) {
    S.pendingSym = { sym, name };
    toast(`${name} is ready. Now open the band.yaml it belongs to.`, 6000);
    renderSymmetry();
    return;
  }
  const m = matches(sym, S.data);
  if (!m.ok) throw new Error(`${name}: ${m.reason}`);
  S.sym = sym; S.symName = name; S.pendingSym = null;
  if (sym.settings.mass_scale !== undefined) $("massscale").checked = !!sym.settings.mass_scale;     // the analysis used these
  if (sym.settings.basis_phase !== undefined) $("basisphase").checked = !!sym.settings.basis_phase;
  plot.setMarks(sym.marks);
  selectMode(S.qi, S.bi);
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const mb = (n) => `${(n / 1e6).toFixed(n > 1e7 ? 0 : 1)} MB`;

/** Read an uploaded file as text, unpacking .gz (by content, not by name). */
async function readUpload(file) {
  const head = new Uint8Array(await file.slice(0, 6).arrayBuffer());
  if (head[0] === 0x1f && head[1] === 0x8b) {
    if (typeof DecompressionStream === "undefined") throw new Error(`${file.name} is gzip-compressed and this browser cannot unpack it. Unzip it (gunzip) and open the result.`);
    return new Response(file.stream().pipeThrough(new DecompressionStream("gzip"))).text();
  }
  if (head[0] === 0xfd && head[1] === 0x37 && head[2] === 0x7a && head[3] === 0x58)
    throw new Error(`${file.name} is xz-compressed, which browsers cannot unpack. Decompress it first (xz -d ${file.name}) and open the result.`);
  if (head[0] === 0x50 && head[1] === 0x4b)
    throw new Error(`${file.name} is a zip archive. Extract it and open the band.yaml inside.`);
  return file.text();
}

async function loadFile(file) {
  await showLoading(file.name, mb(file.size));
  try {
    await loadAny(await readUpload(file), file.name);
  } catch (e) {
    const tooBig = e instanceof RangeError || /allocation|string length|memory/i.test(e.message);
    showError(tooBig ? `${file.name} (${mb(file.size)}) is too large to hold in browser memory. Write the eigenvectors for fewer q-points or a smaller cell and try again.` : e.message);
  }
}

/** Open several files at once: the band structure first, then any symmetry files that belong to it. */
async function loadFiles(files) {
  if (!files.length) return;
  const isData = (x) => /\.(ya?ml|json)(\.gz)?$/i.test(x.name);
  const heads = await Promise.all(files.map(async (f) => ({ f, sym: looksLikeSymmetry(await f.slice(0, 600).text().catch(() => "")) })));
  const bands = heads.filter((h) => !h.sym).map((h) => h.f), syms = heads.filter((h) => h.sym).map((h) => h.f);
  const band = bands.find((x) => /band|phonon/i.test(x.name) && isData(x)) ?? bands.find(isData) ?? bands[0];
  if (band) await loadFile(band);
  for (const s of syms) await loadFile(s);
}

/** `a.yaml` or `a.yaml|a_symmetry.json` (a band structure and the symmetry file made for it). */
async function loadUrl(spec, label) {
  const [url, ...rest] = spec.split("|");
  const name = label || decodeURIComponent(url.split("?")[0].split("/").pop()) || "remote";
  await showLoading(name);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadAny(await res.text(), name);
    for (const extra of rest) {
      const r2 = await fetch(extra);
      if (!r2.ok) throw new Error(`HTTP ${r2.status} for ${extra}`);
      await loadAny(await r2.text(), decodeURIComponent(extra.split("/").pop()));
    }
  } catch (e) {
    showError(`Could not load ${url}: ${e.message}${location.protocol === "file:" ? " (open the viewer through a web server, not file://)" : ""}`);
  }
}

/** At the first q-point, the lowest band that is not a (near-)zero acoustic mode, so the first view shows real motion. */
function firstMovingBand(d) {
  let top = 0;
  for (let b = 0; b < d.nb; b++) top = Math.max(top, d.freq[b]);
  for (let b = 0; b < d.nb; b++) if (d.freq[b] > 0.05 * top) return b;
  return d.nb - 1;
}

function setData(data, name) {
  S.data = data; S.name = name.replace(/\.[^.]+$/, "");
  S.overrides = {}; S.sw = null; S.sym = null; S.symName = null; S.token = (S.token ?? 0) + 1;
  plot.setMarks(null);
  $("welcome").hidden = true;
  $("filename").textContent = name; $("filename").title = `${data.name || S.name} · ${name}`;
  const p = latticeParams(data.lattice);
  $("cellinfo").innerHTML = `<b>${data.formula}</b> · ${data.natom} atom${data.natom > 1 ? "s" : ""} · ${data.nq} q-points · ${data.nb} bands<br>` +
    `a ${p.a.toFixed(3)} b ${p.b.toFixed(3)} c ${p.c.toFixed(3)} Å · α ${p.alpha.toFixed(1)}° β ${p.beta.toFixed(1)}° γ ${p.gamma.toFixed(1)}°`;
  $("massscale").checked = data.eigIsDynmat;
  $("basisphase").checked = data.phaseWithBasis;
  $("qrange").max = data.nq; $("qrange").min = 1; $("qnum").max = data.nq;
  $("brange").max = data.nb; $("bnum").max = data.nb;
  S.baseReps = data.suggestedRepetitions ? [...data.suggestedRepetitions] : defaultSupercell(data);
  S.qi = 0; S.bi = firstMovingBand(data);
  buildElementControls();
  plot.setData(data);
  updateWeights();
  buildStructure_(true);
  scene.defaultView(0);
  scene.fit();
  selectMode(S.qi, S.bi, true);
  if (S.pendingSym) {                         // a symmetry file was opened before its band structure
    const p = S.pendingSym;
    try { attachSymmetry(p.sym, p.name); } catch (e) { S.pendingSym = null; showError(e.message, "Could not use the symmetry file"); }
  }
}

// ---- structure / mode -------------------------------------------------------------------------------------------------------
function currentQ() { const d = S.data; return [d.qpts[S.qi * 3], d.qpts[S.qi * 3 + 1], d.qpts[S.qi * 3 + 2]]; }

function repsFor() {
  return $("autofit").checked ? fitSupercell(currentQ(), S.baseReps) : [...S.baseReps];
}

/** (Re)build the drawn atoms when the supercell or display options changed. Returns whether it did. */
function buildStructure_(force = false) {
  if (!S.data) return false;
  const reps = repsFor();
  const key = `${reps}|${$("wrap").checked}|${$("complete").checked}`;
  if (!force && key === S.structKey) return false;
  S.structKey = key; S.reps = reps;
  ["nx", "ny", "nz"].forEach((id, i) => ($(id).value = reps[i]));
  const struct = buildStructure(S.data, { reps, wrap: $("wrap").checked, complete: $("complete").checked });
  scene.setStructure(S.data, struct, { fit: !force });
  return true;
}

function selectMode(qi, bi, fromUi = true) {
  const d = S.data;
  if (!d) return;
  S.qi = Math.max(0, Math.min(d.nq - 1, qi)); S.bi = Math.max(0, Math.min(d.nb - 1, bi));
  buildStructure_();
  const eig = S.sym && $("useadapted").checked ? adaptedEig(S.sym, S.qi, S.bi, d.natom) : null;   // symmetry-adapted, if the file has one
  scene.setPattern(modePattern(d, scene.struct, S.qi, S.bi, { massScale: $("massscale").checked, basisPhase: $("basisphase").checked, eig }));
  if (fromUi) plot.select(S.qi, S.bi, false);
  $("qrange").value = $("qnum").value = S.qi + 1;
  $("brange").value = $("bnum").value = S.bi + 1;
  refreshReadout();
}

function refreshReadout() {
  const d = S.data;
  if (!d) return;
  const f = d.freq[S.qi * d.nb + S.bi];
  const q = currentQ();
  const qtxt = `(${q.map((x) => +x.toFixed(4)).join(", ")})`;
  const tick = d.ticks.find((t) => t.index === S.qi);
  $("freq").textContent = formatFreq(f, S.unit);
  $("imag").hidden = f > -0.05;
  $("qlabel").textContent = `q = ${qtxt}${tick ? ` · ${tick.label}` : ""} · band ${S.bi + 1} of ${d.nb}`;
  const others = Object.keys(UNITS).filter((u) => u !== S.unit).map((u) => `${formatFreq(f, u)} ${UNITS[u].label}`).join(" · ");
  $("chip").innerHTML = `<b>${formatFreq(f, S.unit)} ${UNITS[S.unit].label}</b>${f < -0.05 ? " <span style='color:var(--bad)'>imaginary</span>" : ""}<br><span class="muted">${others}</span><br>q ${qtxt}${tick ? ` · ${tick.label}` : ""}`;
  $("seam").hidden = isCommensurate(q, S.reps);
  renderSymmetry();
  renderModes();
}

// ---- in-browser symmetry analysis (DistortKit running in Pyodide, in a worker) -------------------------------------------------
const engine = { worker: null, ready: false, busy: null, nextId: 1, error: null };

function startEngine() {
  if (engine.worker) return;
  engine.error = null;
  const w = engine.worker = new Worker("js/symworker.js");
  w.onmessage = onEngineMessage;
  w.onerror = (e) => failEngine(`the analysis engine could not start (${e.message || "worker error"}). It needs to download the Python runtime from cdn.jsdelivr.net; check your connection and try again.`);
  w.postMessage({ type: "init", zipUrl: new URL("py/distortkit.zip", location.href).href });
}

function failEngine(message) {
  engine.worker?.terminate();
  engine.worker = null; engine.ready = false; engine.busy = null; engine.error = message;
  renderSymmetry();
}

function stopEngine() {                        // Cancel: the worker cannot be interrupted, so drop it (the runtime reloads next time)
  engine.worker?.terminate();
  engine.worker = null; engine.ready = false; engine.busy = null;
  renderSymmetry();
}

/** The request for one q-point: the cell, the file's eigenvectors there, and the conventions in force. */
function requestFor(qi) {
  const d = S.data, nb = d.nb, na = d.natom, eig = [];
  for (let b = 0; b < nb; b++)
    for (let a = 0; a < na; a++)
      for (let c = 0; c < 3; c++) { const o = ((qi * nb + b) * na + a) * 6 + 2 * c; eig.push(d.eig[o], d.eig[o + 1]); }
  return {
    lattice: d.lattice, frac: d.frac, species: d.species, masses: Array.from(d.masses), qi, eig,
    q: [d.qpts[3 * qi], d.qpts[3 * qi + 1], d.qpts[3 * qi + 2]], freq: Array.from(d.freq.slice(qi * nb, qi * nb + nb)),
    settings: { max_cells: 16, max_dim: 6, labels: true, tol: S.symTol, mass_scale: $("massscale").checked, basis_phase: $("basisphase").checked },
  };
}

function analyseCurrent() {
  if (!S.data || engine.busy) return;
  if (!S.sym) S.sym = emptySymmetry(S.data, { mass_scale: $("massscale").checked, basis_phase: $("basisphase").checked, max_cells: 16, max_dim: 6 });
  startEngine();
  const id = engine.nextId++;
  engine.busy = { id, qi: S.qi, token: S.token, stage: "Starting", frac: 0 };
  engine.worker.postMessage({ type: "analyse", id, request: requestFor(S.qi) });
  renderSymmetry();
}

function onEngineMessage(e) {
  const m = e.data, busy = engine.busy;
  if (m.type === "ready") engine.ready = true;
  else if (m.type === "progress") {
    if (busy) { busy.stage = m.stage; busy.frac = m.frac; }
    else if (!engine.ready) engine.loading = m;
    const bar = $("symbar");
    if (bar) { bar.style.width = `${Math.round(100 * m.frac)}%`; $("symstage").textContent = m.stage; }
  } else if (m.type === "result" && busy && m.id === busy.id) {
    engine.busy = null;
    if (busy.token === S.token && S.sym) {      // ignore the answer if another band structure was opened meanwhile
      addResult(S.sym, busy.qi, m.result);
      plot.setMarks(S.sym.marks);
      selectMode(S.qi, S.bi);                    // applies the adapted eigenvector and redraws the card and mode list
    } else renderSymmetry();
  } else if (m.type === "error") failEngine(`the analysis failed: ${m.message}`);
}

/** How closely atoms must match to count as symmetric, with a retry: real structures are often only approximately symmetric. */
function renderToleranceRetry(parent, add) {
  const row = add("div", "row", undefined, parent);
  add("span", "small", "Tolerance (Å)", row);
  const inp = add("input", "num", undefined, row);
  inp.type = "number"; inp.step = "0.001"; inp.min = "0.0001"; inp.max = "0.2"; inp.value = S.symTol; inp.style.width = "74px";
  inp.addEventListener("change", () => { const v = parseFloat(inp.value); if (v > 0) S.symTol = v; });
  add("button", "btn small primary", "Try again", row).addEventListener("click", () => { const v = parseFloat(inp.value); if (v > 0) S.symTol = v; analyseCurrent(); });
}

/** Card for a mode whose symmetry has not been worked out: the button that does it, or the progress of doing it. */
function renderAnalyse(box, add) {
  const busy = engine.busy;
  if (engine.error) add("p", "note warn", engine.error.charAt(0).toUpperCase() + engine.error.slice(1));
  if (busy) {
    add("p", "", busy.qi === S.qi ? "Analysing this q-point…" : `Analysing q-point ${busy.qi + 1}…`);
    const prog = add("div", "symprog");
    const track = add("div", "track", undefined, prog);
    const bar = add("div", "", undefined, track);
    bar.id = "symbar"; bar.style.width = `${Math.round(100 * busy.frac)}%`;
    const stage = add("div", "hint", busy.stage); stage.id = "symstage";
    add("button", "btn small", "Cancel").addEventListener("click", stopEngine);
    add("p", "hint", engine.ready ? "A q-point takes from seconds to a few minutes, depending on the size of its symmetry star. The page stays usable." : "The first time, this downloads the Python runtime (about 20 MB, kept by your browser afterwards).");
    return;
  }
  add("p", "", "The symmetry of a mode (its irrep and the structure it leads to) is worked out on request, here in your browser.");
  const go = add("button", "btn primary", "Analyse this q-point");
  go.addEventListener("click", analyseCurrent);
  const adv = add("details", "");
  add("summary", "", "Symmetry tolerance", adv);
  add("p", "hint", "How closely atoms must match to count as symmetric. The default suits exact structures; for a relaxed (only approximately symmetric) one, try 0.01 Å or more.", adv);
  renderToleranceRetry(adv, add);
  add("p", "hint", "Nothing is uploaded: the analysis runs locally. The first time it downloads the Python runtime (about 20 MB, kept by your browser afterwards). Large q-point stars can take a few minutes; some are too large and are skipped.");
}

/** The Symmetry card: irrep, the symmetry the mode leads to, and every symmetry its irrep allows. */
function renderSymmetry() {
  const box = $("symcard");
  box.replaceChildren();
  $("adaptrow").hidden = !S.sym?.qpoints.some((q) => q.bands);
  const add = (tag, cls, text, parent = box) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    parent.appendChild(e);
    return e;
  };
  if (!S.data) {
    add("p", "hint", S.pendingSym ? `${S.pendingSym.name} is loaded; now open the band.yaml it belongs to.` : "Open a band structure to see the symmetry of its modes.");
    return;
  }
  const info = S.sym ? modeInfo(S.sym, S.qi, S.bi) : null;
  if (!S.sym || info?.status === "pending") { renderAnalyse(box, add); return; }
  if (S.sym.parent || S.symName) add("p", "hint", [S.sym.parent && `Parent space group ${prettySG(S.sym.parent)}`, S.symName].filter(Boolean).join(" · "));
  if (!info) { add("p", "hint", "No symmetry data for this mode."); return; }
  if (info.status === "skipped") {
    add("p", "note", `Not analysed: ${info.reason}.`);
    if (/toleran|approximately/i.test(info.reason)) renderToleranceRetry(box, add);
    else add("p", "hint", "Analysed q-points are marked ▲ under the band plot.");
    return;
  }
  const head = add("div", "symhead");
  add("span", "irrep", info.label, head);
  add("span", "dim", `dimension ${info.irrep.dim}${info.irrep.kind !== "real" ? `, ${info.irrep.kind}` : ""} · on ${info.irrep.species.join(", ")}`, head);
  add("p", "hint", `Star of k = (${info.star.k.join(", ")}): ${info.star.star_size} arm${info.star.star_size > 1 ? "s" : ""}, ${info.star.n_cells}-cell supercell. The dimension counts all arms.`);
  if (!info.clean) add("p", "note warn", `Only ${(100 * info.band.weight).toFixed(0)}% of this mode lies in this irrep: another irrep has the same frequency. Tick "Use symmetry-adapted modes" to separate them.`);
  const lines = describe(info);
  const child = add("div", "child");
  const first = lines[0];
  const sub = info.sub;
  if (sub?.space_group) {
    child.append("Freezing it in gives ");
    add("b", "", prettySG(sub.space_group), child);
    child.append(sub.opd.length <= 30 ? `, direction ${sub.opd}${sub.n_conjugates > 1 ? ` (${sub.n_conjugates} equivalent domains)` : ""}.` : ".");
  } else child.textContent = first;
  const rest = lines.slice(1);
  if (rest.length) { const ul = add("ul", ""); rest.forEach((t) => add("li", "", t, ul)); }
  if (info.irrep.subgroups.length) {
    const det = add("details", "");
    add("summary", "", `All ${info.irrep.subgroups.length} child symmetries of ${info.label}`, det);
    const table = add("div", "subtable", undefined, det);
    for (const s of info.irrep.subgroups) {
      const on = sub && s.index === sub.index;
      add("span", `opd${on ? " on" : ""}`, s.opd.length <= 30 ? s.opd : `(${s.fix_dim}-parameter subspace)`, table).title = s.opd;
      add("span", on ? "on" : "", `${s.space_group ? prettySG(s.space_group) : `unidentified, ${s.point_group}`}${s.n_conjugates > 1 ? ` ×${s.n_conjugates}` : ""}`, table);
    }
  } else if (info.irrep.note) add("p", "hint", info.irrep.note);
}

/** The clickable list of all modes at the selected q-point, with the species composition of each. */
function renderModes() {
  const d = S.data, box = $("modelist");
  if (!d || box.offsetParent === null) return; // hidden in the stacked layout
  const { names, w } = (S.sw ??= speciesWeights(d));
  const colors = names.map((n) => S.overrides[n]?.color ?? colorOf(n));
  const frag = document.createDocumentFragment();
  for (let b = 0; b < d.nb; b++) {
    const f = d.freq[S.qi * d.nb + b];
    const row = document.createElement("button");
    row.className = `mrow${b === S.bi ? " sel" : ""}${f < -0.05 ? " imag" : ""}`;
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", b === S.bi);
    row.title = names.map((n, k) => `${esc(n)} ${(100 * w[(S.qi * d.nb + b) * names.length + k]).toFixed(0)}%`).join(" · ");
    const bar = names.map((n, k) => `<i style="width:${100 * w[(S.qi * d.nb + b) * names.length + k]}%;background:${colors[k]}"></i>`).join("");
    const si = S.sym ? modeInfo(S.sym, S.qi, b) : null;
    if (si?.status === "ok") {
      row.classList.add("hasirr");
      row.innerHTML = `<span class="idx">${b + 1}</span><span class="irr">${esc(si.label)}</span><span class="bar">${bar}</span><span class="fr">${formatFreq(f, S.unit)}</span>`;
    } else row.innerHTML = `<span class="idx">${b + 1}</span><span class="bar">${bar}</span><span class="fr">${formatFreq(f, S.unit)}</span>`;
    row.addEventListener("click", () => selectMode(S.qi, b));
    frag.appendChild(row);
  }
  box.replaceChildren(frag);
  box.querySelector(".sel")?.scrollIntoView({ block: "nearest" });
}

// ---- controls: mode ----------------------------------------------------------------------------------------------------------
$("qrange").addEventListener("input", (e) => selectMode(+e.target.value - 1, S.bi));
$("qnum").addEventListener("change", (e) => selectMode(+e.target.value - 1, S.bi));
$("brange").addEventListener("input", (e) => selectMode(S.qi, +e.target.value - 1));
$("bnum").addEventListener("change", (e) => selectMode(S.qi, +e.target.value - 1));
$("prevq").addEventListener("click", () => selectMode(S.qi - 1, S.bi));
$("nextq").addEventListener("click", () => selectMode(S.qi + 1, S.bi));
$("massscale").addEventListener("change", () => selectMode(S.qi, S.bi));
$("basisphase").addEventListener("change", () => selectMode(S.qi, S.bi));
$("unit").value = S.unit;
$("unit").addEventListener("change", (e) => { S.unit = e.target.value; store.set("unit", S.unit); plot.setUnit(S.unit); refreshReadout(); });
plot.setUnit(S.unit);
plot.onZoom = (zoomed) => { $("zoomreset").disabled = !zoomed; };
$("zoomreset").addEventListener("click", () => plot.resetZoom());

// ---- controls: supercell ------------------------------------------------------------------------------------------------------
for (const id of ["nx", "ny", "nz"]) {
  $(id).addEventListener("change", () => {
    S.baseReps = ["nx", "ny", "nz"].map((k) => Math.max(1, Math.min(12, parseInt($(k).value, 10) || 1)));
    buildStructure_(true); scene.fit(); selectMode(S.qi, S.bi);
  });
}
for (const id of ["autofit", "wrap", "complete"]) $(id).addEventListener("change", () => { buildStructure_(true); scene.fit(); selectMode(S.qi, S.bi); });

// ---- controls: animation ---------------------------------------------------------------------------------------------------
function setPlaying(p) { S.playing = p; $("play").textContent = p ? "Pause" : "Play"; $("play2").textContent = p ? "⏸" : "▶"; }
$("play").addEventListener("click", () => setPlaying(!S.playing));
$("play2").addEventListener("click", () => setPlaying(!S.playing));
$("amp").addEventListener("input", (e) => { S.amp = +e.target.value; $("ampout").textContent = `${S.amp.toFixed(2)} Å`; scene.setAmplitude(S.amp); });
$("speed").addEventListener("input", (e) => { S.speed = +e.target.value; $("speedout").textContent = `${S.speed.toFixed(1)}×`; });
$("arrows").addEventListener("change", (e) => { $("arrowopts").hidden = !e.target.checked; scene.setStyle({ arrows: e.target.checked }); });
$("arrowmode").addEventListener("change", (e) => scene.setStyle({ arrowMode: e.target.value }));
$("arrowlen").addEventListener("input", (e) => { $("arrowlenout").textContent = `${(+e.target.value).toFixed(1)} Å`; scene.setStyle({ arrowLength: +e.target.value }); });

// ---- controls: view -----------------------------------------------------------------------------------------------------------
function look(spec) {
  if (!S.data) return;
  const again = scene.lastLook?.spec === spec && !scene.anim;
  scene.lookAlong(spec, { flip: again ? !scene.lastLook.flip : false });
  document.querySelectorAll("[data-look]").forEach((b) => b.classList.toggle("on", b.dataset.look === spec));
}
document.querySelectorAll("[data-look]").forEach((b) => b.addEventListener("click", () => look(b.dataset.look)));
$("resetview").addEventListener("click", () => { scene.defaultView(450); scene.fit(); document.querySelectorAll("[data-look]").forEach((b) => b.classList.remove("on")); });
$("fit").addEventListener("click", () => scene.fit());
$("uvwform").addEventListener("submit", (e) => {
  e.preventDefault();
  if (!S.data) return;
  const txt = $("uvw").value.trim();
  const nums = txt.replace(/[\[\]\(\)<>]/g, "").split(/[\s,]+/).filter(Boolean).map(Number);
  if (nums.length !== 3 || nums.some(Number.isNaN) || nums.every((x) => x === 0)) { toast("Enter three numbers, e.g. [1 1 0] for a direction or (1 1 1) for a plane normal"); return; }
  const lat = S.data.lattice, mix = (basis) => [0, 1, 2].map((k) => nums[0] * basis[0][k] + nums[1] * basis[1][k] + nums[2] * basis[2][k]);
  const plane = txt.startsWith("(");
  scene.lookAlong(mix(plane ? reciprocal(lat) : lat));
  document.querySelectorAll("[data-look]").forEach((b) => b.classList.remove("on"));
});
$("proj").addEventListener("change", (e) => scene.setStyle({ perspective: e.target.value === "persp" }));
$("gizmo").addEventListener("change", (e) => scene.setStyle({ gizmo: e.target.value }));
$("cell").addEventListener("change", (e) => scene.setStyle({ cell: e.target.value }));
$("bg").addEventListener("input", (e) => { S.bgCustom = true; scene.setBackground(e.target.value); });
$("bg").addEventListener("change", () => scene.setGizmo());

// ---- controls: style -----------------------------------------------------------------------------------------------------------
$("look").addEventListener("change", (e) => scene.setStyle({ look: e.target.value }));
$("atomscale").addEventListener("input", (e) => scene.setStyle({ atomScale: +e.target.value }));
$("bonds").addEventListener("change", (e) => scene.setStyle({ bonds: e.target.checked }));
$("bondr").addEventListener("input", (e) => scene.setStyle({ bondRadius: +e.target.value }));
$("bondtol").addEventListener("input", (e) => { $("bondtolout").textContent = (+e.target.value).toFixed(2); scene.setStyle({ bondTol: +e.target.value }); });
$("splitbonds").addEventListener("change", (e) => scene.setStyle({ splitBonds: e.target.checked }));

function buildElementControls() {
  const box = $("elements");
  box.innerHTML = "";
  for (const el of [...new Set(S.data.elements)]) {
    const row = document.createElement("div");
    row.className = "el";
    row.innerHTML = `<span class="sym">${esc(el)}</span><input type="color" value="${colorOf(el)}" aria-label="${esc(el)} colour"><input type="range" min="0.5" max="1.8" step="0.05" value="1" aria-label="${esc(el)} size">`;
    const [color, size] = row.querySelectorAll("input");
    const upd = () => {
      S.overrides[el] = { color: color.value, radiusScale: +size.value };
      scene.setOverrides({ ...S.overrides });
      updateWeights();
      renderModes();
      renderAtomLegend();
    };
    color.addEventListener("input", upd);
    size.addEventListener("input", upd);
    box.appendChild(row);
  }
  scene.setOverrides({});
  renderAtomLegend();
}

/** Colour key for the atoms, shown beside the frequency readout at the top of the 3D view. */
function renderAtomLegend() {
  const box = $("atomlegend");
  if (!S.data) { box.replaceChildren(); return; }
  const frag = document.createDocumentFragment();
  for (const el of [...new Set(S.data.elements)]) {
    const row = document.createElement("span"), dot = document.createElement("i");
    row.className = "atom";
    dot.style.background = scene.atomColor(el);
    row.append(dot, el);
    frag.appendChild(row);
  }
  box.replaceChildren(frag);
}

function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

function updateWeights() {
  const d = S.data;
  if (!d) return;
  if (!$("weights").checked) { plot.setWeights(null); $("legend").innerHTML = ""; return; }
  const { names, w } = (S.sw ??= speciesWeights(d));
  const colors = names.map((n) => hexToRgb(S.overrides[n]?.color ?? colorOf(n)));
  plot.setWeights({ names, w, colors });
  $("legend").innerHTML = names.map((n, i) => `<span><i style="background:rgb(${colors[i]})"></i>${esc(n)}</span>`).join("");
}
$("weights").addEventListener("change", updateWeights);

// ---- export ----------------------------------------------------------------------------------------------------------------------
const stem = () => `${S.name.replace(/[^\w.-]+/g, "_")}_q${S.qi + 1}_b${S.bi + 1}`;

function startJob() {
  const ctl = new AbortController();
  S.busy = ctl;
  $("progress").hidden = false; $("bar").style.width = "0";
  document.querySelectorAll("#gif,#video,#png,#xyz").forEach((b) => (b.disabled = true));
  $("cancel").onclick = () => ctl.abort();
  const wasPlaying = S.playing, phase = S.phase;
  setPlaying(false);
  return {
    signal: ctl.signal,
    onProgress: (f) => ($("bar").style.width = `${Math.round(f * 100)}%`),
    done() {
      S.busy = null;
      $("progress").hidden = true;
      document.querySelectorAll("#gif,#video,#png,#xyz").forEach((b) => (b.disabled = false));
      S.phase = phase; scene.setPhase(phase); setPlaying(wasPlaying);
    },
  };
}

function animSize() {
  const w = +$("animw").value, ar = $("animar").value;
  const aspect = ar === "view" ? scene.size.w / scene.size.h : +ar;
  return { width: w, height: Math.max(2, Math.round(w / aspect / 2) * 2) };
}

$("png").addEventListener("click", async () => {
  if (!S.data) return;
  const sc = +$("pngscale").value;
  try {
    const blob = await exportPNG(scene, { width: Math.round(scene.size.w * sc), height: Math.round(scene.size.h * sc), transparent: $("pngtransp").checked, gizmo: true });
    download(blob, `${stem()}.png`);
  } catch (e) { toast(`PNG export failed: ${e.message}`); }
});
$("pngcopy").addEventListener("click", async () => {
  if (!S.data) return;
  try {
    const sc = +$("pngscale").value;
    await copyPNG(await exportPNG(scene, { width: Math.round(scene.size.w * sc), height: Math.round(scene.size.h * sc), transparent: $("pngtransp").checked, gizmo: true }));
    toast("Copied image to clipboard");
  } catch (e) { toast(`Could not copy: ${e.message}`); }
});

$("gif").addEventListener("click", async () => {
  if (!S.data || S.busy) return;
  const job = startJob();
  try {
    const blob = await exportGIF(scene, (p) => scene.setPhase(p), {
      ...animSize(), frames: +$("gframes").value, fps: +$("gfps").value, gizmo: $("animgizmo").checked,
      background: $("bg").value, onProgress: job.onProgress, signal: job.signal,
    });
    download(blob, `${stem()}.gif`);
    toast(`GIF saved (${(blob.size / 1e6).toFixed(1)} MB)`);
  } catch (e) { if (e.name !== "AbortError") toast(`GIF export failed: ${e.message}`, 6000); }
  job.done();
});

$("video").addEventListener("click", async () => {
  if (!S.data || S.busy) return;
  if (!videoSupport()) { toast("This browser cannot record video; use the GIF export."); return; }
  const job = startJob();
  try {
    const { blob, ext } = await exportVideo(scene, (p) => scene.setPhase(p), {
      ...animSize(), cycles: +$("vcycles").value, periodSec: 2 / S.speed, gizmo: $("animgizmo").checked,
      background: $("bg").value, onProgress: job.onProgress, signal: job.signal,
    });
    download(blob, `${stem()}.${ext}`);
  } catch (e) { if (e.name !== "AbortError") toast(`Video export failed: ${e.message}`, 6000); }
  job.done();
});

$("xyz").addEventListener("click", () => {
  if (!S.data) return;
  const phase = S.phase;
  const blob = exportXYZ(scene, (p) => scene.setPhase(p), { frames: +$("gframes").value, lattice: S.data.lattice });
  scene.setPhase(phase);
  download(blob, `${stem()}.xyz`);
});

$("plotpng").addEventListener("click", async () => {
  if (!S.data) return;
  plot.toCanvas(2000, 1100).toBlob((b) => download(b, `${S.name.replace(/[^\w.-]+/g, "_")}_bands.png`));
});

// ---- file handling -----------------------------------------------------------------------------------------------------------------
$("open").addEventListener("click", () => $("file").click());
$("welcomeopen").addEventListener("click", () => $("file").click());
document.querySelectorAll("[data-demo]").forEach((b) => b.addEventListener("click", () => loadUrl(b.dataset.demo, b.dataset.label)));
$("file").addEventListener("change", (e) => { const fs = [...e.target.files]; e.target.value = ""; loadFiles(fs); });
$("opensym").addEventListener("click", () => $("symfile").click());
$("symfile").addEventListener("change", (e) => { const fs = [...e.target.files]; e.target.value = ""; loadFiles(fs); });
$("useadapted").addEventListener("change", () => selectMode(S.qi, S.bi));
$("demo").addEventListener("change", (e) => {
  const o = e.target.selectedOptions[0];
  if (e.target.value) loadUrl(e.target.value, o.text);
  e.target.value = "";
});

// drag and drop anywhere on the page; the welcome card lights up when it is the target
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes("Files");
addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth++; $("drop").hidden = false; $("welcome").classList.add("over"); });
addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").hidden = true; $("welcome").classList.remove("over"); } });
addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; $("drop").hidden = true; $("welcome").classList.remove("over");
  loadFiles([...e.dataTransfer.files]);
});

// ---- keyboard ----------------------------------------------------------------------------------------------------------------------
addEventListener("keydown", (e) => {
  const t = e.target;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || t.id === "band" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (!S.data) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!S.playing); }
  else if (e.key === "ArrowRight") { e.preventDefault(); selectMode(S.qi + (e.shiftKey ? 5 : 1), S.bi); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); selectMode(S.qi - (e.shiftKey ? 5 : 1), S.bi); }
  else if (e.key === "ArrowUp") { e.preventDefault(); selectMode(S.qi, S.bi + 1); }
  else if (e.key === "ArrowDown") { e.preventDefault(); selectMode(S.qi, S.bi - 1); }
  else if ("123456".includes(e.key) && e.key) look(["a", "b", "c", "a*", "b*", "c*"][+e.key - 1]);
  else if (e.key === "0") $("resetview").click();
});

// ---- main loop ---------------------------------------------------------------------------------------------------------------------------
let last = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (S.playing && S.data) { S.phase = (S.phase + dt * Math.PI * S.speed) % (2 * Math.PI); scene.setPhase(S.phase); }
  scene.stepAnimation(now) && (scene.dirty = true);
  if (scene.dirty) scene.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// start on the welcome screen; ?url=<file> and ?demo=<name> are shortcuts for links and development
const params = new URLSearchParams(location.search);
if (params.get("url")) loadUrl(params.get("url"));
else if (params.get("demo")) loadUrl(`examples/${params.get("demo")}.yaml`, params.get("demo"));

Object.assign(window, { viewer: { S, scene, plot, selectMode } }); // handy for debugging in the console
