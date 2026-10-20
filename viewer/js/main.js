import { parseAuto } from "./parsers.js";
import {
  buildStructure, modePattern, defaultSupercell, fitSupercell, isCommensurate, latticeParams, formatFreq, UNITS,
  reciprocal, speciesWeights,
} from "./physics.js";
import { CrystalScene } from "./scene.js";
import { BandPlot } from "./bandplot.js";
import { colorOf } from "./elements.js";
import { download, exportGIF, exportPNG, exportVideo, exportXYZ, copyPNG, videoSupport } from "./export.js";

const $ = (id) => document.getElementById(id);
const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode: ignore */ } },
};

const S = {
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
    await loadText(await readUpload(file), file.name);
  } catch (e) {
    const tooBig = e instanceof RangeError || /allocation|string length|memory/i.test(e.message);
    showError(tooBig ? `${file.name} (${mb(file.size)}) is too large to hold in browser memory. Write the eigenvectors for fewer q-points or a smaller cell and try again.` : e.message);
  }
}

async function loadUrl(url, label) {
  const name = label || decodeURIComponent(url.split("?")[0].split("/").pop()) || "remote";
  await showLoading(name);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadText(await res.text(), name);
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
  S.overrides = {}; S.sw = null;
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
  scene.setPattern(modePattern(d, scene.struct, S.qi, S.bi, { massScale: $("massscale").checked, basisPhase: $("basisphase").checked }));
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
  renderModes();
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
    row.title = names.map((n, k) => `${n} ${(100 * w[(S.qi * d.nb + b) * names.length + k]).toFixed(0)}%`).join(" · ");
    const bar = names.map((n, k) => `<i style="width:${100 * w[(S.qi * d.nb + b) * names.length + k]}%;background:${colors[k]}"></i>`).join("");
    row.innerHTML = `<span class="idx">${b + 1}</span><span class="bar">${bar}</span><span class="fr">${formatFreq(f, S.unit)}</span>`;
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
    row.innerHTML = `<span class="sym">${el}</span><input type="color" value="${colorOf(el)}" aria-label="${el} colour"><input type="range" min="0.5" max="1.8" step="0.05" value="1" aria-label="${el} size">`;
    const [color, size] = row.querySelectorAll("input");
    const upd = () => {
      S.overrides[el] = { color: color.value, radiusScale: +size.value };
      scene.setOverrides({ ...S.overrides });
      updateWeights();
      renderModes();
    };
    color.addEventListener("input", upd);
    size.addEventListener("input", upd);
    box.appendChild(row);
  }
  scene.setOverrides({});
}

function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

function updateWeights() {
  const d = S.data;
  if (!d) return;
  if (!$("weights").checked) { plot.setWeights(null); $("legend").innerHTML = ""; return; }
  const { names, w } = (S.sw ??= speciesWeights(d));
  const colors = names.map((n) => hexToRgb(S.overrides[n]?.color ?? colorOf(n)));
  plot.setWeights({ names, w, colors });
  $("legend").innerHTML = names.map((n, i) => `<span><i style="background:rgb(${colors[i]})"></i>${n}</span>`).join("");
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
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) loadFile(f); e.target.value = ""; });
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
  const files = [...e.dataTransfer.files];
  // if several files are dropped, prefer the one that looks like phonon data
  const f = files.find((x) => /band|phonon/i.test(x.name) && /\.(ya?ml|json)(\.gz)?$/i.test(x.name)) ?? files.find((x) => /\.(ya?ml|json)(\.gz)?$/i.test(x.name)) ?? files[0];
  if (f) loadFile(f);
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
