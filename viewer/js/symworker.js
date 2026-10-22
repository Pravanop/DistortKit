// Runs DistortKit's phonon symmetry analysis in the browser, in a background thread (Pyodide = CPython compiled to WebAssembly).
//
// Messages in:   {type: "init", zipUrl}                  load the Python runtime, numpy and DistortKit (once)
//                {type: "analyse", id, request}          analyse one q-point (see distortkit/web.py for the request)
// Messages out:  {type: "progress", id?, stage, frac}    while loading / working
//                {type: "ready"}                         the runtime is loaded
//                {type: "result", id, result}           the analysis of one q-point
//                {type: "error", id?, message}
//
// The Pyodide runtime (~10 MB) and numpy (~7 MB) are fetched from the jsDelivr CDN the first time and cached by the browser. The
// band structure never leaves the page: it is handed to the worker in memory and analysed locally.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null, web = null, initPromise = null;
const post = (m) => postMessage(m);

async function init(zipUrl) {
  post({ type: "progress", stage: "Downloading the Python runtime (first time only)", frac: 0.02 });
  py = await loadPyodide({ indexURL: PYODIDE });
  post({ type: "progress", stage: "Loading numpy", frac: 0.35 });
  await py.loadPackage("numpy");
  post({ type: "progress", stage: "Loading DistortKit", frac: 0.7 });
  const res = await fetch(zipUrl);
  if (!res.ok) throw new Error(`could not fetch ${zipUrl} (HTTP ${res.status})`);
  py.unpackArchive(await res.arrayBuffer(), "zip");
  await py.runPythonAsync("import distortkit.web");
  web = py.pyimport("distortkit.web");
  post({ type: "progress", stage: "Ready", frac: 1 });
  post({ type: "ready" });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      initPromise ??= init(m.zipUrl);
      await initPromise;
    } else if (m.type === "analyse") {
      await initPromise;
      const progress = (stage, frac) => post({ type: "progress", id: m.id, stage, frac });
      const out = web.analyze(JSON.stringify(m.request), progress);
      post({ type: "result", id: m.id, result: JSON.parse(out) });
    }
  } catch (err) {
    post({ type: "error", id: m.id, message: String(err && err.message ? err.message : err) });
  }
};
