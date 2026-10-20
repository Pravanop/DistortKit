// Image, GIF and video export. All of them render through CrystalScene.beginCapture/captureFrame, so the output is the view you
// are looking at (same orientation, same style) at the size you ask for, independent of the window size.

import { GIFEncoder, quantize, applyPalette } from "../vendor/gifenc.js";

export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

const toBlob = (canvas, type = "image/png", q) => new Promise((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error("could not encode image"))), type, q));
// yield to the event loop so the progress bar can repaint; a timer (not requestAnimationFrame) so exports also finish in a background tab
const nextFrame = () => new Promise((r) => setTimeout(r, 0));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function exportPNG(scene, { width, height, transparent, gizmo }) {
  scene.beginCapture(width, height, { transparent, gizmo });
  try {
    const canvas = scene.captureFrame();
    return await toBlob(canvas);
  } finally { scene.endCapture(); }
}

export async function copyPNG(blob) {
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

/**
 * One full oscillation as an endless GIF. Colours are quantised to a single 256-colour palette built from samples of every
 * frame, which avoids the colour shimmer of per-frame palettes.
 * `setPhase(phi)` must put the scene in the state for phase phi (radians).
 */
export async function exportGIF(scene, setPhase, { width, height, frames = 36, fps = 24, background = "#ffffff", gizmo = true, onProgress = () => {}, signal }) {
  const prevBg = scene.bg;
  scene.bg = background; // GIF has no usable alpha: render on a solid background
  scene.beginCapture(width, height, { transparent: false, gizmo });
  scene.applyBackground();
  const canvas = Object.assign(document.createElement("canvas"), { width, height });
  try {
    // pass 1: sample pixels from all frames for the palette
    const sample = [];
    const stride = Math.max(1, Math.floor((width * height * frames) / 120000)); // ~120k samples in total keeps quantize() fast
    for (let f = 0; f < frames; f++) {
      if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
      setPhase((2 * Math.PI * f) / frames);
      scene.captureFrame(canvas);
      const px = canvas.getContext("2d").getImageData(0, 0, width, height).data;
      for (let i = 0; i < width * height; i += stride) sample.push(px[4 * i], px[4 * i + 1], px[4 * i + 2], 255);
      onProgress((f + 1) / (2 * frames));
      if (f % 4 === 3) await nextFrame();
    }
    const palette = quantize(new Uint8Array(sample), 256, { format: "rgb565" });
    // pass 2: encode
    const gif = GIFEncoder();
    const delay = Math.max(20, Math.round(1000 / fps / 10) * 10);
    for (let f = 0; f < frames; f++) {
      if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
      setPhase((2 * Math.PI * f) / frames);
      scene.captureFrame(canvas);
      const px = canvas.getContext("2d").getImageData(0, 0, width, height).data;
      gif.writeFrame(applyPalette(px, palette, "rgb565"), width, height, { palette: f === 0 ? palette : undefined, delay, repeat: 0 });
      onProgress(0.5 + (f + 1) / (2 * frames));
      if (f % 4 === 3) await nextFrame();
    }
    gif.finish();
    return new Blob([gif.bytes()], { type: "image/gif" });
  } finally {
    scene.bg = prevBg;
    scene.endCapture();
  }
}

export function videoSupport() {
  if (typeof MediaRecorder === "undefined") return null;
  const tries = [
    ["video/mp4;codecs=avc1.42E01E", "mp4"],
    ["video/mp4", "mp4"],
    ["video/webm;codecs=vp9", "webm"],
    ["video/webm;codecs=vp8", "webm"],
    ["video/webm", "webm"],
  ];
  for (const [mime, ext] of tries) if (MediaRecorder.isTypeSupported(mime)) return { mime, ext };
  return null;
}

/** Record `cycles` oscillations, paced in real time (MediaRecorder timestamps by wall clock). */
export async function exportVideo(scene, setPhase, { width, height, cycles = 3, periodSec = 2, fps = 30, background = "#ffffff", gizmo = true, bitrate = 8e6, onProgress = () => {}, signal }) {
  const sup = videoSupport();
  if (!sup) throw new Error("this browser cannot record video (MediaRecorder unavailable)");
  const prevBg = scene.bg;
  scene.bg = background;
  scene.beginCapture(width, height, { transparent: false, gizmo });
  scene.applyBackground();
  const canvas = Object.assign(document.createElement("canvas"), { width, height });
  const stream = canvas.captureStream(0);
  const track = stream.getVideoTracks()[0];
  const rec = new MediaRecorder(stream, { mimeType: sup.mime, videoBitsPerSecond: bitrate });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const stopped = new Promise((r) => (rec.onstop = r));
  const total = Math.round(cycles * periodSec * fps);
  try {
    rec.start(250);
    const t0 = performance.now();
    for (let f = 0; f < total; f++) {
      if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
      setPhase((2 * Math.PI * f) / (periodSec * fps));
      scene.captureFrame(canvas);
      track.requestFrame?.();
      onProgress((f + 1) / total);
      const due = t0 + ((f + 1) * 1000) / fps;
      await sleep(Math.max(0, due - performance.now()));
    }
    rec.stop();
    await stopped;
    return { blob: new Blob(chunks, { type: sup.mime.split(";")[0] }), ext: sup.ext };
  } finally {
    if (rec.state !== "inactive") rec.stop();
    scene.bg = prevBg;
    scene.endCapture();
  }
}

/** The displaced structure over one cycle as an extended-XYZ trajectory (opens in OVITO, ASE, VESTA via conversion). */
export function exportXYZ(scene, setPhase, { frames = 24, lattice }) {
  const n = scene.struct.n, els = scene.elements;
  const L = lattice.map((r, i) => r.map((x, j) => x * scene.struct.reps[i]));
  const lat = L.flat().map((x) => x.toFixed(8)).join(" ");
  let out = "";
  for (let f = 0; f < frames; f++) {
    setPhase((2 * Math.PI * f) / frames);
    out += `${n}\nLattice="${lat}" Properties=species:S:1:pos:R:3 pbc="T T T" frame=${f}\n`;
    for (let i = 0; i < n; i++) out += `${els[i]} ${scene.pos[3 * i].toFixed(6)} ${scene.pos[3 * i + 1].toFixed(6)} ${scene.pos[3 * i + 2].toFixed(6)}\n`;
  }
  return new Blob([out], { type: "chemical/x-xyz" });
}
