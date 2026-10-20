// Phonon band structure on a <canvas>: click or drag to pick a (q, band) mode, arrow keys to step, optional species colouring.
// Imaginary (negative) frequencies are drawn below a dashed zero line, since they flag a structural instability.

import { toUnit, UNITS } from "./physics.js";

const M = { l: 58, r: 14, t: 12, b: 34 };

function niceStep(range, target = 6) {
  const raw = range / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

const css = (el, name, fallback) => getComputedStyle(el).getPropertyValue(name).trim() || fallback;

export class BandPlot {
  constructor(canvas, tip, onSelect) {
    this.canvas = canvas; this.tip = tip; this.onSelect = onSelect;
    this.data = null; this.unit = "THz"; this.sel = { q: 0, b: 0 }; this.hover = null;
    this.weights = null; // {names, w, colors}
    this.size = { w: 1, h: 1 };
    this.attach();
  }

  setData(data) {
    this.data = data;
    this.sel = { q: 0, b: 0 };
    this.hover = null;
    this.weights = null;
    this.computeRange();
    this.draw();
  }

  setUnit(u) { this.unit = u; this.computeRange(); this.draw(); }
  setWeights(w) { this.weights = w; this.draw(); }
  select(q, b, notify = true) {
    this.sel = { q, b };
    this.draw();
    if (notify) this.onSelect?.(q, b);
  }

  resize(w, h) {
    this.size = { w: Math.max(50, w), h: Math.max(50, h) };
    this.draw();
  }

  computeRange() {
    if (!this.data) return;
    const { freq } = this.data;
    let lo = Infinity, hi = -Infinity;
    for (const f of freq) { if (f < lo) lo = f; if (f > hi) hi = f; }
    lo = toUnit(Math.min(lo, 0), this.unit); hi = toUnit(Math.max(hi, 0), this.unit);
    const step = niceStep(hi - lo || 1);
    // acoustic modes at Gamma come out as -1e-6 THz: only a clearly negative branch counts as imaginary
    this.imaginary = lo < -0.02 * hi;
    this.yRange = { lo: this.imaginary ? Math.floor(lo / step) * step : 0, hi: Math.ceil((hi * 1.03) / step) * step, step };
  }

  // ---- geometry ----------------------------------------------------------------------------------------------------------
  xOf(i) {
    const d = this.data.dist, x0 = d[0], x1 = d[this.data.nq - 1];
    return M.l + ((d[i] - x0) / (x1 - x0 || 1)) * (this.size.w - M.l - M.r);
  }
  yOf(thz) {
    const { lo, hi } = this.yRange;
    return this.size.h - M.b - ((toUnit(thz, this.unit) - lo) / (hi - lo)) * (this.size.h - M.t - M.b);
  }

  nearestQ(px) {
    const d = this.data;
    let best = 0, bd = Infinity;
    for (let i = 0; i < d.nq; i++) { const e = Math.abs(this.xOf(i) - px); if (e < bd) { bd = e; best = i; } }
    return best;
  }
  nearestBand(q, py) {
    const d = this.data;
    let best = 0, bd = Infinity;
    for (let b = 0; b < d.nb; b++) { const e = Math.abs(this.yOf(d.freq[q * d.nb + b]) - py); if (e < bd) { bd = e; best = b; } }
    return best;
  }

  // ---- drawing -------------------------------------------------------------------------------------------------------------
  draw(ctx = null, { w, h, scale = 1, transparent = false } = {}) {
    const c = this.canvas;
    const own = !ctx;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (own) {
      const W = this.size.w, H = this.size.h;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    } else ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const saved = this.size;
    if (!own) this.size = { w, h };
    this.paint(ctx, transparent);
    this.size = saved;
  }

  paint(ctx, transparent = false) {
    const { w, h } = this.size;
    const bgc = css(this.canvas, "--panel", "#fff");
    ctx.clearRect(0, 0, w, h);
    if (!transparent) { ctx.fillStyle = bgc; ctx.fillRect(0, 0, w, h); }
    const d = this.data;
    if (!d) return;
    const ink = css(this.canvas, "--ink", "#222"), muted = css(this.canvas, "--muted", "#777"), grid = css(this.canvas, "--grid", "#e6e6e6");
    const band = css(this.canvas, "--band", "#3b6fd8"), accent = css(this.canvas, "--accent", "#e8590c"), bad = css(this.canvas, "--bad", "#d62f3a");
    const { lo, hi, step } = this.yRange;
    ctx.font = "12px system-ui, sans-serif";
    ctx.lineJoin = "round";

    // horizontal grid + y tick labels
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      const y = this.yOf(v / UNITS[this.unit].factor);
      ctx.strokeStyle = grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(w - M.r, y); ctx.stroke();
      ctx.fillStyle = muted;
      ctx.fillText(Math.abs(v) < 1e-9 ? "0" : String(+v.toPrecision(4)), M.l - 7, y);
    }
    // vertical lines at high-symmetry points
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of d.ticks) {
      const x = this.xOf(t.index);
      ctx.strokeStyle = muted; ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, h - M.b); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = ink; ctx.font = "14px system-ui, 'Segoe UI Symbol', sans-serif";
      ctx.fillText(t.label, x, h - M.b + 8);
    }
    ctx.font = "12px system-ui, sans-serif";
    // frame and axis title
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.strokeRect(M.l, M.t, w - M.l - M.r, h - M.t - M.b);
    ctx.save();
    ctx.translate(14, M.t + (h - M.t - M.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(`Frequency (${UNITS[this.unit].label})`, 0, 0);
    ctx.restore();
    // zero line, dashed red when there are imaginary modes
    if (this.imaginary) {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = bad; ctx.lineWidth = 1.2;
      const y0 = this.yOf(0);
      ctx.beginPath(); ctx.moveTo(M.l, y0); ctx.lineTo(w - M.r, y0); ctx.stroke(); ctx.setLineDash([]);
    }

    // bands, one polyline per path segment
    const weighted = !!this.weights;
    ctx.save();
    ctx.beginPath(); ctx.rect(M.l, M.t, w - M.l - M.r, h - M.t - M.b); ctx.clip();
    for (let b = 0; b < d.nb; b++) {
      for (const seg of d.segments) {
        ctx.beginPath();
        for (let i = seg.start; i <= seg.end; i++) {
          const x = this.xOf(i), y = this.yOf(d.freq[i * d.nb + b]);
          i === seg.start ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = weighted ? muted : b === this.sel.b ? accent : band;
        ctx.globalAlpha = weighted ? 0.35 : b === this.sel.b ? 1 : 0.8;
        ctx.lineWidth = b === this.sel.b ? 2.4 : 1.4;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    if (weighted) {
      const { names, w: W, colors } = this.weights;
      const ns = names.length;
      const rad = Math.max(1.4, Math.min(3, (w - M.l - M.r) / d.nq / 1.8));
      for (let b = 0; b < d.nb; b++)
        for (let i = 0; i < d.nq; i++) {
          let r = 0, g = 0, bl = 0;
          for (let k = 0; k < ns; k++) { const wk = W[(i * d.nb + b) * ns + k]; r += wk * colors[k][0]; g += wk * colors[k][1]; bl += wk * colors[k][2]; }
          ctx.fillStyle = `rgb(${r | 0},${g | 0},${bl | 0})`;
          ctx.beginPath(); ctx.arc(this.xOf(i), this.yOf(d.freq[i * d.nb + b]), rad, 0, 6.2832); ctx.fill();
        }
    }
    ctx.restore();

    // selection marker
    const { q, b } = this.sel;
    const sx = this.xOf(q), sy = this.yOf(d.freq[q * d.nb + b]);
    ctx.strokeStyle = accent; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx, M.t); ctx.lineTo(sx, h - M.b); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = accent; ctx.strokeStyle = bgc; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 5.5, 0, 6.2832); ctx.stroke(); ctx.fill();
    if (this.hover && (this.hover.q !== q || this.hover.b !== b)) {
      ctx.strokeStyle = accent; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(this.xOf(this.hover.q), this.yOf(d.freq[this.hover.q * d.nb + this.hover.b]), 5, 0, 6.2832); ctx.stroke();
    }
  }

  // ---- interaction ------------------------------------------------------------------------------------------------------
  attach() {
    const c = this.canvas;
    let drag = false;
    const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const pick = (e, keepBand) => {
      const [x, y] = pos(e);
      const q = this.nearestQ(x);
      const b = keepBand ? this.sel.b : this.nearestBand(q, y);
      this.select(q, b);
    };
    c.addEventListener("pointerdown", (e) => {
      if (!this.data) return;
      c.setPointerCapture(e.pointerId);
      drag = true; pick(e, false);
      c.focus();
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.data) return;
      if (drag) { pick(e, true); return; }
      const [x, y] = pos(e);
      const q = this.nearestQ(x), b = this.nearestBand(q, y);
      this.hover = { q, b };
      const d = this.data, f = d.freq[q * d.nb + b];
      const qv = [0, 1, 2].map((k) => +d.qpts[q * 3 + k].toFixed(3));
      this.tip.textContent = `${(toUnit(f, this.unit)).toFixed(UNITS[this.unit].digits)} ${UNITS[this.unit].label}  ·  band ${b + 1}  ·  q = (${qv.join(", ")})`;
      this.tip.style.opacity = 1;
      this.draw();
    });
    c.addEventListener("pointerup", () => { drag = false; });
    c.addEventListener("pointerleave", () => { this.hover = null; this.tip.style.opacity = 0; if (this.data) this.draw(); });
    c.addEventListener("keydown", (e) => {
      if (!this.data) return;
      const d = this.data;
      let { q, b } = this.sel;
      if (e.key === "ArrowRight") q = Math.min(d.nq - 1, q + (e.shiftKey ? 5 : 1));
      else if (e.key === "ArrowLeft") q = Math.max(0, q - (e.shiftKey ? 5 : 1));
      else if (e.key === "ArrowUp") b = Math.min(d.nb - 1, b + 1);
      else if (e.key === "ArrowDown") b = Math.max(0, b - 1);
      else if (e.key === "Home") q = 0;
      else if (e.key === "End") q = d.nq - 1;
      else return;
      e.preventDefault();
      this.select(q, b);
    });
    c.tabIndex = 0;
  }

  /** Render to a standalone canvas for export. */
  toCanvas(width = 1600, height = 900, { transparent = false } = {}) {
    const out = document.createElement("canvas");
    out.width = width; out.height = height;
    const scale = width / Math.max(560, this.size.w);
    const logicalW = width / scale, logicalH = height / scale;
    const ctx = out.getContext("2d");
    const prevHover = this.hover; this.hover = null;
    this.draw(ctx, { w: logicalW, h: logicalH, scale, transparent });
    this.hover = prevHover;
    return out;
  }
}
