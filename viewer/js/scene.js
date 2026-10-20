// The 3D view: atoms, bonds, displacement arrows, cell outline, an axes gizmo, and a camera you can orient along crystal
// directions. Rendering is on demand (nothing is drawn unless something changed or the animation is playing).

import * as T from "../vendor/three.js";
import { cellSegments, displace, findBonds, fracToCart, reciprocal, vacuumAxes, viewBasis, unit as vunit } from "./physics.js";
import { colorOf, radiusOf } from "./elements.js";

const Y = new T.Vector3(0, 1, 0);
const _m = new T.Matrix4(), _q = new T.Quaternion(), _p = new T.Vector3(), _s = new T.Vector3(), _d = new T.Vector3();
const _c = new T.Color();

const AXIS_COLORS = ["#e5484d", "#30a46c", "#3e63dd"];

export const DEFAULT_STYLE = {
  look: "ballstick", // ballstick | spacefill | sticks
  atomScale: 1,
  bonds: true,
  bondRadius: 0.09,
  bondTol: 1.15,
  splitBonds: true,
  bondColor: "#9aa0a6",
  arrows: false,
  arrowMode: "peak", // peak: the amplitude vector (fixed) | live: the instantaneous displacement
  arrowLength: 1.5, // A, for the atom that moves most
  arrowRadius: 0.07,
  arrowColor: "#e8590c",
  cell: "unit", // off | unit | super | all
  cellRadius: 0.035,
  cellColor: "#8b8f98",
  gizmo: "lattice", // off | lattice | reciprocal | cartesian
  perspective: false,
};

export class CrystalScene {
  constructor(canvas, onDirty = () => {}) {
    this.canvas = canvas;
    this.onDirty = onDirty;
    this.renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.bg = "#ffffff";
    this.transparent = false;
    this.applyBackground();

    this.scene = new T.Scene();
    this.scene.add(new T.AmbientLight(0xffffff, 0.55));
    this.scene.add(new T.HemisphereLight(0xffffff, 0x8890a0, 0.5));
    this.headlight = new T.Group(); // follows the camera so shading stays put while you rotate
    this.key = new T.DirectionalLight(0xffffff, 1.9);
    this.key.position.set(0.5, 0.8, 1);
    this.headlight.add(this.key, this.key.target);
    this.scene.add(this.headlight);

    this.ortho = new T.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
    this.persp = new T.PerspectiveCamera(28, 1, 0.1, 1000);
    this.view = { q: new T.Quaternion(), target: new T.Vector3(), hh: 5 };
    this.style = { ...DEFAULT_STYLE };
    this.overrides = {}; // element -> {color, radiusScale}

    this.data = null; this.struct = null; this.pattern = null;
    this.elements = []; this.pos = null; this.bondPairs = new Uint32Array(0);
    this.amp = 0.4; this.phase = 0;
    this.meshes = {};
    this.size = { w: 1, h: 1 };
    this.gizmoScene = new T.Scene();
    this.gizmoCam = new T.OrthographicCamera(-1.7, 1.7, 1.7, -1.7, 0.1, 50);
    this.gizmoPx = 104;
    this.anim = null;

    this.sphere = new T.SphereGeometry(1, 40, 28);
    this.cyl = new T.CylinderGeometry(1, 1, 1, 20, 1, false);
    this.cone = new T.ConeGeometry(1, 1, 24, 1);
    this.phong = new T.MeshPhongMaterial({ color: 0xffffff, shininess: 55, specular: 0x303030 });

    this.attachControls();
  }

  // ---- size / camera ----------------------------------------------------------------------------------------------------
  resize(w, h) {
    w = Math.max(1, Math.floor(w)); h = Math.max(1, Math.floor(h));
    this.size = { w, h };
    this.renderer.setSize(w, h, false);
    if (this.struct && !this.userMoved) this.fit(); // keep the crystal framed while the layout settles or the window changes
    this.invalidate();
  }

  get camera() { return this.style.perspective ? this.persp : this.ortho; }

  placeCamera(cam, aspect, hh = this.view.hh) {
    const v = this.view;
    const back = _d.set(0, 0, 1).applyQuaternion(v.q);
    const dist = hh * 6; // far enough that the whole block is in front of the near plane
    if (cam.isOrthographicCamera) {
      cam.left = -hh * aspect; cam.right = hh * aspect; cam.top = hh; cam.bottom = -hh;
      cam.near = 0.1; cam.far = dist * 4;
      cam.position.copy(v.target).addScaledVector(back, dist);
    } else {
      const d = hh / Math.tan((cam.fov * Math.PI) / 360); // same apparent size as the orthographic view at the target
      cam.aspect = aspect; cam.near = Math.max(0.05, d * 0.05); cam.far = d * 10;
      cam.position.copy(v.target).addScaledVector(back, d);
    }
    cam.quaternion.copy(v.q);
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
  }

  invalidate() { this.dirty = true; this.onDirty(); }

  applyBackground() {
    this.renderer.setClearColor(new T.Color(this.bg), this.transparent ? 0 : 1);
  }
  setBackground(hex) { this.bg = hex; this.applyBackground(); this.invalidate(); }

  // ---- data --------------------------------------------------------------------------------------------------------------
  setStructure(data, struct, { fit = true } = {}) {
    this.data = data; this.struct = struct;
    this.elements = Array.from(struct.kappa, (k) => data.elements[k]);
    this.pos = new Float32Array(struct.base);
    this.pattern = this.pattern && this.pattern.re.length === struct.base.length ? this.pattern : null;
    this.computeBonds();
    this.rebuild();
    if (fit) this.fit();
    this.setGizmo();
  }

  computeBonds() {
    this.bondPairs = this.struct ? findBonds(this.struct.base, this.elements, this.style.bondTol) : new Uint32Array(0);
  }

  setPattern(pattern) { this.pattern = pattern; this.update(); }
  setAmplitude(a) { this.amp = a; this.update(); }
  setPhase(p) { this.phase = p; this.update(); }

  setStyle(patch) {
    this.style = { ...this.style, ...patch };
    if ("bondTol" in patch && this.struct) this.computeBonds();
    if ("gizmo" in patch) this.setGizmo();
    if (this.struct) this.rebuild();
    this.invalidate();
  }

  setOverrides(map) { this.overrides = map; if (this.struct) this.rebuild(); }

  atomColor(el) { return this.overrides[el]?.color ?? colorOf(el); }

  atomRadius(el) {
    const s = this.style;
    const cov = radiusOf(el) * (this.overrides[el]?.radiusScale ?? 1);
    if (s.look === "spacefill") return cov * 0.95 * s.atomScale;
    if (s.look === "sticks") return s.bondRadius * 1.02;
    return Math.max(0.16, cov * 0.36) * s.atomScale;
  }

  // ---- meshes --------------------------------------------------------------------------------------------------------------
  disposeMeshes() {
    for (const m of Object.values(this.meshes)) { this.scene.remove(m); m.dispose(); }
    this.meshes = {};
  }

  instanced(name, geo, count, mat = this.phong) {
    const m = new T.InstancedMesh(geo, mat, Math.max(1, count));
    m.instanceMatrix.setUsage(T.DynamicDrawUsage);
    m.count = count;
    m.frustumCulled = false;
    this.scene.add(m);
    this.meshes[name] = m;
    return m;
  }

  rebuild() {
    this.disposeMeshes();
    const s = this.style, n = this.struct.n;
    const atoms = this.instanced("atoms", this.sphere, n);
    for (let i = 0; i < n; i++) atoms.setColorAt(i, _c.set(this.atomColor(this.elements[i])));
    atoms.instanceColor.needsUpdate = true;
    this.radii = Float32Array.from(this.elements, (e) => this.atomRadius(e));

    const nb = this.bondPairs.length / 2;
    if (s.bonds && nb) {
      const bonds = this.instanced("bonds", this.cyl, s.splitBonds ? 2 * nb : nb);
      for (let b = 0; b < nb; b++) {
        if (s.splitBonds) {
          bonds.setColorAt(2 * b, _c.set(this.atomColor(this.elements[this.bondPairs[2 * b]])));
          bonds.setColorAt(2 * b + 1, _c.set(this.atomColor(this.elements[this.bondPairs[2 * b + 1]])));
        } else bonds.setColorAt(b, _c.set(s.bondColor));
      }
      bonds.instanceColor.needsUpdate = true;
    }
    if (s.arrows) {
      const shafts = this.instanced("shafts", this.cyl, n);
      const heads = this.instanced("heads", this.cone, n);
      for (let i = 0; i < n; i++) { shafts.setColorAt(i, _c.set(s.arrowColor)); heads.setColorAt(i, _c.set(s.arrowColor)); }
      shafts.instanceColor.needsUpdate = heads.instanceColor.needsUpdate = true;
    }
    const segs = cellSegments(this.data.lattice, this.struct.reps, s.cell);
    if (segs.length) {
      const cell = this.instanced("cell", this.cyl, segs.length);
      segs.forEach(([a, b], i) => {
        cell.setColorAt(i, _c.set(s.cellColor));
        this.segment(cell, i, a[0], a[1], a[2], b[0], b[1], b[2], s.cellRadius);
      });
      cell.instanceColor.needsUpdate = true;
      cell.instanceMatrix.needsUpdate = true;
    }
    this.update();
  }

  /** Orient instance `i` of a unit cylinder (radius 1, height 1, axis +y) to run from a to b with radius r. */
  segment(mesh, i, ax, ay, az, bx, by, bz, r) {
    _d.set(bx - ax, by - ay, bz - az);
    const L = _d.length();
    if (L < 1e-9) { _m.makeScale(0, 0, 0); mesh.setMatrixAt(i, _m); return; }
    _d.multiplyScalar(1 / L);
    _q.setFromUnitVectors(Y, _d);
    _p.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2);
    _s.set(r, L, r);
    mesh.setMatrixAt(i, _m.compose(_p, _q, _s));
  }

  /** Recompute positions for the current phase/amplitude and write the instance matrices. */
  update() {
    if (!this.struct) return;
    const { struct, style: s } = this;
    const n = struct.n, pos = this.pos, base = struct.base;
    if (this.pattern) displace(base, this.pattern, this.amp, this.phase, pos);
    else pos.set(base);

    const a = this.meshes.atoms, arr = a.instanceMatrix.array;
    for (let i = 0; i < n; i++) {
      const r = this.radii[i], o = 16 * i;
      arr[o] = r; arr[o + 1] = 0; arr[o + 2] = 0; arr[o + 3] = 0;
      arr[o + 4] = 0; arr[o + 5] = r; arr[o + 6] = 0; arr[o + 7] = 0;
      arr[o + 8] = 0; arr[o + 9] = 0; arr[o + 10] = r; arr[o + 11] = 0;
      arr[o + 12] = pos[3 * i]; arr[o + 13] = pos[3 * i + 1]; arr[o + 14] = pos[3 * i + 2]; arr[o + 15] = 1;
    }
    a.instanceMatrix.needsUpdate = true;

    const bm = this.meshes.bonds;
    if (bm) {
      const P = this.bondPairs, nb = P.length / 2, br = s.bondRadius;
      for (let b = 0; b < nb; b++) {
        const i = 3 * P[2 * b], j = 3 * P[2 * b + 1];
        if (s.splitBonds) {
          const mx = (pos[i] + pos[j]) / 2, my = (pos[i + 1] + pos[j + 1]) / 2, mz = (pos[i + 2] + pos[j + 2]) / 2;
          this.segment(bm, 2 * b, pos[i], pos[i + 1], pos[i + 2], mx, my, mz, br);
          this.segment(bm, 2 * b + 1, mx, my, mz, pos[j], pos[j + 1], pos[j + 2], br);
        } else this.segment(bm, b, pos[i], pos[i + 1], pos[i + 2], pos[j], pos[j + 1], pos[j + 2], br);
      }
      bm.instanceMatrix.needsUpdate = true;
    }

    const sh = this.meshes.shafts;
    if (sh && this.pattern) {
      const he = this.meshes.heads, ar = s.arrowRadius, headLen = Math.max(0.22, ar * 4.2), headR = ar * 2.4;
      const live = s.arrowMode === "live";
      const cp = Math.cos(this.phase), sp = Math.sin(this.phase);
      const { re, im } = this.pattern;
      for (let i = 0; i < n; i++) {
        let vx, vy, vz;
        if (live) {
          vx = re[3 * i] * cp + im[3 * i] * sp; vy = re[3 * i + 1] * cp + im[3 * i + 1] * sp; vz = re[3 * i + 2] * cp + im[3 * i + 2] * sp;
        } else { vx = re[3 * i]; vy = re[3 * i + 1]; vz = re[3 * i + 2]; }
        const m = Math.hypot(vx, vy, vz), L = m * s.arrowLength;
        const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        if (L < 0.08) { _m.makeScale(0, 0, 0); sh.setMatrixAt(i, _m); he.setMatrixAt(i, _m); continue; }
        const ux = vx / m, uy = vy / m, uz = vz / m;
        const hl = Math.min(headLen, L * 0.6), sl = L - hl;
        this.segment(sh, i, x, y, z, x + ux * sl, y + uy * sl, z + uz * sl, ar);
        this.segment(he, i, x + ux * sl, y + uy * sl, z + uz * sl, x + ux * L, y + uy * L, z + uz * L, headR);
      }
      sh.instanceMatrix.needsUpdate = true; he.instanceMatrix.needsUpdate = true;
    }
    this.invalidate();
  }

  // ---- camera control -------------------------------------------------------------------------------------------------------
  blockCorners() {
    const { reps } = this.struct, lat = this.data.lattice, pts = [];
    const vac = vacuumAxes(this.data); // do not let 15 A of vacuum dominate the framing of a slab
    const ext = reps.map((n, i) => (vac[i] ? 0 : n));
    for (const i of [0, ext[0]]) for (const j of [0, ext[1]]) for (const k of [0, ext[2]]) pts.push(fracToCart([i, j, k], lat));
    return pts;
  }

  /** Camera target and half-height that frame every atom (and the cell block) tightly for orientation `q`. */
  fitFor(q) {
    const pts = this.blockCorners();
    const b = this.struct.base;
    for (let i = 0; i < b.length; i += 3) pts.push([b[i], b[i + 1], b[i + 2]]);
    const r = new T.Vector3(1, 0, 0).applyQuaternion(q), u = new T.Vector3(0, 1, 0).applyQuaternion(q), k = new T.Vector3(0, 0, 1).applyQuaternion(q);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) {
      const c = [r.x * p[0] + r.y * p[1] + r.z * p[2], u.x * p[0] + u.y * p[1] + u.z * p[2], k.x * p[0] + k.y * p[1] + k.z * p[2]];
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], c[a]); hi[a] = Math.max(hi[a], c[a]); }
    }
    const pad = Math.max(...this.radii) + this.amp + 0.25;
    const mid = lo.map((x, a) => (x + hi[a]) / 2);
    const target = new T.Vector3().addScaledVector(r, mid[0]).addScaledVector(u, mid[1]).addScaledVector(k, mid[2]);
    const aspect = this.size.w / this.size.h;
    const hh = Math.max((hi[1] - lo[1]) / 2 + pad, ((hi[0] - lo[0]) / 2 + pad) / aspect) * 1.06;
    return { target, hh };
  }

  fit() {
    if (!this.struct) return;
    this.userMoved = false;
    const f = this.fitFor(this.view.q);
    this.view.target.copy(f.target);
    this.view.hh = f.hh;
    this.invalidate();
  }

  /** Orientation quaternion looking down `dir` with the crystal's own axes as the screen's right axis. */
  quatLooking(dir, refs) {
    const { right, up, back } = viewBasis(dir, refs);
    const m = new T.Matrix4().makeBasis(new T.Vector3(...right), new T.Vector3(...up), new T.Vector3(...back));
    return new T.Quaternion().setFromRotationMatrix(m);
  }

  /** Look down a lattice direction ('a'|'b'|'c'|'a*'|'b*'|'c*'), a cartesian axis ('x'|'y'|'z'), or an explicit [x,y,z]. */
  lookAlong(spec, { flip = false, ms = 450 } = {}) {
    const lat = this.data.lattice, rec = reciprocal(lat);
    let dir, refs = lat;
    const named = { a: lat[0], b: lat[1], c: lat[2], "a*": rec[0], "b*": rec[1], "c*": rec[2], x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
    if (typeof spec === "string") { dir = named[spec]; if ("xyz".includes(spec)) refs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; }
    else dir = spec;
    if (!dir) return;
    if (flip) dir = dir.map((x) => -x);
    this.animateTo(this.quatLooking(dir, refs), ms);
    this.lastLook = { spec: typeof spec === "string" ? spec : null, flip };
  }

  defaultView(ms = 0) {
    const lat = this.data.lattice;
    const base = this.quatLooking(lat[2], lat);
    const tilt = vacuumAxes(this.data)[2] ? [0, 0] : [-0.5, 0.38]; // a slab is shown face-on, a bulk crystal slightly tilted
    const dq = new T.Quaternion().setFromAxisAngle(new T.Vector3(0, 1, 0), tilt[0]).multiply(new T.Quaternion().setFromAxisAngle(new T.Vector3(1, 0, 0), tilt[1]));
    this.animateTo(base.clone().multiply(dq), ms);
    this.lastLook = { spec: null, flip: false };
  }

  animateTo(q, ms = 400) {
    this.userMoved = false;
    const f = this.struct ? this.fitFor(q) : null; // frame the new orientation as we turn to it
    if (!ms) { this.view.q.copy(q); if (f) { this.view.target.copy(f.target); this.view.hh = f.hh; } this.invalidate(); return; }
    this.anim = { from: this.view.q.clone(), to: q.clone(), t0: performance.now(), ms, fit: f, tFrom: this.view.target.clone(), hFrom: this.view.hh };
    this.invalidate();
  }

  stepAnimation(now) {
    if (!this.anim) return false;
    const a = this.anim;
    let t = Math.min(1, (now - a.t0) / a.ms);
    t = t * t * (3 - 2 * t);
    this.view.q.copy(a.from).slerp(a.to, t);
    if (a.fit) { this.view.target.copy(a.tFrom).lerp(a.fit.target, t); this.view.hh = a.hFrom + (a.fit.hh - a.hFrom) * t; }
    if (now - a.t0 >= a.ms) this.anim = null;
    return true;
  }

  attachControls() {
    const el = this.canvas, v = this.view;
    const ptrs = new Map();
    let last = null;
    const k = () => Math.PI / (0.7 * Math.min(this.size.w, this.size.h));
    const rot = (dx, dy, roll) => {
      this.anim = null; this.userMoved = true;
      if (roll) _q.setFromAxisAngle(_d.set(0, 0, 1), dx * k());
      else _q.setFromAxisAngle(_d.set(0, 1, 0), -dx * k()).multiply(new T.Quaternion().setFromAxisAngle(_s.set(1, 0, 0), -dy * k()));
      v.q.multiply(_q).normalize();
      this.invalidate();
    };
    const pan = (dx, dy) => {
      this.userMoved = true;
      const upp = (2 * v.hh) / this.size.h;
      const r = _p.set(1, 0, 0).applyQuaternion(v.q), u = _d.set(0, 1, 0).applyQuaternion(v.q);
      v.target.addScaledVector(r, -dx * upp).addScaledVector(u, dy * upp);
      this.invalidate();
    };
    const zoom = (f) => { this.userMoved = true; v.hh = Math.min(500, Math.max(0.3, v.hh * f)); this.invalidate(); };
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      last = { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey };
      this.canvas.style.cursor = "grabbing";
    });
    el.addEventListener("pointermove", (e) => {
      if (!ptrs.has(e.pointerId)) return;
      const prev = ptrs.get(e.pointerId);
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
      if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()];
        const before = Math.hypot(a.x - b.x, a.y - b.y);
        ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
        const [a2, b2] = [...ptrs.values()];
        const after = Math.hypot(a2.x - b2.x, a2.y - b2.y);
        if (before > 0 && after > 0) zoom(before / after);
        pan(dx / 2, dy / 2);
        return;
      }
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (last.button === 2 || last.shift || last.ctrl) pan(dx, dy);
      else rot(dx, dy, last.alt);
    });
    const up = (e) => { ptrs.delete(e.pointerId); this.canvas.style.cursor = "grab"; };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", (e) => { e.preventDefault(); zoom(Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0012))); }, { passive: false });
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("dblclick", () => { this.fit(); });
    el.style.cursor = "grab";
    el.style.touchAction = "none";
  }

  // ---- gizmo -------------------------------------------------------------------------------------------------------------
  setGizmo() {
    const gs = this.gizmoScene;
    while (gs.children.length) { const c = gs.children.pop(); c.geometry?.dispose?.(); c.material?.map?.dispose?.(); c.material?.dispose?.(); }
    this.gizmoVecs = [];
    const mode = this.style.gizmo;
    if (mode === "off" || !this.data) return;
    const lat = this.data.lattice;
    const vecs = mode === "cartesian" ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : mode === "reciprocal" ? reciprocal(lat) : lat;
    const names = mode === "cartesian" ? ["x", "y", "z"] : mode === "reciprocal" ? ["a*", "b*", "c*"] : ["a", "b", "c"];
    vecs.forEach((vec, i) => {
      const u = vunit(vec), col = AXIS_COLORS[i];
      const mat = new T.MeshBasicMaterial({ color: col });
      const shaft = new T.Mesh(this.cyl, mat), head = new T.Mesh(this.cone, mat);
      const L = 1.0, hl = 0.3;
      const place = (mesh, a, b, r) => {
        _d.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        const len = _d.length();
        mesh.quaternion.setFromUnitVectors(Y, _d.multiplyScalar(1 / len));
        mesh.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
        mesh.scale.set(r, len, r);
      };
      const s0 = [0, 0, 0], s1 = u.map((x) => x * (L - hl)), tip = u.map((x) => x * L);
      place(shaft, s0, s1, 0.045);
      place(head, s1, tip, 0.12);
      gs.add(shaft, head);
      const sprite = this.label(names[i], col);
      sprite.position.set(...u.map((x) => x * (L + 0.34)));
      gs.add(sprite);
    });
    gs.add(new T.AmbientLight(0xffffff, 1));
  }

  label(text, color) {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d");
    g.font = "italic 700 84px Georgia, 'Times New Roman', serif";
    g.textAlign = "center"; g.textBaseline = "middle";
    g.lineWidth = 10; g.strokeStyle = this.bg === "transparent" ? "#fff" : this.bg; g.lineJoin = "round";
    g.strokeText(text, 64, 70); g.fillStyle = color; g.fillText(text, 64, 70);
    const tex = new T.CanvasTexture(c);
    tex.colorSpace = T.SRGBColorSpace;
    const sp = new T.Sprite(new T.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(0.62, 0.62, 1);
    return sp;
  }

  renderGizmo(w, h, px) {
    if (!this.gizmoScene.children.length) return;
    const m = Math.round(8 * (px / this.gizmoPx));
    this.gizmoCam.quaternion.copy(this.view.q);
    this.gizmoCam.position.set(0, 0, 8).applyQuaternion(this.view.q);
    this.gizmoCam.updateMatrixWorld();
    const r = this.renderer;
    r.setScissorTest(true);
    r.setViewport(m, m, px, px);
    r.setScissor(m, m, px, px);
    r.autoClear = false;
    r.clearDepth();
    r.render(this.gizmoScene, this.gizmoCam);
    r.setScissorTest(false);
    r.setViewport(0, 0, w, h);
    r.autoClear = true;
  }

  // ---- drawing -------------------------------------------------------------------------------------------------------------
  render() {
    if (!this.struct) { this.renderer.clear(); return; }
    const { w, h } = this.size;
    this.placeCamera(this.camera, w / h);
    this.headlight.quaternion.copy(this.view.q);
    this.headlight.position.copy(this.view.target);
    this.renderer.render(this.scene, this.camera);
    if (this.style.gizmo !== "off") this.renderGizmo(w, h, this.gizmoPx);
    this.dirty = false;
  }

  // ---- offscreen capture (stills and animation frames) -----------------------------------------------------------------
  /**
   * Switch the renderer to a fixed output size for a batch of captures. The view is kept whole: if the output is narrower than
   * the window, the visible height grows so nothing is cropped.
   */
  beginCapture(width, height, { transparent = false, gizmo = true } = {}) {
    this.saved = { pr: this.renderer.getPixelRatio(), transparent: this.transparent, gizmoStyle: this.style.gizmo, hh: this.view.hh };
    const viewAspect = this.size.w / this.size.h, outAspect = width / height;
    this.capHH = this.view.hh * Math.max(1, viewAspect / outAspect);
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.transparent = transparent;
    this.applyBackground();
    this.cap = { width, height, gizmo };
  }

  /** Render the current state into a fresh 2D canvas (or draw it into `into`). */
  captureFrame(into = null) {
    const { width, height, gizmo } = this.cap;
    this.placeCamera(this.camera, width / height, this.capHH);
    this.headlight.quaternion.copy(this.view.q);
    this.headlight.position.copy(this.view.target);
    this.renderer.render(this.scene, this.camera);
    if (gizmo && this.style.gizmo !== "off") this.renderGizmo(width, height, Math.round(this.gizmoPx * (Math.min(width, height) / Math.min(this.size.w, this.size.h))));
    const out = into ?? Object.assign(document.createElement("canvas"), { width, height });
    const ctx = out.getContext("2d", { willReadFrequently: true });
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(this.renderer.domElement, 0, 0, width, height);
    return out;
  }

  endCapture() {
    const s = this.saved;
    this.renderer.setPixelRatio(s.pr);
    this.renderer.setSize(this.size.w, this.size.h, false);
    this.transparent = s.transparent;
    this.applyBackground();
    this.cap = null;
    this.invalidate();
  }
}
