"""Animated GIFs of the atoms moving in a mode (or a selection of modes/atoms).

Uses the same selections as ``freeze`` and the arrow files (modes, atoms, parts, amplitudes).  The atoms oscillate
about their **parent** positions, 0 -> +max -> 0 -> -max -> 0, where ``max`` is a fraction (or percentage) of the amplitude
found in the child; the structure is drawn in the child's supercell with element colours, bonds and the cell outline.

Rendering needs matplotlib and Pillow, which may live in another environment (see ``external.py``): the frames are
drawn by a worker run in that interpreter.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np

from .external import find_renderer_python
from .freeze import SelectionError, slug

# covalent radii (A, Cordero 2008) and Jmol colours for common elements
RADIUS = dict(H=.31, He=.28, Li=1.28, Be=.96, B=.84, C=.76, N=.71, O=.66, F=.57, Ne=.58, Na=1.66, Mg=1.41, Al=1.21, Si=1.11,
              P=1.07, S=1.05, Cl=1.02, Ar=1.06, K=2.03, Ca=1.76, Sc=1.70, Ti=1.60, V=1.53, Cr=1.39, Mn=1.39, Fe=1.32,
              Co=1.26, Ni=1.24, Cu=1.32, Zn=1.22, Ga=1.22, Ge=1.20, As=1.19, Se=1.20, Br=1.20, Kr=1.16, Rb=2.20, Sr=1.95,
              Y=1.90, Zr=1.75, Nb=1.64, Mo=1.54, Tc=1.47, Ru=1.46, Rh=1.42, Pd=1.39, Ag=1.45, Cd=1.44, In=1.42, Sn=1.39,
              Sb=1.39, Te=1.38, I=1.39, Xe=1.40, Cs=2.44, Ba=2.15, La=2.07, Ce=2.04, Pr=2.03, Nd=2.01, Sm=1.98, Eu=1.98,
              Gd=1.96, Tb=1.94, Dy=1.92, Ho=1.92, Er=1.89, Tm=1.90, Yb=1.87, Lu=1.87, Hf=1.75, Ta=1.70, W=1.62, Re=1.51,
              Os=1.44, Ir=1.41, Pt=1.36, Au=1.36, Hg=1.32, Tl=1.45, Pb=1.46, Bi=1.48)
COLOR = dict(H="FFFFFF", Li="CC80FF", Be="C2FF00", B="FFB5B5", C="909090", N="3050F8", O="FF0D0D", F="90E050", Na="AB5CF2",
             Mg="8AFF00", Al="BFA6A6", Si="F0C8A0", P="FF8000", S="FFFF30", Cl="1FF01F", K="8F40D4", Ca="3DFF00", Sc="E6E6E6",
             Ti="BFC2C7", V="A6A6AB", Cr="8A99C7", Mn="9C7AC7", Fe="E06633", Co="F090A0", Ni="50D050", Cu="C88033",
             Zn="7D80B0", Ga="C28F8F", Ge="668F8F", As="BD80E3", Se="FFA100", Br="A62929", Rb="702EB0", Sr="00FF00",
             Y="94FFFF", Zr="94E0E0", Nb="73C2C9", Mo="54B5B5", Ru="248F8F", Rh="0A7D8C", Pd="006985", Ag="C0C0C0",
             Cd="FFD98F", In="A67573", Sn="668080", Sb="9E63B5", Te="D47A00", I="940094", Cs="57178F", Ba="00C900",
             La="70D4FF", Hf="4DC2FF", Ta="4DA6FF", W="2194D6", Re="267DAB", Os="266696", Ir="175487", Pt="D0D0E0",
             Au="FFD123", Hg="B8B8D0", Tl="A6544D", Pb="575961", Bi="9E4FB5")


def element(species):
    import re
    m = re.match(r"[A-Z][a-z]?", species)
    return m.group(0) if m else species


def parse_fraction(text):
    """'50%' -> 0.5, '0.5' -> 0.5, '150%' -> 1.5."""
    t = str(text).strip()
    try:
        return float(t[:-1]) / 100.0 if t.endswith("%") else float(t)
    except ValueError:
        raise SelectionError(f"cannot read '{text}' as a fraction or percentage (e.g. 0.5 or 50%)")


def frame_scales(n_frames, oneway=False):
    """Phase factors in [-1, 1] (or [0, 1] for ``oneway``) for the frames of one cycle."""
    if n_frames < 2:
        raise SelectionError("an animation needs at least 2 frames")
    ph = np.arange(n_frames) / n_frames
    return 0.5 * (1.0 - np.cos(2 * np.pi * ph)) if oneway else np.sin(2 * np.pi * ph)


def view_basis(P, view="iso", elev=None, azim=None):
    """Orthonormal camera basis ``(right, up, toward)`` for a named view or elevation/azimuth (degrees).

    The frame is tied to the supercell: 'up' is the c axis, azimuth is measured from a towards b.  Named views look
    down a lattice axis ('a', 'b', 'c', or '-a' ...), 'iso' is an oblique view.
    """
    unit = P / np.linalg.norm(P, axis=1)[:, None]
    a, b, c = unit
    z = c
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if view in ("a", "b", "c", "-a", "-b", "-c"):
        sign = -1.0 if view.startswith("-") else 1.0
        n = sign * {"a": a, "b": b, "c": c}[view.lstrip("-")]
        n = n / np.linalg.norm(n)
        up = c if abs(n @ c) < 0.99 else b
    else:
        el = np.radians(22.0 if elev is None else elev)
        az = np.radians(-50.0 if azim is None else azim)
        n = np.cos(el) * np.cos(az) * x + np.cos(el) * np.sin(az) * y + np.sin(el) * z
        up = z if abs(n @ z) < 0.99 else y
    right = np.cross(up, n)
    right /= np.linalg.norm(right)
    upv = np.cross(n, right)
    return right, upv, n


class Scene:
    """Atoms (with periodic images near the cell faces), bonds and the cell outline for drawing."""

    def __init__(self, fz, field, margin=2.8, face=0.3, bonds=True):
        A = fz.parent.lattice
        P = fz.M @ A
        Minv = np.linalg.inv(fz.M)
        self.P = P
        base, disp, labels, species, src = [], [], [], [], []
        frac_all = []
        for i, x in enumerate(fz.atoms):
            f = (fz.parent.frac[x.a] + fz.cells[x.c]) @ Minv
            frac_all.append(f - np.floor(f + 1e-9))
        lens = np.linalg.norm(P, axis=1)
        delta, dface = margin / lens, face / lens
        shown = []
        for i, x in enumerate(fz.atoms):
            for n in itertools.product((-1, 0, 1), repeat=3):
                f = frac_all[i] + np.array(n)
                if np.all(f >= -delta) and np.all(f <= 1 + delta):
                    base.append(f @ P)
                    disp.append(field[x.a, x.c])
                    labels.append(x.label)
                    species.append(x.species)
                    src.append(i)
                    shown.append(bool(np.all(f >= -dface) and np.all(f <= 1 + dface)))      # inside the cell or on a face
        self.shown = np.array(shown)
        self.base, self.disp = np.array(base), np.array(disp)
        self.labels, self.species, self.src = labels, species, np.array(src)
        el = [element(s) for s in species]
        self.colors = ["#" + COLOR.get(e, "AAAAAA") for e in el]
        self.cov = np.array([RADIUS.get(e, 1.2) for e in el])
        self.bonds = []
        if bonds:
            for i in range(len(self.base)):
                d = np.linalg.norm(self.base[i + 1:] - self.base[i], axis=1)
                for k in np.where(d < 1.2 * (self.cov[i] + self.cov[i + 1:]))[0]:
                    j = i + 1 + int(k)
                    if el[i] != el[j] and (self.shown[i] or self.shown[j]):       # half-bonds reach out of the cell
                        self.bonds.append((i, j))
        cube = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
        corners = cube @ P
        self.edges = [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
                      if np.abs(cube[i] - cube[j]).sum() == 1]


_WORKER = r'''
import json, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

req = json.load(sys.stdin)
base = np.array(req["base"]); disp = np.array(req["disp"])
right, up, toward = (np.array(req[k]) for k in ("right", "up", "toward"))
edges = np.array(req["edges"]).reshape(-1, 2, 3)
size = req["size"]; dpi = 100
colors = req["colors"]; radii = np.array(req["radii"]); bonds = req["bonds"]; shown = np.array(req["shown"], dtype=bool)
scales = req["scales"]; fps = req["fps"]

def proj(p):
    return np.stack([p @ right, p @ up], axis=-1), p @ toward

# fixed window: the cell, the atoms at the largest excursion, with padding
allp = [(base + s * disp)[shown] for s in (min(scales), max(scales))] + [edges.reshape(-1, 3), base[shown]]
xy = np.vstack([proj(p)[0] for p in allp])
lo, hi = xy.min(axis=0), xy.max(axis=0)
span = float(max(hi - lo)) * 1.12 + 2 * radii.max()
mid = (lo + hi) / 2
xlim = (mid[0] - span / 2, mid[0] + span / 2)
ylim = (mid[1] - span / 2 - 0.06 * span, mid[1] + span / 2 + 0.04 * span)
pts_per_A = size / dpi * 72 * 0.94 / (1.10 * span)        # the y range (1.10 x span) limits the equal-aspect scale
frames = []
for f, s in enumerate(scales):
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    pos = base + s * disp
    xy, depth = proj(pos)
    exy, _ = proj(edges.reshape(-1, 3))
    ax.add_collection(LineCollection(exy.reshape(-1, 2, 2), colors="#8a8a8a", linewidths=0.9, zorder=1))
    if bonds:
        segs, cols = [], []
        for i, j in bonds:
            m = (xy[i] + xy[j]) / 2
            if shown[i]:
                segs.append([xy[i], m]); cols.append(colors[i])
            if shown[j]:
                segs.append([m, xy[j]]); cols.append(colors[j])
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.6, zorder=2, capstyle="round"))
    order = np.array([k for k in np.argsort(depth) if shown[k]])
    ax.scatter(xy[order, 0], xy[order, 1], s=((2 * radii[order] * pts_per_A) ** 2),
               c=[colors[k] for k in order], edgecolors="#222222", linewidths=0.7, zorder=3)
    if req.get("arrows"):
        vec = np.array(req["arrow_vectors"])
        d2 = vec @ np.stack([right, up], axis=1)
        keep = (np.linalg.norm(vec, axis=1) > 1e-9) & shown          # no arrows on the out-of-cell images used for bonds
        b0 = proj(base)[0]
        ax.quiver(b0[keep, 0], b0[keep, 1], d2[keep, 0], d2[keep, 1], angles="xy", scale_units="xy", scale=1,
                  color="#111111", width=0.004, zorder=4, alpha=0.55)
    ax.text(0.01, 0.995, req["title"], transform=ax.transAxes, va="top", ha="left", fontsize=10.5, weight="bold")
    ax.text(0.01, 0.945, req["subtitle"].format(pct=100.0 * s * req["fraction"]), transform=ax.transAxes, va="top",
            ha="left", fontsize=9.5, family="monospace")
    for k, (name, col) in enumerate(req["legend"]):
        ax.scatter([], [], s=70, color=col, edgecolors="#222222", label=name)
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=len(req["legend"]), handletextpad=0.2, columnspacing=1.0)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    frames.append(Image.fromarray(rgb))
    plt.close(fig)

# one palette for every frame (built from all of them, so the background is the same white in each) -> no flicker
w, h = frames[0].size
montage = Image.new("RGB", (w, h * len(frames)))
for k, fr in enumerate(frames):
    montage.paste(fr, (0, k * h))
pal = montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
pl = pal.getpalette()[:3 * 255]
for k in range(len(pl) // 3):                                      # snap near-white entries to exact white (the lookup is bucketed)
    if min(pl[3 * k:3 * k + 3]) >= 248:
        pl[3 * k:3 * k + 3] = [255, 255, 255]
pal.putpalette(pl)
q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
q[0].save(req["out"], save_all=True, append_images=q[1:], duration=int(round(1000.0 / fps)), loop=req["loops"], optimize=False)
print(json.dumps({"frames": len(frames), "size": [frames[0].width, frames[0].height]}))
'''


def render_gif(scene, scales, path, *, title, subtitle, fraction, view="iso", elev=None, azim=None, size=640, fps=12,
               loops=0, arrows=False, arrow_scale=1.0, bonds=True, legend=None):
    """Render the animation to ``path`` via the matplotlib/Pillow worker; returns the worker's summary dict."""
    py = find_renderer_python()
    if not py:
        raise RuntimeError("animations need matplotlib and Pillow, which no known Python interpreter has. Install them "
                           "(e.g. `~/.venvs/distortkit/bin/pip install matplotlib pillow`) or set DISTORTKIT_PYTHON to an "
                           "interpreter that has them")
    right, up, toward = view_basis(scene.P, view, elev, azim)
    species = list(dict.fromkeys(scene.species))
    legend = legend or [(sp, "#" + COLOR.get(element(sp), "AAAAAA")) for sp in species]
    req = {
        "base": scene.base.tolist(), "disp": scene.disp.tolist(), "right": right.tolist(), "up": up.tolist(),
        "toward": toward.tolist(), "edges": np.array([[a, b] for a, b in scene.edges]).tolist(), "size": int(size),
        "colors": scene.colors, "radii": (0.33 * scene.cov).tolist(), "shown": scene.shown.tolist(), "bonds": scene.bonds if bonds else [],
        "scales": [float(s) for s in scales], "fps": fps, "loops": int(loops), "out": os.path.abspath(path), "title": title,
        "subtitle": subtitle, "fraction": float(fraction), "legend": legend, "arrows": bool(arrows),
        "arrow_vectors": (scene.disp * arrow_scale).tolist(),
    }
    r = subprocess.run([py, "-c", _WORKER], input=json.dumps(req), capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("rendering failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
    return json.loads(r.stdout.strip().splitlines()[-1])


def gif_info(path):
    """(width, height, frames, loops) read straight from a GIF file by walking its blocks (no imaging library needed)."""
    with open(path, "rb") as fh:
        d = fh.read()
    if d[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("not a GIF file")
    w, h = int.from_bytes(d[6:8], "little"), int.from_bytes(d[8:10], "little")
    pos = 13 + (3 * 2 ** ((d[10] & 7) + 1) if d[10] & 0x80 else 0)
    frames, loops = 0, None

    def skip_sub(p):
        while d[p] != 0:
            p += 1 + d[p]
        return p + 1

    while pos < len(d) and d[pos] != 0x3B:
        if d[pos] == 0x21:                                   # extension
            label = d[pos + 1]
            if label == 0xFF and d[pos + 3:pos + 14] == b"NETSCAPE2.0":
                loops = int.from_bytes(d[pos + 16:pos + 18], "little")
            pos = skip_sub(pos + 2)
        elif d[pos] == 0x2C:                                 # image
            frames += 1
            flags = d[pos + 9]
            pos += 10 + (3 * 2 ** ((flags & 7) + 1) if flags & 0x80 else 0)
            pos = skip_sub(pos + 1)                          # skip the LZW minimum-code-size byte, then the data
        else:
            raise ValueError("unexpected block in GIF")
    return w, h, frames, loops


def run_gif(fz, recipes, *, fraction=1.0, amplitude=None, n_frames=24, fps=12, oneway=False, view="iso", elev=None,
            azim=None, size=640, loops=0, arrows=False, arrow_scale=3.0, bonds=True, outdir=".", prefix=None, log=print):
    """One GIF per recipe.  The largest excursion is ``fraction`` x the amplitude found in the child, or ``amplitude`` A."""
    os.makedirs(outdir, exist_ok=True)
    prefix = prefix or slug(os.path.splitext(str(fz.doc["meta"].get("child", "child")))[0].split("/")[-1]) or "anim"
    scales = frame_scales(n_frames, oneway)
    rows = []
    for i, rc in enumerate(recipes, start=1):
        rss1 = fz.rss(rc.parts, rc.atoms, rc.exclude)
        if rss1 < 1e-12:
            raise SelectionError(f"'{rc.description}' selects no displacement (all selected atoms stay at their parent positions)")
        frac = fraction if amplitude is None else float(amplitude) / rss1
        field = fz.field(rc.parts, rc.atoms, rc.exclude, frac)
        scene = Scene(fz, field, bonds=bonds)
        path = os.path.abspath(os.path.join(outdir, f"{prefix}_{i:02d}_{rc.name}.gif"))
        pg = fz.doc.get("parent_space_group")
        title = f"{rc.description}" + (f"   (parent {pg['symbol']})" if pg else "")
        sub = f"{{pct:+6.1f}}% of the child's amplitude   (max {frac * rss1:.3f} A)"
        info = render_gif(scene, scales, path, title=title, subtitle=sub, fraction=frac, view=view, elev=elev, azim=azim,
                          size=size, fps=fps, loops=loops, arrows=arrows, arrow_scale=arrow_scale, bonds=bonds)
        rows.append({"file": path, "recipe": rc.description, "fraction": frac, "max_amplitude": frac * rss1,
                     "frames": info["frames"], "fps": fps, "oneway": oneway, "view": view, "size": info["size"],
                     "atoms_drawn": int(scene.shown.sum()), "bonds": int(len(scene.bonds))})
    mpath = os.path.join(outdir, "animations_manifest.json")
    allrows = {}
    if os.path.exists(mpath):
        try:
            with open(mpath) as fh:
                allrows = {os.path.basename(r["file"]): r for r in json.load(fh)}
        except (ValueError, KeyError):
            allrows = {}
    for r in rows:
        allrows[os.path.basename(r["file"])] = r
    allrows = {k: v for k, v in allrows.items() if os.path.exists(v["file"])}
    with open(mpath, "w") as fh:
        json.dump(list(allrows.values()), fh, indent=1)
    head = f"{'file':<46s} {'selection':<30s} {'max %':>7s} {'max(A)':>8s} {'frames':>6s} {'fps':>4s}"
    fmt = lambda r: (f"{os.path.basename(r['file']):<46s} {r['recipe'][:30]:<30s} {100 * r['fraction']:7.1f} "
                     f"{r['max_amplitude']:8.4f} {r['frames']:>6d} {r['fps']:>4d}")
    with open(os.path.join(outdir, "animations_manifest.txt"), "w") as fh:
        fh.write("\n".join([head] + [fmt(r) for r in allrows.values()]) + "\n")
    log("\n".join([head] + [fmt(r) for r in rows]))
    return rows
