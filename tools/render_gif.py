#!/usr/bin/env python3
"""Render viewer frames (frame_*.json + manifest.json) into an animated GIF.

Liquid particles are drawn as a 3-D point cloud coloured by speed (or
temperature); vapour and walls are omitted.  Used for the README hero image.

  python tools/render_gif.py out/maneuver docs/hero.gif --field speed --fps 20
"""
import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("output")
    ap.add_argument("--field", default="speed", choices=["speed", "temperature", "pressure"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", type=float, default=5.0, help="figure size [in]")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--max-points", type=int, default=40000)
    ap.add_argument("--title", default="")
    ap.add_argument("--box", type=float, nargs=6, action="append", default=[],
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="draw an axis-aligned wireframe box (obstacles, tank outline); repeatable")
    ap.add_argument("--zoom", type=float, default=1.3)
    ap.add_argument("--bounds", type=float, nargs=6, default=None, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="fixed plot bounds (default: liquid extent over all frames)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    with open(os.path.join(args.frames_dir, "manifest.json")) as f:
        manifest = json.load(f)
    frames = []
    for fr in manifest["frames"]:
        with open(os.path.join(args.frames_dir, fr["file"])) as f:
            d = json.load(f)
        pos = np.asarray(d["position"], dtype=np.float32).reshape(-1, 3)
        ph = np.asarray(d["phase"])
        val = np.asarray(d[args.field], dtype=np.float32)
        liq = ph == 0
        p, v = pos[liq], val[liq]
        if len(p) > args.max_points:
            idx = np.random.default_rng(0).choice(len(p), args.max_points, replace=False)
            p, v = p[idx], v[idx]
        frames.append((d["time"], p, v))
    allp = np.concatenate([f[1] for f in frames])
    lo, hi = allp.min(0), allp.max(0)
    if args.bounds:
        b = args.bounds
        lo, hi = np.array([b[0], b[2], b[4]]), np.array([b[1], b[3], b[5]])
    vmax = np.percentile(np.concatenate([f[2] for f in frames]), 99)
    vmin = 0.0 if args.field == "speed" else np.percentile(np.concatenate([f[2] for f in frames]), 1)

    fig = plt.figure(figsize=(args.size * 1.25, args.size), facecolor="#0f1218")
    ax = fig.add_subplot(projection="3d", facecolor="#0f1218")
    ax.set_box_aspect(hi - lo, zoom=args.zoom)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def wire(b):
        x0, x1, y0, y1, z0, z1 = b
        for (xa, ya, za), (xb, yb, zb) in [
                ((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)), ((x0, y0, z1), (x1, y0, z1)),
                ((x0, y1, z1), (x1, y1, z1)), ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)),
                ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x0, y0, z0), (x0, y0, z1)),
                ((x1, y0, z0), (x1, y0, z1)), ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1))]:
            ax.plot([xa, xb], [ya, yb], [za, zb], color="#8b93a3", lw=0.8, alpha=0.8)

    def draw(k):
        ax.cla()
        t, p, v = frames[k]
        for b in args.box:
            wire(b)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=v, cmap="turbo", vmin=vmin, vmax=vmax, s=2, depthshade=False,
                   linewidths=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        ax.text2D(0.03, 0.95, f"{args.title}  t = {t:5.2f} s", transform=ax.transAxes, color="#e6e9ef", fontsize=11)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    anim.save(args.output, writer=PillowWriter(fps=args.fps), dpi=90)
    print(f"wrote {args.output} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
