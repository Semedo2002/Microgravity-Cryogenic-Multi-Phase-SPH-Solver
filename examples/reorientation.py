#!/usr/bin/env python3
"""Propellant reorientation after a thrust step in an LH2 capsule tank.

The liquid starts as a wetting film around a central vapour bubble (the
typical low-g configuration).  At t = 0 a settling thrust of `--accel` m/s^2
along -z switches on: the ullage bubble rises towards the forward dome while
the liquid collects aft.  Whether and how fast this happens is governed by the
Bond number Bo = rho_l a R^2 / sigma (capillary forces dominate for Bo < 1).
The script prints Bo and tracks the ullage-bubble centroid and the liquid
centre of mass.

When the bubble approaches the forward dome the liquid film between them
thins to ~1-2 dx; the transport-velocity formulation (on by default) keeps
that film stable.

Run with the module on the path, e.g.
  PYTHONPATH=build/python python examples/reorientation.py
"""
import argparse
import json
import os
import sys

import numpy as np

import CryoSPH as cs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--accel", type=float, default=0.05, help="settling acceleration [m/s^2]")
    ap.add_argument("--fill", type=float, default=0.85,
                    help="liquid fill fraction (below ~0.5 the bubble touches the barrel and the liquid "
                         "above it can only fall through a slow Rayleigh-Taylor instability)")
    ap.add_argument("--t-end", type=float, default=1.5)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", default="out/reorientation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lh2 = cs.FluidProperties.liquid_hydrogen()
    cfg = cs.SolverConfig()
    cfg.dx = args.dx
    cfg.gravity = (0.0, 0.0, 0.0)
    cfg.contact_angle_deg = 2.0
    cfg.thermal = False
    cfg.phase_change = False
    cfg.velocity_scale = (2 * args.accel * 0.2) ** 0.5  # settling speed over ~0.2 m

    solver = cs.Solver(cfg, lh2)
    radius = 0.06
    tank = cs.CapsuleTank(radius=radius, length=0.10)
    fill = cs.FillSpec()
    fill.mode = cs.FillMode.CentralBubble
    fill.fill_fraction = args.fill
    solver.build_tank(tank, fill)
    bond = lh2.rho_liquid * args.accel * radius**2 / lh2.sigma
    print(f"{solver.num_particles} particles on {solver.device}; Bond number = {bond:.2f}")
    solver.relax(0.05)

    solver.set_gravity((0.0, 0.0, -args.accel))
    frames = []
    for k in range(args.frames + 1):
        solver.run_until(args.t_end * k / args.frames)
        d = solver.diagnostics()
        parts = solver.particles()
        vap = parts["phase"] == 1
        z_bubble = float(np.average(parts["position"][vap, 2], weights=parts["mass"][vap])) if vap.any() else float("nan")
        name = f"frame_{k:05d}.json"
        solver.write_json(os.path.join(args.out, name))
        frames.append({"file": name, "time": d.time})
        print(f"t={d.time:6.3f}s  bubble z={z_bubble * 1e3:8.3f} mm  liquid z_com={d.liquid_com[2] * 1e3:8.3f} mm  "
              f"|v|max={d.max_speed:.3f} m/s")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"title": f"LH2 reorientation, {args.accel} m/s^2 settling", "fluid": lh2.name,
                   "T_sat": lh2.T_sat, "frames": frames}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
