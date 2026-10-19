#!/usr/bin/env python3
"""Slosh loads on a half-full LH2 tank during a spacecraft manoeuvre.

The tank sits in a 1e-4 g residual acceleration.  The manoeuvre, prescribed in
the tank frame through `ManeuverProfile`, is
  1. a lateral thruster pulse: 0.5 m/s^2 along x for 0.2 s (50 ms ramps);
  2. a slew about y: angular rate ramped to 0.6 rad/s and back, 0.8-1.6 s.
The solver integrates the liquid in the accelerating, rotating tank frame
(Euler, centrifugal and Coriolis terms) and reports the force and moment the
liquid exerts on the tank -- the inputs a GNC engineer needs for attitude
control design.

Outputs in --out: maneuver.csv, maneuver.png and viewer frames.

  PYTHONPATH=build/python python examples/maneuver_slosh.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import CryoSPH as cs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx", type=float, default=0.006)
    ap.add_argument("--t-end", type=float, default=2.5)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--velocity-scale", type=float, default=1.0,
                    help="sets c0 = 10 x this; must keep the tank's acoustic modes well above the slosh modes")
    ap.add_argument("--out", default="out/maneuver")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lh2 = cs.FluidProperties.liquid_hydrogen()
    cfg = cs.SolverConfig()
    cfg.dx = args.dx
    cfg.gravity = (0.0, 0.0, -9.81e-4)
    cfg.contact_angle_deg = 5.0
    cfg.thermal = False
    cfg.phase_change = False
    cfg.velocity_scale = args.velocity_scale
    solver = cs.Solver(cfg, lh2)
    fill = cs.FillSpec()
    fill.fill_fraction = 0.5
    solver.build_tank(cs.CapsuleTank(radius=0.08, length=0.12), fill)
    print(f"{solver.num_particles} particles on {solver.device}", flush=True)
    solver.relax(0.1)

    t = [0.0, 0.05, 0.10, 0.30, 0.35, 0.8, 1.2, 1.6, 10.0]
    acc = [[0, 0, 0], [0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    omega = [[0, 0, 0]] * 5 + [[0, 0, 0], [0, 0.6, 0], [0, 0, 0], [0, 0, 0]]
    solver.set_maneuver(cs.ManeuverProfile(t, np.array(acc, float), np.array(omega, float), origin=(0, 0, 0)))

    rows, frames = [], []
    with open(os.path.join(args.out, "maneuver.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "ax", "omega_y", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "com_x", "com_z"])
        n = int(round(args.t_end / 0.005))
        for k in range(n + 1):
            solver.run_until(k * 0.005)
            d = solver.diagnostics()
            row = [d.time, d.frame_acceleration[0], d.frame_angular_velocity[1], *d.slosh_force, *d.slosh_moment,
                   d.liquid_com[0], d.liquid_com[2]]
            rows.append(row)
            w.writerow([f"{v:.6g}" for v in row])
            if k % max(1, n // args.frames) == 0:
                name = f"frame_{len(frames):05d}.json"
                solver.write_json(os.path.join(args.out, name))
                frames.append({"file": name, "time": d.time})
            if k % 100 == 0:
                print(f"t={d.time:5.2f}s  a_x={row[1]:.2f}  w_y={row[2]:.2f}  F=({row[3]:+.3f},{row[4]:+.3f},{row[5]:+.3f}) N  "
                      f"My={row[7]:+.4f} N m", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"title": "LH2 tank: thruster pulse and slew", "fluid": lh2.name, "frames": frames}, f)

    d0 = solver.diagnostics()
    print(f"liquid mass {d0.mass_liquid:.3f} kg -> rigid-liquid pulse load M*a = {d0.mass_liquid * 0.5:.3f} N")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    r = np.array(rows)
    # The weakly compressible liquid rings at the tank's acoustic modes
    # (c0 / 2L ~ 18-30 Hz here); those are artefacts of the artificial sound
    # speed, far above the slosh dynamics, so loads are also shown low-passed.
    win = max(1, int(round(0.1 / 0.005)))
    lp = lambda y: np.convolve(y, np.ones(win) / win, mode="same")
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    ax[0].plot(r[:, 0], r[:, 1], color="#d9480f", label="tank accel. a_x [m/s²]")
    ax[0].plot(r[:, 0], r[:, 2], color="#7048e8", label="slew rate ω_y [rad/s]")
    ax[0].legend(frameon=False, loc="upper right")
    fz = r[:, 5] - r[0, 5]
    for y, c, lab in ((r[:, 3], "#2a6fdb", "F_x"), (fz, "#12b886", "F_z − F_z(0)")):
        ax[1].plot(r[:, 0], y, color=c, lw=0.6, alpha=0.3)
        ax[1].plot(r[:, 0], lp(y), color=c, lw=1.8, label=lab + " (0.1 s average)")
    ax[1].axhline(-d0.mass_liquid * 0.5, color="#9aa3b2", ls="--", lw=1, label="−M a (rigid liquid)")
    ax[1].set_ylabel("slosh force on tank [N]")
    ax[1].legend(frameon=False, loc="lower right", ncol=2, fontsize=8)
    ax[2].plot(r[:, 0], r[:, 7], color="#2a6fdb", lw=0.6, alpha=0.3)
    ax[2].plot(r[:, 0], lp(r[:, 7]), color="#2a6fdb", lw=1.8)
    ax[2].set_ylabel("slosh moment M_y [N m]")
    ax[2].set_xlabel("time [s]")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Half-full LH2 tank, 1e-4 g: thruster pulse then slew (faint: raw, incl. acoustic ringing)", fontsize=10)
    fig.savefig(os.path.join(args.out, "maneuver.png"), dpi=150)
    print(f"wrote {args.out}/maneuver.csv, maneuver.png and {len(frames)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
