#!/usr/bin/env python3
"""Heated LN2 cylinder in milli-g: boil-off, self-pressurisation and wetting.

A half-filled cylinder (wetting liquid, 5 deg contact angle) sits in a residual
acceleration of 1e-4 g.  A heater band on the lower barrel wall delivers a
constant power; the tank is closed, so evaporation raises the ullage pressure
and with it the saturation temperature.

Outputs (in --out):
  frame_XXXXX.json   particle frames for viz/index.html
  manifest.json      frame list read by the viewer
  state_XXXXX.vtk    full particle state for ParaView (every --vtk-every frames)
  diagnostics.csv    time series of the global state
  pressure.png       ullage pressure and liquid temperature history

Run with the module on the path, e.g.
  PYTHONPATH=build/python python examples/boiloff_cylinder.py --t-end 0.5
"""
import argparse
import csv
import json
import os
import sys
import time

import CryoSPH as cs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx", type=float, default=0.004, help="particle spacing [m]")
    ap.add_argument("--t-end", type=float, default=0.3, help="simulated time [s]")
    ap.add_argument("--frames", type=int, default=60, help="number of output frames")
    ap.add_argument("--heater", type=float, default=20.0, help="heater power [W]")
    ap.add_argument("--vtk-every", type=int, default=5, help="write a VTK file every N frames (0 = never)")
    ap.add_argument("--out", default="out/boiloff", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ln2 = cs.FluidProperties.liquid_nitrogen()
    cfg = cs.SolverConfig()
    cfg.dx = args.dx
    cfg.gravity = (0.0, 0.0, -9.81e-4)
    cfg.contact_angle_deg = 5.0
    # velocity_scale left at 0: c0 is derived from the capillary and gravity
    # scales, which keeps WCSPH noise (a few % of c0) small in milli-g.
    cfg.thermo_mode = cs.TankThermoMode.Closed
    cfg.heater_power = args.heater
    cfg.lee_evaporation = 1.0
    cfg.lee_condensation = 1.0

    solver = cs.Solver(cfg, ln2)
    tank = cs.CylinderTank(radius=0.05, height=0.12)
    fill = cs.FillSpec()
    fill.fill_fraction = 0.5
    fill.T_liquid = ln2.T_sat
    fill.heater_zmin, fill.heater_zmax = -0.06, -0.01
    solver.build_tank(tank, fill)
    print(f"{solver.num_particles} particles on {solver.device}; c0 = {solver.sound_speed:.2f} m/s")

    t0 = time.perf_counter()
    solver.relax(0.05)
    print(f"relaxed initial lattice in {time.perf_counter() - t0:.1f} s")

    frames, rows = [], []
    for k in range(args.frames + 1):
        t_target = args.t_end * k / args.frames
        solver.run_until(t_target)
        d = solver.diagnostics()
        rows.append(d.as_dict())
        name = f"frame_{k:05d}.json"
        solver.write_json(os.path.join(args.out, name))
        frames.append({"file": name, "time": d.time})
        if args.vtk_every > 0 and k % args.vtk_every == 0:
            solver.write_vtk(os.path.join(args.out, f"state_{k:05d}.vtk"))
        print(f"t={d.time:7.4f}s  steps={d.step:6d}  p_ullage={d.p_ullage / 1e3:8.3f} kPa  "
              f"T_sat={d.T_sat:7.3f} K  m_evap={d.evaporated_mass:.3e} kg  |v|max={d.max_speed:.3f} m/s")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"title": "LN2 boil-off, closed cylinder, 1e-4 g", "fluid": ln2.name,
                   "T_sat": ln2.T_sat, "frames": frames}, f, indent=1)

    keys = [k for k in rows[0] if k not in ("liquid_com", "liquid_momentum")]
    with open(os.path.join(args.out, "diagnostics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plot")
        return 0
    t = [r["time"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax[0].plot(t, [r["p_ullage"] / 1e3 for r in rows], color="#2a6fdb")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("ullage pressure [kPa]")
    ax[1].plot(t, [r["T_liquid_mean"] for r in rows], label="liquid mean", color="#2a6fdb")
    ax[1].plot(t, [r["T_sat"] for r in rows], label="T_sat", color="#d9480f", linestyle="--")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("temperature [K]")
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(args.out, "pressure.png"), dpi=150)
    print(f"wrote {len(frames)} frames, VTK states, diagnostics.csv and pressure.png to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
