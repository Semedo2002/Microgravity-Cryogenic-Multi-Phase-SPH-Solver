#!/usr/bin/env python3
"""Stability of the liquid/vapour interface under strong evaporation.

A closed LN2 box (4 cm, half full, 0.5 m/s^2 settling acceleration) starts with
superheated liquid.  The script sweeps the Lee coefficient and the superheat
with the defaults and with vapour mass rebalancing switched off (the previous
behaviour), and records for each case:

  * interface integrity: fraction of fluid particles on the "wrong" side of the
    initial interface (liquid above it or vapour below it, 2 dx tolerance);
  * peak and 99th-percentile speeds;
  * mass balance error and energy balance error relative to the latent heat
    moved by phase change.

Writes out/high_evaporation/summary.csv.

  PYTHONPATH=build/python python examples/high_evaporation.py
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

import CryoSPH as cs


VARIANTS = {
    "default": {},
    "no_rebalance": {"vapor_mass_rebalance": 0.0},
}


def run_case(lee, superheat, variant, t_end, dx):
    ln2 = cs.FluidProperties.liquid_nitrogen()
    cfg = cs.SolverConfig()
    cfg.dx = dx
    cfg.gravity = (0.0, 0.0, -0.5)
    cfg.contact_angle_deg = 10.0
    cfg.velocity_scale = 1.0
    cfg.thermo_mode = cs.TankThermoMode.Closed
    cfg.lee_evaporation = lee
    cfg.lee_condensation = lee
    for k, v in VARIANTS[variant].items():
        setattr(cfg, k, v)
    solver = cs.Solver(cfg, ln2)
    fill = cs.FillSpec()
    fill.fill_fraction = 0.5
    fill.T_liquid = ln2.T_sat + superheat
    solver.build_tank(cs.BoxTank(0.04, 0.04, 0.04), fill)
    solver.relax(0.02)
    d0 = solver.diagnostics()
    wall = time.perf_counter()
    solver.run_until(t_end)
    wall = time.perf_counter() - wall
    d1 = solver.diagnostics()

    p = solver.particles()
    ph, z = p["phase"], p["position"][:, 2]
    speed = np.linalg.norm(p["velocity"], axis=1)
    fluid = ph < 2
    wrong = np.sum((ph == 0) & (z > 2 * dx)) + np.sum((ph == 1) & (z < -2 * dx))
    latent = max(d1.evaporated_mass + d1.condensed_mass, 1e-30) * ln2.h_fg
    return {
        "lee": lee, "superheat_K": superheat, "variant": variant,
        "steps": d1.step, "wall_s": round(wall, 2),
        "evaporated_kg": d1.evaporated_mass, "condensed_kg": d1.condensed_mass,
        "p_ullage_kPa": d1.p_ullage / 1e3,
        "T_liquid_mean_K": d1.T_liquid_mean,
        "misplaced_fraction": wrong / np.sum(fluid),
        "v_max": float(speed[fluid].max()),
        "v99_liquid": float(np.percentile(speed[ph == 0], 99)),
        "v99_vapor": float(np.percentile(speed[ph == 1], 99)),
        "mass_rel_error": (d1.mass_total - d0.mass_total) / d0.mass_total,
        "energy_error_rel_latent": (d1.enthalpy - d0.enthalpy) / latent,
        "splits": d1.splits, "merges": d1.merges,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lee", default="2,5,10")
    ap.add_argument("--superheat", default="3,5")
    ap.add_argument("--t-end", type=float, default=0.1)
    ap.add_argument("--dx", type=float, default=0.004)
    ap.add_argument("--out", default="out/high_evaporation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for lee in [float(v) for v in args.lee.split(",")]:
        for sh in [float(v) for v in args.superheat.split(",")]:
            for variant in VARIANTS:
                r = run_case(lee, sh, variant, args.t_end, args.dx)
                rows.append(r)
                print(f"Lee {lee:4.1f}  dT {sh:3.1f} K  {variant:16s} | "
                      f"evap {r['evaporated_kg']:.2e} kg  misplaced {100 * r['misplaced_fraction']:5.2f} %  "
                      f"v99 L/V {r['v99_liquid']:.3f}/{r['v99_vapor']:.3f}  vmax {r['v_max']:.2f} m/s  "
                      f"mass err {r['mass_rel_error']:+.1e}  energy err {100 * r['energy_error_rel_latent']:+.2f} % of latent",
                      flush=True)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.join(args.out, 'summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
