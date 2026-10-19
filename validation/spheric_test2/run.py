#!/usr/bin/env python3
"""SPHERIC Test 2: 3D dam break with an obstacle (Kleefsman et al. 2005).

Geometry (SPHERIC test description, origin at the upstream wall, z = 0 at the
floor, y centred): tank 3.22 x 1 x 1 m; water column x in [1.992, 3.22] m,
0.55 m deep; box 0.161 x 0.403 x 0.161 m with its front (downstream) face at
x = 0.744 m.  Pressure sensors P1-P4 on the front face at z = 0.021, 0.061,
0.101, 0.141 m and P5-P8 on the top face 0.021, 0.061, 0.101, 0.141 m behind
the front edge, all 0.176 m from the box side (y = 0.0255 m).  Wave probes
H1-H4 at x = 0.496, 0.992, 1.488, 2.638 m, y = 0.

Writes out/spheric_test2/simulation.csv (time, P1..P8, H1..H4) and, if the
experimental data are present (run fetch_data.py first), the comparison plot
validation/spheric_test2/comparison.png and error metrics.

  PYTHONPATH=build/python python validation/spheric_test2/run.py --dx 0.02
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

import CryoSPH as cs

HERE = os.path.dirname(os.path.abspath(__file__))
X_FRONT = 0.744
BOX = (0.161, 0.403, 0.161)
SENSOR_Y = BOX[1] / 2 - 0.176
OFFS = (0.021, 0.061, 0.101, 0.141)
H_X = (0.496, 0.992, 1.488, 2.638)


def build(dx, free_slip=False, clamp=False, alpha=0.02, delta=0.1):
    water = cs.FluidProperties.water()
    water.mu_liquid = 1.0e-3
    cfg = cs.SolverConfig()
    cfg.dx = dx
    cfg.gravity = (0.0, 0.0, -9.81)
    cfg.surface_tension = cs.SurfaceTensionModel.NoTension
    cfg.thermal = False
    cfg.phase_change = False
    cfg.density_mode = cs.DensityMode.Continuity
    cfg.velocity_scale = (2 * 9.81 * 0.55) ** 0.5
    cfg.artificial_viscosity = alpha
    cfg.delta_sph = delta
    cfg.no_slip = not free_slip
    cfg.clamp_negative_pressure = clamp
    solver = cs.Solver(cfg, water)

    tank = cs.Translated(cs.BoxTank(3.22, 1.0, 1.0), (1.61, 0.0, 0.5))
    box = cs.Translated(cs.BoxTank(*BOX), (X_FRONT - BOX[0] / 2, 0.0, BOX[2] / 2))
    geo = cs.Difference(tank, box)
    fill = cs.FillSpec()
    fill.mode = cs.FillMode.Full
    solver.build_tank(geo, fill)
    p = solver.particles()
    x, ph = p["position"], p["phase"]
    keep = (ph == 2) | ((x[:, 0] > 3.22 - 1.228) & (x[:, 2] < 0.55))
    pad = 4 * dx
    solver.set_particles(x[keep], p["mass"][keep], p["density"][keep], p["temperature"][keep], ph[keep],
                         domain_lo=(-pad, -0.5 - pad, -pad), domain_hi=(3.22 + pad, 0.5 + pad, 1.0 + pad),
                         normal=p["normal"][keep])
    return solver


def sensors(dx):
    front = [(X_FRONT + 0.5 * dx, SENSOR_Y, z) for z in OFFS]
    top = [(X_FRONT - o, SENSOR_Y, BOX[2] + 0.5 * dx) for o in OFFS]
    return np.array(front + top)


def heights(parts, dx):
    x, ph = parts["position"], parts["phase"]
    liq = x[ph == 0]
    out = []
    for xp in H_X:
        near = (np.abs(liq[:, 0] - xp) < dx) & (np.abs(liq[:, 1]) < dx)
        out.append(float(liq[near, 2].max() + 0.5 * dx) if near.any() else 0.0)
    return out


def smooth(t, y, window):
    """Centred moving average over `window` seconds (applied to both signals)."""
    if window <= 0:
        return y
    dt = np.median(np.diff(t))
    n = max(1, int(round(window / dt)))
    k = np.ones(n) / n
    return np.convolve(np.nan_to_num(y), k, mode="same")


def compare(sim_csv, exp_csv, png, window=0.02):
    sim = np.genfromtxt(sim_csv, delimiter=",", names=True)
    exp = np.genfromtxt(exp_csv, delimiter=",", names=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("H2", "water height H2 [m]"), ("H4", "water height H4 [m]"),
              ("P1", "pressure P1 [Pa]"), ("P3", "pressure P3 [Pa]"),
              ("P5", "pressure P5 [Pa]"), ("P7", "pressure P7 [Pa]")]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    metrics = {}
    for ax, (key, label) in zip(axes.flat, panels):
        ys = smooth(sim["time"], sim[key], window)
        ye = smooth(exp["time"], exp[key], window)
        ax.plot(exp["time"], ye, color="#9aa3b2", lw=1.0, label="experiment (MARIN)")
        ax.plot(sim["time"], ys, color="#2a6fdb", lw=1.4, label="CryoSPH-CUDA")
        ax.set_xlim(0, 6)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        # error metrics on the common time base (experiment resampled to the simulation)
        t = sim["time"][(sim["time"] > 0) & (sim["time"] <= min(6.0, exp["time"].max()))]
        e = np.interp(t, exp["time"], ye)
        s = np.interp(t, sim["time"], ys)
        ok = np.isfinite(s)
        rng = np.nanmax(e) - np.nanmin(e)
        metrics[key] = {
            "nrmse": float(np.sqrt(np.mean((s[ok] - e[ok]) ** 2)) / rng),
            "peak_exp": float(np.nanmax(e)), "peak_sim": float(np.nanmax(s[ok])),
            "t_peak_exp": float(t[np.nanargmax(e)]), "t_peak_sim": float(t[ok][np.nanargmax(s[ok])]),
        }
    axes[0, 0].legend(frameon=False)
    fig.suptitle("SPHERIC Test 2 — 3D dam break with obstacle (Kleefsman et al. 2005); "
                 f"both signals {window * 1e3:.0f} ms moving average")
    fig.savefig(png, dpi=150)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--t-end", type=float, default=6.0)
    ap.add_argument("--sample", type=float, default=0.005, help="probe sampling interval [s]")
    ap.add_argument("--free-slip", action="store_true", help="free-slip walls (default: no-slip)")
    ap.add_argument("--clamp", action="store_true", help="clamp negative pressure to zero")
    ap.add_argument("--alpha", type=float, default=0.02, help="artificial viscosity")
    ap.add_argument("--window", type=float, default=0.02, help="moving-average window for the comparison [s]")
    ap.add_argument("--compare-only", action="store_true", help="re-plot an existing simulation.csv")
    ap.add_argument("--delta", type=float, default=0.1, help="delta-SPH density diffusion")
    ap.add_argument("--out", default="out/spheric_test2")
    ap.add_argument("--frames-every", type=float, default=0.0, help="write viewer frames every N s (0 = off)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    exp_csv = os.path.join(HERE, "data", "experiment.csv")
    sim_csv = os.path.join(args.out, "simulation.csv")
    if args.compare_only:
        for k, m in compare(sim_csv, exp_csv, os.path.join(HERE, "comparison.png"), args.window).items():
            print(f"{k}: NRMSE {m['nrmse']:.3f}  peak exp {m['peak_exp']:.3g} @ {m['t_peak_exp']:.2f}s  "
                  f"sim {m['peak_sim']:.3g} @ {m['t_peak_sim']:.2f}s")
        return 0
    solver = build(args.dx, args.free_slip, args.clamp, args.alpha, args.delta)
    pts = sensors(args.dx)
    print(f"{solver.num_particles} particles on {solver.device}; c0 = {solver.sound_speed:.1f} m/s", flush=True)
    t0 = time.perf_counter()
    frames, next_frame = [], 0.0
    with open(sim_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + [f"P{i}" for i in range(1, 9)] + [f"H{i}" for i in range(1, 5)])
        k = 0
        while True:
            t = k * args.sample
            if t > args.t_end + 1e-9:
                break
            solver.run_until(t)
            p = solver.probe_pressure(pts)
            parts = solver.particles()
            w.writerow([f"{solver.time:.5f}"] + [f"{v:.2f}" for v in np.nan_to_num(p, nan=0.0)] +
                       [f"{v:.4f}" for v in heights(parts, args.dx)])
            if args.frames_every > 0 and solver.time >= next_frame - 1e-9:
                name = f"frame_{len(frames):05d}.json"
                solver.write_json(os.path.join(args.out, name))
                frames.append({"file": name, "time": solver.time})
                next_frame += args.frames_every
            if k % 100 == 0:
                print(f"t={solver.time:5.2f}s steps={solver.step_count:6d} wall={time.perf_counter() - t0:6.0f}s "
                      f"P1={p[0]:8.1f} Pa", flush=True)
            k += 1
    if frames:
        import json
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump({"title": "SPHERIC Test 2 dam break", "frames": frames}, f)
    print(f"wrote {sim_csv} in {time.perf_counter() - t0:.0f} s")

    if os.path.exists(exp_csv):
        metrics = compare(sim_csv, exp_csv, os.path.join(HERE, "comparison.png"), args.window)
        for k, m in metrics.items():
            print(f"{k}: NRMSE {m['nrmse']:.3f}  peak exp {m['peak_exp']:.3g} @ {m['t_peak_exp']:.2f}s  "
                  f"sim {m['peak_sim']:.3g} @ {m['t_peak_sim']:.2f}s")
    else:
        print("no experimental data: run validation/spheric_test2/fetch_data.py to add the comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
