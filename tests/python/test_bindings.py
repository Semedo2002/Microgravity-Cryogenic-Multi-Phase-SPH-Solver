"""Smoke tests for the CryoSPH Python module.

Run with the built module on the path:
    PYTHONPATH=build/python python -m pytest tests/python -q
"""
import json
import math

import numpy as np
import pytest

import CryoSPH as cs


def small_config(**overrides):
    cfg = cs.SolverConfig()
    cfg.dx = 0.005
    cfg.gravity = (0.0, 0.0, 0.0)
    cfg.velocity_scale = 1.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_module_metadata():
    assert cs.__version__
    assert isinstance(cs.cuda_backend, bool)
    assert cs.device_name()


def test_fluid_presets_and_clausius_clapeyron():
    ln2 = cs.FluidProperties.liquid_nitrogen()
    assert ln2.name == "LN2"
    assert ln2.rho_liquid / ln2.rho_vapor > 100
    assert ln2.saturation_temperature(ln2.p_sat) == pytest.approx(ln2.T_sat, rel=1e-6)
    p2 = ln2.saturation_pressure(ln2.T_sat + 2.0)
    assert p2 > ln2.p_sat
    assert ln2.saturation_temperature(p2) == pytest.approx(ln2.T_sat + 2.0, rel=1e-5)


def test_geometry_sdf():
    tank = cs.CylinderTank(radius=0.1, height=0.3)
    assert tank.sdf(0, 0, 0) == pytest.approx(-0.1)
    assert tank.volume() == pytest.approx(math.pi * 0.01 * 0.3)
    lo, hi = tank.bounds
    assert lo[2] == pytest.approx(-0.15) and hi[0] == pytest.approx(0.1)


def test_config_roundtrip():
    cfg = small_config(contact_angle_deg=12.5, surface_tension=cs.SurfaceTensionModel.Akinci)
    assert cfg.gravity == (0.0, 0.0, 0.0)
    assert cfg.contact_angle_deg == 12.5
    assert cfg.surface_tension == cs.SurfaceTensionModel.Akinci


def test_tank_run_conserves_mass(tmp_path):
    ln2 = cs.FluidProperties.liquid_nitrogen()
    solver = cs.Solver(small_config(thermo_mode=cs.TankThermoMode.Closed, lee_evaporation=2.0), ln2)
    fill = cs.FillSpec()
    fill.fill_fraction = 0.5
    fill.T_liquid = ln2.T_sat + 2.0
    solver.build_tank(cs.BoxTank(0.04, 0.04, 0.04), fill)
    solver.relax(0.01)
    d0 = solver.diagnostics()
    solver.run_steps(100)
    d1 = solver.diagnostics()
    assert d1.step == 100
    assert d1.mass_total == pytest.approx(d0.mass_total, rel=1e-5)
    assert d1.evaporated_mass > 0
    assert d1.p_ullage > d0.p_ullage

    parts = solver.particles()
    n = solver.num_particles
    assert parts["position"].shape == (n, 3)
    assert parts["phase"].dtype == np.uint8
    assert set(np.unique(parts["phase"])) <= {0, 1, 2}
    assert solver.pressure().shape == (n,)
    assert np.isfinite(parts["temperature"]).all()

    path = tmp_path / "frame.json"
    solver.write_json(str(path))
    frame = json.loads(path.read_text())
    assert frame["step"] == 100
    assert len(frame["position"]) == 3 * frame["n"]
    assert 2 not in frame["phase"]  # walls omitted by default
    solver.write_vtk(str(tmp_path / "frame.vtk"))
    assert (tmp_path / "frame.vtk").stat().st_size > 0


def test_custom_particles_and_telemetry():
    ln2 = cs.FluidProperties.liquid_nitrogen()
    solver = cs.Solver(small_config(thermal=False, phase_change=False,
                                    surface_tension=cs.SurfaceTensionModel.NoTension), ln2)
    dx = 0.005
    g = (np.arange(6) + 0.5) * dx
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    n = len(pos)
    rho = np.full(n, ln2.rho_liquid)
    solver.set_particles(pos, rho * dx**3, rho, np.full(n, ln2.T_sat), np.zeros(n, np.uint8),
                         domain_lo=(-0.01, -0.01, -0.01), domain_hi=(0.04, 0.04, 0.04))
    assert solver.num_particles == n
    for mode in (cs.TelemetryMode.AsyncCopy, cs.TelemetryMode.ZeroCopy):
        solver.configure_telemetry(mode)
        solver.run_steps(3)
        assert solver.request_telemetry()
        assert not solver.request_telemetry()  # one frame in flight at a time
        frame = solver.fetch_telemetry()
        assert frame["position"].shape == (n, 3)
        assert frame["step"] == solver.step_count
        assert solver.fetch_telemetry() is None


def test_bad_inputs_raise():
    ln2 = cs.FluidProperties.liquid_nitrogen()
    with pytest.raises(ValueError):
        cs.Solver(small_config(contact_angle_deg=200.0), ln2)
    solver = cs.Solver(small_config(), ln2)
    with pytest.raises(ValueError):
        solver.set_particles(np.zeros((4, 2)), np.ones(4), np.ones(4), np.ones(4), np.zeros(4, np.uint8),
                             domain_lo=(0, 0, 0), domain_hi=(1, 1, 1))
    with pytest.raises(ValueError):
        solver.set_contact_angle(-5.0)


def test_csg_geometry_and_probes():
    tank = cs.Translated(cs.BoxTank(0.1, 0.1, 0.1), (0.05, 0.0, 0.05))
    block = cs.Translated(cs.BoxTank(0.02, 0.02, 0.02), (0.05, 0.0, 0.01))
    geo = cs.Difference(tank, block)
    assert geo.sdf(0.05, 0.0, 0.01) > 0      # inside the obstacle: solid
    assert geo.sdf(0.02, 0.0, 0.05) < 0      # open fluid region
    assert cs.Union(tank, block).sdf(0.05, 0.0, 0.01) < 0
    water = cs.FluidProperties.water()
    cfg = small_config(surface_tension=cs.SurfaceTensionModel.NoTension, thermal=False, phase_change=False,
                       gravity=(0.0, 0.0, -9.81))
    cfg.dx = 0.01
    solver = cs.Solver(cfg, water)
    fill = cs.FillSpec(); fill.mode = cs.FillMode.Full
    solver.build_tank(geo, fill)
    solver.run_steps(5)
    p = solver.probe_pressure(np.array([[0.02, 0.0, 0.02], [0.02, 0.0, 0.08], [0.5, 0.5, 0.5]]))
    assert p.shape == (3,)
    assert np.isfinite(p[0]) and np.isfinite(p[1])
    assert np.isnan(p[2])                     # far outside the tank: no liquid in range
    assert p[0] > p[1]                        # hydrostatic: deeper probe reads higher


def test_maneuver_profile_and_slosh_loads():
    prof = cs.ManeuverProfile([0.0, 1.0], np.array([[0, 0, 0], [2.0, 0, 0]]),
                              np.array([[0, 0, 0], [0, 0, 1.0]]), origin=(0, 0, 0))
    a, w, wd = prof.evaluate(0.5)
    assert a == pytest.approx((1.0, 0.0, 0.0))
    assert w[2] == pytest.approx(0.5) and wd[2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cs.ManeuverProfile([0.0, 0.0], np.zeros((2, 3)))
    water = cs.FluidProperties.water()
    cfg = small_config(surface_tension=cs.SurfaceTensionModel.NoTension, thermal=False, phase_change=False)
    cfg.dx = 0.01
    solver = cs.Solver(cfg, water)
    fill = cs.FillSpec(); fill.mode = cs.FillMode.Full
    solver.build_tank(cs.BoxTank(0.06, 0.06, 0.06), fill)
    solver.set_frame_motion(acceleration=(1.0, 0.0, 0.0))
    solver.run_steps(20)
    d = solver.diagnostics()
    assert d.frame_acceleration == pytest.approx((1.0, 0.0, 0.0))
    assert d.slosh_force[0] < 0               # liquid pushes on the trailing wall
    assert set(d.as_dict()) >= {"slosh_force", "slosh_moment"}
