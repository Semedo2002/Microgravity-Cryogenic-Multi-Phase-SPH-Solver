// Manoeuvre-driven tanks: prescribed frame motion and slosh loads, checked
// against analytic rigid-body / hydrostatic answers.
#include <gtest/gtest.h>

#include <cmath>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

SolverConfig water_config(real velocity_scale, real dx = real(0.01)) {
  SolverConfig cfg;
  cfg.dx = dx;
  cfg.gravity = Vec3{0, 0, 0};
  cfg.surface_tension = SurfaceTensionModel::None;
  cfg.thermal = false;
  cfg.phase_change = false;
  cfg.artificial_viscosity = real(0.1);
  cfg.velocity_scale = velocity_scale;
  return cfg;
}

/// Damped settling: the fluid comes to rest in the (possibly moving) tank frame.
void settle(Solver& solver, int cycles) {
  for (int k = 0; k < cycles; ++k) {
    solver.run_until(solver.time() + 0.02);
    solver.damp_velocities(real(0.5));
  }
  solver.run_until(solver.time() + 0.05);
}

TEST(Maneuver, ProfileInterpolation) {
  ManeuverProfile p;
  p.time = {0.0, 1.0, 3.0};
  p.acceleration = {Vec3{0, 0, 0}, Vec3{2, 0, 0}, Vec3{2, 0, 0}};
  p.angular_velocity = {Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{0, 0, 0}};
  Vec3 a, w, wd;
  p.evaluate(0.5, a, w, wd);
  EXPECT_NEAR(a.x, 1.0, 1e-6);
  EXPECT_NEAR(w.z, 0.5, 1e-6);
  EXPECT_NEAR(wd.z, 1.0, 1e-6);
  p.evaluate(2.0, a, w, wd);
  EXPECT_NEAR(w.z, 0.5, 1e-6);
  EXPECT_NEAR(wd.z, -0.5, 1e-6);
  p.evaluate(10.0, a, w, wd);  // held after the table
  EXPECT_NEAR(a.x, 2.0, 1e-6);
  EXPECT_NEAR(wd.z, 0.0, 1e-6);
  ManeuverProfile bad = p;
  bad.time = {0.0, 0.0, 1.0};
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(Maneuver, GravityAndLateralAccelerationLoads) {
  // A full, settled water box loads the tank with exactly M (g - a).
  const FluidProperties water = FluidProperties::water();
  Solver solver(water_config(real(1.0)), water);
  FillSpec fill;
  fill.mode = FillMode::Full;
  solver.build_tank(BoxTank(0.1, 0.1, 0.1), fill);

  solver.set_gravity(Vec3{0, 0, real(-9.81)});
  settle(solver, 30);
  Diagnostics d = solver.diagnostics();
  const double M = d.mass_liquid;
  EXPECT_NEAR(d.slosh_force.z / (-M * 9.81), 1.0, 0.03) << "Fz = " << d.slosh_force.z;
  EXPECT_NEAR(d.slosh_force.x, 0.0, 0.01 * M * 9.81);

  solver.set_gravity(Vec3{0, 0, 0});
  solver.set_frame_motion(Vec3{real(2.0), 0, 0}, Vec3{0, 0, 0}, Vec3{0, 0, 0}, Vec3{0, 0, 0});
  settle(solver, 30);
  d = solver.diagnostics();
  EXPECT_NEAR(d.slosh_force.x / (-M * 2.0), 1.0, 0.03) << "Fx = " << d.slosh_force.x;
  EXPECT_NEAR(d.frame_acceleration.x, 2.0, 1e-6);
}

TEST(Maneuver, SteadyRotationGivesCentrifugalPressure) {
  // Spinning full cylinder: p = p0 + rho w^2 r^2 / 2. Converges with resolution
  // (7.1 / 3.7 / 1.2 % error at R/dx = 5 / 7.5 / 10), so R = 10 dx here.
  const FluidProperties water = FluidProperties::water();
  const real omega = 20, R = real(0.05);
  Solver solver(water_config(omega * R, real(0.005)), water);
  FillSpec fill;
  fill.mode = FillMode::Full;
  solver.build_tank(CylinderTank(R, 0.05), fill);
  solver.set_frame_motion(Vec3{0, 0, 0}, Vec3{0, 0, omega}, Vec3{0, 0, 0}, Vec3{0, 0, 0});
  settle(solver, 40);

  const HostParticles hp = solver.particles();
  const std::vector<real> p = solver.pressure();
  // Least-squares fit p = p0 + k r^2 over interior particles.
  double s1 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < hp.size(); ++i) {
    if (hp.phase[i] != kLiquid) continue;
    const double r2 = double(hp.x[i].x) * hp.x[i].x + double(hp.x[i].y) * hp.x[i].y;
    if (r2 > 0.035 * 0.035 || std::abs(hp.x[i].z) > 0.012) continue;
    s1 += 1;
    sx += r2;
    sy += p[i];
    sxx += r2 * r2;
    sxy += r2 * p[i];
  }
  ASSERT_GT(s1, 300);
  const double k = (s1 * sxy - sx * sy) / (s1 * sxx - sx * sx);
  const double expected = 0.5 * water.rho_liquid * omega * omega;
  EXPECT_NEAR(k / expected, 1.0, 0.04) << "dp/d(r^2) = " << k << ", expected " << expected;

  const Diagnostics d = solver.diagnostics();
  const double scale = d.mass_liquid * omega * omega * R;  // centrifugal load magnitude
  EXPECT_NEAR(d.slosh_force.x, 0.0, 0.02 * scale);
  EXPECT_NEAR(d.slosh_force.y, 0.0, 0.02 * scale);
  EXPECT_NEAR(d.slosh_moment.z, 0.0, 0.02 * scale * R);
}

}  // namespace
