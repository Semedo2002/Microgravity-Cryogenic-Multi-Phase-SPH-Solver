// Hydrostatic equilibrium in a closed box: after relaxation the vertical
// pressure gradient must equal rho g and the fluid must come to rest.
#include <gtest/gtest.h>

#include <cmath>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

TEST(Hydrostatic, PressureGradientMatchesRhoG) {
  FluidProperties water = FluidProperties::water();
  SolverConfig cfg;
  cfg.dx = real(0.01);
  cfg.gravity = Vec3{0, 0, real(-9.81)};
  cfg.surface_tension = SurfaceTensionModel::None;
  cfg.thermal = false;
  cfg.phase_change = false;
  cfg.artificial_viscosity = real(0.1);
  Solver solver(cfg, water);

  const BoxTank box(0.1, 0.1, 0.2);
  FillSpec fill;
  fill.mode = FillMode::Full;
  solver.build_tank(box, fill);

  // Damped relaxation, then free evolution.
  for (int k = 0; k < 40; ++k) {
    solver.run_until(solver.time() + 0.02);
    solver.damp_velocities(real(0.5));
  }
  solver.run_until(solver.time() + 0.2);
  const Diagnostics d = solver.diagnostics();
  EXPECT_LT(d.max_speed, 0.05);
  EXPECT_TRUE(std::isfinite(d.kinetic_energy));

  // Least-squares slope dp/dz over interior liquid particles.
  const HostParticles hp = solver.particles();
  const std::vector<real> p = solver.pressure();
  double sz = 0, sp = 0, szz = 0, szp = 0;
  int n = 0;
  for (std::size_t i = 0; i < hp.size(); ++i) {
    if (hp.phase[i] != kLiquid) continue;
    const Vec3 x = hp.x[i];
    if (std::abs(x.x) > 0.03 || std::abs(x.y) > 0.03 || std::abs(x.z) > 0.07) continue;
    sz += x.z;
    sp += p[i];
    szz += double(x.z) * x.z;
    szp += double(x.z) * p[i];
    ++n;
  }
  ASSERT_GT(n, 50);
  const double slope = (n * szp - sz * sp) / (n * szz - sz * sz);
  const double expected = -double(water.rho_liquid) * 9.81;
  EXPECT_NEAR(slope / expected, 1.0, 0.05) << "dp/dz = " << slope << " expected " << expected;
}

}  // namespace
