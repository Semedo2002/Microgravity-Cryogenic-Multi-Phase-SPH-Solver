// Transport-velocity formulation (Adami et al. 2013): momentum conservation of
// the physical velocity, and a buoyant bubble that rises without particles
// leaving the tank.
#include <gtest/gtest.h>

#include <cmath>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

TEST(TransportVelocity, ConservesLinearMomentumWithoutWalls) {
  // Free liquid block in zero g with a shearing velocity field: all pairwise
  // terms (pressure, viscosity, TVF extra stress) are antisymmetric, so the
  // total momentum must not change although every particle is accelerated.
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  SolverConfig cfg;
  cfg.dx = real(0.004);
  cfg.gravity = Vec3{0, 0, 0};
  cfg.surface_tension = SurfaceTensionModel::None;
  cfg.thermal = false;
  cfg.phase_change = false;
  cfg.velocity_scale = real(0.5);
  Solver solver(cfg, ln2);
  HostParticles hp = make_block(Vec3{0, 0, 0}, Vec3{real(0.04), real(0.04), real(0.04)}, cfg.dx, Phase::Liquid,
                                ln2.rho_liquid, ln2.T_sat);
  for (std::size_t i = 0; i < hp.size(); ++i)
    hp.v[i] = Vec3{real(0.2) * std::sin(real(150) * hp.x[i].z), real(0.1) * std::cos(real(120) * hp.x[i].x), 0};
  solver.set_particles(hp, AABB{Vec3{-0.02f, -0.02f, -0.02f}, Vec3{0.06f, 0.06f, 0.06f}});
  const Diagnostics d0 = solver.diagnostics();
  solver.run_steps(200);
  const Diagnostics d1 = solver.diagnostics();
  const double p_scale = d0.mass_liquid * 0.2;  // |m v| scale of the initial field
  EXPECT_NEAR(d1.liquid_momentum.x, d0.liquid_momentum.x, 1e-4 * p_scale);
  EXPECT_NEAR(d1.liquid_momentum.y, d0.liquid_momentum.y, 1e-4 * p_scale);
  EXPECT_NEAR(d1.liquid_momentum.z, d0.liquid_momentum.z, 1e-4 * p_scale);
  EXPECT_NEAR(d1.mass_total, d0.mass_total, 1e-6 * d0.mass_total);
}

TEST(TransportVelocity, BubbleRisesAndStaysInsideTank) {
  // LH2 capsule at 1 g with a central vapour bubble: the bubble must rise
  // (Davies-Taylor with wall correction ~0.25 m/s) and no particle may leave.
  const FluidProperties lh2 = FluidProperties::liquid_hydrogen();
  SolverConfig cfg;
  cfg.dx = real(0.008);
  cfg.gravity = Vec3{0, 0, 0};
  cfg.surface_tension = SurfaceTensionModel::None;
  cfg.thermal = false;
  cfg.phase_change = false;
  cfg.velocity_scale = std::sqrt(real(2 * 9.81 * 0.2));
  Solver solver(cfg, lh2);
  const CapsuleTank tank(0.06, 0.10);
  FillSpec fill;
  fill.mode = FillMode::CentralBubble;
  fill.fill_fraction = 0.9;
  solver.build_tank(tank, fill);
  solver.relax(0.02);
  solver.set_gravity(Vec3{0, 0, real(-9.81)});

  auto bubble_z = [&]() {
    const HostParticles hp = solver.particles();
    double mz = 0, m = 0;
    for (std::size_t i = 0; i < hp.size(); ++i)
      if (hp.phase[i] == kVapor) {
        mz += double(hp.m[i]) * hp.x[i].z;
        m += hp.m[i];
      }
    return mz / m;
  };
  const double z0 = bubble_z();
  solver.run_until(0.2);
  const double rise = bubble_z() - z0;
  EXPECT_GT(rise, 0.025) << "bubble rise " << rise * 1e3 << " mm";
  EXPECT_LT(rise, 0.09) << "bubble rise " << rise * 1e3 << " mm";

  const HostParticles hp = solver.particles();
  int outside = 0;
  for (std::size_t i = 0; i < hp.size(); ++i)
    if (hp.phase[i] != kWall && tank.sdf(hp.x[i].x, hp.x[i].y, hp.x[i].z) > 2.0 * cfg.dx) ++outside;
  EXPECT_EQ(outside, 0);
  EXPECT_LT(solver.diagnostics().max_speed, 3.0);
}

}  // namespace
