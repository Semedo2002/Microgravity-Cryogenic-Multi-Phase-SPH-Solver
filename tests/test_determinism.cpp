// Reproducibility: two identical runs must give bitwise-identical states.
// Guards against stream-ordering races (memsets on the default stream racing
// solver kernels) and scheduling-dependent writes (atomic slot assignment).
#include <gtest/gtest.h>

#include <cstring>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

HostParticles run_once(int steps, Diagnostics& d) {
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  SolverConfig cfg;
  cfg.dx = real(0.004);
  cfg.gravity = Vec3{0, 0, real(-0.5)};
  cfg.contact_angle_deg = 10;
  cfg.velocity_scale = 1;
  cfg.thermo_mode = TankThermoMode::Closed;
  cfg.lee_evaporation = 5;
  cfg.lee_condensation = 5;
  cfg.split_ratio = real(1.05);  // make splitting happen despite vapour mass rebalancing
  Solver solver(cfg, ln2);
  FillSpec fill;
  fill.fill_fraction = 0.5;
  fill.T_liquid = ln2.T_sat + 3;
  solver.build_tank(BoxTank(0.04, 0.04, 0.04), fill);
  solver.run_steps(steps);
  d = solver.diagnostics();
  return solver.particles();
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

TEST(Determinism, IdenticalRunsAreBitwiseEqual) {
  Diagnostics d1, d2;
  const HostParticles a = run_once(300, d1);
  const HostParticles b = run_once(300, d2);
  EXPECT_GT(d1.splits, 0) << "the scenario should exercise particle splitting";
  ASSERT_EQ(a.size(), b.size());
  EXPECT_TRUE(same_bits(a.x, b.x));
  EXPECT_TRUE(same_bits(a.v, b.v));
  EXPECT_TRUE(same_bits(a.m, b.m));
  EXPECT_TRUE(same_bits(a.T, b.T));
  EXPECT_TRUE(same_bits(a.id, b.id));
  EXPECT_EQ(d1.splits, d2.splits);
  EXPECT_EQ(d1.mass_total, d2.mass_total);
  EXPECT_EQ(d1.enthalpy, d2.enthalpy);
}

}  // namespace
