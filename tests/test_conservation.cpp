// Conservation through heat conduction and Lee-model phase change in a closed,
// adiabatic tank: total mass and total enthalpy must be invariant while mass
// moves between the phases.
#include <gtest/gtest.h>

#include <cmath>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

SolverConfig base_config() {
  SolverConfig cfg;
  cfg.dx = real(0.004);
  cfg.gravity = Vec3{0, 0, 0};
  cfg.surface_tension = SurfaceTensionModel::CSF;
  cfg.contact_angle_deg = 5;
  cfg.thermo_mode = TankThermoMode::Closed;
  cfg.wall_thermal_bc = WallThermalBC::Adiabatic;
  cfg.velocity_scale = 1;  // phase change drives the flow: c0 = 10 m/s
  return cfg;
}

TEST(Conservation, HeatConductionConservesEnergy) {
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  SolverConfig cfg = base_config();
  cfg.phase_change = false;
  // Fast diffusion: the diffusion length after the run is ~2 dx, so the
  // discrete operator is resolved (the time step stays CFL-limited).
  cfg.conductivity_scale = 2e4;
  Solver solver(cfg, ln2);
  // Two liquid blocks at different temperatures, no walls needed in zero g.
  HostParticles hp = make_block(Vec3{0, 0, 0}, Vec3{real(0.04), real(0.04), real(0.02)}, cfg.dx, Phase::Liquid,
                                ln2.rho_liquid, ln2.T_sat - 4);
  hp.append(make_block(Vec3{0, 0, real(0.02)}, Vec3{real(0.04), real(0.04), real(0.04)}, cfg.dx, Phase::Liquid,
                       ln2.rho_liquid, ln2.T_sat - 2));
  solver.set_particles(hp, AABB{Vec3{-0.01f, -0.01f, -0.01f}, Vec3{0.05f, 0.05f, 0.05f}});
  const Diagnostics d0 = solver.diagnostics();
  solver.run_steps(300);
  const Diagnostics d1 = solver.diagnostics();
  EXPECT_NEAR(d1.mass_total, d0.mass_total, 1e-6 * d0.mass_total);
  const double scale = d0.mass_total * ln2.cp_liquid * 4.0;
  EXPECT_NEAR(d1.enthalpy, d0.enthalpy, 2e-3 * scale);
  // Heat flowed from the warm upper block into the cold lower one.
  EXPECT_NEAR(d1.T_liquid_mean, d0.T_liquid_mean, 0.01);
  const HostParticles out = solver.particles();
  double t_lo = 0, t_hi = 0;
  int n_lo = 0, n_hi = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (out.x[i].z < real(0.02)) {
      t_lo += out.T[i];
      ++n_lo;
    } else {
      t_hi += out.T[i];
      ++n_hi;
    }
  }
  // Two semi-infinite slabs at T1, T2 in contact: each block's mean changes by
  //   dT = (T2 - T1) sqrt(alpha t / pi) / L.
  const double alpha = cfg.conductivity_scale * ln2.k_liquid / (double(ln2.rho_liquid) * ln2.cp_liquid);
  const double expected = 2.0 * std::sqrt(alpha * d1.time / 3.14159265358979323846) / 0.02;
  EXPECT_NEAR((t_lo / n_lo - (ln2.T_sat - 4)) / expected, 1.0, 0.3);
  EXPECT_NEAR(((ln2.T_sat - 2) - t_hi / n_hi) / expected, 1.0, 0.3);
}

TEST(Conservation, PhaseChangeConservesMassAndEnthalpy) {
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  SolverConfig cfg = base_config();
  cfg.lee_evaporation = 2;
  cfg.lee_condensation = 2;
  Solver solver(cfg, ln2);
  const BoxTank box(0.04, 0.04, 0.04);
  FillSpec fill;
  fill.fill_fraction = 0.5;
  fill.T_liquid = ln2.T_sat + 3;  // superheated liquid -> evaporation
  fill.T_vapor = ln2.T_sat;
  solver.build_tank(box, fill);
  solver.relax(0.03);

  const Diagnostics d0 = solver.diagnostics();
  solver.run_until(0.04);
  const Diagnostics d1 = solver.diagnostics();

  ASSERT_TRUE(std::isfinite(d1.enthalpy));
  EXPECT_LT(d1.max_speed, 2.0);  // bounded: no runaway of the vapour acoustics
  EXPECT_GT(d1.evaporated_mass, 0.0);
  EXPECT_GT(d1.mass_vapor, d0.mass_vapor);
  EXPECT_LT(d1.mass_liquid, d0.mass_liquid);
  EXPECT_NEAR(d1.mass_total, d0.mass_total, 1e-5 * d0.mass_total);
  // Energy budget: the evaporated mass alone carries h_fg * m_evap, so a
  // missing latent term would be a 100 % error on that scale.  Temperatures
  // are float32 (~6e-8 relative), hence the round-off allowance on m cp T.
  const double latent = d1.evaporated_mass * ln2.h_fg;
  const double roundoff = 1e-6 * d0.mass_total * ln2.cp_liquid * ln2.T_sat;
  EXPECT_NEAR(d1.enthalpy, d0.enthalpy, 0.02 * latent + roundoff);
  EXPECT_GT(latent, 10 * roundoff);  // the check above is meaningful
  // The liquid cools towards saturation.
  EXPECT_LT(d1.T_liquid_mean, d0.T_liquid_mean);
}

TEST(Conservation, VentingRemovesExcessVapour) {
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  SolverConfig cfg = base_config();
  cfg.thermo_mode = TankThermoMode::Vented;
  cfg.vent_relaxation = real(0.2);
  Solver solver(cfg, ln2);
  const BoxTank box(0.04, 0.04, 0.04);
  FillSpec fill;
  fill.fill_fraction = 0.5;
  fill.T_liquid = ln2.T_sat + 3;
  solver.build_tank(box, fill);
  solver.relax(0.03);
  const Diagnostics d0 = solver.diagnostics();
  solver.run_until(0.03);
  const Diagnostics d1 = solver.diagnostics();
  // Everything that left the liquid is either still vapour or was vented.
  EXPECT_NEAR(d1.mass_total + d1.vented_mass, d0.mass_total, 1e-5 * d0.mass_total);
  EXPECT_GT(d1.vented_mass, 0.0);
  const double roundoff = 1e-6 * d0.mass_total * ln2.cp_liquid * ln2.T_sat;
  EXPECT_NEAR(d1.enthalpy + d1.vented_enthalpy, d0.enthalpy, 0.02 * d1.evaporated_mass * ln2.h_fg + roundoff);
}

}  // namespace
