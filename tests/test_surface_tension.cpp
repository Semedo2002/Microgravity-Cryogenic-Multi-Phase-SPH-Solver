// Surface tension: a static spherical drop in vapour at zero gravity must
// develop the Laplace pressure jump  dp = 2 sigma / R.
#include <gtest/gtest.h>

#include <cmath>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

TEST(SurfaceTension, LaplacePressureJump) {
  FluidProperties fl = FluidProperties::water();
  fl.rho_vapor = 100;  // moderate density ratio keeps the test small and stable
  fl.mu_liquid = real(0.05);
  fl.mu_vapor = real(0.005);
  fl.sigma = real(1.0);
  SolverConfig cfg;
  cfg.dx = real(0.002);
  cfg.gravity = Vec3{0, 0, 0};
  cfg.thermal = false;
  cfg.phase_change = false;
  cfg.surface_tension = SurfaceTensionModel::CSF;
  cfg.artificial_viscosity = real(0.05);
  Solver solver(cfg, fl);

  const real R = real(0.012);
  const real L = real(0.048);
  const real dx = cfg.dx;
  HostParticles hp;
  const int n = static_cast<int>(std::lround(L / dx));
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const Vec3 x{(i + real(0.5)) * dx - L / 2, (j + real(0.5)) * dx - L / 2, (k + real(0.5)) * dx - L / 2};
        const bool liquid = norm(x) < R;
        const real rho = liquid ? fl.rho_liquid : fl.rho_vapor;
        hp.push(x, Vec3{0, 0, 0}, rho * dx * dx * dx, rho, fl.T_sat, liquid ? Phase::Liquid : Phase::Vapor);
      }
  // Enclose in three wall layers.
  for (int k = -3; k < n + 3; ++k)
    for (int j = -3; j < n + 3; ++j)
      for (int i = -3; i < n + 3; ++i) {
        if (i >= 0 && i < n && j >= 0 && j < n && k >= 0 && k < n) continue;
        const Vec3 x{(i + real(0.5)) * dx - L / 2, (j + real(0.5)) * dx - L / 2, (k + real(0.5)) * dx - L / 2};
        const Vec3 nrm = normalized(Vec3{i < 0 ? real(1) : (i >= n ? real(-1) : real(0)),
                                         j < 0 ? real(1) : (j >= n ? real(-1) : real(0)),
                                         k < 0 ? real(1) : (k >= n ? real(-1) : real(0))});
        hp.push(x, Vec3{0, 0, 0}, fl.rho_liquid * dx * dx * dx, fl.rho_liquid, fl.T_sat, Phase::Wall, nrm);
      }
  const real pad = L / 2 + 4 * dx;
  solver.set_particles(hp, AABB{Vec3{-pad, -pad, -pad}, Vec3{pad, pad, pad}});

  for (int k = 0; k < 30; ++k) {
    solver.run_until(solver.time() + 0.004);
    solver.damp_velocities(real(0.7));
  }
  solver.run_until(solver.time() + 0.02);

  const HostParticles out = solver.particles();
  const std::vector<real> p = solver.pressure();
  double pin = 0, pout = 0;
  int nin = 0, nout = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double r = norm(out.x[i]);
    if (out.phase[i] == kLiquid && r < 0.5 * R) {
      pin += p[i];
      ++nin;
    } else if (out.phase[i] == kVapor && r > 1.6 * R && r < 2.0 * R) {
      pout += p[i];
      ++nout;
    }
  }
  ASSERT_GT(nin, 10);
  ASSERT_GT(nout, 10);
  const double dp = pin / nin - pout / nout;
  const double laplace = 2.0 * fl.sigma / R;
  EXPECT_NEAR(dp / laplace, 1.0, 0.3) << "dp = " << dp << " Pa, Laplace = " << laplace << " Pa";
  const Diagnostics d = solver.diagnostics();
  EXPECT_TRUE(std::isfinite(d.kinetic_energy));
}

}  // namespace
