#include "cryosph/physics/fluid_properties.hpp"

#include <cmath>

namespace cryosph {

real FluidProperties::saturation_temperature(real p) const {
  const double inv_T = 1.0 / T_sat - (double(R_gas) / h_fg) * std::log(double(p) / p_sat);
  return static_cast<real>(1.0 / inv_T);
}

real FluidProperties::saturation_pressure(real T) const {
  return static_cast<real>(p_sat * std::exp(double(h_fg) / R_gas * (1.0 / T_sat - 1.0 / double(T))));
}

FluidProperties FluidProperties::liquid_hydrogen() {
  FluidProperties f;
  f.name = "LH2";
  f.rho_liquid = 70.85f;  f.rho_vapor = 1.338f;
  f.mu_liquid = 1.33e-5f; f.mu_vapor = 1.11e-6f;
  f.k_liquid = 0.1034f;   f.k_vapor = 0.0166f;
  f.cp_liquid = 9770.f;   f.cp_vapor = 12200.f;
  f.sigma = 1.93e-3f;     f.h_fg = 445.6e3f;
  f.T_sat = 20.37f;       f.p_sat = 101325.f;
  f.R_gas = 4124.2f;
  return f;
}

FluidProperties FluidProperties::liquid_nitrogen() {
  FluidProperties f;
  f.name = "LN2";
  f.rho_liquid = 806.1f;  f.rho_vapor = 4.61f;
  f.mu_liquid = 1.61e-4f; f.mu_vapor = 5.4e-6f;
  f.k_liquid = 0.1446f;   f.k_vapor = 0.0074f;
  f.cp_liquid = 2042.f;   f.cp_vapor = 1123.f;
  f.sigma = 8.85e-3f;     f.h_fg = 199.2e3f;
  f.T_sat = 77.36f;       f.p_sat = 101325.f;
  f.R_gas = 296.8f;
  return f;
}

FluidProperties FluidProperties::liquid_oxygen() {
  FluidProperties f;
  f.name = "LOX";
  f.rho_liquid = 1141.f;  f.rho_vapor = 4.47f;
  f.mu_liquid = 1.96e-4f; f.mu_vapor = 6.9e-6f;
  f.k_liquid = 0.152f;    f.k_vapor = 0.0084f;
  f.cp_liquid = 1699.f;   f.cp_vapor = 972.f;
  f.sigma = 13.2e-3f;     f.h_fg = 213.1e3f;
  f.T_sat = 90.19f;       f.p_sat = 101325.f;
  f.R_gas = 259.8f;
  return f;
}

FluidProperties FluidProperties::liquid_methane() {
  FluidProperties f;
  f.name = "LCH4";
  f.rho_liquid = 422.4f;  f.rho_vapor = 1.816f;
  f.mu_liquid = 1.17e-4f; f.mu_vapor = 4.4e-6f;
  f.k_liquid = 0.184f;    f.k_vapor = 0.0119f;
  f.cp_liquid = 3481.f;   f.cp_vapor = 2218.f;
  f.sigma = 13.4e-3f;     f.h_fg = 510.8e3f;
  f.T_sat = 111.67f;      f.p_sat = 101325.f;
  f.R_gas = 518.3f;
  return f;
}

FluidProperties FluidProperties::water() {
  FluidProperties f;
  f.name = "H2O";
  f.rho_liquid = 998.2f;  f.rho_vapor = 0.598f;
  f.mu_liquid = 1.0e-3f;  f.mu_vapor = 1.23e-5f;
  f.k_liquid = 0.598f;    f.k_vapor = 0.0248f;
  f.cp_liquid = 4182.f;   f.cp_vapor = 2080.f;
  f.sigma = 0.0728f;      f.h_fg = 2.257e6f;
  f.T_sat = 373.15f;      f.p_sat = 101325.f;
  f.R_gas = 461.5f;
  return f;
}

}  // namespace cryosph
