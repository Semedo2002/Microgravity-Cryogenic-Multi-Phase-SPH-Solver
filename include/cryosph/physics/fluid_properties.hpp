// Saturated liquid/vapour properties at 1 atm (NIST values, rounded).
// Constant-property approximation for moderate temperature changes.
#pragma once

#include <string>

#include "cryosph/core/cuda_utils.cuh"

namespace cryosph {

struct FluidProperties {
  std::string name = "custom";
  real rho_liquid = 1000;  ///< saturated liquid density [kg/m^3]
  real rho_vapor = 1;  ///< saturated vapour density [kg/m^3]
  real mu_liquid = 1e-3;  ///< liquid dynamic viscosity [Pa s]
  real mu_vapor = 1e-5;  ///< vapour dynamic viscosity [Pa s]
  real k_liquid = 0.6;  ///< liquid thermal conductivity [W/m/K]
  real k_vapor = 0.025;  ///< vapour thermal conductivity [W/m/K]
  real cp_liquid = 4180;  ///< liquid specific heat [J/kg/K]
  real cp_vapor = 2000;  ///< vapour specific heat [J/kg/K]
  real sigma = 0.072;  ///< surface tension [N/m]
  real h_fg = 2.26e6;  ///< latent heat of vaporisation [J/kg]
  real T_sat = 373.15;  ///< saturation temperature at p_sat [K]
  real p_sat = 101325;  ///< reference saturation pressure [Pa]
  real R_gas = 461.5;  ///< specific gas constant of the vapour [J/kg/K]

  /// Clausius-Clapeyron saturation temperature at pressure p:
  ///   1/T = 1/T_ref - (R / h_fg) ln(p / p_ref)
  real saturation_temperature(real p) const;
  /// Inverse of saturation_temperature.
  real saturation_pressure(real T) const;

  static FluidProperties liquid_hydrogen();
  static FluidProperties liquid_nitrogen();
  static FluidProperties liquid_oxygen();
  static FluidProperties liquid_methane();
  static FluidProperties water();
};

}  // namespace cryosph
