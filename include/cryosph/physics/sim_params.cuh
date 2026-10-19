// Parameter block passed by value to every kernel, plus the equations of
// state: Tait liquid and a linearised ideal-gas vapour sharing a background
// pressure. The ullage (thermodynamic) pressure is tracked separately.
#pragma once

#include "cryosph/core/cuda_utils.cuh"
#include "cryosph/core/particle_system.hpp"
#include "cryosph/physics/SPH_kernels.cuh"

namespace cryosph {

enum class DensityMode : int { Summation = 0, Continuity = 1 };
enum class SurfaceTensionModel : int { None = 0, CSF = 1, Akinci = 2 };
enum class WallThermalBC : int { Adiabatic = 0, Isothermal = 1 };
enum class TankThermoMode : int { Vented = 0, Closed = 1 };
// How the contact angle is imposed: rotate normals near walls (Brackbill),
// an explicit contact-line force (Huber et al. 2016), or both.
enum class WettingModel : int { NormalRotation = 0, ContactLineForce = 1, Both = 2 };

struct SimParams {
  KernelParams kernel;
  real dx = 0;  ///< initial particle spacing
  real vol0 = 0;  ///< dx^3, volume of a wall particle
  real support2 = 0;  ///< (2h)^2

  // per phase (index 0 = liquid, 1 = vapour)
  real rho0[2] = {1, 1};  ///< reference densities
  real mu[2] = {0, 0};  ///< dynamic viscosity
  real k[2] = {0, 0};  ///< thermal conductivity (incl. scale factor)
  real cp[2] = {1, 1};  ///< specific heat
  real mref[2] = {1, 1};  ///< reference particle mass rho0 * dx^3

  // equation of state
  real c0 = 1;  ///< artificial sound speed
  real tait_B = 1;
  real tait_gamma = 7;
  real p_bg = 0;  ///< background pressure
  real gas_R = 1;  ///< effective gas constant of the mechanical EOS (= vapor_K / (rho_v0 T_ref))
  real vapor_K = 1;  ///< vapour stiffness rho_v0 c_v^2; vapour p = p_bg + vapor_K (rho T / (rho_v0 T_ref) - 1)
  int clamp_negative_pressure = 0;

  // momentum
  Vec3 gravity{0, 0, 0};  ///< body acceleration in the tank frame: g - a_tank
  int frame_rot = 0;  ///< rotating tank frame on/off
  Vec3 omega{0, 0, 0};  ///< tank angular velocity [rad/s] (tank frame)
  Vec3 omega_dot{0, 0, 0};  ///< tank angular acceleration [rad/s^2]
  Vec3 frame_origin{0, 0, 0};  ///< rotation centre / moment reference point
  real alpha_av = 0;  ///< Monaghan artificial-viscosity coefficient (liquid)
  real alpha_av_vap = 0;  ///< same for the vapour (and vapour-wall pairs)
  real delta_sph = 0.1f;  ///< density diffusion coefficient (continuity mode)
  int density_mode = 0;
  int no_slip = 1;
  int sps = 0;  ///< Smagorinsky sub-particle-scale turbulence on/off
  // transport-velocity formulation (Adami, Hu & Adams 2013)
  int tvf = 1;  ///< advect particles with v + dv instead of v
  int tvf_stress = 0;  ///< include the extra stress A = rho v (vt - v) in the momentum equation
  real tvf_factor = 2.0f;  ///< p_b,i = tvf_factor * rho0_phase * c_i^2
  real tvf_sigma_min = 0;  ///< kernel sum below which dv is suppressed (free surfaces); 0 = never
  real shift_strength = 0.25f;  ///< A in the delta+-SPH shift (0 = no shifting)
  int tvf_interface_only = 1;  ///< apply dv only where the other fluid phase is within the kernel
  real smagorinsky_cs = 0.12f;

  // surface tension
  int st_model = 1;
  real sigma = 0;  ///< surface tension coefficient [N/m]
  real cos_theta = 0;  ///< cos(static contact angle)
  real sin_theta = 1;
  real normal_eps = 0;  ///< |grad c| threshold for a valid interface normal
  real wall_adhesion_weight = 0.01f;  ///< wall kernel weight above which the contact angle is fully imposed
  int wetting_model = 0;  ///< 0 normal rotation (Brackbill), 1 contact-line force (Huber), 2 both
  int wall_colour = 1;  ///< wall particles carry an extrapolated colour in the colour gradient
  int csf_smooth_normals = 0;  ///< kernel-smooth the unit normals before the curvature
  int csf_balanced = 0;  ///< balanced-force (pressure-operator) CSF, Adami et al. 2010
  real akinci_gamma = 0;  ///< Akinci cohesion strength
  real akinci_beta = 0;  ///< Akinci adhesion strength

  // thermodynamics & phase change
  int thermal = 1;
  int phase_change = 1;
  int wall_bc = 0;  ///< WallThermalBC
  real T_wall = 0;
  real T_ref = 0;  ///< enthalpy reference temperature (= T_sat at p_ref)
  real T_sat = 0;  ///< current saturation temperature
  real h_fg = 0;
  real lee_evap = 0;  ///< Lee evaporation coefficient r_v [1/s]
  real lee_cond = 0;  ///< Lee condensation coefficient r_c [1/s]
  real max_transfer_fraction = 0.25f;  ///< cap on mass emitted per step
  real vapor_rebalance = 1.0f;  ///< per-step vapour mass diffusion coefficient (0 = off)
  real pc_density_limit = 0.01f;  ///< max relative density change of a vapour particle per step (0 = off)
  real merge_ratio = 0.3f;  ///< particles lighter than merge_ratio*mref merge away
  real split_ratio = 2.0f;  ///< particles heavier than split_ratio*mref split
  real heat_power = 0;  ///< heater power [W]
  real heat_inv_norm = 0;  ///< 1 / sum_i heat_w_i (computed every step)
  real dt = 0;  ///< current time step (used by phase change)
};

CRYOSPH_HD CRYOSPH_INLINE int phase_index(std::uint8_t ph) { return ph == kVapor ? 1 : 0; }

/// Mechanical equation of state.
CRYOSPH_HD CRYOSPH_INLINE real eos_pressure(const SimParams& P, std::uint8_t ph, real rho, real T) {
  real p;
  if (ph == kVapor) {
    // Linearised ideal gas with its own stiffness; identical to p = rho R T
    // when vapor_K == p_bg.
    p = rho * P.gas_R * T + (P.p_bg - P.vapor_K);
  } else {
    const real ratio = rho / P.rho0[0];
    // gamma = 7 is by far the common case: avoid pow() on the GPU.
    real r_g;
    if (P.tait_gamma == real(7)) {
      const real r2 = ratio * ratio;
      r_g = r2 * r2 * r2 * ratio;
    } else {
      r_g = std::pow(ratio, P.tait_gamma);
    }
    p = P.tait_B * (r_g - real(1)) + P.p_bg;
  }
  return (P.clamp_negative_pressure && p < real(0)) ? real(0) : p;
}

/// Isothermal / Tait sound speed of a particle.
CRYOSPH_HD CRYOSPH_INLINE real sound_speed(const SimParams& P, std::uint8_t ph, real T) {
  if (ph == kVapor) return std::sqrt(P.gas_R * (T > real(0) ? T : real(0)));
  return P.c0;
}

/// Specific enthalpy relative to saturated liquid at T_ref:
///   h_l(T) = cp_l (T - T_ref),   h_v(T) = h_fg + cp_v (T - T_ref).
CRYOSPH_HD CRYOSPH_INLINE real specific_enthalpy(const SimParams& P, std::uint8_t ph, real T) {
  return ph == kVapor ? P.h_fg + P.cp[1] * (T - P.T_ref) : P.cp[0] * (T - P.T_ref);
}

CRYOSPH_HD CRYOSPH_INLINE real temperature_from_enthalpy(const SimParams& P, std::uint8_t ph, real h) {
  return ph == kVapor ? P.T_ref + (h - P.h_fg) / P.cp[1] : P.T_ref + h / P.cp[0];
}

}  // namespace cryosph
