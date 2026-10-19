// Smoothing kernels (cubic spline, Wendland C2; support 2h), Akinci
// splines, and the density / transport-velocity / shifting passes.
#pragma once

#include "cryosph/core/cuda_utils.cuh"
#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/spatial_hash.hpp"

namespace cryosph {

enum class KernelType : int { CubicSpline = 0, WendlandC2 = 1 };

/// Precomputed kernel constants (cheap to pass by value into kernels).
struct KernelParams {
  KernelType type = KernelType::WendlandC2;
  real h = 1;  ///< smoothing length
  real inv_h = 1;
  real support = 2;  ///< compact support radius (2h)
  real norm = 1;  ///< 3-D normalisation constant / h^3

  static KernelParams make(KernelType t, real h) {
    KernelParams k;
    k.type = t;
    k.h = h;
    k.inv_h = real(1) / h;
    k.support = real(2) * h;
    const real h3 = h * h * h;
    k.norm = (t == KernelType::CubicSpline) ? real(1) / (kPi * h3) : real(21) / (real(16) * kPi * h3);
    return k;
  }
};

/// Kernel value W(r).
CRYOSPH_HD CRYOSPH_INLINE real kernel_W(const KernelParams& k, real r) {
  const real q = r * k.inv_h;
  if (q >= real(2)) return real(0);
  if (k.type == KernelType::CubicSpline) {
    if (q < real(1)) return k.norm * (real(1) - real(1.5) * q * q + real(0.75) * q * q * q);
    const real t = real(2) - q;
    return k.norm * real(0.25) * t * t * t;
  }
  const real t = real(1) - real(0.5) * q;
  const real t2 = t * t;
  return k.norm * t2 * t2 * (real(2) * q + real(1));
}

/// Radial derivative dW/dr (non-positive).
CRYOSPH_HD CRYOSPH_INLINE real kernel_dWdr(const KernelParams& k, real r) {
  const real q = r * k.inv_h;
  if (q >= real(2)) return real(0);
  if (k.type == KernelType::CubicSpline) {
    if (q < real(1)) return k.norm * k.inv_h * (real(-3) * q + real(2.25) * q * q);
    const real t = real(2) - q;
    return k.norm * k.inv_h * real(-0.75) * t * t;
  }
  const real t = real(1) - real(0.5) * q;
  return k.norm * k.inv_h * real(-5) * q * t * t * t;
}

/// grad_i W(|x_i - x_j|) given r_ij = x_i - x_j and r = |r_ij|.
CRYOSPH_HD CRYOSPH_INLINE Vec3 kernel_gradW(const KernelParams& k, Vec3 rij, real r) {
  if (r <= real(1e-12) * k.h) return Vec3{0, 0, 0};
  return rij * (kernel_dWdr(k, r) / r);
}

// Akinci et al. (2013) cohesion and adhesion splines, support c

/// Cohesion spline C(r) (positive = attraction for r > c/2).
CRYOSPH_HD CRYOSPH_INLINE real akinci_cohesion(real r, real c) {
  if (r >= c || r <= real(0)) return real(0);
  const real c3 = c * c * c;
  const real c9 = c3 * c3 * c3;
  const real coeff = real(32) / (kPi * c9);
  const real t = (c - r);
  const real base = t * t * t * r * r * r;
  if (real(2) * r > c) return coeff * base;
  return coeff * (real(2) * base - c3 * c3 / real(64));
}

/// Adhesion spline A(r), non-zero for c/2 < r < c.
CRYOSPH_HD CRYOSPH_INLINE real akinci_adhesion(real r, real c) {
  if (r >= c || real(2) * r <= c) return real(0);
  const real arg = real(-4) * r * r / c + real(6) * r - real(2) * c;
  if (arg <= real(0)) return real(0);
  return real(0.007) / std::pow(c, real(3.25)) * std::pow(arg, real(0.25));
}

// density evaluation (src/physics/SPH_kernels.cu)

struct SimParams;

/// Number-density summation  sigma_i = sum_j W_ij  (fluid + wall particles);
/// in summation mode the fluid density is rho_i = m_i sigma_i (Hu & Adams 2006),
/// which stays sharp across liquid/vapour interfaces of large density ratio.
void compute_density_summation(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// m_i = rho_i / sum_j W_ij, so the initial density is exact (removes the ~2 %
/// lattice-sum error the stiff Tait EOS would turn into an interface pressure jump).
void normalize_initial_masses(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Transport-velocity acceleration (Adami et al. 2013), interface band only:
///   ab_i = -(p_b / m_i) sum_j (V_i^2 + V_j^2) grad W_ij
/// Walls continue the fluid's volume field. Call after compute_wall_boundary.
void compute_transport_acceleration(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// delta+-SPH particle shifting (Sun et al. 2017): positions only, tangential at
/// interfaces, none at free surfaces, capped at 0.1 dx.
void apply_particle_shifting(ParticleView pv, GridView g, const SimParams& P, real mach, stream_t s);

/// Continuity equation  drho_i/dt = rho_i sum_j V_j (v_i - v_j) . grad_i W_ij
/// plus the delta-SPH density diffusion term of Molteni & Colagrossi (2009)
/// restricted to same-phase pairs.
void compute_density_rate(ParticleView pv, GridView g, const SimParams& P, stream_t s);

}  // namespace cryosph
