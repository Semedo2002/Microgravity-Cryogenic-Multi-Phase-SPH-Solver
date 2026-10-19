// Structure-of-arrays particle storage. Kernels get a ParticleView (raw
// pointers, passed by value); the neighbour sort permutes the persistent arrays.
#pragma once

#include <cstdint>
#include <vector>

#include "cryosph/core/cuda_utils.cuh"

namespace cryosph {

/// Thermodynamic / role state of a particle.
enum class Phase : std::uint8_t {
  Liquid = 0,  ///< propellant liquid (Tait EOS)
  Vapor = 1,  ///< ullage vapour (ideal-gas EOS)
  Wall = 2,  ///< fixed boundary particle (Adami generalized wall BC)
  Dead = 3  ///< removed by merging; compacted away at the next sort
};

inline constexpr std::uint8_t kLiquid = static_cast<std::uint8_t>(Phase::Liquid);
inline constexpr std::uint8_t kVapor = static_cast<std::uint8_t>(Phase::Vapor);
inline constexpr std::uint8_t kWall = static_cast<std::uint8_t>(Phase::Wall);
inline constexpr std::uint8_t kDead = static_cast<std::uint8_t>(Phase::Dead);

CRYOSPH_HD CRYOSPH_INLINE bool is_fluid(std::uint8_t ph) { return ph <= kVapor; }

/// Bit flags stored per particle.
namespace pflag {
inline constexpr std::uint8_t Heated = 1u << 0;  ///< wall particle receives heater power
inline constexpr std::uint8_t Source = 1u << 1;  ///< scratch: sends mass this step
inline constexpr std::uint8_t TargetVapor = 1u << 2;  ///< scratch: mass goes to vapour (else liquid)
inline constexpr std::uint8_t Dying = 1u << 3;  ///< scratch: merging away this step
inline constexpr std::uint8_t ScratchMask = Source | TargetVapor | Dying;
}  // namespace pflag

/// Kernel-side handle: raw pointers into the SoA arrays plus the count.
struct ParticleView {
  int n = 0;
  int capacity = 0;
  // persistent
  Vec3* x = nullptr;  ///< position [m]
  Vec3* v = nullptr;  ///< velocity [m/s]
  Vec3* vt = nullptr;  ///< transport velocity v + dv (Adami et al. 2013) used to advect x [m/s]
  Vec3* nrm = nullptr;  ///< fluid: colour gradient (CSF) or Akinci normal; wall: unit normal into fluid
  real* m = nullptr;  ///< mass [kg]
  real* m0 = nullptr;  ///< reference mass: liquid summation density is m0 * sum_j W_ij
  real* rho = nullptr;  ///< density [kg/m^3]
  real* T = nullptr;  ///< temperature [K]
  std::uint8_t* phase = nullptr;
  std::uint8_t* flags = nullptr;
  std::uint32_t* id = nullptr;
  // per-step scratch
  Vec3* a = nullptr;  ///< acceleration [m/s^2]
  Vec3* ab = nullptr;  ///< background-pressure (transport) acceleration [m/s^2]
  Vec3* fw = nullptr;  ///< force exerted on the particle by wall particles [N]
  Vec3* mw = nullptr;  ///< moment of that force about the frame origin (wall positions) [N m]
  Vec3* ns = nullptr;  ///< smoothed unit interface normal (CSF), 0 where invalid
  Vec3* vw = nullptr;  ///< wall: extrapolated (ghost) velocity for no-slip
  real* p = nullptr;  ///< mechanical pressure [Pa]
  real* drho = nullptr;  ///< density rate (continuity mode) [kg/m^3/s]
  real* dTdt = nullptr;  ///< temperature rate [K/s]
  real* sigma = nullptr;  ///< particle number density sum_j W_ij [1/m^3]
  real* kappa = nullptr;  ///< interface curvature [1/m]
  real* nu_t = nullptr;  ///< SPS eddy viscosity [m^2/s]
  real* alpha = nullptr;  ///< volume fraction of the opposite fluid phase in the kernel
  real* wall_w = nullptr;  ///< kernel-weighted wall proximity sum_w V_w W_iw
  real* heat_w = nullptr;  ///< same, restricted to heated walls
  real* out_m = nullptr;  ///< mass emitted this step [kg]
  real* out_h = nullptr;  ///< specific enthalpy carried by emitted mass [J/kg]
  real* w_norm = nullptr;  ///< receiver weight normalisation for emitted mass
  real* rmin = nullptr;  ///< distance to the nearest other particle (fluid and wall)
  real* p_wv = nullptr;  ///< wall: pressure extrapolated from vapour neighbours (p holds the liquid side)
};

/// Host-side mirror of the persistent attributes (used for I/O and setup).
struct HostParticles {
  std::vector<Vec3> x, v, nrm;
  std::vector<real> m, rho, T;
  std::vector<std::uint8_t> phase, flags;
  std::vector<std::uint32_t> id;

  std::size_t size() const { return x.size(); }
  void reserve(std::size_t n);
  void clear();
  /// Appends one particle. Returns its index.
  std::size_t push(Vec3 pos, Vec3 vel, real mass, real density, real temperature, Phase ph,
                   Vec3 normal = {0, 0, 0}, std::uint8_t flag = 0);
  /// Appends all particles of `other`, re-numbering ids after the current ones.
  void append(const HostParticles& other);
};

/// Owns all particle arrays on the compute device.
class ParticleSystem {
 public:
  ParticleSystem() = default;
  ParticleSystem(const ParticleSystem&) = delete;
  ParticleSystem& operator=(const ParticleSystem&) = delete;

  /// Allocates every array for `capacity` particles (contents undefined).
  void allocate(std::size_t capacity);
  /// Grows the capacity while preserving the first size() particles.
  void grow(std::size_t new_capacity);
  /// Replaces the particle set with `host` (capacity grows as needed).
  void upload(const HostParticles& host, double growth_factor = 2.0);
  /// Copies persistent attributes of all live particles back to the host.
  HostParticles download() const;

  std::size_t size() const noexcept { return n_; }
  std::size_t capacity() const noexcept { return capacity_; }
  void set_size(std::size_t n);

  ParticleView view();

  /// Host-side copies of individual scratch fields (for diagnostics / output).
  std::vector<real> download_pressure() const { return p.to_host(n_); }
  std::vector<real> download_curvature() const { return kappa.to_host(n_); }
  std::vector<Vec3> download_acceleration() const { return a.to_host(n_); }

  // persistent arrays (permuted by the neighbour sort)
  DeviceArray<Vec3> x, v, vt, nrm;
  DeviceArray<real> m, m0, rho, T;
  DeviceArray<std::uint8_t> phase, flags;
  DeviceArray<std::uint32_t> id;
  // scratch arrays
  DeviceArray<Vec3> a, ab, ns, vw, fw, mw;
  DeviceArray<real> p, drho, dTdt, sigma, kappa, nu_t, alpha, wall_w, heat_w, out_m, out_h, w_norm, p_wv, rmin;
  // spare buffers used for out-of-place permutation
  DeviceArray<Vec3> spare_vec3;
  DeviceArray<real> spare_real;
  DeviceArray<std::uint8_t> spare_u8;
  DeviceArray<std::uint32_t> spare_u32;

  /// Next unused particle id (ids stay unique when particles are split).
  std::uint32_t next_id = 0;

 private:
  template <class F>
  void for_each_array(F&& f);

  std::size_t n_ = 0;
  std::size_t capacity_ = 0;
};

}  // namespace cryosph
