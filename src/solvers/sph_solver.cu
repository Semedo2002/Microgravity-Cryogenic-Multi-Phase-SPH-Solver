// Solver orchestration. One step:
//   kick + drift (transport velocity) -> rebuild grid -> rates at the new
//   positions -> kick + heat -> phase change -> ullage state.
#include "cryosph/solvers/sph_solver.hpp"

#include <algorithm>
#include <cmath>

#include "cryosph/physics/SPH_kernels.cuh"
#include "cryosph/physics/navier_stokes.cuh"
#include "cryosph/physics/phase_change.cuh"
#include "cryosph/physics/surface_tension.cuh"

namespace cryosph {

// Kernel launches live in free functions: CUDA extended lambdas may not be
// defined inside private member functions.
namespace solver_detail {

void kick_drift(ParticleView pv, real dt, int continuity, int tvf, stream_t s) {
  const real half = real(0.5) * dt;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (!is_fluid(pv.phase[i])) return;
        const Vec3 v = pv.v[i] + pv.a[i] * half;
        pv.v[i] = v;
        const Vec3 vt = tvf ? v + pv.ab[i] * half : v;
        pv.vt[i] = vt;
        pv.x[i] += vt * dt;
        if (continuity) pv.rho[i] += pv.drho[i] * dt;
      },
      s);
}

void kick(ParticleView pv, real dt, int thermal, stream_t s) {
  const real half = real(0.5) * dt;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (!is_fluid(pv.phase[i])) return;
        pv.v[i] += pv.a[i] * half;
        if (thermal) pv.T[i] += pv.dTdt[i] * dt;
      },
      s);
}

void scale_velocities(ParticleView pv, real factor, stream_t s) {
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (!is_fluid(pv.phase[i])) return;
        pv.v[i] *= factor;
        pv.vt[i] *= factor;
      },
      s);
}

/// max(1, max_i 0.5 dx / r_min,i) over fluid particles: > 1 only when some
/// pair is closer than half the lattice spacing (fresh split pairs), not for
/// ordinary compression.
real max_spacing_ratio(ParticleView pv, real dx, stream_t s) {
  return reduce_max<real>(
      pv.n,
      [=] CRYOSPH_HD(int i) -> real {
        if (!is_fluid(pv.phase[i])) return real(0);
        const real r = pv.rmin[i] > real(0.05) * dx ? pv.rmin[i] : real(0.05) * dx;
        return real(0.5) * dx / r;
      },
      real(1), s);
}

real max_fluid_speed(ParticleView pv, stream_t s) {
  return reduce_max<real>(
      pv.n, [=] CRYOSPH_HD(int i) -> real { return is_fluid(pv.phase[i]) ? norm(pv.v[i]) : real(0); }, real(0), s);
}

real max_fluid_accel(ParticleView pv, stream_t s) {
  // Includes the transport (background-pressure) acceleration, which moves
  // particles just like a physical force does.
  return reduce_max<real>(
      pv.n,
      [=] CRYOSPH_HD(int i) -> real {
        if (!is_fluid(pv.phase[i])) return real(0);
        const real a = norm(pv.a[i]), b = norm(pv.ab[i]);
        return a > b ? a : b;
      },
      real(0), s);
}

real max_vapor_temperature(ParticleView pv, stream_t s) {
  return reduce_max<real>(
      pv.n, [=] CRYOSPH_HD(int i) -> real { return pv.phase[i] == kVapor ? pv.T[i] : real(0); }, real(0), s);
}

real max_eddy_viscosity(ParticleView pv, stream_t s) {
  return reduce_max<real>(
      pv.n, [=] CRYOSPH_HD(int i) -> real { return is_fluid(pv.phase[i]) ? pv.nu_t[i] : real(0); }, real(0), s);
}

double sum_heat_weight(ParticleView pv, stream_t s) {
  return reduce_sum<double>(
      pv.n, [=] CRYOSPH_HD(int i) -> double { return is_fluid(pv.phase[i]) ? double(pv.heat_w[i]) : 0.0; }, 0.0, s);
}

/// Mass emitted by phase-change sources: evaporation (target vapour) or
/// condensation (target liquid); merges are excluded.
double sum_phase_change(ParticleView pv, bool evaporation, stream_t s) {
  return reduce_sum<double>(
      pv.n,
      [=] CRYOSPH_HD(int i) -> double {
        const std::uint8_t fl = pv.flags[i];
        if (!(fl & pflag::Source) || (fl & pflag::Dying)) return 0.0;
        const bool to_vapor = (fl & pflag::TargetVapor) != 0;
        return to_vapor == evaporation ? double(pv.out_m[i]) : 0.0;
      },
      0.0, s);
}

int count_merges(ParticleView pv, stream_t s) {
  return reduce_sum<int>(
      pv.n,
      [=] CRYOSPH_HD(int i) -> int { return ((pv.flags[i] & pflag::Dying) && pv.out_m[i] > real(0)) ? 1 : 0; }, 0, s);
}

double sum_out_mass(ParticleView pv, stream_t s) {
  return reduce_sum<double>(pv.n, [=] CRYOSPH_HD(int i) -> double { return double(pv.out_m[i]); }, 0.0, s);
}

double sum_out_enthalpy(ParticleView pv, stream_t s) {
  return reduce_sum<double>(
      pv.n, [=] CRYOSPH_HD(int i) -> double { return double(pv.out_m[i]) * double(pv.out_h[i]); }, 0.0, s);
}

/// Phase inventory for the lumped ullage model: liquid mass, vapour mass and
/// vapour sum m T.
void phase_inventory(ParticleView pv, double& m_liq, double& m_vap, double& mT_vap, stream_t s) {
  m_liq = reduce_sum<double>(
      pv.n, [=] CRYOSPH_HD(int i) -> double { return pv.phase[i] == kLiquid ? double(pv.m[i]) : 0.0; }, 0.0, s);
  m_vap = reduce_sum<double>(
      pv.n, [=] CRYOSPH_HD(int i) -> double { return pv.phase[i] == kVapor ? double(pv.m[i]) : 0.0; }, 0.0, s);
  mT_vap = reduce_sum<double>(
      pv.n, [=] CRYOSPH_HD(int i) -> double { return pv.phase[i] == kVapor ? double(pv.m[i]) * pv.T[i] : 0.0; },
      0.0, s);
}

int count_phase(ParticleView pv, std::uint8_t ph, stream_t s) {
  return reduce_sum<int>(pv.n, [=] CRYOSPH_HD(int i) -> int { return pv.phase[i] == ph ? 1 : 0; }, 0, s);
}

/// sum over particles of phase `ph` of field(i), selected by `what`:
/// 0 mass, 1 m*T, 2 kinetic energy, 3 m*x, 4 m*y, 5 m*z, 6 m*vx, 7 m*vy, 8 m*vz, 9 m*h
double phase_sum(ParticleView pv, std::uint8_t ph, int what, SimParams P, stream_t s) {
  return reduce_sum<double>(
      pv.n,
      [=] CRYOSPH_HD(int i) -> double {
        if (pv.phase[i] != ph) return 0.0;
        const double m = pv.m[i];
        switch (what) {
          case 0: return m;
          case 1: return m * pv.T[i];
          case 2: return 0.5 * m * double(norm2(pv.v[i]));
          case 3: return m * pv.x[i].x;
          case 4: return m * pv.x[i].y;
          case 5: return m * pv.x[i].z;
          case 6: return m * pv.v[i].x;
          case 7: return m * pv.v[i].y;
          case 8: return m * pv.v[i].z;
          case 9: return m * double(specific_enthalpy(P, ph, pv.T[i]));
          default: return 0.0;
        }
      },
      0.0, s);
}

real phase_max_temperature(ParticleView pv, std::uint8_t ph, stream_t s) {
  return reduce_max<real>(
      pv.n, [=] CRYOSPH_HD(int i) -> real { return pv.phase[i] == ph ? pv.T[i] : real(0); }, real(0), s);
}

/// Fowler mean-field integral  int_0^c r^4 f(r) dr  (Simpson's rule).
template <class F>
double radial_moment4(F f, double c) {
  const int n = 2000;
  const double hstep = c / n;
  double acc = 0;
  for (int k = 0; k <= n; ++k) {
    const double r = k * hstep;
    const double w = (k == 0 || k == n) ? 1.0 : (k % 2 ? 4.0 : 2.0);
    acc += w * r * r * r * r * f(r);
  }
  return acc * hstep / 3.0;
}

}  // namespace solver_detail

void probe_liquid_pressure(ParticleView pv, GridView g, const SimParams& Pref, const Vec3* pts, real* out, int np,
                           stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      np,
      [=] CRYOSPH_HD(int k) {
        const Vec3 x = pts[k];
        real num = 0, den = 0;
        for_each_neighbor(g, x, [&](int j) {
          if (pv.phase[j] != kLiquid) return;
          const real r2 = norm2(x - pv.x[j]);
          if (r2 >= P.support2) return;
          const real w = (pv.m[j] / pv.rho[j]) * kernel_W(P.kernel, std::sqrt(r2));
          num += pv.p[j] * w;
          den += w;
        });
        out[k] = den > real(0) ? num / den : std::numeric_limits<real>::quiet_NaN();
      },
      s);
}

/// Sum over fluid particles of one Vec3 field (double precision).
Vec3 sum_fluid_vec3(ParticleView pv, const Vec3* f, stream_t s) {
  double c[3];
  for (int k = 0; k < 3; ++k) {
    c[k] = reduce_sum<double>(
        pv.n,
        [=] CRYOSPH_HD(int i) -> double {
          if (!is_fluid(pv.phase[i])) return 0.0;
          const Vec3 v = f[i];
          return k == 0 ? double(v.x) : (k == 1 ? double(v.y) : double(v.z));
        },
        0.0, s);
  }
  return Vec3{static_cast<real>(c[0]), static_cast<real>(c[1]), static_cast<real>(c[2])};
}

void ManeuverProfile::validate() const {
  if (time.empty()) throw std::invalid_argument("ManeuverProfile: empty time table");
  if (acceleration.size() != time.size())
    throw std::invalid_argument("ManeuverProfile: acceleration must have one entry per time");
  if (!angular_velocity.empty() && angular_velocity.size() != time.size())
    throw std::invalid_argument("ManeuverProfile: angular_velocity must be empty or have one entry per time");
  for (std::size_t k = 1; k < time.size(); ++k)
    if (!(time[k] > time[k - 1])) throw std::invalid_argument("ManeuverProfile: time must be strictly increasing");
}

void ManeuverProfile::evaluate(double t, Vec3& a, Vec3& omega, Vec3& omega_dot) const {
  const std::size_t n = time.size();
  const bool rot = !angular_velocity.empty();
  omega_dot = Vec3{0, 0, 0};
  if (t <= time.front() || n == 1) {
    a = acceleration.front();
    omega = rot ? angular_velocity.front() : Vec3{0, 0, 0};
    return;
  }
  if (t >= time.back()) {
    a = acceleration.back();
    omega = rot ? angular_velocity.back() : Vec3{0, 0, 0};
    return;
  }
  const std::size_t k = static_cast<std::size_t>(std::upper_bound(time.begin(), time.end(), t) - time.begin()) - 1;
  const double dt = time[k + 1] - time[k];
  const real f = static_cast<real>((t - time[k]) / dt);
  a = acceleration[k] * (real(1) - f) + acceleration[k + 1] * f;
  if (rot) {
    omega = angular_velocity[k] * (real(1) - f) + angular_velocity[k + 1] * f;
    omega_dot = (angular_velocity[k + 1] - angular_velocity[k]) / static_cast<real>(dt);
  } else {
    omega = Vec3{0, 0, 0};
  }
}

using namespace solver_detail;

// Solver
Solver::Solver(const SolverConfig& config, const FluidProperties& fluid) : cfg_(config), fluid_(fluid) {
  if (!(cfg_.dx > 0)) throw std::invalid_argument("SolverConfig.dx must be > 0");
  if (cfg_.contact_angle_deg < 0 || cfg_.contact_angle_deg > 180)
    throw std::invalid_argument("contact angle must be within [0, 180] degrees");
  stream_ = mem::create_stream();
  p_ullage_ = fluid_.p_sat;
}

Solver::~Solver() {
  try {
    mem::synchronize(stream_);
    telemetry_.configure(0, telemetry_.mode());
  } catch (...) {
  }
  mem::destroy_stream(stream_);
}

std::string Solver::device() const { return mem::device_name(); }

void Solver::init_params() {
  const SolverConfig& c = cfg_;
  SimParams& P = P_;
  const real dx = c.dx;
  const real h = c.h_factor * dx;
  P.kernel = KernelParams::make(c.kernel, h);
  P.dx = dx;
  P.support2 = P.kernel.support * P.kernel.support;
  {
    // Lattice volume 1/sigma0 (sigma0 = sum_j W_ij on the initial cubic
    // lattice): wall particles then carry exactly the volume of the fluid
    // particles they face, keeping the wall-fluid pressure force balanced.
    double sigma0 = 0;
    const int r = static_cast<int>(std::ceil(P.kernel.support / dx));
    for (int k = -r; k <= r; ++k)
      for (int j = -r; j <= r; ++j)
        for (int i = -r; i <= r; ++i)
          sigma0 += kernel_W(P.kernel, dx * std::sqrt(real(i * i + j * j + k * k)));
    P.vol0 = static_cast<real>(1.0 / sigma0);
  }

  P.rho0[0] = fluid_.rho_liquid;
  P.rho0[1] = fluid_.rho_vapor;
  P.mu[0] = fluid_.mu_liquid;
  P.mu[1] = fluid_.mu_vapor;
  P.k[0] = fluid_.k_liquid * c.conductivity_scale;
  P.k[1] = fluid_.k_vapor * c.conductivity_scale;
  P.cp[0] = fluid_.cp_liquid;
  P.cp[1] = fluid_.cp_vapor;
  P.mref[0] = P.rho0[0] * P.vol0;
  P.mref[1] = P.rho0[1] * P.vol0;

  const bool st_on = c.surface_tension != SurfaceTensionModel::None;
  const real g = norm(c.gravity);
  real vref = c.velocity_scale;
  if (!(vref > 0)) {
    vref = std::sqrt(g * char_length_);
    if (st_on) vref = std::max(vref, std::sqrt(fluid_.sigma / (fluid_.rho_liquid * h)));
    vref = std::max(vref, real(1e-3));
  }
  P.c0 = c.sound_speed > 0 ? c.sound_speed : real(10) * vref;
  P.tait_gamma = c.tait_gamma;
  P.tait_B = P.rho0[0] * P.c0 * P.c0 / P.tait_gamma;
  // 0.1 rho_v c0^2 with vapour. With p_bg = K (plain ideal gas) a still vapour
  // never came to rest -- SPH's pressure error scales with absolute pressure.
  P.p_bg = c.background_pressure >= 0 ? c.background_pressure
                                      : (has_vapor_ ? real(0.1) * P.rho0[1] * P.c0 * P.c0 : real(0));
  // Vapour stiffness K; p_bg below K keeps the sound speed but lowers the noise.
  P.vapor_K = c.vapor_stiffness > 0 ? c.vapor_stiffness : (has_vapor_ ? P.rho0[1] * P.c0 * P.c0 : P.p_bg);
  P.gas_R = P.vapor_K > 0 ? P.vapor_K / (P.rho0[1] * fluid_.T_sat) : fluid_.R_gas;
  P.clamp_negative_pressure = c.clamp_negative_pressure ? 1 : 0;

  P.gravity = c.gravity - frame_acc_;  // keeps a frame acceleration set before loading
  P.alpha_av = c.artificial_viscosity;
  P.alpha_av_vap = c.artificial_viscosity_vapor;
  P.delta_sph = c.delta_sph;
  P.density_mode = static_cast<int>(c.density_mode);
  P.no_slip = c.no_slip ? 1 : 0;
  P.sps = c.sps_turbulence ? 1 : 0;
  P.smagorinsky_cs = c.smagorinsky_cs;
  P.tvf = c.transport_velocity ? 1 : 0;
  P.tvf_stress = c.transport_stress ? 1 : 0;
  P.tvf_factor = c.transport_background_factor;
  P.tvf_interface_only = c.transport_interface_only ? 1 : 0;
  P.shift_strength = c.particle_shifting ? c.shifting_strength : real(0);
  // Without vapour the liquid has a free surface, where the truncated kernel
  // sum would push particles outwards: suppress the shift there.
  P.tvf_sigma_min = has_vapor_ ? real(0) : real(0.95) / P.vol0;

  P.st_model = static_cast<int>(c.surface_tension);
  P.sigma = st_on ? fluid_.sigma : real(0);
  const real theta = c.contact_angle_deg * kPi / real(180);
  P.cos_theta = std::cos(theta);
  P.sin_theta = std::sin(theta);
  P.wall_adhesion_weight = c.contact_angle_wall_weight;
  P.wall_colour = c.wall_colour ? 1 : 0;
  P.wetting_model = static_cast<int>(c.wetting_model);
  P.csf_smooth_normals = c.csf_smooth_normals ? 1 : 0;
  P.csf_balanced = c.csf_balanced ? 1 : 0;
  P.normal_eps = real(0.1) / h;
  {
    // Akinci strengths from the Fowler mean-field surface energy
    //   sigma = (pi/8) gamma rho^2 int r^4 C(r) dr
    // and the Young-Dupre work of adhesion sigma (1 + cos theta).
    const double sup = P.kernel.support;
    const double i_coh = radial_moment4([&](double r) { return double(akinci_cohesion(real(r), real(sup))); }, sup);
    const double i_adh = radial_moment4([&](double r) { return double(akinci_adhesion(real(r), real(sup))); }, sup);
    const double rho = fluid_.rho_liquid;
    const double gamma = c.akinci_gamma >= 0 ? c.akinci_gamma
                                             : (i_coh > 0 ? 8.0 * fluid_.sigma / (kPi * rho * rho * i_coh) : 0.0);
    P.akinci_gamma = static_cast<real>(gamma);
    P.akinci_beta = c.akinci_beta >= 0
                        ? c.akinci_beta
                        : static_cast<real>(i_adh > 0 ? gamma * (i_coh / i_adh) * 0.5 * (1.0 + P.cos_theta) : 0.0);
  }

  P.thermal = c.thermal ? 1 : 0;
  P.phase_change = c.phase_change ? 1 : 0;
  P.wall_bc = static_cast<int>(c.wall_thermal_bc);
  P.T_wall = fluid_.T_sat;
  P.T_ref = fluid_.T_sat;
  P.T_sat = fluid_.T_sat;
  P.h_fg = fluid_.h_fg;
  P.lee_evap = c.lee_evaporation;
  P.lee_cond = c.lee_condensation;
  P.max_transfer_fraction = c.max_transfer_fraction;
  P.pc_density_limit = c.phase_change_density_limit;
  P.vapor_rebalance = c.vapor_mass_rebalance;
  P.merge_ratio = c.merge_ratio;
  P.split_ratio = c.split_ratio;
  P.heat_power = c.heater_power;
  P.heat_inv_norm = 0;
  P.dt = 0;
}

void Solver::set_particles(const HostParticles& particles, const AABB& domain) {
  if (particles.size() == 0) throw std::invalid_argument("Solver::set_particles: no particles");
  has_vapor_ = std::any_of(particles.phase.begin(), particles.phase.end(), [](std::uint8_t p) { return p == kVapor; });
  if (has_vapor_ && cfg_.surface_tension == SurfaceTensionModel::Akinci)
    throw std::invalid_argument(
        "Solver: the Akinci surface-tension model is a free-surface (liquid-only) model; use CSF when vapour "
        "particles are present (FillSpec.include_vapor = false for Akinci runs)");
  const Vec3 ext = domain.hi - domain.lo;
  char_length_ = std::max({ext.x, ext.y, ext.z});
  init_params();
  ps_.upload(particles);
  grid_.configure(domain.lo, domain.hi, P_.kernel.support);
  telemetry_.configure(ps_.capacity(), TelemetryMode::AsyncCopy);
  initialized_ = false;
  time_ = 0;
  step_ = 0;
  last_dt_ = 0;
  heat_input_ = evaporated_ = condensed_ = vented_ = vented_h_ = 0;
  splits_ = merges_ = 0;
  p_ullage_ = fluid_.p_sat;
  fluid_volume0_ = 0;
}

void Solver::build_tank(const Geometry& geometry, const FillSpec& fill) {
  const HostParticles hp = build_tank_particles(geometry, fluid_, cfg_.dx, fill);
  AABB dom = geometry.bounds();
  const real pad = (fill.wall_layers + 1) * cfg_.dx;
  dom.lo = dom.lo - Vec3{pad, pad, pad};
  dom.hi = dom.hi + Vec3{pad, pad, pad};
  set_particles(hp, dom);
}

void Solver::ensure_initialized() {
  if (initialized_) return;
  if (ps_.size() == 0) throw std::runtime_error("Solver: no particles loaded");
  grid_.build(ps_, stream_);
  if (cfg_.normalize_initial_masses) normalize_initial_masses(ps_.view(), grid_.view(), P_, stream_);
  evaluate_rates();  // sigma must be known before the vapour volume is summed
  update_saturation();
  initialized_ = true;
}

void Solver::evaluate_rates() {
  ParticleView pv = ps_.view();
  const GridView gv = grid_.view();
  compute_density_summation(pv, gv, P_, stream_);
  compute_equation_of_state(pv, P_, stream_);
  compute_wall_boundary(pv, gv, P_, stream_);
  compute_interface_fields(pv, gv, P_, stream_);
  compute_transport_acceleration(pv, gv, P_, stream_);  // needs wall pressures and alpha
  if (P_.sps) compute_sps_viscosity(pv, gv, P_, stream_);
  compute_momentum(pv, gv, P_, stream_);
  if (cfg_.surface_tension == SurfaceTensionModel::CSF && has_vapor_) {
    smooth_interface_normals(pv, gv, P_, stream_);
    apply_csf_surface_tension(pv, gv, P_, stream_);
  } else if (cfg_.surface_tension == SurfaceTensionModel::Akinci) {
    apply_akinci_surface_tension(pv, gv, P_, stream_);
  }
  if (cfg_.density_mode == DensityMode::Continuity) compute_density_rate(pv, gv, P_, stream_);
  if (P_.thermal) {
    if (P_.heat_power != real(0)) {
      const double S = sum_heat_weight(pv, stream_);
      P_.heat_inv_norm = S > 0 ? static_cast<real>(1.0 / S) : real(0);
    } else {
      P_.heat_inv_norm = 0;
    }
    compute_thermal_rates(pv, gv, P_, stream_);
  }
  v_max_ = max_fluid_speed(pv, stream_);
  spacing_ratio_ = max_spacing_ratio(pv, P_.dx, stream_);
  a_max_ = max_fluid_accel(pv, stream_);
  Tv_max_ = has_vapor_ ? max_vapor_temperature(pv, stream_) : real(0);
  nu_t_max_ = P_.sps ? max_eddy_viscosity(pv, stream_) : real(0);
}

real Solver::compute_dt() const {
  if (cfg_.fixed_dt > 0) return cfg_.fixed_dt;
  const real h = P_.kernel.h;
  real c_max = P_.c0;
  if (has_vapor_) c_max = std::max(c_max, std::sqrt(P_.gas_R * std::max(Tv_max_, P_.T_ref)));
  // Acoustic CFL on the smoothing length, tightened when a pair comes closer
  // than half the lattice spacing (fresh split pairs): the pair stiffness then
  // scales with 1 / r_min, not 1 / h.
  real dt = cfg_.cfl * h / ((c_max + v_max_) * std::max(real(1), spacing_ratio_));
  const real nu = std::max(P_.mu[0] / P_.rho0[0], P_.mu[1] / P_.rho0[1]) + nu_t_max_;
  if (nu > 0) dt = std::min(dt, cfg_.viscous_number * h * h / nu);
  if (a_max_ > 0) dt = std::min(dt, cfg_.force_number * std::sqrt(h / a_max_));
  if (P_.sigma > 0) {
    const real rho_avg = real(0.5) * (P_.rho0[0] + (has_vapor_ ? P_.rho0[1] : P_.rho0[0]));
    dt = std::min(dt, cfg_.capillary_number * std::sqrt(rho_avg * h * h * h / (real(2) * kPi * P_.sigma)));
  }
  if (P_.thermal) {
    real alpha = P_.k[0] / (P_.rho0[0] * P_.cp[0]);
    if (has_vapor_) alpha = std::max(alpha, P_.k[1] / (P_.rho0[1] * P_.cp[1]));
    if (alpha > 0) dt = std::min(dt, cfg_.thermal_number * h * h / alpha);
  }
  return std::clamp(dt, cfg_.dt_min, cfg_.dt_max);
}

double Solver::next_dt() {
  ensure_initialized();
  return compute_dt();
}

void Solver::step() {
  ensure_initialized();
  const real dt = compute_dt();
  P_.dt = dt;
  ParticleView pv = ps_.view();
  kick_drift(pv, dt, cfg_.density_mode == DensityMode::Continuity, P_.tvf, stream_);
  grid_.build(ps_, stream_);
  if (has_maneuver_) {
    Vec3 a, w, wdot;
    maneuver_.evaluate(time_ + dt, a, w, wdot);  // frame at the new time level
    apply_frame(a, w, wdot, maneuver_.origin);
  }
  evaluate_rates();
  pv = ps_.view();
  kick(pv, dt, P_.thermal, stream_);
  if (P_.shift_strength > real(0)) apply_particle_shifting(ps_.view(), grid_.view(), P_, v_max_ / P_.c0, stream_);
  if (P_.thermal && P_.heat_inv_norm > 0) heat_input_ += double(P_.heat_power) * dt;
  if (cfg_.phase_change) phase_change_step();
  update_saturation();
  time_ += dt;
  ++step_;
  last_dt_ = dt;
}

void Solver::run_steps(long long n) {
  for (long long k = 0; k < n; ++k) step();
  mem::synchronize(stream_);
}

void Solver::run_until(double t_end) {
  while (time_ < t_end) step();
  mem::synchronize(stream_);
}

void Solver::phase_change_step() {
  const GridView gv = grid_.view();
  ParticleView pv = ps_.view();
  phase_change_sources(pv, P_, stream_);
  phase_change_normalize(pv, gv, P_, stream_);
  evaporated_ += sum_phase_change(pv, true, stream_);
  condensed_ += sum_phase_change(pv, false, stream_);
  merges_ += count_merges(pv, stream_);
  phase_change_apply(pv, gv, P_, stream_);
  if (P_.vapor_rebalance > real(0) && has_vapor_) rebalance_vapor_mass(ps_.view(), gv, P_, stream_);

  // Splitting may append up to n particles: keep head-room.
  if (ps_.size() * 4 > ps_.capacity() * 3) {
    ps_.grow(ps_.capacity() * 3 / 2 + 1024);
    telemetry_.configure(ps_.capacity(), telemetry_.mode());
  }
  pv = ps_.view();
  if (split_flag_.size() < ps_.capacity()) {
    split_flag_.resize(ps_.capacity());
    split_offset_.resize(ps_.capacity());
  }
  mark_heavy_particles(pv, P_, split_flag_.data(), stream_);
  const unsigned requested = exclusive_scan(pv.n, split_flag_.data(), split_offset_.data(), stream_);
  if (requested > 0) {
    split_heavy_particles(pv, P_, split_flag_.data(), split_offset_.data(), ps_.next_id,
                          static_cast<std::uint32_t>(step_), stream_);
    const std::size_t room = ps_.capacity() - ps_.size();
    const std::size_t added = std::min<std::size_t>(requested, room);
    ps_.set_size(ps_.size() + added);
    ps_.next_id += requested;
    splits_ += static_cast<int>(added);
  }

  if (cfg_.thermo_mode == TankThermoMode::Vented && has_vapor_) {
    pv = ps_.view();
    apply_venting(pv, P_, cfg_.vent_relaxation, stream_);
    vented_ += sum_out_mass(pv, stream_);
    vented_h_ += sum_out_enthalpy(pv, stream_);
  }
}

void Solver::update_saturation() {
  if (cfg_.thermo_mode == TankThermoMode::Closed && has_vapor_) {
    // The ullage volume is the fixed fluid volume minus the (thermodynamically
    // incompressible) liquid volume; the weakly compressible SPH volumes are
    // not used because the artificial sound speed makes them far too soft.
    double m_liq = 0, m_vap = 0, mT_vap = 0;
    phase_inventory(ps_.view(), m_liq, m_vap, mT_vap, stream_);
    if (!(fluid_volume0_ > 0)) fluid_volume0_ = m_liq / fluid_.rho_liquid + m_vap / fluid_.rho_vapor;
    const double v_ullage = fluid_volume0_ - m_liq / fluid_.rho_liquid;
    // Relative ideal-gas law around the saturated reference state (keeps the
    // reference compressibility factor of the real vapour).
    const double ratio =
        v_ullage > 0 && m_vap > 0 ? (mT_vap / v_ullage) / (double(fluid_.rho_vapor) * fluid_.T_sat) : 0.0;
    if (std::isfinite(ratio) && ratio > 0) {
      p_ullage_ = fluid_.p_sat * ratio;
      P_.T_sat = fluid_.saturation_temperature(static_cast<real>(p_ullage_));
      return;
    }
  }
  p_ullage_ = fluid_.p_sat;
  P_.T_sat = fluid_.T_sat;
}

Diagnostics Solver::diagnostics() {
  ensure_initialized();
  ParticleView pv = ps_.view();
  const stream_t s = stream_;
  Diagnostics d;
  d.time = time_;
  d.step = step_;
  d.dt = last_dt_;
  d.n_total = pv.n;
  d.n_liquid = count_phase(pv, kLiquid, s);
  d.n_vapor = count_phase(pv, kVapor, s);
  d.n_wall = count_phase(pv, kWall, s);
  d.mass_liquid = phase_sum(pv, kLiquid, 0, P_, s);
  d.mass_vapor = phase_sum(pv, kVapor, 0, P_, s);
  d.mass_total = d.mass_liquid + d.mass_vapor;
  d.kinetic_energy = phase_sum(pv, kLiquid, 2, P_, s) + phase_sum(pv, kVapor, 2, P_, s);
  d.enthalpy = phase_sum(pv, kLiquid, 9, P_, s) + phase_sum(pv, kVapor, 9, P_, s);
  d.heat_input = heat_input_;
  d.evaporated_mass = evaporated_;
  d.condensed_mass = condensed_;
  d.vented_mass = vented_;
  d.vented_enthalpy = vented_h_;
  d.T_sat = P_.T_sat;
  d.p_ullage = p_ullage_;
  d.max_speed = max_fluid_speed(pv, s);
  if (d.mass_liquid > 0) {
    d.T_liquid_mean = phase_sum(pv, kLiquid, 1, P_, s) / d.mass_liquid;
    d.T_liquid_max = phase_max_temperature(pv, kLiquid, s);
    d.liquid_com = Vec3{static_cast<real>(phase_sum(pv, kLiquid, 3, P_, s) / d.mass_liquid),
                        static_cast<real>(phase_sum(pv, kLiquid, 4, P_, s) / d.mass_liquid),
                        static_cast<real>(phase_sum(pv, kLiquid, 5, P_, s) / d.mass_liquid)};
    d.liquid_momentum = Vec3{static_cast<real>(phase_sum(pv, kLiquid, 6, P_, s)),
                             static_cast<real>(phase_sum(pv, kLiquid, 7, P_, s)),
                             static_cast<real>(phase_sum(pv, kLiquid, 8, P_, s))};
  }
  if (d.mass_vapor > 0) d.T_vapor_mean = phase_sum(pv, kVapor, 1, P_, s) / d.mass_vapor;
  d.splits = splits_;
  d.merges = merges_;
  // Reaction on the tank: minus the wall forces on the fluid.
  d.slosh_force = -sum_fluid_vec3(pv, pv.fw, s);
  d.slosh_moment = -sum_fluid_vec3(pv, pv.mw, s);
  d.frame_acceleration = frame_acc_;
  d.frame_angular_velocity = P_.omega;
  return d;
}

void Solver::set_gravity(Vec3 g) {
  cfg_.gravity = g;
  P_.gravity = g - frame_acc_;
}

void Solver::apply_frame(Vec3 a, Vec3 w, Vec3 wdot, Vec3 origin) {
  frame_acc_ = a;
  P_.gravity = cfg_.gravity - a;
  P_.omega = w;
  P_.omega_dot = wdot;
  P_.frame_origin = origin;
  P_.frame_rot = (norm2(w) > real(0) || norm2(wdot) > real(0)) ? 1 : 0;
}

void Solver::set_frame_motion(Vec3 acceleration, Vec3 angular_velocity, Vec3 angular_acceleration, Vec3 origin) {
  has_maneuver_ = false;
  apply_frame(acceleration, angular_velocity, angular_acceleration, origin);
}

void Solver::set_maneuver(const ManeuverProfile& profile) {
  profile.validate();
  maneuver_ = profile;
  has_maneuver_ = true;
  Vec3 a, w, wdot;
  maneuver_.evaluate(time_, a, w, wdot);
  apply_frame(a, w, wdot, maneuver_.origin);
}

void Solver::clear_frame_motion() {
  has_maneuver_ = false;
  apply_frame(Vec3{0, 0, 0}, Vec3{0, 0, 0}, Vec3{0, 0, 0}, Vec3{0, 0, 0});
}

void Solver::set_heater_power(real watts) {
  cfg_.heater_power = watts;
  P_.heat_power = watts;
}

void Solver::set_contact_angle(real degrees) {
  if (degrees < 0 || degrees > 180) throw std::invalid_argument("contact angle must be within [0, 180] degrees");
  cfg_.contact_angle_deg = degrees;
  const real theta = degrees * kPi / real(180);
  P_.cos_theta = std::cos(theta);
  P_.sin_theta = std::sin(theta);
}

void Solver::damp_velocities(real factor) {
  scale_velocities(ps_.view(), factor, stream_);
  mem::synchronize(stream_);
  v_max_ *= factor;
}

void Solver::relax(double duration, real damping) {
  ensure_initialized();
  const int thermal = P_.thermal;
  const bool phase_change = cfg_.phase_change;
  P_.thermal = 0;
  cfg_.phase_change = false;
  const double t_end = time_ + duration;
  try {
    while (time_ < t_end) {
      step();
      scale_velocities(ps_.view(), damping, stream_);
    }
  } catch (...) {
    P_.thermal = thermal;
    cfg_.phase_change = phase_change;
    throw;
  }
  P_.thermal = thermal;
  cfg_.phase_change = phase_change;
  scale_velocities(ps_.view(), real(0), stream_);
  mem::synchronize(stream_);
  time_ = 0;
  step_ = 0;
  last_dt_ = 0;
  heat_input_ = evaporated_ = condensed_ = vented_ = vented_h_ = 0;
  splits_ = merges_ = 0;
  // Refresh the rates (and dt limits) for the state at rest.
  evaluate_rates();
  update_saturation();
}

std::vector<real> Solver::probe_pressure(const std::vector<Vec3>& points) {
  ensure_initialized();
  const int np = static_cast<int>(points.size());
  if (np == 0) return {};
  DeviceArray<Vec3> d_pts(points.size());
  DeviceArray<real> d_out(points.size());
  d_pts.upload(points);
  probe_liquid_pressure(ps_.view(), grid_.view(), P_, d_pts.data(), d_out.data(), np, stream_);
  mem::synchronize(stream_);
  return d_out.to_host(points.size());
}

bool Solver::request_telemetry() {
  ensure_initialized();
  if (ps_.size() > ps_.capacity() || telemetry_.pending()) return false;
  return telemetry_.request(ps_, stream_, time_, step_);
}

}  // namespace cryosph
