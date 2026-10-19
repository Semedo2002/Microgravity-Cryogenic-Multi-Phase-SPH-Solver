// Top-level solver: owns particles, grid and physics, and advances them with
// kick-drift-kick under adaptive time-step limits.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cryosph/core/cuda_utils.cuh"
#include "cryosph/core/geometry.hpp"
#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/spatial_hash.hpp"
#include "cryosph/core/telemetry.hpp"
#include "cryosph/physics/fluid_properties.hpp"
#include "cryosph/physics/sim_params.cuh"

namespace cryosph {

struct SolverConfig {
  // discretisation
  real dx = real(0.005);  // particle spacing [m]
  real h_factor = real(1.3);  // h = h_factor * dx
  KernelType kernel = KernelType::WendlandC2;
  DensityMode density_mode = DensityMode::Summation;
  bool normalize_initial_masses = true;  // set m = rho / sum(W) at start-up

  // body force and weak compressibility
  Vec3 gravity{0, 0, real(-9.81)};
  real velocity_scale = 0;  // expected max speed; <= 0 estimates it from g, sigma and tank size
  real sound_speed = 0;  // <= 0: 10 * velocity_scale
  real tait_gamma = 7;
  real background_pressure = -1;  // < 0: 0.1 rho_v c0^2 with vapour, 0 without
  real vapor_stiffness = -1;  // <= 0: rho_v c0^2
  bool clamp_negative_pressure = false;

  // transport velocity (Adami et al. 2013), applied in the interface band
  bool transport_velocity = true;
  real transport_background_factor = real(2.0);  // p_b = factor * rho0 * c^2
  bool transport_stress = false;  // destabilised evaporating interfaces, so off
  bool transport_interface_only = true;

  // particle shifting (Sun et al. 2017) -- off: no noise benefit, unstable in a 1 g column
  bool particle_shifting = false;
  real shifting_strength = real(0.25);

  // viscosity
  real artificial_viscosity = real(0.02);  // liquid alpha
  real artificial_viscosity_vapor = real(0.2);  // the vapour needs much more acoustic damping
  real delta_sph = real(0.1);  // density diffusion, continuity mode only
  bool no_slip = true;
  bool sps_turbulence = false;
  real smagorinsky_cs = real(0.12);

  // surface tension and wetting
  SurfaceTensionModel surface_tension = SurfaceTensionModel::CSF;
  real contact_angle_deg = 90;  // measured through the liquid
  real contact_angle_wall_weight = real(0.01);  // wall weight above which the angle is fully imposed
  bool wall_colour = true;  // walls carry an extrapolated colour in grad(c)
  WettingModel wetting_model = WettingModel::NormalRotation;
  bool csf_smooth_normals = false;
  bool csf_balanced = false;
  real akinci_gamma = -1;  // < 0: derived from sigma
  real akinci_beta = -1;  // < 0: derived from the contact angle

  // heat transfer and phase change
  bool thermal = true;
  bool phase_change = true;
  WallThermalBC wall_thermal_bc = WallThermalBC::Adiabatic;
  real heater_power = 0;  // [W], spread over heated wall particles
  real conductivity_scale = 1;
  real lee_evaporation = 10;  // [1/s]
  real lee_condensation = 10;  // [1/s]
  TankThermoMode thermo_mode = TankThermoMode::Vented;
  real vent_relaxation = real(0.02);  // fraction of excess vapour vented per step
  real merge_ratio = real(0.3);
  real split_ratio = real(2.0);
  real max_transfer_fraction = real(0.25);
  real phase_change_density_limit = real(0.01);  // max vapour density change per step from phase change
  real vapor_mass_rebalance = real(1.0);  // vapour mass diffusion, see rebalance_vapor_mass

  // time stepping
  real cfl = real(0.25);
  real viscous_number = real(0.125);
  real force_number = real(0.25);
  real capillary_number = real(0.25);
  real thermal_number = real(0.1);
  real dt_max = real(1e-2);
  real dt_min = real(1e-9);
  real fixed_dt = 0;  // > 0 disables adaptive stepping
};

// Tank motion in the tank frame: acceleration a(t) and angular velocity w(t),
// piecewise linear, held at the end values outside the table. The liquid sees
// g - a - dw/dt x r - w x (w x r) - 2 w x v, with r measured from `origin`.
struct ManeuverProfile {
  std::vector<double> time;  ///< strictly increasing [s]
  std::vector<Vec3> acceleration;  ///< tank acceleration [m/s^2], same length as time
  std::vector<Vec3> angular_velocity;  ///< tank angular velocity [rad/s], same length (or empty)
  Vec3 origin{0, 0, 0};
  /// Frame state at time t.
  void evaluate(double t, Vec3& a, Vec3& omega, Vec3& omega_dot) const;
  /// Throws std::invalid_argument when the table is malformed.
  void validate() const;
};

/// Global state summary (all sums in double precision).
struct Diagnostics {
  double time = 0;
  long long step = 0;
  double dt = 0;
  int n_total = 0, n_liquid = 0, n_vapor = 0, n_wall = 0;
  double mass_liquid = 0, mass_vapor = 0, mass_total = 0;
  double kinetic_energy = 0;
  double enthalpy = 0;  ///< sum m h (reference: saturated liquid at T_ref)
  double heat_input = 0;  ///< cumulative heater energy [J]
  double evaporated_mass = 0;  ///< cumulative [kg]
  double condensed_mass = 0;  ///< cumulative [kg]
  double vented_mass = 0;  ///< cumulative [kg]
  double vented_enthalpy = 0;  ///< cumulative [J]
  double T_sat = 0;
  double p_ullage = 0;  ///< thermodynamic ullage pressure [Pa]
  double max_speed = 0;
  double T_liquid_mean = 0, T_liquid_max = 0, T_vapor_mean = 0;
  Vec3 liquid_com{0, 0, 0};
  Vec3 liquid_momentum{0, 0, 0};
  int splits = 0, merges = 0;  ///< cumulative topology events
  Vec3 slosh_force{0, 0, 0};  ///< force of the fluid on the tank walls [N] (tank frame)
  Vec3 slosh_moment{0, 0, 0};  ///< its moment about the frame origin [N m]
  Vec3 frame_acceleration{0, 0, 0};  ///< current tank acceleration [m/s^2]
  Vec3 frame_angular_velocity{0, 0, 0};  ///< current tank angular velocity [rad/s]
};

class Solver {
 public:
  Solver(const SolverConfig& config, const FluidProperties& fluid);
  ~Solver();
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  /// Loads particles; `domain` must enclose every particle (walls included).
  void set_particles(const HostParticles& particles, const AABB& domain);
  /// Convenience: generates and loads a filled tank.
  void build_tank(const Geometry& geometry, const FillSpec& fill);

  /// Advances one time step.
  void step();
  /// Advances by n steps.
  void run_steps(long long n);
  /// Advances until time() >= t_end (last step is not truncated).
  void run_until(double t_end);

  double time() const noexcept { return time_; }
  long long step_count() const noexcept { return step_; }
  double last_dt() const noexcept { return last_dt_; }
  /// Time step the next call to step() will use.
  double next_dt();

  Diagnostics diagnostics();
  HostParticles particles() const { return ps_.download(); }
  std::vector<real> pressure() const { return ps_.download_pressure(); }
  std::vector<real> curvature() const { return ps_.download_curvature(); }
  std::size_t num_particles() const { return ps_.size(); }

  // run-time controls
  void set_gravity(Vec3 g);
  /// Tank-frame motion held constant from now on (overrides a maneuver).
  void set_frame_motion(Vec3 acceleration, Vec3 angular_velocity, Vec3 angular_acceleration, Vec3 origin);
  /// Prescribed motion, evaluated every step at the new time level.
  void set_maneuver(const ManeuverProfile& profile);
  /// Back to an inertial tank (no frame acceleration or rotation).
  void clear_frame_motion();
  void set_heater_power(real watts);
  void set_contact_angle(real degrees);
  /// Multiplies all fluid velocities by `factor` (damped relaxation of an
  /// initial particle distribution).
  void damp_velocities(real factor);
  /// Settle the start lattice: run `duration` seconds with heat and phase change
  /// frozen and velocities damped, then reset velocities, clock and counters.
  void relax(double duration, real damping = real(0.98));

  const SolverConfig& config() const noexcept { return cfg_; }
  const FluidProperties& fluid() const noexcept { return fluid_; }
  const SimParams& params() const noexcept { return P_; }
  ParticleSystem& particle_system() noexcept { return ps_; }
  SpatialHash& grid() noexcept { return grid_; }
  TelemetryStream& telemetry() noexcept { return telemetry_; }
  stream_t stream() const noexcept { return stream_; }
  std::string device() const;

  /// Captures a telemetry frame asynchronously (see TelemetryStream).
  bool request_telemetry();

  /// Pressure at arbitrary points (Shepard average over nearby liquid; NaN if dry).
  /// Put sensor points about dx/2 off a wall on the fluid side.
  std::vector<real> probe_pressure(const std::vector<Vec3>& points);

 private:
  void init_params();
  void apply_frame(Vec3 a, Vec3 w, Vec3 wdot, Vec3 origin);
  void ensure_initialized();
  void evaluate_rates();
  void phase_change_step();
  void update_saturation();
  real compute_dt() const;

  SolverConfig cfg_;
  FluidProperties fluid_;
  SimParams P_{};
  ParticleSystem ps_;
  SpatialHash grid_;
  TelemetryStream telemetry_;
  stream_t stream_ = nullptr;
  DeviceArray<unsigned> split_flag_, split_offset_;  // deterministic split slots

  bool initialized_ = false;
  bool has_vapor_ = false;
  double time_ = 0;
  long long step_ = 0;
  double last_dt_ = 0;
  real char_length_ = 0;
  // maxima from the last rate evaluation (for the adaptive time step)
  real v_max_ = 0, a_max_ = 0, Tv_max_ = 0, nu_t_max_ = 0, spacing_ratio_ = 1;
  // cumulative bookkeeping
  double heat_input_ = 0, evaporated_ = 0, condensed_ = 0, vented_ = 0, vented_h_ = 0;
  double p_ullage_ = 0;
  bool has_maneuver_ = false;
  ManeuverProfile maneuver_;
  Vec3 frame_acc_{0, 0, 0};
  double fluid_volume0_ = 0;  ///< liquid + vapour volume at start (closed tank)
  int splits_ = 0, merges_ = 0;
};

}  // namespace cryosph
