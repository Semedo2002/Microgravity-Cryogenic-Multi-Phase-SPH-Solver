// Python module `CryoSPH`. Arrays cross as NumPy copies; long calls drop the GIL.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "cryosph/core/geometry.hpp"
#include "cryosph/core/io.hpp"
#include "cryosph/solvers/sph_solver.hpp"

namespace py = pybind11;
using namespace cryosph;

namespace {

using Tuple3 = std::array<double, 3>;

Vec3 to_vec(const Tuple3& t) {
  return Vec3{static_cast<real>(t[0]), static_cast<real>(t[1]), static_cast<real>(t[2])};
}
py::tuple to_tuple(Vec3 v) { return py::make_tuple(double(v.x), double(v.y), double(v.z)); }

template <class T>
py::array_t<T> to_array(const std::vector<T>& v) {
  py::array_t<T> out(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
  return out;
}

py::array_t<real> vec3_array(const std::vector<Vec3>& v) {
  py::array_t<real> out({static_cast<py::ssize_t>(v.size()), py::ssize_t{3}});
  auto r = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size()); ++i) {
    r(i, 0) = v[i].x;
    r(i, 1) = v[i].y;
    r(i, 2) = v[i].z;
  }
  return out;
}

py::dict particles_to_dict(const HostParticles& hp) {
  py::dict d;
  d["position"] = vec3_array(hp.x);
  d["velocity"] = vec3_array(hp.v);
  d["normal"] = vec3_array(hp.nrm);
  d["mass"] = to_array(hp.m);
  d["density"] = to_array(hp.rho);
  d["temperature"] = to_array(hp.T);
  d["phase"] = to_array(hp.phase);
  d["flags"] = to_array(hp.flags);
  d["id"] = to_array(hp.id);
  return d;
}

py::dict frame_to_dict(const TelemetryFrame& f) {
  py::dict d;
  d["time"] = f.time;
  d["step"] = f.step;
  py::array_t<float> pos({static_cast<py::ssize_t>(f.n), py::ssize_t{3}});
  if (f.n > 0) std::memcpy(pos.mutable_data(), f.position.data(), f.position.size() * sizeof(float));
  d["position"] = pos;
  d["temperature"] = to_array(f.temperature);
  d["pressure"] = to_array(f.pressure);
  d["speed"] = to_array(f.speed);
  d["phase"] = to_array(f.phase);
  return d;
}

/// Builds HostParticles from NumPy inputs (all arrays of length N).
HostParticles particles_from_arrays(py::array_t<double, py::array::c_style | py::array::forcecast> position,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> mass,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> density,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> temperature,
                                    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> phase,
                                    py::object velocity, py::object normal, py::object flags) {
  if (position.ndim() != 2 || position.shape(1) != 3) throw std::invalid_argument("position must have shape (N, 3)");
  const py::ssize_t n = position.shape(0);
  auto check = [n](const py::array& a, const char* name) {
    if (a.size() != n) throw std::invalid_argument(std::string(name) + " must have length N");
  };
  check(mass, "mass");
  check(density, "density");
  check(temperature, "temperature");
  check(phase, "phase");
  using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
  DArr vel, nrm;
  if (!velocity.is_none()) {
    vel = velocity.cast<DArr>();
    if (vel.ndim() != 2 || vel.shape(0) != n || vel.shape(1) != 3)
      throw std::invalid_argument("velocity must have shape (N, 3)");
  }
  if (!normal.is_none()) {
    nrm = normal.cast<DArr>();
    if (nrm.ndim() != 2 || nrm.shape(0) != n || nrm.shape(1) != 3)
      throw std::invalid_argument("normal must have shape (N, 3)");
  }
  py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> fl;
  if (!flags.is_none()) {
    fl = flags.cast<py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>>();
    check(fl, "flags");
  }
  auto P = position.unchecked<2>();
  const double* m = mass.data();
  const double* rho = density.data();
  const double* T = temperature.data();
  const std::uint8_t* ph = phase.data();
  HostParticles hp;
  hp.reserve(static_cast<std::size_t>(n));
  for (py::ssize_t i = 0; i < n; ++i) {
    if (ph[i] > kWall) throw std::invalid_argument("phase values must be 0 (liquid), 1 (vapour) or 2 (wall)");
    const Vec3 x{static_cast<real>(P(i, 0)), static_cast<real>(P(i, 1)), static_cast<real>(P(i, 2))};
    Vec3 v{0, 0, 0}, nn{0, 0, 0};
    if (vel.size()) {
      const double* vv = vel.data() + 3 * i;
      v = Vec3{static_cast<real>(vv[0]), static_cast<real>(vv[1]), static_cast<real>(vv[2])};
    }
    if (nrm.size()) {
      const double* q = nrm.data() + 3 * i;
      nn = Vec3{static_cast<real>(q[0]), static_cast<real>(q[1]), static_cast<real>(q[2])};
    }
    hp.push(x, v, static_cast<real>(m[i]), static_cast<real>(rho[i]), static_cast<real>(T[i]),
            static_cast<Phase>(ph[i]), nn, fl.size() ? fl.data()[i] : std::uint8_t{0});
  }
  return hp;
}

}  // namespace

PYBIND11_MODULE(CryoSPH, m) {
  m.doc() = "CryoSPH-CUDA: GPU multiphase SPH for cryogenic propellant management in microgravity";
  m.attr("__version__") = "0.2.0";
  m.attr("cuda_backend") = kCudaBackend;
  m.attr("double_precision") = sizeof(real) == sizeof(double);
  m.def("device_name", &mem::device_name, "Name of the compute device used by the solver.");
  m.def("device_count", &mem::device_count, "Number of CUDA devices (0 on the CPU backend).");

  // enums
  py::enum_<Phase>(m, "Phase")
      .value("Liquid", Phase::Liquid)
      .value("Vapor", Phase::Vapor)
      .value("Wall", Phase::Wall);
  py::enum_<KernelType>(m, "KernelType")
      .value("CubicSpline", KernelType::CubicSpline)
      .value("WendlandC2", KernelType::WendlandC2);
  py::enum_<DensityMode>(m, "DensityMode")
      .value("Summation", DensityMode::Summation)
      .value("Continuity", DensityMode::Continuity);
  py::enum_<SurfaceTensionModel>(m, "SurfaceTensionModel")
      .value("NoTension", SurfaceTensionModel::None)
      .value("CSF", SurfaceTensionModel::CSF)
      .value("Akinci", SurfaceTensionModel::Akinci);
  py::enum_<WallThermalBC>(m, "WallThermalBC")
      .value("Adiabatic", WallThermalBC::Adiabatic)
      .value("Isothermal", WallThermalBC::Isothermal);
  py::enum_<TankThermoMode>(m, "TankThermoMode")
      .value("Vented", TankThermoMode::Vented)
      .value("Closed", TankThermoMode::Closed);
  py::enum_<FillMode>(m, "FillMode")
      .value("Settled", FillMode::Settled)
      .value("CentralBubble", FillMode::CentralBubble)
      .value("Full", FillMode::Full);
  py::enum_<WettingModel>(m, "WettingModel")
      .value("NormalRotation", WettingModel::NormalRotation)
      .value("ContactLineForce", WettingModel::ContactLineForce)
      .value("Both", WettingModel::Both);
  py::enum_<TelemetryMode>(m, "TelemetryMode")
      .value("AsyncCopy", TelemetryMode::AsyncCopy)
      .value("ZeroCopy", TelemetryMode::ZeroCopy);

  // fluids
  py::class_<FluidProperties>(m, "FluidProperties", "Saturated liquid/vapour property set (SI units).")
      .def(py::init<>())
      .def_readwrite("name", &FluidProperties::name)
      .def_readwrite("rho_liquid", &FluidProperties::rho_liquid)
      .def_readwrite("rho_vapor", &FluidProperties::rho_vapor)
      .def_readwrite("mu_liquid", &FluidProperties::mu_liquid)
      .def_readwrite("mu_vapor", &FluidProperties::mu_vapor)
      .def_readwrite("k_liquid", &FluidProperties::k_liquid)
      .def_readwrite("k_vapor", &FluidProperties::k_vapor)
      .def_readwrite("cp_liquid", &FluidProperties::cp_liquid)
      .def_readwrite("cp_vapor", &FluidProperties::cp_vapor)
      .def_readwrite("sigma", &FluidProperties::sigma)
      .def_readwrite("h_fg", &FluidProperties::h_fg)
      .def_readwrite("T_sat", &FluidProperties::T_sat)
      .def_readwrite("p_sat", &FluidProperties::p_sat)
      .def_readwrite("R_gas", &FluidProperties::R_gas)
      .def("saturation_temperature", &FluidProperties::saturation_temperature, py::arg("p"))
      .def("saturation_pressure", &FluidProperties::saturation_pressure, py::arg("T"))
      .def_static("liquid_hydrogen", &FluidProperties::liquid_hydrogen)
      .def_static("liquid_nitrogen", &FluidProperties::liquid_nitrogen)
      .def_static("liquid_oxygen", &FluidProperties::liquid_oxygen)
      .def_static("liquid_methane", &FluidProperties::liquid_methane)
      .def_static("water", &FluidProperties::water)
      .def("__repr__", [](const FluidProperties& f) {
        return "<FluidProperties " + f.name + " rho_l=" + std::to_string(f.rho_liquid) +
               " rho_v=" + std::to_string(f.rho_vapor) + " T_sat=" + std::to_string(f.T_sat) + ">";
      });

  // geometry
  py::class_<Geometry, std::shared_ptr<Geometry>>(m, "Geometry", "Tank geometry given by a signed distance function.")
      .def("sdf", &Geometry::sdf, py::arg("x"), py::arg("y"), py::arg("z"))
      .def("volume", &Geometry::volume)
      .def_property_readonly("name", &Geometry::name)
      .def_property_readonly("bounds", [](const Geometry& g) {
        const AABB b = g.bounds();
        return std::make_pair(to_tuple(b.lo), to_tuple(b.hi));
      });
  py::class_<CylinderTank, Geometry, std::shared_ptr<CylinderTank>>(m, "CylinderTank")
      .def(py::init<double, double>(), py::arg("radius"), py::arg("height"));
  py::class_<CapsuleTank, Geometry, std::shared_ptr<CapsuleTank>>(m, "CapsuleTank")
      .def(py::init<double, double>(), py::arg("radius"), py::arg("length"));
  py::class_<SphereTank, Geometry, std::shared_ptr<SphereTank>>(m, "SphereTank")
      .def(py::init<double>(), py::arg("radius"));
  py::class_<BoxTank, Geometry, std::shared_ptr<BoxTank>>(m, "BoxTank")
      .def(py::init<double, double, double>(), py::arg("lx"), py::arg("ly"), py::arg("lz"));
  py::class_<Translated, Geometry, std::shared_ptr<Translated>>(m, "Translated",
                                                               "Geometry shifted by a constant offset.")
      .def(py::init([](std::shared_ptr<Geometry> g, const Tuple3& o) { return std::make_shared<Translated>(g, to_vec(o)); }),
           py::arg("geometry"), py::arg("offset"));
  py::class_<Difference, Geometry, std::shared_ptr<Difference>>(
      m, "Difference", "Fluid domain of `a` minus the solid `b` (obstacles, baffles, PMD vanes).")
      .def(py::init<std::shared_ptr<const Geometry>, std::shared_ptr<const Geometry>>(), py::arg("a"), py::arg("b"));
  py::class_<Union, Geometry, std::shared_ptr<Union>>(m, "Union", "Fluid domain of `a` united with `b`.")
      .def(py::init<std::shared_ptr<const Geometry>, std::shared_ptr<const Geometry>>(), py::arg("a"), py::arg("b"));
  py::class_<MeshTank, Geometry, std::shared_ptr<MeshTank>>(m, "MeshTank")
      .def(py::init([](const std::string& path, double scale, const Tuple3& offset) {
             return std::make_shared<MeshTank>(path, scale, to_vec(offset));
           }),
           py::arg("obj_path"), py::arg("scale") = 1.0, py::arg("offset") = Tuple3{0, 0, 0})
      .def_property_readonly("num_triangles", &MeshTank::num_triangles)
      .def("winding_number", &MeshTank::winding_number, py::arg("x"), py::arg("y"), py::arg("z"));

  py::class_<FillSpec>(m, "FillSpec", "How the generator fills a tank with liquid and vapour.")
      .def(py::init<>())
      .def_readwrite("mode", &FillSpec::mode)
      .def_readwrite("fill_fraction", &FillSpec::fill_fraction)
      .def_property(
          "settle_direction", [](const FillSpec& f) { return to_tuple(f.settle_direction); },
          [](FillSpec& f, const Tuple3& t) { f.settle_direction = to_vec(t); })
      .def_property(
          "bubble_center", [](const FillSpec& f) { return to_tuple(f.bubble_center); },
          [](FillSpec& f, const Tuple3& t) { f.bubble_center = to_vec(t); })
      .def_readwrite("include_vapor", &FillSpec::include_vapor)
      .def_readwrite("wall_layers", &FillSpec::wall_layers)
      .def_readwrite("T_liquid", &FillSpec::T_liquid)
      .def_readwrite("T_vapor", &FillSpec::T_vapor)
      .def_readwrite("T_wall", &FillSpec::T_wall)
      .def_readwrite("heater_zmin", &FillSpec::heater_zmin)
      .def_readwrite("heater_zmax", &FillSpec::heater_zmax);

  // config
  py::class_<SolverConfig> cfg(m, "SolverConfig", "Solver parameters; see sph_solver.hpp for documentation.");
  cfg.def(py::init<>())
      .def_property(
          "gravity", [](const SolverConfig& c) { return to_tuple(c.gravity); },
          [](SolverConfig& c, const Tuple3& t) { c.gravity = to_vec(t); });
#define CRYOSPH_RW(field) cfg.def_readwrite(#field, &SolverConfig::field)
  CRYOSPH_RW(dx);
  CRYOSPH_RW(h_factor);
  CRYOSPH_RW(kernel);
  CRYOSPH_RW(density_mode);
  CRYOSPH_RW(normalize_initial_masses);
  CRYOSPH_RW(velocity_scale);
  CRYOSPH_RW(sound_speed);
  CRYOSPH_RW(tait_gamma);
  CRYOSPH_RW(background_pressure);
  CRYOSPH_RW(vapor_stiffness);
  CRYOSPH_RW(clamp_negative_pressure);
  CRYOSPH_RW(transport_velocity);
  CRYOSPH_RW(transport_background_factor);
  CRYOSPH_RW(transport_stress);
  CRYOSPH_RW(transport_interface_only);
  CRYOSPH_RW(particle_shifting);
  CRYOSPH_RW(shifting_strength);
  CRYOSPH_RW(artificial_viscosity);
  CRYOSPH_RW(artificial_viscosity_vapor);
  CRYOSPH_RW(delta_sph);
  CRYOSPH_RW(no_slip);
  CRYOSPH_RW(sps_turbulence);
  CRYOSPH_RW(smagorinsky_cs);
  CRYOSPH_RW(surface_tension);
  CRYOSPH_RW(contact_angle_deg);
  CRYOSPH_RW(contact_angle_wall_weight);
  CRYOSPH_RW(wall_colour);
  CRYOSPH_RW(wetting_model);
  CRYOSPH_RW(csf_smooth_normals);
  CRYOSPH_RW(csf_balanced);
  CRYOSPH_RW(akinci_gamma);
  CRYOSPH_RW(akinci_beta);
  CRYOSPH_RW(thermal);
  CRYOSPH_RW(phase_change);
  CRYOSPH_RW(wall_thermal_bc);
  CRYOSPH_RW(heater_power);
  CRYOSPH_RW(conductivity_scale);
  CRYOSPH_RW(lee_evaporation);
  CRYOSPH_RW(lee_condensation);
  CRYOSPH_RW(thermo_mode);
  CRYOSPH_RW(vent_relaxation);
  CRYOSPH_RW(merge_ratio);
  CRYOSPH_RW(split_ratio);
  CRYOSPH_RW(max_transfer_fraction);
  CRYOSPH_RW(phase_change_density_limit);
  CRYOSPH_RW(vapor_mass_rebalance);
  CRYOSPH_RW(cfl);
  CRYOSPH_RW(viscous_number);
  CRYOSPH_RW(force_number);
  CRYOSPH_RW(capillary_number);
  CRYOSPH_RW(thermal_number);
  CRYOSPH_RW(dt_max);
  CRYOSPH_RW(dt_min);
  CRYOSPH_RW(fixed_dt);
#undef CRYOSPH_RW

  // diagnostics
  py::class_<Diagnostics> diag(m, "Diagnostics", "Global state summary returned by Solver.diagnostics().");
#define CRYOSPH_RO(field) diag.def_readonly(#field, &Diagnostics::field)
  CRYOSPH_RO(time);
  CRYOSPH_RO(step);
  CRYOSPH_RO(dt);
  CRYOSPH_RO(n_total);
  CRYOSPH_RO(n_liquid);
  CRYOSPH_RO(n_vapor);
  CRYOSPH_RO(n_wall);
  CRYOSPH_RO(mass_liquid);
  CRYOSPH_RO(mass_vapor);
  CRYOSPH_RO(mass_total);
  CRYOSPH_RO(kinetic_energy);
  CRYOSPH_RO(enthalpy);
  CRYOSPH_RO(heat_input);
  CRYOSPH_RO(evaporated_mass);
  CRYOSPH_RO(condensed_mass);
  CRYOSPH_RO(vented_mass);
  CRYOSPH_RO(vented_enthalpy);
  CRYOSPH_RO(T_sat);
  CRYOSPH_RO(p_ullage);
  CRYOSPH_RO(max_speed);
  CRYOSPH_RO(T_liquid_mean);
  CRYOSPH_RO(T_liquid_max);
  CRYOSPH_RO(T_vapor_mean);
  CRYOSPH_RO(splits);
  CRYOSPH_RO(merges);
#undef CRYOSPH_RO
  diag.def_property_readonly("liquid_com", [](const Diagnostics& d) { return to_tuple(d.liquid_com); })
      .def_property_readonly("slosh_force", [](const Diagnostics& d) { return to_tuple(d.slosh_force); })
      .def_property_readonly("slosh_moment", [](const Diagnostics& d) { return to_tuple(d.slosh_moment); })
      .def_property_readonly("frame_acceleration", [](const Diagnostics& d) { return to_tuple(d.frame_acceleration); })
      .def_property_readonly("frame_angular_velocity",
                             [](const Diagnostics& d) { return to_tuple(d.frame_angular_velocity); })
      .def_property_readonly("liquid_momentum", [](const Diagnostics& d) { return to_tuple(d.liquid_momentum); })
      .def("as_dict", [](const Diagnostics& d) {
        py::dict r;
        py::object self = py::cast(d);
        for (const char* k :
             {"time", "step", "dt", "n_total", "n_liquid", "n_vapor", "n_wall", "mass_liquid", "mass_vapor",
              "mass_total", "kinetic_energy", "enthalpy", "heat_input", "evaporated_mass", "condensed_mass",
              "vented_mass", "vented_enthalpy", "T_sat", "p_ullage", "max_speed", "T_liquid_mean", "T_liquid_max",
              "T_vapor_mean", "splits", "merges", "liquid_com", "liquid_momentum", "slosh_force", "slosh_moment",
              "frame_acceleration", "frame_angular_velocity"})
          r[k] = self.attr(k);
        return r;
      });

  // maneuver
  py::class_<ManeuverProfile>(m, "ManeuverProfile",
                              "Prescribed tank motion (tank frame): piecewise-linear acceleration and angular velocity.")
      .def(py::init([](std::vector<double> time, py::array_t<double, py::array::c_style | py::array::forcecast> acc,
                       py::object omega, const Tuple3& origin) {
             auto rows = [](const py::array_t<double, py::array::c_style | py::array::forcecast>& a, std::size_t n,
                            const char* what) {
               if (a.ndim() != 2 || a.shape(1) != 3 || static_cast<std::size_t>(a.shape(0)) != n)
                 throw std::invalid_argument(std::string(what) + " must have shape (len(time), 3)");
               std::vector<Vec3> v(n);
               auto r = a.unchecked<2>();
               for (std::size_t i = 0; i < n; ++i)
                 v[i] = Vec3{static_cast<real>(r(i, 0)), static_cast<real>(r(i, 1)), static_cast<real>(r(i, 2))};
               return v;
             };
             ManeuverProfile p;
             p.time = std::move(time);
             p.acceleration = rows(acc, p.time.size(), "acceleration");
             if (!omega.is_none())
               p.angular_velocity = rows(omega.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
                                         p.time.size(), "angular_velocity");
             p.origin = to_vec(origin);
             p.validate();
             return p;
           }),
           py::arg("time"), py::arg("acceleration"), py::arg("angular_velocity") = py::none(),
           py::arg("origin") = Tuple3{0, 0, 0})
      .def("evaluate", [](const ManeuverProfile& p, double t) {
        Vec3 a, w, wd;
        p.evaluate(t, a, w, wd);
        return py::make_tuple(to_tuple(a), to_tuple(w), to_tuple(wd));
      }, py::arg("t"), "Returns (acceleration, angular_velocity, angular_acceleration) at time t.");

  // solver
  py::class_<Solver>(m, "Solver", "Weakly-compressible multiphase SPH solver.")
      .def(py::init<const SolverConfig&, const FluidProperties&>(), py::arg("config"), py::arg("fluid"))
      .def(
          "build_tank", [](Solver& s, const Geometry& g, const FillSpec& f) { s.build_tank(g, f); },
          py::arg("geometry"), py::arg("fill") = FillSpec{}, "Generates wall/liquid/vapour particles for a tank.")
      .def(
          "set_particles",
          [](Solver& s, py::array_t<double, py::array::c_style | py::array::forcecast> position,
             py::array_t<double, py::array::c_style | py::array::forcecast> mass,
             py::array_t<double, py::array::c_style | py::array::forcecast> density,
             py::array_t<double, py::array::c_style | py::array::forcecast> temperature,
             py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> phase, const Tuple3& domain_lo,
             const Tuple3& domain_hi, py::object velocity, py::object normal, py::object flags) {
            const HostParticles hp =
                particles_from_arrays(position, mass, density, temperature, phase, velocity, normal, flags);
            s.set_particles(hp, AABB{to_vec(domain_lo), to_vec(domain_hi)});
          },
          py::arg("position"), py::arg("mass"), py::arg("density"), py::arg("temperature"), py::arg("phase"),
          py::arg("domain_lo"), py::arg("domain_hi"), py::arg("velocity") = py::none(),
          py::arg("normal") = py::none(), py::arg("flags") = py::none(),
          "Loads a custom particle set (arrays of length N; position (N,3)).")
      .def("step", &Solver::step, py::call_guard<py::gil_scoped_release>())
      .def("run_steps", &Solver::run_steps, py::arg("n"), py::call_guard<py::gil_scoped_release>())
      .def("run_until", &Solver::run_until, py::arg("t_end"), py::call_guard<py::gil_scoped_release>())
      .def("relax", &Solver::relax, py::arg("duration"), py::arg("damping") = real(0.98),
           py::call_guard<py::gil_scoped_release>())
      .def("next_dt", &Solver::next_dt)
      .def("diagnostics", &Solver::diagnostics)
      .def(
          "particles", [](const Solver& s) { return particles_to_dict(s.particles()); },
          "Copies the particle state into a dict of NumPy arrays.")
      .def("pressure", [](const Solver& s) { return to_array(s.pressure()); })
      .def("curvature", [](const Solver& s) { return to_array(s.curvature()); })
      .def_property_readonly("time", &Solver::time)
      .def_property_readonly("step_count", &Solver::step_count)
      .def_property_readonly("last_dt", &Solver::last_dt)
      .def_property_readonly("num_particles", &Solver::num_particles)
      .def_property_readonly("device", &Solver::device)
      .def_property_readonly("config", &Solver::config)
      .def_property_readonly("fluid", &Solver::fluid)
      .def_property_readonly("sound_speed", [](const Solver& s) { return s.params().c0; })
      .def_property_readonly("smoothing_length", [](const Solver& s) { return s.params().kernel.h; })
      .def("set_gravity", [](Solver& s, const Tuple3& g) { s.set_gravity(to_vec(g)); }, py::arg("g"))
      .def("set_heater_power", &Solver::set_heater_power, py::arg("watts"))
      .def(
          "set_frame_motion",
          [](Solver& s, const Tuple3& acc, const Tuple3& w, const Tuple3& wdot, const Tuple3& origin) {
            s.set_frame_motion(to_vec(acc), to_vec(w), to_vec(wdot), to_vec(origin));
          },
          py::arg("acceleration") = Tuple3{0, 0, 0}, py::arg("angular_velocity") = Tuple3{0, 0, 0},
          py::arg("angular_acceleration") = Tuple3{0, 0, 0}, py::arg("origin") = Tuple3{0, 0, 0},
          "Constant tank-frame motion: linear acceleration of the tank, angular velocity and acceleration.")
      .def("set_maneuver", &Solver::set_maneuver, py::arg("profile"))
      .def("clear_frame_motion", &Solver::clear_frame_motion)
      .def("set_contact_angle", &Solver::set_contact_angle, py::arg("degrees"))
      .def("damp_velocities", &Solver::damp_velocities, py::arg("factor"))
      .def(
          "configure_telemetry",
          [](Solver& s, TelemetryMode mode) { s.telemetry().configure(s.particle_system().capacity(), mode); },
          py::arg("mode"), "Selects AsyncCopy (pinned DMA) or ZeroCopy (mapped host memory) telemetry.")
      .def_property_readonly("telemetry_mode", [](Solver& s) { return s.telemetry().mode(); })
      .def(
          "probe_pressure",
          [](Solver& s, py::array_t<double, py::array::c_style | py::array::forcecast> pts) {
            if (pts.ndim() != 2 || pts.shape(1) != 3) throw std::invalid_argument("points must have shape (N, 3)");
            std::vector<Vec3> v(static_cast<std::size_t>(pts.shape(0)));
            auto r = pts.unchecked<2>();
            for (py::ssize_t i = 0; i < pts.shape(0); ++i)
              v[i] = Vec3{static_cast<real>(r(i, 0)), static_cast<real>(r(i, 1)), static_cast<real>(r(i, 2))};
            return to_array(s.probe_pressure(v));
          },
          py::arg("points"),
          "Virtual pressure sensors: Shepard-weighted liquid pressure at each point (N, 3); NaN where dry.")
      .def("request_telemetry", &Solver::request_telemetry,
           "Enqueues a non-blocking capture of the current state; returns False if one is pending.")
      .def("telemetry_ready", [](Solver& s) { return s.telemetry().ready(); })
      .def(
          "fetch_telemetry",
          [](Solver& s) -> py::object {
            TelemetryFrame f;
            bool ok;
            {
              py::gil_scoped_release release;
              ok = s.telemetry().fetch(f);
            }
            if (!ok) return py::none();
            return frame_to_dict(f);
          },
          "Waits for the pending capture and returns it as a dict (None if nothing was requested).")
      .def(
          "write_vtk", [](const Solver& s, const std::string& path) { write_vtk(path, s.particles(), s.pressure()); },
          py::arg("path"))
      .def(
          "write_json",
          [](const Solver& s, const std::string& path, bool include_walls) {
            write_json_frame(path, make_frame(s.particles(), s.pressure(), s.time(), s.step_count()), include_walls);
          },
          py::arg("path"), py::arg("include_walls") = false);
}
