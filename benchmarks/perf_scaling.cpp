// Throughput benchmark: half-full LN2 cylinder at several resolutions,
// optionally with telemetry capture every N steps.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "cryosph/solvers/sph_solver.hpp"

using namespace cryosph;

namespace {

struct Options {
  int steps = 200;
  std::vector<double> dx{0.006, 0.004, 0.003, 0.0025};
  bool telemetry = false;
  int every = 10;
  std::string csv;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--steps") {
      o.steps = std::atoi(next().c_str());
    } else if (a == "--dx") {
      o.dx.clear();
      std::stringstream ss(next());
      std::string tok;
      while (std::getline(ss, tok, ',')) o.dx.push_back(std::atof(tok.c_str()));
    } else if (a == "--telemetry") {
      o.telemetry = true;
    } else if (a == "--every") {
      o.every = std::max(1, std::atoi(next().c_str()));
    } else if (a == "--csv") {
      o.csv = next();
    } else {
      std::fprintf(stderr, "usage: %s [--steps N] [--dx a,b,c] [--telemetry] [--every K] [--csv file]\n", argv[0]);
      std::exit(a == "--help" || a == "-h" ? 0 : 2);
    }
  }
  return o;
}

struct Result {
  double dx;
  std::size_t n;
  std::string mode;
  double ms_per_step;
  double mpups;  // million particle updates per second
  int frames;
};

Result run_case(double dx, int steps, const char* mode, int every) {
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  SolverConfig cfg;
  cfg.dx = static_cast<real>(dx);
  cfg.gravity = Vec3{0, 0, real(-1e-4)};  // residual milli-g
  cfg.contact_angle_deg = 5;
  cfg.velocity_scale = real(0.5);
  cfg.lee_evaporation = real(0.5);
  cfg.lee_condensation = real(0.5);
  cfg.heater_power = real(5);
  Solver solver(cfg, ln2);
  const CylinderTank tank(0.05, 0.12);
  FillSpec fill;
  fill.fill_fraction = 0.5;
  fill.T_liquid = ln2.T_sat + real(0.5);
  solver.build_tank(tank, fill);
  if (std::strcmp(mode, "zerocopy") == 0) solver.telemetry().configure(solver.particle_system().capacity(), TelemetryMode::ZeroCopy);
  const bool capture = std::strcmp(mode, "none") != 0;

  solver.run_steps(5);  // warm-up (allocations, first sort, JIT of kernels)
  TelemetryFrame frame;
  int frames = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < steps; ++k) {
    solver.step();
    if (capture && k % every == 0) {
      if (solver.telemetry().pending()) {
        solver.telemetry().fetch(frame);
        ++frames;
      }
      solver.request_telemetry();
    }
  }
  if (capture && solver.telemetry().fetch(frame)) ++frames;
  mem::synchronize(solver.stream());
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const std::size_t n = solver.num_particles();
  return {dx, n, mode, 1e3 * sec / steps, double(n) * steps / sec / 1e6, frames};
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  std::printf("CryoSPH-CUDA performance benchmark on %s\n", mem::device_name().c_str());
  std::printf("%-8s %10s %-9s %12s %10s %7s\n", "dx [m]", "particles", "telemetry", "ms/step", "Mpart/s", "frames");
  std::vector<Result> results;
  std::vector<const char*> modes{"none"};
  if (opt.telemetry) {
    modes.push_back("async");
    modes.push_back("zerocopy");
  }
  for (double dx : opt.dx) {
    for (const char* mode : modes) {
      try {
        const Result r = run_case(dx, opt.steps, mode, opt.every);
        std::printf("%-8.4f %10zu %-9s %12.3f %10.2f %7d\n", r.dx, r.n, r.mode.c_str(), r.ms_per_step, r.mpups,
                    r.frames);
        std::fflush(stdout);
        results.push_back(r);
      } catch (const std::exception& e) {
        std::printf("%-8.4f %10s %-9s failed: %s\n", dx, "-", mode, e.what());
      }
    }
  }
  if (!opt.csv.empty()) {
    std::FILE* f = std::fopen(opt.csv.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "cannot write %s\n", opt.csv.c_str());
      return 1;
    }
    std::fprintf(f, "device,dx,particles,telemetry,ms_per_step,mparticles_per_s,frames\n");
    for (const Result& r : results)
      std::fprintf(f, "\"%s\",%g,%zu,%s,%.4f,%.4f,%d\n", mem::device_name().c_str(), r.dx, r.n, r.mode.c_str(),
                   r.ms_per_step, r.mpups, r.frames);
    std::fclose(f);
  }
  return 0;
}
