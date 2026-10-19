// VTK output for ParaView and JSON frames for the web viewer in viz/.
#pragma once

#include <string>
#include <vector>

#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/telemetry.hpp"

namespace cryosph {

/// Writes a legacy binary VTK POLYDATA file with point fields
/// temperature, pressure (if given), mass, phase and velocity.
void write_vtk(const std::string& path, const HostParticles& particles, const std::vector<real>& pressure = {});

/// One JSON frame (time, step, n, position, phase, temperature, pressure,
/// speed) for the web viewer. Walls are skipped unless include_walls is set.
void write_json_frame(const std::string& path, const TelemetryFrame& frame, bool include_walls = true);

/// Builds a TelemetryFrame from host particles (synchronous path).
TelemetryFrame make_frame(const HostParticles& particles, const std::vector<real>& pressure, double time,
                          long long step);

}  // namespace cryosph
