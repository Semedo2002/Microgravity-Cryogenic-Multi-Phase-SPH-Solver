// Momentum equation (Hu & Adams 2006 multiphase form) with Morris viscosity,
// Monaghan artificial viscosity and optional Smagorinsky SPS; walls follow
// Adami et al. 2012 with per-phase pressure extrapolation.
#pragma once

#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/spatial_hash.hpp"
#include "cryosph/physics/sim_params.cuh"

namespace cryosph {

/// Mechanical pressure of every fluid particle from the EOS.
void compute_equation_of_state(ParticleView pv, const SimParams& P, stream_t s);

/// Wall pressure, density and ghost velocity extrapolated from the fluid.
void compute_wall_boundary(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Smagorinsky eddy viscosity nu_t from the SPH velocity-gradient tensor.
void compute_sps_viscosity(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Writes a_i = g + pressure + viscous accelerations for every fluid particle
/// (walls get a = 0).  Surface-tension models add to `a` afterwards.
void compute_momentum(ParticleView pv, GridView g, const SimParams& P, stream_t s);

}  // namespace cryosph
