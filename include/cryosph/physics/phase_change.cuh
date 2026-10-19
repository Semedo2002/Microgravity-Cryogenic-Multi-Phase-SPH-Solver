// Heat conduction (Cleary & Monaghan 1999) and Lee-model evaporation /
// condensation. Every transfer carries its enthalpy and is shared among
// receivers with weights that sum to one, so mass and energy are conserved.
#pragma once

#include <cstdint>

#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/spatial_hash.hpp"
#include "cryosph/physics/sim_params.cuh"

namespace cryosph {

/// Temperature rate from conduction and heater input (fluid particles).
void compute_thermal_rates(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Stage 1 (local): decides which particles emit mass this step (phase change
/// or merging), sets out_m / out_h and the scratch flags.
void phase_change_sources(ParticleView pv, const SimParams& P, stream_t s);

/// Stage 2: receiver-weight normalisation; cancels emissions that have no
/// eligible receiver.
void phase_change_normalize(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Stage 3: gather of emitted mass/enthalpy, update of m and T, removal of
/// merged particles (phase -> Dead).
void phase_change_apply(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Stage 3b: conservative mass exchange between vapour neighbours,
///   dm_i = beta sum_j V0 W_ij (m_j - m_i)
/// Evaporated mass piles up in the first vapour layer; spreading it keeps the
/// liquid's neighbour count intact so the interface doesn't tear.
void rebalance_vapor_mass(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Stage 4a: flag[i] = 1 for particles heavier than split_ratio * m_ref.
void mark_heavy_particles(ParticleView pv, const SimParams& P, unsigned* flag, stream_t s);

/// Stage 4b: split flagged particles; the new half goes to slot n + offset[i]
/// (prefix sum of the flags, so the result doesn't depend on thread order).
void split_heavy_particles(ParticleView pv, const SimParams& P, const unsigned* flag, const unsigned* offset,
                           std::uint32_t id_base, std::uint32_t seed, stream_t s);

/// Pressure-controlled vent: vapour particles denser than rho_v0 lose the
/// excess mass with relaxation factor `relax` in (0, 1].  Writes the removed
/// mass to out_m and its enthalpy to out_h (for bookkeeping by the caller).
void apply_venting(ParticleView pv, const SimParams& P, real relax, stream_t s);

}  // namespace cryosph
