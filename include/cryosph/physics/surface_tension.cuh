// Interface detection, CSF surface tension (classic or balanced-force),
// contact angles and the Akinci model for liquid-only runs.
#pragma once

#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/spatial_hash.hpp"
#include "cryosph/physics/sim_params.cuh"

namespace cryosph {

/// Per fluid particle: colour gradient / Akinci normal (stored in nrm, with
/// contact-angle correction), opposite-phase volume fraction alpha, wall
/// proximity weight wall_w and heated-wall weight heat_w.
void compute_interface_fields(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Smoothed unit interface normals ns (Shepard average of the neighbours' unit
/// colour-gradient normals; zero where the colour gradient is not resolved).
void smooth_interface_normals(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// CSF curvature and force (adds to a, writes kappa). The balanced variant
/// builds grad(c) with the pressure operator (Adami et al. 2010).
void apply_csf_surface_tension(ParticleView pv, GridView g, const SimParams& P, stream_t s);

/// Akinci cohesion, curvature-minimisation and wall adhesion (adds to a).
void apply_akinci_surface_tension(ParticleView pv, GridView g, const SimParams& P, stream_t s);

}  // namespace cryosph
