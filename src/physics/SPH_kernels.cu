#include "cryosph/physics/SPH_kernels.cuh"
#include "cryosph/physics/sim_params.cuh"

namespace cryosph {

void compute_density_summation(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) {
          pv.sigma[i] = real(1) / P.vol0;
          pv.rmin[i] = P.kernel.support;
          return;
        }
        const Vec3 xi = pv.x[i];
        real sum = 0, r2min = P.support2;
        for_each_neighbor(g, xi, [&](int j) {
          const real r2 = norm2(xi - pv.x[j]);
          if (r2 >= P.support2) return;
          sum += kernel_W(P.kernel, std::sqrt(r2));
          if (j != i && r2 < r2min) r2min = r2;
        });
        pv.sigma[i] = sum;
        pv.rmin[i] = std::sqrt(r2min);
        // Liquid uses its reference mass: mass lost to evaporation shrinks the
        // particle (by dm / rho_l, negligible next to the vapour's dm / rho_v)
        // instead of putting the stiff liquid under spurious tension.
        if (P.density_mode == static_cast<int>(DensityMode::Summation))
          pv.rho[i] = (ph == kLiquid ? pv.m0[i] : pv.m[i]) * sum;
      },
      s);
}

void normalize_initial_masses(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (!is_fluid(pv.phase[i])) return;
        const Vec3 xi = pv.x[i];
        real sum = 0;
        for_each_neighbor(g, xi, [&](int j) {
          const real r2 = norm2(xi - pv.x[j]);
          if (r2 < P.support2) sum += kernel_W(P.kernel, std::sqrt(r2));
        });
        if (sum > real(0)) {
          pv.m[i] = pv.rho[i] / sum;
          pv.m0[i] = pv.m[i];
        }
      },
      s);
}

void compute_transport_acceleration(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph) || !P.tvf || pv.sigma[i] < P.tvf_sigma_min ||
            (P.tvf_interface_only && !(pv.alpha[i] > real(0)))) {
          pv.ab[i] = Vec3{0, 0, 0};
          return;
        }
        const Vec3 xi = pv.x[i];
        // Volumes from 1/sigma, not m/rho: an evaporating particle would otherwise
        // look like a growing hole and keep pulling its neighbours in.
        const real rhoi = pv.rho[i];
        const real Vi = real(1) / pv.sigma[i];
        const real Vi2 = Vi * Vi;
        const real mi = rhoi * Vi;  // mass consistent with that volume
        const real ci = sound_speed(P, ph, pv.T[i]);
        const real pb = P.tvf_factor * P.rho0[phase_index(ph)] * ci * ci;
        const real inv_gamma = real(1) / P.tait_gamma;
        Vec3 sum{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const std::uint8_t phj = pv.phase[j];
          real Vj;
          if (phj == kWall) {
            // Wall volume from its extrapolated pressure, so walls don't push on a
            // stratified (hydrostatic) fluid.
            real rho_w;
            if (ph == kVapor) {
              rho_w = (pv.p_wv[j] - (P.p_bg - P.vapor_K)) / (P.gas_R * pv.T[i]);
            } else {
              const real ratio = (pv.p[j] - P.p_bg) / P.tait_B + real(1);
              rho_w = P.rho0[0] * std::pow(ratio > real(0.5) ? ratio : real(0.5), inv_gamma);
            }
            Vj = rho_w > real(0) ? Vi * rhoi / rho_w : Vi;
          } else {
            Vj = real(1) / pv.sigma[j];
          }
          sum += kernel_gradW(P.kernel, rij, std::sqrt(r2)) * (Vi2 + Vj * Vj);
        });
        pv.ab[i] = sum * (-pb / mi);
      },
      s);
}

void apply_particle_shifting(ParticleView pv, GridView g, const SimParams& Pref, real mach, stream_t s) {
  const SimParams P = Pref;
  const real h = P.kernel.h;
  const real w_dx = kernel_W(P.kernel, P.dx);
  const real coeff = P.shift_strength * mach * real(4) * h * h;  // A Ma (2h)^2
  const real cap = real(0.1) * P.dx;
  // Pass 1: shift vectors into ns (free scratch at the end of a step).
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        pv.ns[i] = Vec3{0, 0, 0};
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph) || pv.sigma[i] < P.tvf_sigma_min) return;  // free surface: no shift
        const Vec3 xi = pv.x[i];
        Vec3 sum{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const std::uint8_t phj = pv.phase[j];
          if (phj == kDead) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const real r = std::sqrt(r2);
          // Walls mirror the facing particle's volume: with the fixed lattice
          // volume a compressed (e.g. hydrostatic) fluid sees a net wall push.
          const real Vj = phj == kWall ? real(1) / pv.sigma[i] : real(1) / pv.sigma[j];
          const real q = kernel_W(P.kernel, r) / w_dx;
          const real q2 = q * q;
          sum += kernel_gradW(P.kernel, rij, r) * (Vj * (real(1) + real(0.2) * q2 * q2));
        });
        Vec3 dr = sum * (-coeff);
        // Interface particles keep only the tangential part: no shifting
        // across the liquid/vapour interface.
        if (pv.alpha[i] > real(0)) {
          const real gm = norm(pv.nrm[i]);
          if (gm > P.normal_eps) {
            const Vec3 n = pv.nrm[i] / gm;
            dr -= n * dot(dr, n);
          } else {
            dr = Vec3{0, 0, 0};
          }
        }
        const real len = norm(dr);
        if (len > cap) dr *= cap / len;
        pv.ns[i] = dr;
      },
      s);
  // Pass 2: apply.
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (is_fluid(pv.phase[i])) pv.x[i] += pv.ns[i];
      },
      s);
}

void compute_density_rate(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) {
          pv.drho[i] = 0;
          return;
        }
        const Vec3 xi = pv.x[i];
        // Particles move with the transport velocity, so it sets the divergence.
        const Vec3* vel = P.tvf ? pv.vt : pv.v;
        const Vec3 vi = vel[i];
        const real rhoi = pv.rho[i];
        const real h = P.kernel.h;
        const real eta2 = real(0.01) * h * h;
        real div = 0, diff = 0;
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const real r = std::sqrt(r2);
          const Vec3 gw = kernel_gradW(P.kernel, rij, r);
          const std::uint8_t phj = pv.phase[j];
          if (phj == kWall) {
            // Static walls: v_j = 0, wall volume dx^3.
            div += P.vol0 * dot(vi, gw);
            return;
          }
          const real Vj = pv.m[j] / pv.rho[j];
          div += Vj * dot(vi - vel[j], gw);
          if (phj == ph) {
            // delta-SPH: psi_ij = 2 (rho_j - rho_i) (x_j - x_i) / (r^2 + eta^2)
            diff += Vj * real(2) * (pv.rho[j] - rhoi) * dot(-rij, gw) / (r2 + eta2);
          }
        });
        pv.drho[i] = rhoi * div + P.delta_sph * h * P.c0 * diff;
      },
      s);
}

}  // namespace cryosph
