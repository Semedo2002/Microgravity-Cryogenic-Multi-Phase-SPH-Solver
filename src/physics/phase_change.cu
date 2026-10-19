#include "cryosph/physics/phase_change.cuh"

namespace cryosph {

namespace {

CRYOSPH_HD CRYOSPH_INLINE std::uint32_t hash_u32(std::uint32_t x) {
  // lowbias32 (Chris Wellons): good avalanche, cheap on GPUs.
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

CRYOSPH_HD CRYOSPH_INLINE real hash_unit(std::uint32_t x) {
  return static_cast<real>(hash_u32(x) & 0x00FFFFFFu) / static_cast<real>(0x01000000u);
}

/// Deterministic pseudo-random unit vector.
CRYOSPH_HD CRYOSPH_INLINE Vec3 random_direction(std::uint32_t id, std::uint32_t seed) {
  const real u = real(2) * hash_unit(id * 2654435761u ^ seed) - real(1);
  const real phi = real(2) * kPi * hash_unit(hash_u32(id + 0x9e3779b9u) ^ (seed * 747796405u));
  const real s = std::sqrt(real(1) - u * u > real(0) ? real(1) - u * u : real(0));
  return Vec3{s * std::cos(phi), s * std::sin(phi), u};
}

}  // namespace

void compute_thermal_rates(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) {
          pv.dTdt[i] = 0;
          return;
        }
        const int ii = phase_index(ph);
        const Vec3 xi = pv.x[i];
        const real Ti = pv.T[i];
        const real mi = pv.m[i];
        const real Vi = mi / pv.rho[i];
        const real ki = P.k[ii];
        const real eta2 = real(0.01) * P.kernel.h * P.kernel.h;
        const bool isothermal_walls = P.wall_bc == static_cast<int>(WallThermalBC::Isothermal);
        real q = 0;  // heat rate into particle i [W]
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const std::uint8_t phj = pv.phase[j];
          real Vj, kij;
          if (phj == kWall) {
            if (!isothermal_walls) return;
            Vj = P.vol0;
            kij = real(2) * ki;
          } else {
            const real kj = P.k[phase_index(phj)];
            Vj = pv.m[j] / pv.rho[j];
            kij = real(4) * ki * kj / (ki + kj);
          }
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const real r = std::sqrt(r2);
          // r_ij . grad_i W_ij = r dW/dr  (<= 0)
          const real rgw = r * kernel_dWdr(P.kernel, r);
          q += Vi * Vj * kij * (Ti - pv.T[j]) * rgw / (r2 + eta2);
        });
        q += P.heat_power * pv.heat_w[i] * P.heat_inv_norm;
        pv.dTdt[i] = q / (mi * P.cp[ii]);
      },
      s);
}

void phase_change_sources(ParticleView pv, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        std::uint8_t fl = static_cast<std::uint8_t>(pv.flags[i] & ~pflag::ScratchMask);
        pv.out_m[i] = 0;
        pv.out_h[i] = 0;
        pv.w_norm[i] = 0;
        if (!is_fluid(ph)) {
          pv.flags[i] = fl;
          return;
        }
        const int ii = phase_index(ph);
        const real m = pv.m[i];
        const real T = pv.T[i];
        if (m < P.merge_ratio * P.mref[ii]) {
          // Remnant particle: merge into same-phase neighbours.
          pv.out_m[i] = m;
          pv.out_h[i] = specific_enthalpy(P, ph, T);
          fl |= pflag::Source | pflag::Dying;
          if (ph == kVapor) fl |= pflag::TargetVapor;
          pv.flags[i] = fl;
          return;
        }
        const real alpha = pv.alpha[i];
        if (P.phase_change && alpha > real(0) && P.dt > real(0)) {
          if (ph == kLiquid && T > P.T_sat) {
            const real h_out = P.h_fg + P.cp[1] * (P.T_sat - P.T_ref);  // saturated vapour
            const real latent = h_out - P.cp[0] * (P.T_sat - P.T_ref);
            const real rate = P.lee_evap * m * alpha * (T - P.T_sat) / P.T_sat;
            real dm = rate * P.dt;
            const real dm_energy = m * P.cp[0] * (T - P.T_sat) / latent;  // keeps T >= T_sat
            dm = dm < dm_energy ? dm : dm_energy;
            const real dm_cap = P.max_transfer_fraction * m;
            dm = dm < dm_cap ? dm : dm_cap;
            if (dm > real(0)) {
              pv.out_m[i] = dm;
              pv.out_h[i] = h_out;
              fl |= pflag::Source | pflag::TargetVapor;
            }
          } else if (ph == kVapor && T < P.T_sat) {
            const real rate = P.lee_cond * m * alpha * (P.T_sat - T) / P.T_sat;
            real dm = rate * P.dt;
            real dm_cap = P.max_transfer_fraction * m;
            // Flux limiter, emitter side: a condensing vapour particle loses
            // at most pc_density_limit of its mass (and density) per step.
            if (P.pc_density_limit > real(0) && P.pc_density_limit * m < dm_cap) dm_cap = P.pc_density_limit * m;
            dm = dm < dm_cap ? dm : dm_cap;
            if (dm > real(0)) {
              pv.out_m[i] = dm;
              pv.out_h[i] = specific_enthalpy(P, kVapor, T);
              fl |= pflag::Source;
            }
          }
        }
        pv.flags[i] = fl;
      },
      s);
}

void phase_change_normalize(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t fl = pv.flags[i];
        if (!(fl & pflag::Source)) return;
        const std::uint8_t target = (fl & pflag::TargetVapor) ? kVapor : kLiquid;
        const Vec3 xi = pv.x[i];
        real wsum = 0, wr_max = 0;  // wr = W_ij / rho_j of the receivers
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i || pv.phase[j] != target || (pv.flags[j] & pflag::Dying)) return;
          const real r2 = norm2(xi - pv.x[j]);
          if (r2 >= P.support2) return;
          const real w = kernel_W(P.kernel, std::sqrt(r2));
          const real rhoj = pv.rho[j];
          wsum += (pv.m[j] / rhoj) * w;
          const real wr = w / rhoj;
          wr_max = wr > wr_max ? wr : wr_max;
        });
        pv.w_norm[i] = wsum;
        if (!(wsum > real(0))) {
          pv.out_m[i] = 0;  // no receiver: emission cancelled
          return;
        }
        // Cap the emission so no vapour receiver gains more than pc_density_limit
        // of its mass in one step. Merges are never capped.
        if (target == kVapor && P.pc_density_limit > real(0) && !(fl & pflag::Dying) && wr_max > real(0)) {
          const real cap = P.pc_density_limit * wsum / wr_max;
          if (pv.out_m[i] > cap) pv.out_m[i] = cap;
        }
      },
      s);
}

void phase_change_apply(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) return;
        const std::uint8_t fl = pv.flags[i];
        const real mi = pv.m[i];
        const real Vi = mi / pv.rho[i];
        const Vec3 xi = pv.x[i];
        real in_m = 0, in_H = 0;
        if (!(fl & pflag::Dying)) {
          for_each_neighbor(g, xi, [&](int j) {
            if (j == i) return;
            const real om = pv.out_m[j];
            if (!(om > real(0))) return;
            const std::uint8_t target = (pv.flags[j] & pflag::TargetVapor) ? kVapor : kLiquid;
            if (target != ph) return;
            const real r2 = norm2(xi - pv.x[j]);
            if (r2 >= P.support2) return;
            const real share = om * Vi * kernel_W(P.kernel, std::sqrt(r2)) / pv.w_norm[j];
            in_m += share;
            in_H += share * pv.out_h[j];
          });
        }
        const real om_i = pv.out_m[i];
        if ((fl & pflag::Dying) && om_i > real(0)) {
          pv.m[i] = 0;
          pv.phase[i] = kDead;
          return;
        }
        if (!(om_i > real(0)) && !(in_m > real(0))) return;
        const real m_new = mi - om_i + in_m;
        const real H_new = mi * specific_enthalpy(P, ph, pv.T[i]) - om_i * pv.out_h[i] + in_H;
        pv.T[i] = temperature_from_enthalpy(P, ph, H_new / m_new);
        // Density is intensive: the transferred mass changes the particle volume
        // V = m / rho (continuity mode), not its density and hence not its pressure.
        pv.m[i] = m_new;
      },
      s);
}

void rebalance_vapor_mass(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  const real beta = P.vapor_rebalance;
  // Pass 1: pairwise exchange with upwinded enthalpy and momentum, so the totals
  // are conserved. Scratch: out_m, out_h, ns (free at this point).
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        pv.out_m[i] = 0;
        pv.out_h[i] = 0;
        pv.ns[i] = Vec3{0, 0, 0};
        if (pv.phase[i] != kVapor) return;
        const Vec3 xi = pv.x[i];
        const real mi = pv.m[i];
        const real hi = specific_enthalpy(P, kVapor, pv.T[i]);
        const Vec3 vi = pv.v[i];
        real dm = 0, dH = 0;
        Vec3 dP{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i || pv.phase[j] != kVapor) return;
          const real r2 = norm2(xi - pv.x[j]);
          if (r2 >= P.support2) return;
          const real flow = beta * P.vol0 * kernel_W(P.kernel, std::sqrt(r2)) * (pv.m[j] - mi);  // into i
          const bool from_j = flow > real(0);
          dm += flow;
          dH += flow * (from_j ? specific_enthalpy(P, kVapor, pv.T[j]) : hi);
          dP += (from_j ? pv.v[j] : vi) * flow;
        });
        pv.out_m[i] = dm;
        pv.out_h[i] = dH;
        pv.ns[i] = dP;
      },
      s);
  // Pass 2: apply.
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (pv.phase[i] != kVapor) return;
        const real dm = pv.out_m[i];
        if (dm == real(0)) return;
        const real m = pv.m[i];
        const real m_new = m + dm;
        const real H = m * specific_enthalpy(P, kVapor, pv.T[i]) + pv.out_h[i];
        pv.T[i] = temperature_from_enthalpy(P, kVapor, H / m_new);
        pv.v[i] = (pv.v[i] * m + pv.ns[i]) / m_new;
        pv.vt[i] = pv.v[i];
        pv.m[i] = m_new;
        pv.out_m[i] = 0;
        pv.out_h[i] = 0;
      },
      s);
}

void mark_heavy_particles(ParticleView pv, const SimParams& Pref, unsigned* flag, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        flag[i] = (is_fluid(ph) && pv.m[i] > P.split_ratio * P.mref[phase_index(ph)]) ? 1u : 0u;
      },
      s);
}

void split_heavy_particles(ParticleView pv, const SimParams& Pref, const unsigned* flag, const unsigned* offset,
                           std::uint32_t id_base, std::uint32_t seed, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (!flag[i]) return;
        const std::uint8_t ph = pv.phase[i];
        const real m = pv.m[i];
        const unsigned k = offset[i];
        const int dst = pv.n + static_cast<int>(k);
        if (dst >= pv.capacity) return;
        const real vol = m / pv.rho[i];
        Vec3 e = random_direction(pv.id[i], seed);
        const Vec3 xi = pv.x[i];
        // At an interface, put the new half along the interface and away from the
        // other phase (never inside it); in the bulk, split symmetrically.
        const Vec3 gc = pv.nrm[i];
        const real gm = norm(gc);
        if (gm > P.normal_eps) {
          const Vec3 n = gc / gm;
          const Vec3 t = e - n * dot(e, n);
          e = normalized(t - n * real(0.5));
          pv.x[dst] = xi + e * (real(0.5) * std::cbrt(vol));
        } else {
          const real delta = real(0.25) * std::cbrt(vol);
          pv.x[dst] = xi + e * delta;
          pv.x[i] = xi - e * delta;
        }
        pv.v[dst] = pv.v[i];
        pv.vt[dst] = pv.vt[i];
        pv.ab[dst] = pv.ab[i];
        pv.a[dst] = pv.a[i];
        pv.nrm[dst] = pv.nrm[i];
        pv.m[dst] = real(0.5) * m;
        // Liquid volume changes from phase change are neglected (see density
        // summation), so the reference mass is shared like the mass itself.
        pv.m0[i] *= real(0.5);
        pv.m0[dst] = pv.m0[i];
        pv.m[i] = real(0.5) * m;
        pv.rho[dst] = pv.rho[i];
        pv.T[dst] = pv.T[i];
        pv.phase[dst] = ph;
        pv.flags[dst] = static_cast<std::uint8_t>(pv.flags[i] & ~pflag::ScratchMask);
        pv.id[dst] = id_base + k;
        pv.p[dst] = pv.p[i];
        // Each half occupies half the volume until the next density summation.
        pv.sigma[i] *= real(2);
        pv.sigma[dst] = pv.sigma[i];
        pv.dTdt[dst] = 0;
        pv.out_m[dst] = 0;
        pv.out_h[dst] = 0;
      },
      s);
}

void apply_venting(ParticleView pv, const SimParams& Pref, real relax, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        pv.out_m[i] = 0;
        pv.out_h[i] = 0;
        if (pv.phase[i] != kVapor) return;
        const real m = pv.m[i];
        // Particle volume m / rho (rho from this step's density evaluation; in
        // summation mode a vapour particle's volume grows with its mass).
        const real vol = m / pv.rho[i];
        const real target = P.rho0[1] * vol;
        if (!(m > target)) return;
        const real dm = relax * (m - target);
        pv.out_m[i] = dm;
        pv.out_h[i] = specific_enthalpy(P, kVapor, pv.T[i]);
        if (P.density_mode == static_cast<int>(DensityMode::Continuity)) pv.rho[i] *= (m - dm) / m;
        pv.m[i] = m - dm;
      },
      s);
}

}  // namespace cryosph
