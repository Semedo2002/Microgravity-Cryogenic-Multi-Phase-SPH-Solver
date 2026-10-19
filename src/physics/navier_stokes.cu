#include "cryosph/physics/navier_stokes.cuh"

namespace cryosph {

void compute_equation_of_state(ParticleView pv, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (is_fluid(ph)) pv.p[i] = eos_pressure(P, ph, pv.rho[i], pv.T[i]);
      },
      s);
}

void compute_wall_boundary(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (pv.phase[i] != kWall) return;
        const Vec3 xi = pv.x[i];
        // Pressure is extrapolated separately from each phase: a single mixed
        // value would let small liquid compressions (stiff Tait EOS) push the
        // light vapour through the wall at large density ratios.
        real sw[2] = {0, 0}, sp[2] = {0, 0}, sgr[2] = {0, 0};
        real srho = 0;
        Vec3 sv{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          const std::uint8_t phj = pv.phase[j];
          if (!is_fluid(phj)) return;
          const Vec3 rwf = xi - pv.x[j];
          const real r2 = norm2(rwf);
          if (r2 >= P.support2) return;
          const real w = kernel_W(P.kernel, std::sqrt(r2));
          const real rhoj = pv.rho[j];
          const int k = phase_index(phj);
          sw[k] += w;
          sp[k] += pv.p[j] * w;
          // Static walls: a_wall = 0, so only the body force enters.
          sgr[k] += dot(P.gravity, rwf) * rhoj * w;
          srho += rhoj * w;
          sv += pv.v[j] * w;
        });
        real pw[2];
        for (int k = 0; k < 2; ++k) {
          pw[k] = sw[k] > real(0) ? (sp[k] + sgr[k]) / sw[k] : P.p_bg;
          if (P.clamp_negative_pressure && pw[k] < real(0)) pw[k] = 0;
        }
        // A phase absent from the wall's support borrows the other one's value.
        if (!(sw[0] > real(0)) && sw[1] > real(0)) pw[0] = pw[1];
        if (!(sw[1] > real(0)) && sw[0] > real(0)) pw[1] = pw[0];
        pv.p[i] = pw[0];
        pv.p_wv[i] = pw[1];
        // Liquid fraction seen by the wall, stored in alpha (unused for walls);
        // used in the colour gradient near contact lines.
        pv.alpha[i] = (sw[0] + sw[1]) > real(0) ? sw[0] / (sw[0] + sw[1]) : real(0);
        const real swt = sw[0] + sw[1];
        if (swt > real(0)) {
          pv.rho[i] = srho / swt;
          pv.vw[i] = sv * (real(-1) / swt);
        } else {
          pv.rho[i] = P.rho0[0];
          pv.vw[i] = Vec3{0, 0, 0};
        }
        pv.a[i] = Vec3{0, 0, 0};
      },
      s);
}

void compute_sps_viscosity(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) {
          pv.nu_t[i] = 0;
          return;
        }
        const Vec3 xi = pv.x[i];
        const Vec3 vi = pv.v[i];
        // Velocity-gradient tensor L_ab = sum_j V_j (v_j - v_i)_a (grad_i W)_b
        real L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const Vec3 gw = kernel_gradW(P.kernel, rij, std::sqrt(r2));
          const std::uint8_t phj = pv.phase[j];
          Vec3 vj;
          real Vj;
          if (phj == kWall) {
            if (!P.no_slip) return;
            vj = pv.vw[j];
            Vj = P.vol0;
          } else {
            vj = pv.v[j];
            Vj = pv.m[j] / pv.rho[j];
          }
          const Vec3 dv = (vj - vi) * Vj;
          const real dvc[3] = {dv.x, dv.y, dv.z};
          const real gwc[3] = {gw.x, gw.y, gw.z};
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) L[a][b] += dvc[a] * gwc[b];
        });
        real ss = 0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            const real sab = real(0.5) * (L[a][b] + L[b][a]);
            ss += sab * sab;
          }
        const real smag = std::sqrt(real(2) * ss);
        const real l = P.smagorinsky_cs * P.dx;
        pv.nu_t[i] = l * l * smag;
      },
      s);
}

void compute_momentum(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) {
          pv.a[i] = Vec3{0, 0, 0};
          pv.fw[i] = Vec3{0, 0, 0};
          pv.mw[i] = Vec3{0, 0, 0};
          return;
        }
        const int pi_idx = phase_index(ph);
        const Vec3 xi = pv.x[i];
        const Vec3 vi = pv.v[i];
        const real mi = pv.m[i];
        const real rhoi = pv.rho[i];
        const real Vi = mi / rhoi;
        const real Vi2 = Vi * Vi;
        const real pi = pv.p[i];
        const real ci = sound_speed(P, ph, pv.T[i]);
        const real etai = P.mu[pi_idx] + (P.sps ? rhoi * pv.nu_t[i] : real(0));
        const real alpha_i = ph == kVapor ? P.alpha_av_vap : P.alpha_av;
        const real h = P.kernel.h;
        const real eta2 = real(0.01) * h * h;

        Vec3 force{0, 0, 0};  // sum of pairwise pressure + shear forces [N]
        Vec3 acc_av{0, 0, 0};  // artificial-viscosity (+ wall guard) acceleration [m/s^2]
        // Penetration guard: fluid and wall particles never come closer than
        // ~dx in regular flow; below 0.6 dx a stiff repulsion keeps a particle
        // that was accelerated by an under-resolved force from tunnelling.
        const real r_guard = real(0.6) * P.dx;
        // Transport-velocity extra stress A = rho v (vt - v)^T (Adami et al.
        // 2013); (A . grad W) = rho v ((vt - v) . grad W).
        const bool tvf_stress = P.tvf && P.tvf_stress;
        const Vec3 dvi = pv.vt[i] - vi;
        // Slosh loads: every wall interaction's force on i (pressure, shear,
        // artificial viscosity, guard) and its moment about the frame origin,
        // with the wall particle's position as the point of application.
        Vec3 fwall{0, 0, 0}, mwall{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const real r = std::sqrt(r2);
          const real dW = kernel_dWdr(P.kernel, r);
          const bool separated = r > real(1e-12) * h;
          const Vec3 gw = separated ? rij * (dW / r) : Vec3{0, 0, 0};
          const std::uint8_t phj = pv.phase[j];

          if (phj == kWall) {
            const Vec3 f_before = force, a_before = acc_av;
            const real Vj = P.vol0;
            const real vsum = Vi2 + Vj * Vj;
            const real pw = ph == kVapor ? pv.p_wv[j] : pv.p[j];
            force -= gw * (vsum * real(0.5) * (pi + pw));
            if (tvf_stress) force += vi * (real(0.5) * vsum * rhoi * dot(dvi, gw));  // A_wall = 0
            if (r < r_guard && separated) {
              const real q = real(1) - r / r_guard;
              const Vec3 n = rij / r;
              acc_av += n * (real(4) * ci * ci * q * q / r_guard);
              // Kinematic part: cancel the approach velocity within one step
              // (effective even when the particle moves faster than c0).
              const real vn = dot(vi, n);  // walls are static
              if (vn < real(0) && P.dt > real(0)) acc_av -= n * (q * vn / P.dt);
            }
            if (P.no_slip) {
              const Vec3 vij = vi - pv.vw[j];
              if (separated) force += vij * (vsum * etai * dW / r);
              const real vr = dot(vij, rij);
              if (alpha_i > real(0) && vr < real(0)) {
                const real mu_ij = h * vr / (r2 + eta2);
                const real Pi = -alpha_i * ci * mu_ij / rhoi;
                acc_av -= gw * (rhoi * P.vol0 * Pi);
              }
            }
            const Vec3 dF = (force - f_before) + (acc_av - a_before) * mi;
            fwall += dF;
            mwall += cross(pv.x[j] - P.frame_origin, dF);
            return;
          }

          const real mj = pv.m[j];
          const real rhoj = pv.rho[j];
          const real Vj = mj / rhoj;
          const real vsum = Vi2 + Vj * Vj;
          const real pt = (rhoj * pi + rhoi * pv.p[j]) / (rhoi + rhoj);
          force -= gw * (vsum * pt);
          if (tvf_stress) {
            const Vec3 vj = pv.v[j];
            const Vec3 Ag = vi * (rhoi * dot(dvi, gw)) + vj * (rhoj * dot(pv.vt[j] - vj, gw));
            force += Ag * (real(0.5) * vsum);
          }

          const Vec3 vij = vi - pv.v[j];
          const real etaj = P.mu[phase_index(phj)] + (P.sps ? rhoj * pv.nu_t[j] : real(0));
          const real eta_sum = etai + etaj;
          if (eta_sum > real(0) && separated) {
            const real eta_ij = real(2) * etai * etaj / eta_sum;
            force += vij * (vsum * eta_ij * dW / r);
          }

          if (alpha_i > real(0) && phj == ph) {
            const real vr = dot(vij, rij);
            if (vr < real(0)) {
              const real cj = sound_speed(P, phj, pv.T[j]);
              const real mu_ij = h * vr / (r2 + eta2);
              const real Pi = -alpha_i * real(0.5) * (ci + cj) * mu_ij / (real(0.5) * (rhoi + rhoj));
              acc_av -= gw * (mj * Pi);
            }
          }
        });
        Vec3 acc = P.gravity + force / mi + acc_av;
        if (P.frame_rot) {
          // Non-inertial tank frame: Euler, centrifugal and Coriolis accelerations.
          const Vec3 r = xi - P.frame_origin;
          acc -= cross(P.omega_dot, r) + cross(P.omega, cross(P.omega, r)) + cross(P.omega, vi) * real(2);
        }
        pv.a[i] = acc;
        pv.fw[i] = fwall;
        pv.mw[i] = mwall;
      },
      s);
}

}  // namespace cryosph
