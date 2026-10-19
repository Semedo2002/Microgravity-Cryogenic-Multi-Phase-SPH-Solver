#include "cryosph/physics/surface_tension.cuh"

namespace cryosph {

void compute_interface_fields(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) return;  // walls keep their geometric normal in nrm
        const Vec3 xi = pv.x[i];
        const real ci = ph == kLiquid ? real(1) : real(0);
        Vec3 gradc{0, 0, 0}, wall_n{0, 0, 0}, akn{0, 0, 0}, gradphi{0, 0, 0};
        real s_all = 0, s_other = 0, ww = 0, hw = 0;
        for_each_neighbor(g, xi, [&](int j) {
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const real r = std::sqrt(r2);
          const real w = kernel_W(P.kernel, r);
          const std::uint8_t phj = pv.phase[j];
          if (phj == kWall) {
            const real vw = P.vol0 * w;
            ww += vw;
            if (pv.flags[j] & pflag::Heated) hw += vw;
            wall_n += pv.nrm[j] * vw;
            gradphi += kernel_gradW(P.kernel, rij, r) * P.vol0;
            if (P.wall_colour) gradc += kernel_gradW(P.kernel, rij, r) * (P.vol0 * (pv.alpha[j] - ci));
            return;
          }
          const real Vj = pv.m[j] / pv.rho[j];
          s_all += Vj * w;
          if (phj != ph) s_other += Vj * w;
          if (j == i) return;
          const Vec3 gw = kernel_gradW(P.kernel, rij, r);
          const real cj = phj == kLiquid ? real(1) : real(0);
          gradc += gw * (Vj * (cj - ci));
          if (phj == kLiquid) akn += gw * Vj;
        });
        pv.alpha[i] = s_all > real(0) ? s_other / s_all : real(0);
        pv.wall_w[i] = ww;
        pv.heat_w[i] = hw;

        if (P.st_model == static_cast<int>(SurfaceTensionModel::Akinci)) {
          pv.nrm[i] = ph == kLiquid ? akn * (-P.kernel.h) : Vec3{0, 0, 0};
          return;
        }

        // Contact-line force (Huber et al. 2016):
        //   f = sigma (cos theta_s - cos theta_d) t |grad c| delta_w
        // Kept in vw (unused for fluid) and added after the momentum equation.
        const real gmag = norm(gradc);
        pv.vw[i] = Vec3{0, 0, 0};
        if (P.wetting_model != 0 && ww > real(0) && gmag > P.normal_eps && norm2(wall_n) > real(0)) {
          const Vec3 nw = normalized(wall_n);
          const Vec3 nh = gradc / gmag;
          const Vec3 tan_in = nh - nw * dot(nh, nw);  // towards the liquid, in the wall plane
          const real tm = norm(tan_in);
          if (tm > real(1e-6)) {
            const real cos_d = -dot(nh, nw);
            const real delta_w = real(2) * norm(gradphi);
            const real coeff = real(2) * P.sigma / (P.rho0[0] + P.rho0[1]);
            pv.vw[i] = tan_in * (-(P.cos_theta - cos_d) * coeff * gmag * delta_w / tm);
          }
        }

        // Wall adhesion (Brackbill): rotate the interface normal to the prescribed angle.
        if (P.wetting_model != 1 && ww > real(0) && gmag > P.normal_eps) {
          const Vec3 nw = normalized(wall_n);
          const Vec3 nh = gradc / gmag;
          Vec3 t = nh - nw * dot(nh, nw);
          const real tmag = norm(t);
          if (norm2(nw) > real(0) && tmag > real(1e-6)) {
            t = t / tmag;
            const Vec3 target = nw * (-P.cos_theta) + t * P.sin_theta;
            const real beta = clamp_val(ww / P.wall_adhesion_weight, real(0), real(1));
            const Vec3 blended = normalized(nh * (real(1) - beta) + target * beta);
            gradc = blended * gmag;
          }
        }
        pv.nrm[i] = gradc;
      },
      s);
}

void smooth_interface_normals(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (!is_fluid(pv.phase[i])) return;
        const Vec3 gi = pv.nrm[i];
        const real gmag = norm(gi);
        if (!(gmag > P.normal_eps)) {
          pv.ns[i] = Vec3{0, 0, 0};
          return;
        }
        if (!P.csf_smooth_normals) {
          pv.ns[i] = gi / gmag;
          return;
        }
        // Average of neighbour normals; removes particle-scale noise in kappa.
        const Vec3 xi = pv.x[i];
        Vec3 acc{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (!is_fluid(pv.phase[j])) return;
          const Vec3 gj = pv.nrm[j];
          const real gjm = norm(gj);
          if (!(gjm > P.normal_eps)) return;
          const real r2 = norm2(xi - pv.x[j]);
          if (r2 >= P.support2) return;
          acc += gj * ((pv.m[j] / pv.rho[j]) * kernel_W(P.kernel, std::sqrt(r2)) / gjm);
        });
        pv.ns[i] = normalized(acc);
      },
      s);
}

void apply_csf_surface_tension(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  const real coeff = real(2) * P.sigma / (P.rho0[0] + P.rho0[1]);
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        const std::uint8_t ph = pv.phase[i];
        if (!is_fluid(ph)) return;
        const Vec3 ni = pv.ns[i];
        if (!(norm2(ni) > real(0.25))) {
          pv.kappa[i] = 0;
          return;
        }
        const Vec3 xi = pv.x[i];
        const real rhoi = pv.rho[i];
        const real mi = pv.m[i];
        const real Vi = mi / rhoi;
        const real ci = ph == kLiquid ? real(1) : real(0);
        // kappa = -div(n); G is grad(c) built with the pressure operator.
        real num = 0, den = 0;
        Vec3 G{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const std::uint8_t phj = pv.phase[j];
          if (!is_fluid(phj)) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= P.support2) return;
          const real r = std::sqrt(r2);
          const real dW = kernel_dWdr(P.kernel, r);
          const Vec3 gw = rij * (dW / r);
          const real rhoj = pv.rho[j];
          const real Vj = pv.m[j] / rhoj;
          const Vec3 nj = pv.ns[j];
          if (norm2(nj) > real(0.25)) {
            num += Vj * dot(ni - nj, gw);
            den += Vj * r * (-dW);
          }
          if (P.csf_balanced) {
            const real cj = phj == kLiquid ? real(1) : real(0);
            G += gw * ((Vi * Vi + Vj * Vj) * (rhoj * ci + rhoi * cj) / (rhoi + rhoj));
          }
        });
        const real kappa = den > real(0) ? real(3) * num / den : real(0);
        pv.kappa[i] = kappa;
        if (P.wetting_model != 0) pv.a[i] += pv.vw[i];
        if (P.csf_balanced) {
          // a_i = sigma kappa_i G_i / m_i is cancelled exactly by a pressure
          // field p = p_v + sigma kappa c, so a static interface is a discrete
          // equilibrium (no parasitic driving where kappa is uniform).
          pv.a[i] += G * (P.sigma * kappa / mi);
        } else {
          // Classic density-scaled CSF along the (wall-corrected) colour gradient.
          pv.a[i] += pv.nrm[i] * (coeff * kappa);
        }
      },
      s);
}

void apply_akinci_surface_tension(ParticleView pv, GridView g, const SimParams& Pref, stream_t s) {
  const SimParams P = Pref;
  parallel_for(
      pv.n,
      [=] CRYOSPH_HD(int i) {
        if (pv.phase[i] != kLiquid) return;
        const Vec3 xi = pv.x[i];
        const Vec3 ni = pv.nrm[i];
        const real mi = pv.m[i];
        const real rhoi = pv.rho[i];
        const real c = P.kernel.support;
        const real rho0 = P.rho0[0];
        const real psi = rho0 * P.vol0;
        Vec3 acc{0, 0, 0};
        for_each_neighbor(g, xi, [&](int j) {
          if (j == i) return;
          const Vec3 rij = xi - pv.x[j];
          const real r2 = norm2(rij);
          if (r2 >= c * c || r2 <= real(0)) return;
          const real r = std::sqrt(r2);
          const Vec3 rhat = rij / r;
          const std::uint8_t phj = pv.phase[j];
          if (phj == kLiquid) {
            const real K = real(2) * rho0 / (rhoi + pv.rho[j]);
            const Vec3 f_coh = rhat * (-P.akinci_gamma * mi * pv.m[j] * akinci_cohesion(r, c));
            const Vec3 f_curv = (ni - pv.nrm[j]) * (-P.akinci_gamma * mi);
            acc += (f_coh + f_curv) * (K / mi);
          } else if (phj == kWall) {
            acc -= rhat * (P.akinci_beta * psi * akinci_adhesion(r, c));
          }
        });
        pv.a[i] += acc;
      },
      s);
}

}  // namespace cryosph
