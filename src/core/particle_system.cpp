#include "cryosph/core/particle_system.hpp"

#include <algorithm>
#include <stdexcept>

namespace cryosph {

void HostParticles::reserve(std::size_t n) {
  x.reserve(n); v.reserve(n); nrm.reserve(n);
  m.reserve(n); rho.reserve(n); T.reserve(n);
  phase.reserve(n); flags.reserve(n); id.reserve(n);
}

void HostParticles::clear() {
  x.clear(); v.clear(); nrm.clear();
  m.clear(); rho.clear(); T.clear();
  phase.clear(); flags.clear(); id.clear();
}

std::size_t HostParticles::push(Vec3 pos, Vec3 vel, real mass, real density, real temperature, Phase ph,
                                Vec3 normal, std::uint8_t flag) {
  x.push_back(pos);
  v.push_back(vel);
  nrm.push_back(normal);
  m.push_back(mass);
  rho.push_back(density);
  T.push_back(temperature);
  phase.push_back(static_cast<std::uint8_t>(ph));
  flags.push_back(flag);
  id.push_back(static_cast<std::uint32_t>(id.size()));
  return x.size() - 1;
}

void HostParticles::append(const HostParticles& o) {
  const auto base = static_cast<std::uint32_t>(size());
  x.insert(x.end(), o.x.begin(), o.x.end());
  v.insert(v.end(), o.v.begin(), o.v.end());
  nrm.insert(nrm.end(), o.nrm.begin(), o.nrm.end());
  m.insert(m.end(), o.m.begin(), o.m.end());
  rho.insert(rho.end(), o.rho.begin(), o.rho.end());
  T.insert(T.end(), o.T.begin(), o.T.end());
  phase.insert(phase.end(), o.phase.begin(), o.phase.end());
  flags.insert(flags.end(), o.flags.begin(), o.flags.end());
  for (std::uint32_t k : o.id) id.push_back(base + k);
}

template <class F>
void ParticleSystem::for_each_array(F&& f) {
  f(x); f(v); f(vt); f(nrm); f(m); f(m0); f(rho); f(T); f(phase); f(flags); f(id);
  f(a); f(ab); f(ns); f(vw); f(fw); f(mw);
  f(p); f(drho); f(dTdt); f(sigma); f(kappa); f(nu_t); f(alpha); f(wall_w); f(heat_w);
  f(out_m); f(out_h); f(w_norm); f(p_wv); f(rmin);
  f(spare_vec3); f(spare_real); f(spare_u8); f(spare_u32);
}

void ParticleSystem::allocate(std::size_t capacity) {
  for_each_array([capacity](auto& arr) { arr.resize(capacity); });
  capacity_ = capacity;
  n_ = 0;
}

void ParticleSystem::grow(std::size_t new_capacity) {
  if (new_capacity <= capacity_) return;
  const std::size_t keep = n_;
  for_each_array([&](auto& arr) { arr.resize_preserve(new_capacity, keep); });
  capacity_ = new_capacity;
}

void ParticleSystem::set_size(std::size_t n) {
  if (n > capacity_) throw std::out_of_range("ParticleSystem::set_size exceeds capacity");
  n_ = n;
}

void ParticleSystem::upload(const HostParticles& h, double growth_factor) {
  const std::size_t n = h.size();
  const auto cap = std::max<std::size_t>(static_cast<std::size_t>(static_cast<double>(n) * growth_factor), n + 1024);
  if (cap > capacity_) allocate(cap);
  x.upload(h.x); v.upload(h.v); vt.upload(h.v); nrm.upload(h.nrm);
  m.upload(h.m); m0.upload(h.m); rho.upload(h.rho); T.upload(h.T);
  phase.upload(h.phase); flags.upload(h.flags); id.upload(h.id);
  // Scratch fields start from a clean state.
  a.zero(); ab.zero(); ns.zero(); vw.zero(); fw.zero(); mw.zero(); p.zero(); drho.zero(); dTdt.zero(); sigma.zero(); kappa.zero();
  nu_t.zero(); alpha.zero(); wall_w.zero(); heat_w.zero(); out_m.zero(); out_h.zero(); w_norm.zero(); p_wv.zero();
  n_ = n;
  next_id = 0;
  for (std::uint32_t k : h.id) next_id = std::max(next_id, k + 1);
}

HostParticles ParticleSystem::download() const {
  HostParticles h;
  h.x = x.to_host(n_); h.v = v.to_host(n_); h.nrm = nrm.to_host(n_);
  h.m = m.to_host(n_); h.rho = rho.to_host(n_); h.T = T.to_host(n_);
  h.phase = phase.to_host(n_); h.flags = flags.to_host(n_); h.id = id.to_host(n_);
  return h;
}

ParticleView ParticleSystem::view() {
  ParticleView pv;
  pv.n = static_cast<int>(n_);
  pv.capacity = static_cast<int>(capacity_);
  pv.x = x.data(); pv.v = v.data(); pv.vt = vt.data(); pv.nrm = nrm.data();
  pv.m = m.data(); pv.m0 = m0.data(); pv.rho = rho.data(); pv.T = T.data();
  pv.phase = phase.data(); pv.flags = flags.data(); pv.id = id.data();
  pv.a = a.data(); pv.ab = ab.data(); pv.ns = ns.data(); pv.vw = vw.data();
  pv.fw = fw.data(); pv.mw = mw.data();
  pv.p = p.data(); pv.drho = drho.data(); pv.dTdt = dTdt.data(); pv.sigma = sigma.data();
  pv.kappa = kappa.data(); pv.nu_t = nu_t.data(); pv.alpha = alpha.data();
  pv.wall_w = wall_w.data(); pv.heat_w = heat_w.data();
  pv.out_m = out_m.data(); pv.out_h = out_h.data(); pv.w_norm = w_norm.data();
  pv.p_wv = p_wv.data();
  pv.rmin = rmin.data();
  return pv;
}

}  // namespace cryosph
