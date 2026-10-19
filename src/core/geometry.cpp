#include "cryosph/core/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cryosph {

namespace {

constexpr double kPiD = 3.14159265358979323846;

struct D3 {
  double x, y, z;
};
inline D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot3(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline D3 cross3(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double len3(D3 a) { return std::sqrt(dot3(a, a)); }
inline D3 to_d3(const std::array<double, 3>& a) { return {a[0], a[1], a[2]}; }

/// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5).
D3 closest_point_on_triangle(D3 p, D3 a, D3 b, D3 c) {
  const D3 ab = b - a, ac = c - a, ap = p - a;
  const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const D3 bp = p - b;
  const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
  const D3 cp = p - c;
  const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const double denom = 1.0 / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

}  // namespace

// Geometry
Vec3 Geometry::outward_normal(double x, double y, double z, double eps) const {
  const double gx = sdf(x + eps, y, z) - sdf(x - eps, y, z);
  const double gy = sdf(x, y + eps, z) - sdf(x, y - eps, z);
  const double gz = sdf(x, y, z + eps) - sdf(x, y, z - eps);
  const double n = std::sqrt(gx * gx + gy * gy + gz * gz);
  if (n < 1e-300) return Vec3{0, 0, 0};
  return Vec3{static_cast<real>(gx / n), static_cast<real>(gy / n), static_cast<real>(gz / n)};
}

// CylinderTank
CylinderTank::CylinderTank(double radius, double height) : R_(radius), H_(height) {
  if (!(radius > 0 && height > 0)) throw std::invalid_argument("CylinderTank: radius and height must be > 0");
}
double CylinderTank::sdf(double x, double y, double z) const {
  const double qx = std::sqrt(x * x + y * y) - R_;
  const double qy = std::abs(z) - 0.5 * H_;
  const double ox = std::max(qx, 0.0), oy = std::max(qy, 0.0);
  return std::min(std::max(qx, qy), 0.0) + std::sqrt(ox * ox + oy * oy);
}
AABB CylinderTank::bounds() const {
  const auto r = static_cast<real>(R_), h = static_cast<real>(0.5 * H_);
  return {{-r, -r, -h}, {r, r, h}};
}
double CylinderTank::volume() const { return kPiD * R_ * R_ * H_; }

// CapsuleTank
CapsuleTank::CapsuleTank(double radius, double length) : R_(radius), L_(length) {
  if (!(radius > 0 && length >= 0)) throw std::invalid_argument("CapsuleTank: invalid dimensions");
}
double CapsuleTank::sdf(double x, double y, double z) const {
  const double hz = 0.5 * L_;
  const double dz = z - std::clamp(z, -hz, hz);
  return std::sqrt(x * x + y * y + dz * dz) - R_;
}
AABB CapsuleTank::bounds() const {
  const auto r = static_cast<real>(R_), h = static_cast<real>(0.5 * L_ + R_);
  return {{-r, -r, -h}, {r, r, h}};
}
double CapsuleTank::volume() const { return kPiD * R_ * R_ * L_ + 4.0 / 3.0 * kPiD * R_ * R_ * R_; }

// SphereTank
SphereTank::SphereTank(double radius) : R_(radius) {
  if (!(radius > 0)) throw std::invalid_argument("SphereTank: radius must be > 0");
}
double SphereTank::sdf(double x, double y, double z) const { return std::sqrt(x * x + y * y + z * z) - R_; }
AABB SphereTank::bounds() const {
  const auto r = static_cast<real>(R_);
  return {{-r, -r, -r}, {r, r, r}};
}
double SphereTank::volume() const { return 4.0 / 3.0 * kPiD * R_ * R_ * R_; }

// BoxTank
BoxTank::BoxTank(double lx, double ly, double lz) : hx_(0.5 * lx), hy_(0.5 * ly), hz_(0.5 * lz) {
  if (!(lx > 0 && ly > 0 && lz > 0)) throw std::invalid_argument("BoxTank: side lengths must be > 0");
}
double BoxTank::sdf(double x, double y, double z) const {
  const double qx = std::abs(x) - hx_, qy = std::abs(y) - hy_, qz = std::abs(z) - hz_;
  const double ox = std::max(qx, 0.0), oy = std::max(qy, 0.0), oz = std::max(qz, 0.0);
  return std::sqrt(ox * ox + oy * oy + oz * oz) + std::min(std::max({qx, qy, qz}), 0.0);
}
AABB BoxTank::bounds() const {
  return {{static_cast<real>(-hx_), static_cast<real>(-hy_), static_cast<real>(-hz_)},
          {static_cast<real>(hx_), static_cast<real>(hy_), static_cast<real>(hz_)}};
}
double BoxTank::volume() const { return 8.0 * hx_ * hy_ * hz_; }

// CSG / affine
Translated::Translated(std::shared_ptr<const Geometry> g, Vec3 offset)
    : g_(std::move(g)), ox_(offset.x), oy_(offset.y), oz_(offset.z) {
  if (!g_) throw std::invalid_argument("Translated: null geometry");
}
double Translated::sdf(double x, double y, double z) const { return g_->sdf(x - ox_, y - oy_, z - oz_); }
AABB Translated::bounds() const {
  AABB b = g_->bounds();
  const Vec3 o{static_cast<real>(ox_), static_cast<real>(oy_), static_cast<real>(oz_)};
  return {b.lo + o, b.hi + o};
}

Difference::Difference(std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b)
    : a_(std::move(a)), b_(std::move(b)) {
  if (!a_ || !b_) throw std::invalid_argument("Difference: null geometry");
}
double Difference::sdf(double x, double y, double z) const { return std::max(a_->sdf(x, y, z), -b_->sdf(x, y, z)); }

Union::Union(std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b) : a_(std::move(a)), b_(std::move(b)) {
  if (!a_ || !b_) throw std::invalid_argument("Union: null geometry");
}
double Union::sdf(double x, double y, double z) const { return std::min(a_->sdf(x, y, z), b_->sdf(x, y, z)); }
AABB Union::bounds() const {
  const AABB p = a_->bounds(), q = b_->bounds();
  return {Vec3{std::min(p.lo.x, q.lo.x), std::min(p.lo.y, q.lo.y), std::min(p.lo.z, q.lo.z)},
          Vec3{std::max(p.hi.x, q.hi.x), std::max(p.hi.y, q.hi.y), std::max(p.hi.z, q.hi.z)}};
}

// MeshTank
void load_obj(const std::string& path, std::vector<std::array<double, 3>>& vertices,
              std::vector<std::array<int, 3>>& triangles) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("load_obj: cannot open '" + path + "'");
  vertices.clear();
  triangles.clear();
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string tag;
    if (!(ls >> tag)) continue;
    if (tag == "v") {
      std::array<double, 3> v{};
      if (!(ls >> v[0] >> v[1] >> v[2])) throw std::runtime_error("load_obj: malformed vertex in '" + path + "'");
      vertices.push_back(v);
    } else if (tag == "f") {
      std::vector<int> idx;
      std::string tok;
      while (ls >> tok) {
        // Accept "i", "i/t", "i//n", "i/t/n"; negative indices are relative.
        const int k = std::stoi(tok.substr(0, tok.find('/')));
        const int resolved = k > 0 ? k - 1 : static_cast<int>(vertices.size()) + k;
        if (resolved < 0 || resolved >= static_cast<int>(vertices.size()))
          throw std::runtime_error("load_obj: face index out of range in '" + path + "'");
        idx.push_back(resolved);
      }
      for (std::size_t t = 1; t + 1 < idx.size(); ++t) triangles.push_back({idx[0], idx[t], idx[t + 1]});
    }
  }
  if (triangles.empty()) throw std::runtime_error("load_obj: no faces found in '" + path + "'");
}

MeshTank::MeshTank(const std::string& obj_path, double scale, Vec3 offset) {
  load_obj(obj_path, verts_, tris_);
  for (auto& v : verts_) {
    v[0] = v[0] * scale + offset.x;
    v[1] = v[1] * scale + offset.y;
    v[2] = v[2] * scale + offset.z;
  }
  finalize();
}

MeshTank::MeshTank(std::vector<std::array<double, 3>> vertices, std::vector<std::array<int, 3>> triangles)
    : verts_(std::move(vertices)), tris_(std::move(triangles)) {
  for (const auto& t : tris_)
    for (int k : t)
      if (k < 0 || k >= static_cast<int>(verts_.size())) throw std::invalid_argument("MeshTank: bad triangle index");
  finalize();
}

void MeshTank::finalize() {
  if (verts_.empty() || tris_.empty()) throw std::invalid_argument("MeshTank: empty mesh");
  D3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
  for (const auto& v : verts_) {
    lo = {std::min(lo.x, v[0]), std::min(lo.y, v[1]), std::min(lo.z, v[2])};
    hi = {std::max(hi.x, v[0]), std::max(hi.y, v[1]), std::max(hi.z, v[2])};
  }
  box_ = {{static_cast<real>(lo.x), static_cast<real>(lo.y), static_cast<real>(lo.z)},
          {static_cast<real>(hi.x), static_cast<real>(hi.y), static_cast<real>(hi.z)}};
}

double MeshTank::winding_number(double x, double y, double z) const {
  // Van Oosterom & Strackee solid angle of each triangle seen from p.
  const D3 p{x, y, z};
  double omega = 0;
  for (const auto& t : tris_) {
    const D3 a = to_d3(verts_[t[0]]) - p, b = to_d3(verts_[t[1]]) - p, c = to_d3(verts_[t[2]]) - p;
    const double la = len3(a), lb = len3(b), lc = len3(c);
    const double num = dot3(a, cross3(b, c));
    const double den = la * lb * lc + dot3(a, b) * lc + dot3(b, c) * la + dot3(c, a) * lb;
    omega += 2.0 * std::atan2(num, den);
  }
  return omega / (4.0 * kPiD);
}

double MeshTank::sdf(double x, double y, double z) const {
  const D3 p{x, y, z};
  double best = 1e300;
  for (const auto& t : tris_) {
    const D3 q = closest_point_on_triangle(p, to_d3(verts_[t[0]]), to_d3(verts_[t[1]]), to_d3(verts_[t[2]]));
    const D3 d = p - q;
    best = std::min(best, dot3(d, d));
  }
  const double dist = std::sqrt(best);
  return std::abs(winding_number(x, y, z)) > 0.5 ? -dist : dist;
}

AABB MeshTank::bounds() const { return box_; }

double MeshTank::volume() const {
  double v6 = 0;
  for (const auto& t : tris_) v6 += dot3(to_d3(verts_[t[0]]), cross3(to_d3(verts_[t[1]]), to_d3(verts_[t[2]])));
  return std::abs(v6) / 6.0;
}

// particle generation
HostParticles build_tank_particles(const Geometry& geo, const FluidProperties& fluid, real dx, const FillSpec& fill) {
  if (!(dx > 0)) throw std::invalid_argument("build_tank_particles: dx must be > 0");
  if (fill.wall_layers < 1) throw std::invalid_argument("build_tank_particles: wall_layers must be >= 1");
  if (fill.fill_fraction < 0 || fill.fill_fraction > 1)
    throw std::invalid_argument("build_tank_particles: fill_fraction must be in [0, 1]");

  const double h = dx;
  const double wall_thick = fill.wall_layers * h;
  const AABB b = geo.bounds();
  // Lattice anchored at the origin: nodes at (i + 1/2) dx.  For tanks whose
  // walls sit on a multiple of dx this places the wall surface exactly
  // half-way between the last fluid and the first wall layer.
  const int i0 = static_cast<int>(std::floor((b.lo.x - wall_thick) / h)) - 1;
  const int j0 = static_cast<int>(std::floor((b.lo.y - wall_thick) / h)) - 1;
  const int k0 = static_cast<int>(std::floor((b.lo.z - wall_thick) / h)) - 1;
  const int i1 = static_cast<int>(std::ceil((b.hi.x + wall_thick) / h)) + 1;
  const int j1 = static_cast<int>(std::ceil((b.hi.y + wall_thick) / h)) + 1;
  const int k1 = static_cast<int>(std::ceil((b.hi.z + wall_thick) / h)) + 1;
  const int nx = i1 - i0, ny = j1 - j0, nz = k1 - k0;
  const long long total = static_cast<long long>(nx) * ny * nz;
  if (total > 400'000'000LL) throw std::runtime_error("build_tank_particles: lattice too large, increase dx");

  std::vector<double> phi(static_cast<std::size_t>(total));
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const double x = (i0 + i + 0.5) * h, y = (j0 + j + 0.5) * h, z = (k0 + k + 0.5) * h;
        phi[(static_cast<std::size_t>(k) * ny + j) * nx + i] = geo.sdf(x, y, z);
      }

  const real T_sat = fluid.T_sat;
  const real T_liq = std::isnan(fill.T_liquid) ? T_sat : fill.T_liquid;
  const real T_vap = std::isnan(fill.T_vapor) ? T_sat : fill.T_vapor;
  const real T_wall = std::isnan(fill.T_wall) ? T_sat : fill.T_wall;
  const real vol = dx * dx * dx;

  HostParticles out;
  std::vector<Vec3> fluid_pts;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const double d = phi[(static_cast<std::size_t>(k) * ny + j) * nx + i];
        const double x = (i0 + i + 0.5) * h, y = (j0 + j + 0.5) * h, z = (k0 + k + 0.5) * h;
        const Vec3 p{static_cast<real>(x), static_cast<real>(y), static_cast<real>(z)};
        if (d < 0) {
          fluid_pts.push_back(p);
        } else if (d < wall_thick) {
          const Vec3 n_out = geo.outward_normal(x, y, z, 0.5 * h);
          const bool heated = z >= fill.heater_zmin && z <= fill.heater_zmax;
          out.push(p, {0, 0, 0}, fluid.rho_liquid * vol, fluid.rho_liquid, T_wall, Phase::Wall, -n_out,
                   heated ? pflag::Heated : std::uint8_t{0});
        }
      }
  if (fluid_pts.empty()) throw std::runtime_error("build_tank_particles: no fluid particles; dx too large?");

  // Classify fluid lattice nodes as liquid or vapour.
  const std::size_t nf = fluid_pts.size();
  std::vector<std::uint8_t> is_liquid(nf, 1);
  if (fill.mode != FillMode::Full) {
    // Liquid fills the nodes with the largest key: the "lowest" nodes along the
    // settle direction, or the nodes farthest from the bubble centre.
    std::vector<double> key(nf);
    const Vec3 g = normalized(fill.settle_direction);
    for (std::size_t q = 0; q < nf; ++q) {
      key[q] = fill.mode == FillMode::Settled ? static_cast<double>(dot(fluid_pts[q], g))
                                              : static_cast<double>(norm(fluid_pts[q] - fill.bubble_center));
    }
    std::vector<std::size_t> order(nf);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t c) { return key[a] > key[c]; });
    const auto n_liq = static_cast<std::size_t>(std::llround(fill.fill_fraction * static_cast<double>(nf)));
    std::fill(is_liquid.begin(), is_liquid.end(), 0);
    if (n_liq > 0) {
      // Round to complete lattice planes/shells for a clean initial interface.
      const double cut = key[order[n_liq - 1]] - 1e-6 * h;
      for (std::size_t q = 0; q < nf; ++q) is_liquid[q] = key[q] >= cut ? 1 : 0;
    }
  }

  for (std::size_t q = 0; q < nf; ++q) {
    if (is_liquid[q]) {
      out.push(fluid_pts[q], {0, 0, 0}, fluid.rho_liquid * vol, fluid.rho_liquid, T_liq, Phase::Liquid);
    } else if (fill.include_vapor) {
      out.push(fluid_pts[q], {0, 0, 0}, fluid.rho_vapor * vol, fluid.rho_vapor, T_vap, Phase::Vapor);
    }
  }
  return out;
}

HostParticles make_block(Vec3 lo, Vec3 hi, real dx, Phase phase, real density, real temperature) {
  HostParticles out;
  const int nx = static_cast<int>(std::floor((hi.x - lo.x) / dx + real(1e-4)));
  const int ny = static_cast<int>(std::floor((hi.y - lo.y) / dx + real(1e-4)));
  const int nz = static_cast<int>(std::floor((hi.z - lo.z) / dx + real(1e-4)));
  out.reserve(static_cast<std::size_t>(std::max(0, nx)) * std::max(0, ny) * std::max(0, nz));
  const real mass = density * dx * dx * dx;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const Vec3 p{lo.x + (i + real(0.5)) * dx, lo.y + (j + real(0.5)) * dx, lo.z + (k + real(0.5)) * dx};
        out.push(p, {0, 0, 0}, mass, density, temperature, phase);
      }
  return out;
}

}  // namespace cryosph
