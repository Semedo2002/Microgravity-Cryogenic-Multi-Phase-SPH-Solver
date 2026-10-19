// Tank geometries as signed distance functions (negative inside the fluid
// domain) and the lattice generator that fills them with particles.
#pragma once

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "cryosph/core/cuda_utils.cuh"
#include "cryosph/core/particle_system.hpp"
#include "cryosph/physics/fluid_properties.hpp"

namespace cryosph {

struct AABB {
  Vec3 lo{0, 0, 0};
  Vec3 hi{0, 0, 0};
};

class Geometry {
 public:
  virtual ~Geometry() = default;
  /// Signed distance to the tank wall (negative inside).
  virtual double sdf(double x, double y, double z) const = 0;
  /// Tight bounding box of the fluid domain.
  virtual AABB bounds() const = 0;
  virtual std::string name() const = 0;
  /// Analytic interior volume, or a negative value when unknown.
  virtual double volume() const { return -1.0; }
  /// Unit outward normal (gradient of the SDF, central differences).
  Vec3 outward_normal(double x, double y, double z, double eps) const;
};

/// Right circular cylinder with flat end caps, axis along z, centred at origin.
class CylinderTank final : public Geometry {
 public:
  CylinderTank(double radius, double height);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "cylinder"; }
  double volume() const override;
  double radius() const { return R_; }
  double height() const { return H_; }

 private:
  double R_, H_;
};

/// Cylinder with hemispherical domes (a capsule); `length` is the straight
/// barrel length, total length = length + 2 radius.  Axis z, centred.
class CapsuleTank final : public Geometry {
 public:
  CapsuleTank(double radius, double length);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "capsule"; }
  double volume() const override;

 private:
  double R_, L_;
};

/// Spherical propellant depot, centred at the origin.
class SphereTank final : public Geometry {
 public:
  explicit SphereTank(double radius);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "sphere"; }
  double volume() const override;

 private:
  double R_;
};

/// Rectangular box centred at the origin with the given full side lengths.
class BoxTank final : public Geometry {
 public:
  BoxTank(double lx, double ly, double lz);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "box"; }
  double volume() const override;

 private:
  double hx_, hy_, hz_;
};

/// Geometry shifted by a constant offset.
class Translated final : public Geometry {
 public:
  Translated(std::shared_ptr<const Geometry> g, Vec3 offset);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "translated(" + g_->name() + ")"; }
  double volume() const override { return g_->volume(); }

 private:
  std::shared_ptr<const Geometry> g_;
  double ox_, oy_, oz_;
};

/// Fluid domain of A minus the solid B: internal obstacles, baffles and
/// propellant-management vanes (B's interior becomes wall).  sdf = max(sdf_A, -sdf_B).
class Difference final : public Geometry {
 public:
  Difference(std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override { return a_->bounds(); }
  std::string name() const override { return "difference(" + a_->name() + ", " + b_->name() + ")"; }

 private:
  std::shared_ptr<const Geometry> a_, b_;
};

/// Fluid domain A united with B (e.g. a tank with an attached sump).  sdf = min(sdf_A, sdf_B).
class Union final : public Geometry {
 public:
  Union(std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "union(" + a_->name() + ", " + b_->name() + ")"; }

 private:
  std::shared_ptr<const Geometry> a_, b_;
};

/// Closed triangle mesh loaded from a Wavefront OBJ file.  The sign of the
/// SDF uses the generalized winding number, which is robust to small gaps and
/// inconsistent triangle orientation.
class MeshTank final : public Geometry {
 public:
  MeshTank(const std::string& obj_path, double scale = 1.0, Vec3 offset = {0, 0, 0});
  MeshTank(std::vector<std::array<double, 3>> vertices, std::vector<std::array<int, 3>> triangles);
  double sdf(double x, double y, double z) const override;
  AABB bounds() const override;
  std::string name() const override { return "mesh"; }
  double volume() const override;
  std::size_t num_triangles() const { return tris_.size(); }
  /// Generalized winding number at p (~1 inside, ~0 outside).
  double winding_number(double x, double y, double z) const;

 private:
  void finalize();
  std::vector<std::array<double, 3>> verts_;
  std::vector<std::array<int, 3>> tris_;
  AABB box_;
};

/// Parses a Wavefront OBJ file (v / f records; polygons are fan-triangulated).
void load_obj(const std::string& path, std::vector<std::array<double, 3>>& vertices,
              std::vector<std::array<int, 3>>& triangles);

enum class FillMode : int {
  Settled = 0,  ///< flat interface normal to settle_direction (liquid "below")
  CentralBubble = 1,  ///< vapour bubble around bubble_center, wetting liquid at the walls
  Full = 2  ///< tank completely filled with liquid
};

struct FillSpec {
  FillMode mode = FillMode::Settled;
  double fill_fraction = 0.5;  ///< liquid volume / tank volume
  Vec3 settle_direction{0, 0, -1};  ///< direction in which the liquid has settled
  Vec3 bubble_center{0, 0, 0};
  bool include_vapor = true;  ///< false: free-surface single-phase run
  int wall_layers = 3;  ///< must cover the kernel support (2h)
  real T_liquid = std::numeric_limits<real>::quiet_NaN();  ///< NaN -> T_sat
  real T_vapor = std::numeric_limits<real>::quiet_NaN();  ///< NaN -> T_sat
  real T_wall = std::numeric_limits<real>::quiet_NaN();  ///< NaN -> T_sat
  /// Heated wall band along z (walls outside get no heater power).
  double heater_zmin = -std::numeric_limits<double>::infinity();
  double heater_zmax = std::numeric_limits<double>::infinity();
};

/// Generates wall, liquid and vapour particles on a cubic lattice of spacing dx.
HostParticles build_tank_particles(const Geometry& geo, const FluidProperties& fluid, real dx, const FillSpec& fill);

/// Lattice block of liquid or vapour particles filling an axis-aligned box
/// (handy for tests and custom set-ups).
HostParticles make_block(Vec3 lo, Vec3 hi, real dx, Phase phase, real density, real temperature);

}  // namespace cryosph
