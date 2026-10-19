// Signed distance functions, OBJ meshes and the particle generator.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include "cryosph/core/geometry.hpp"

using namespace cryosph;

namespace {

constexpr double kPiD = 3.14159265358979323846;

TEST(Geometry, AnalyticSdfSigns) {
  const CylinderTank cyl(0.5, 2.0);
  EXPECT_NEAR(cyl.sdf(0, 0, 0), -0.5, 1e-12);
  EXPECT_NEAR(cyl.sdf(0.7, 0, 0), 0.2, 1e-12);
  EXPECT_NEAR(cyl.sdf(0, 0, 1.3), 0.3, 1e-12);
  EXPECT_NEAR(cyl.volume(), kPiD * 0.25 * 2.0, 1e-12);

  const SphereTank sph(1.0);
  EXPECT_NEAR(sph.sdf(0, 0, 0.5), -0.5, 1e-12);
  const CapsuleTank cap(0.5, 1.0);
  EXPECT_NEAR(cap.sdf(0, 0, 1.0), 0.0, 1e-12);
  EXPECT_NEAR(cap.sdf(0.5, 0, 0.2), 0.0, 1e-12);
  const BoxTank box(1, 2, 3);
  EXPECT_NEAR(box.sdf(0, 0, 0), -0.5, 1e-12);
  EXPECT_NEAR(box.volume(), 6.0, 1e-12);

  const Vec3 n = cyl.outward_normal(0.5, 0, 0, 1e-4);
  EXPECT_NEAR(n.x, 1.0, 1e-4);
}

TEST(Geometry, ObjMeshMatchesBox) {
  const std::string path = std::string(CRYOSPH_TEST_DATA_DIR) + "/unit_cube.obj";
  {
    std::ofstream f(path);
    f << "v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5\n"
         "v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5\n"
         "f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";
  }
  const MeshTank mesh(path);
  const BoxTank box(1, 1, 1);
  EXPECT_EQ(mesh.num_triangles(), 12u);
  EXPECT_NEAR(mesh.volume(), 1.0, 1e-12);
  EXPECT_NEAR(mesh.winding_number(0.1, 0.2, -0.1), 1.0, 1e-9);
  EXPECT_NEAR(mesh.winding_number(2, 0, 0), 0.0, 1e-9);
  for (double z : {-0.8, -0.3, 0.0, 0.45, 0.9})
    for (double x : {-0.7, 0.0, 0.3}) EXPECT_NEAR(mesh.sdf(x, 0.1, z), box.sdf(x, 0.1, z), 1e-9) << x << ' ' << z;
}

TEST(Geometry, TankFillFractionAndPhases) {
  const FluidProperties lh2 = FluidProperties::liquid_hydrogen();
  const BoxTank box(0.1, 0.1, 0.1);
  FillSpec fill;
  fill.fill_fraction = 0.4;
  fill.wall_layers = 3;
  const real dx = real(0.01);
  const HostParticles hp = build_tank_particles(box, lh2, dx, fill);
  int nl = 0, nv = 0, nw = 0;
  for (auto ph : hp.phase) {
    nl += ph == kLiquid;
    nv += ph == kVapor;
    nw += ph == kWall;
  }
  EXPECT_EQ(nl + nv, 1000);
  EXPECT_EQ(nl, 400);  // whole lattice planes: 4 of 10
  // Faces and edges are complete; the Euclidean SDF trims the outer corners.
  EXPECT_GT(nw, 6 * 10 * 10 * 3 + 12 * 10 * 9);
  EXPECT_LT(nw, 16 * 16 * 16 - 1000);
  // Liquid settles towards -z: every liquid particle lies below every vapour particle.
  real zl = -1e9, zv = 1e9;
  for (std::size_t i = 0; i < hp.size(); ++i) {
    if (hp.phase[i] == kLiquid) zl = std::max(zl, hp.x[i].z);
    if (hp.phase[i] == kVapor) zv = std::min(zv, hp.x[i].z);
  }
  EXPECT_LT(zl, zv);
  // Wall normals point into the fluid.
  for (std::size_t i = 0; i < hp.size(); ++i)
    if (hp.phase[i] == kWall && hp.x[i].z < real(-0.06) && std::abs(hp.x[i].x) < real(0.04) &&
        std::abs(hp.x[i].y) < real(0.04))
      EXPECT_NEAR(hp.nrm[i].z, 1.0, 1e-4);
}

TEST(Geometry, CentralBubbleFill) {
  const FluidProperties ln2 = FluidProperties::liquid_nitrogen();
  const SphereTank sph(0.05);
  FillSpec fill;
  fill.mode = FillMode::CentralBubble;
  fill.fill_fraction = 0.7;
  const HostParticles hp = build_tank_particles(sph, ln2, real(0.005), fill);
  double rv = 0, rl = 1e9;
  int nl = 0, nf = 0;
  for (std::size_t i = 0; i < hp.size(); ++i) {
    if (hp.phase[i] == kWall) continue;
    ++nf;
    const double r = norm(hp.x[i]);
    if (hp.phase[i] == kVapor) rv = std::max(rv, r);
    if (hp.phase[i] == kLiquid) {
      rl = std::min(rl, r);
      ++nl;
    }
  }
  EXPECT_LT(rv, rl + 1e-6);
  EXPECT_NEAR(double(nl) / nf, 0.7, 0.03);
}

}  // namespace
