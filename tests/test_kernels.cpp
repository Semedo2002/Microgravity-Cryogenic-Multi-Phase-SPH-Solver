// Smoothing-kernel properties: normalisation, gradient consistency, support.
#include <gtest/gtest.h>

#include <cmath>

#include "cryosph/physics/SPH_kernels.cuh"

using namespace cryosph;

namespace {

class KernelTest : public ::testing::TestWithParam<KernelType> {};

/// Lattice sum  dx^3 sum_j W(|x_j|)  approximates the integral of W.
TEST_P(KernelTest, PartitionOfUnityOnLattice) {
  const real dx = real(0.01);
  const KernelParams k = KernelParams::make(GetParam(), real(1.3) * dx);
  const int n = 4;
  double sum = 0;
  for (int i = -n; i <= n; ++i)
    for (int j = -n; j <= n; ++j)
      for (int l = -n; l <= n; ++l) {
        const real r = dx * std::sqrt(real(i * i + j * j + l * l));
        sum += kernel_W(k, r);
      }
  EXPECT_NEAR(sum * dx * dx * dx, 1.0, 0.02);
}

/// Radial integral 4 pi int_0^2h r^2 W dr == 1 (exact normalisation).
TEST_P(KernelTest, ContinuousNormalisation) {
  const KernelParams k = KernelParams::make(GetParam(), real(0.5));
  const int steps = 20000;
  const double dr = double(k.support) / steps;
  double acc = 0;
  for (int s = 0; s < steps; ++s) {
    const double r = (s + 0.5) * dr;
    acc += 4.0 * 3.14159265358979323846 * r * r * double(kernel_W(k, real(r))) * dr;
  }
  EXPECT_NEAR(acc, 1.0, 1e-4);
}

TEST_P(KernelTest, DerivativeMatchesFiniteDifference) {
  const KernelParams k = KernelParams::make(GetParam(), real(1));
  for (double r = 0.05; r < 1.95; r += 0.1) {
    const double eps = 1e-3;
    const double fd = (double(kernel_W(k, real(r + eps))) - double(kernel_W(k, real(r - eps)))) / (2 * eps);
    EXPECT_NEAR(double(kernel_dWdr(k, real(r))), fd, 1e-3 * double(k.norm) + 1e-4) << "r = " << r;
    EXPECT_LE(kernel_dWdr(k, real(r)), real(0));
  }
}

TEST_P(KernelTest, CompactSupportAndGradientDirection) {
  const KernelParams k = KernelParams::make(GetParam(), real(0.1));
  EXPECT_EQ(kernel_W(k, real(0.2)), real(0));
  EXPECT_EQ(kernel_W(k, real(0.25)), real(0));
  EXPECT_EQ(kernel_dWdr(k, real(0.21)), real(0));
  const Vec3 rij{real(0.05), 0, 0};
  const Vec3 g = kernel_gradW(k, rij, real(0.05));
  EXPECT_LT(g.x, real(0));  // points from i towards j
  EXPECT_EQ(g.y, real(0));
  const Vec3 g0 = kernel_gradW(k, Vec3{0, 0, 0}, real(0));
  EXPECT_EQ(norm(g0), real(0));
}

INSTANTIATE_TEST_SUITE_P(Kernels, KernelTest,
                         ::testing::Values(KernelType::CubicSpline, KernelType::WendlandC2),
                         [](const auto& info) {
                           return info.param == KernelType::CubicSpline ? std::string("CubicSpline")
                                                                        : std::string("WendlandC2");
                         });

TEST(Akinci, CohesionSplineShape) {
  const real c = real(1);
  EXPECT_EQ(akinci_cohesion(real(1.0), c), real(0));
  EXPECT_LT(akinci_cohesion(real(0.1), c), real(0));  // repulsive at short range
  EXPECT_GT(akinci_cohesion(real(0.7), c), real(0));  // attractive further out
  EXPECT_EQ(akinci_adhesion(real(0.4), c), real(0));
  EXPECT_GT(akinci_adhesion(real(0.75), c), real(0));
}

}  // namespace
