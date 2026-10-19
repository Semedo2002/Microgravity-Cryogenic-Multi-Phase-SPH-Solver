// Neighbour search: cell-list results must match brute force, and the sort
// must permute every persistent attribute consistently.
#include <gtest/gtest.h>

#include <random>

#include "cryosph/core/particle_system.hpp"
#include "cryosph/core/spatial_hash.hpp"

using namespace cryosph;

namespace {

HostParticles random_cloud(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  HostParticles hp;
  for (int i = 0; i < n; ++i) {
    const Vec3 p{u(rng), u(rng), u(rng)};
    hp.push(p, Vec3{p.z, p.x, p.y}, real(1) + p.x, real(1000), real(20) + p.y, i % 3 == 0 ? Phase::Vapor : Phase::Liquid);
  }
  return hp;
}

TEST(SpatialHash, NeighbourCountsMatchBruteForce) {
  const HostParticles hp = random_cloud(3000, 7);
  ParticleSystem ps;
  ps.upload(hp);
  SpatialHash grid;
  const real radius = real(0.08);
  grid.configure(Vec3{0, 0, 0}, Vec3{1, 1, 1}, radius);
  stream_t s = mem::create_stream();
  grid.build(ps, s);
  std::vector<int> counts;
  std::vector<std::uint64_t> id_sums;
  grid.neighbor_stats(ps, radius, counts, id_sums, s);
  const HostParticles sorted = ps.download();
  ASSERT_EQ(sorted.size(), hp.size());
  for (std::size_t i = 0; i < sorted.size(); i += 37) {
    int bc = 0;
    std::uint64_t bs = 0;
    for (std::size_t j = 0; j < sorted.size(); ++j)
      if (norm2(sorted.x[i] - sorted.x[j]) < radius * radius) {
        ++bc;
        bs += sorted.id[j];
      }
    EXPECT_EQ(counts[i], bc) << "particle " << i;
    EXPECT_EQ(id_sums[i], bs) << "particle " << i;
  }
  mem::destroy_stream(s);
}

TEST(SpatialHash, SortPermutesAttributesConsistently) {
  const HostParticles hp = random_cloud(2000, 11);
  ParticleSystem ps;
  ps.upload(hp);
  SpatialHash grid;
  grid.configure(Vec3{0, 0, 0}, Vec3{1, 1, 1}, real(0.1));
  stream_t s = mem::create_stream();
  grid.build(ps, s);
  const HostParticles out = ps.download();
  ASSERT_EQ(out.size(), hp.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::uint32_t id = out.id[i];
    ASSERT_LT(id, hp.size());
    EXPECT_EQ(out.x[i].x, hp.x[id].x);
    EXPECT_EQ(out.v[i].y, hp.v[id].y);
    EXPECT_EQ(out.m[i], hp.m[id]);
    EXPECT_EQ(out.T[i], hp.T[id]);
    EXPECT_EQ(out.phase[i], hp.phase[id]);
  }
  mem::destroy_stream(s);
}

TEST(SpatialHash, DeadParticlesAreCompacted) {
  HostParticles hp = random_cloud(500, 3);
  int dead = 0;
  for (std::size_t i = 0; i < hp.size(); i += 5) {
    hp.phase[i] = kDead;
    ++dead;
  }
  ParticleSystem ps;
  ps.upload(hp);
  SpatialHash grid;
  grid.configure(Vec3{0, 0, 0}, Vec3{1, 1, 1}, real(0.1));
  stream_t s = mem::create_stream();
  const int removed = grid.build(ps, s);
  EXPECT_EQ(removed, dead);
  EXPECT_EQ(ps.size(), hp.size() - static_cast<std::size_t>(dead));
  const HostParticles out = ps.download();
  for (auto ph : out.phase) EXPECT_NE(ph, kDead);
  mem::destroy_stream(s);
}

}  // namespace
