// VTK and JSON writers.
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "cryosph/core/geometry.hpp"
#include "cryosph/core/io.hpp"

using namespace cryosph;

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

TEST(IO, VtkHeaderAndSize) {
  const HostParticles hp = make_block(Vec3{0, 0, 0}, Vec3{real(0.03), real(0.02), real(0.01)}, real(0.01),
                                      Phase::Liquid, real(70.8), real(20.3));
  ASSERT_EQ(hp.size(), 6u);
  const std::string path = std::string(CRYOSPH_TEST_DATA_DIR) + "/block.vtk";
  write_vtk(path, hp, std::vector<real>(hp.size(), real(1e5)));
  const std::string s = slurp(path);
  EXPECT_EQ(s.rfind("# vtk DataFile Version 3.0", 0), 0u);
  EXPECT_NE(s.find("POINTS 6 float"), std::string::npos);
  EXPECT_NE(s.find("SCALARS pressure float 1"), std::string::npos);
  EXPECT_NE(s.find("VECTORS velocity float"), std::string::npos);
}

TEST(IO, JsonFrameFiltersWalls) {
  HostParticles hp = make_block(Vec3{0, 0, 0}, Vec3{real(0.02), real(0.01), real(0.01)}, real(0.01), Phase::Liquid,
                                real(1000), real(300));
  hp.push(Vec3{1, 1, 1}, Vec3{0, 0, 0}, 1, 1000, 300, Phase::Wall);
  const TelemetryFrame fr = make_frame(hp, {}, 0.5, 42);
  const std::string path = std::string(CRYOSPH_TEST_DATA_DIR) + "/frame.json";
  write_json_frame(path, fr, /*include_walls=*/false);
  const std::string s = slurp(path);
  EXPECT_NE(s.find("\"step\":42"), std::string::npos);
  EXPECT_NE(s.find("\"n\":2"), std::string::npos);
  EXPECT_NE(s.find("\"phase\":[0,0]"), std::string::npos);
  EXPECT_NE(s.find("\"temperature\":[300,300]"), std::string::npos);
}

}  // namespace
