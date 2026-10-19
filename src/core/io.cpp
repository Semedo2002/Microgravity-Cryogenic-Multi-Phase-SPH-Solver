#include "cryosph/core/io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace cryosph {

namespace {

/// Legacy VTK binary data is big-endian.
template <class T>
T to_big_endian(T v) {
  if constexpr (std::endian::native == std::endian::little) {
    auto bytes = std::bit_cast<std::array<unsigned char, sizeof(T)>>(v);
    std::reverse(bytes.begin(), bytes.end());
    return std::bit_cast<T>(bytes);
  } else {
    return v;
  }
}

template <class T>
void write_be(std::ofstream& out, T v) {
  const T b = to_big_endian(v);
  out.write(reinterpret_cast<const char*>(&b), sizeof(T));
}

struct FileCloser {
  void operator()(std::FILE* f) const noexcept { if (f) std::fclose(f); }
};

void write_float_array(std::FILE* f, const char* name, const std::vector<float>& vals, const std::vector<int>& keep) {
  std::fprintf(f, ",\"%s\":[", name);
  bool first = true;
  for (int i : keep) {
    const float v = vals[static_cast<std::size_t>(i)];
    std::fprintf(f, first ? "%.6g" : ",%.6g", std::isfinite(v) ? static_cast<double>(v) : 0.0);
    first = false;
  }
  std::fputc(']', f);
}

}  // namespace

void write_vtk(const std::string& path, const HostParticles& hp, const std::vector<real>& pressure) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("write_vtk: cannot open '" + path + "'");
  const std::size_t n = hp.size();
  out << "# vtk DataFile Version 3.0\nCryoSPH-CUDA particles\nBINARY\nDATASET POLYDATA\n";
  out << "POINTS " << n << " float\n";
  for (const Vec3& p : hp.x) {
    write_be(out, static_cast<float>(p.x));
    write_be(out, static_cast<float>(p.y));
    write_be(out, static_cast<float>(p.z));
  }
  out << "\nVERTICES " << n << ' ' << 2 * n << '\n';
  for (std::size_t i = 0; i < n; ++i) {
    write_be(out, std::int32_t{1});
    write_be(out, static_cast<std::int32_t>(i));
  }
  out << "\nPOINT_DATA " << n << '\n';
  auto scalar = [&](const char* name, auto getter) {
    out << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < n; ++i) write_be(out, static_cast<float>(getter(i)));
    out << '\n';
  };
  scalar("temperature", [&](std::size_t i) { return hp.T[i]; });
  scalar("mass", [&](std::size_t i) { return hp.m[i]; });
  scalar("density", [&](std::size_t i) { return hp.rho[i]; });
  scalar("phase", [&](std::size_t i) { return static_cast<float>(hp.phase[i]); });
  if (pressure.size() == n) scalar("pressure", [&](std::size_t i) { return pressure[i]; });
  out << "VECTORS velocity float\n";
  for (const Vec3& v : hp.v) {
    write_be(out, static_cast<float>(v.x));
    write_be(out, static_cast<float>(v.y));
    write_be(out, static_cast<float>(v.z));
  }
  out << '\n';
  if (!out) throw std::runtime_error("write_vtk: write failed for '" + path + "'");
}

TelemetryFrame make_frame(const HostParticles& hp, const std::vector<real>& pressure, double time, long long step) {
  TelemetryFrame f;
  const std::size_t n = hp.size();
  f.time = time;
  f.step = step;
  f.n = static_cast<int>(n);
  f.position.resize(3 * n);
  f.temperature.resize(n);
  f.pressure.assign(n, 0.0f);
  f.speed.resize(n);
  f.phase.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    f.position[3 * i] = static_cast<float>(hp.x[i].x);
    f.position[3 * i + 1] = static_cast<float>(hp.x[i].y);
    f.position[3 * i + 2] = static_cast<float>(hp.x[i].z);
    f.temperature[i] = static_cast<float>(hp.T[i]);
    if (pressure.size() == n) f.pressure[i] = static_cast<float>(pressure[i]);
    f.speed[i] = static_cast<float>(norm(hp.v[i]));
    f.phase[i] = hp.phase[i];
  }
  return f;
}

void write_json_frame(const std::string& path, const TelemetryFrame& fr, bool include_walls) {
  std::unique_ptr<std::FILE, FileCloser> fp(std::fopen(path.c_str(), "wb"));
  if (!fp) throw std::runtime_error("write_json_frame: cannot open '" + path + "'");
  std::FILE* f = fp.get();
  std::vector<int> keep;
  keep.reserve(static_cast<std::size_t>(fr.n));
  for (int i = 0; i < fr.n; ++i)
    if (include_walls || fr.phase[static_cast<std::size_t>(i)] != kWall) keep.push_back(i);

  std::fprintf(f, "{\"time\":%.9g,\"step\":%lld,\"n\":%zu", fr.time, fr.step, keep.size());
  std::fputs(",\"position\":[", f);
  bool first = true;
  for (int i : keep) {
    const auto k = static_cast<std::size_t>(i);
    std::fprintf(f, first ? "%.6g,%.6g,%.6g" : ",%.6g,%.6g,%.6g", static_cast<double>(fr.position[3 * k]),
                 static_cast<double>(fr.position[3 * k + 1]), static_cast<double>(fr.position[3 * k + 2]));
    first = false;
  }
  std::fputs("],\"phase\":[", f);
  first = true;
  for (int i : keep) {
    std::fprintf(f, first ? "%u" : ",%u", static_cast<unsigned>(fr.phase[static_cast<std::size_t>(i)]));
    first = false;
  }
  std::fputc(']', f);
  write_float_array(f, "temperature", fr.temperature, keep);
  write_float_array(f, "pressure", fr.pressure, keep);
  write_float_array(f, "speed", fr.speed, keep);
  std::fputs("}\n", f);
  if (std::ferror(f)) throw std::runtime_error("write_json_frame: write failed for '" + path + "'");
}

}  // namespace cryosph
