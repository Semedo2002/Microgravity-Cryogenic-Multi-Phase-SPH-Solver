// Uniform-grid neighbour search: cell keys, sort, cell ranges, reorder.
// Cell size = kernel support, so neighbours live in the 27 surrounding cells.
#pragma once

#include <cstdint>
#include <vector>

#include "cryosph/core/cuda_utils.cuh"
#include "cryosph/core/particle_system.hpp"

namespace cryosph {

/// Kernel-side view of the cell lists.
struct GridView {
  Vec3 origin{0, 0, 0};
  real inv_cell = 1;
  int nx = 1, ny = 1, nz = 1;
  const int* cell_start = nullptr;
  const int* cell_end = nullptr;
};

CRYOSPH_HD CRYOSPH_INLINE void grid_cell_coords(const GridView& g, Vec3 p, int& cx, int& cy, int& cz) {
  cx = clamp_val(static_cast<int>(std::floor((p.x - g.origin.x) * g.inv_cell)), 0, g.nx - 1);
  cy = clamp_val(static_cast<int>(std::floor((p.y - g.origin.y) * g.inv_cell)), 0, g.ny - 1);
  cz = clamp_val(static_cast<int>(std::floor((p.z - g.origin.z) * g.inv_cell)), 0, g.nz - 1);
}

CRYOSPH_HD CRYOSPH_INLINE int grid_cell_index(const GridView& g, Vec3 p) {
  int cx, cy, cz;
  grid_cell_coords(g, p, cx, cy, cz);
  return (cz * g.ny + cy) * g.nx + cx;
}

/// Calls f(j) for every particle j stored in the 27 cells around `p`
/// (including the particle itself when p is a particle position).  The caller
/// is responsible for the distance test against the support radius.
template <class F>
CRYOSPH_HD CRYOSPH_INLINE void for_each_neighbor(const GridView& g, Vec3 p, F&& f) {
  int cx, cy, cz;
  grid_cell_coords(g, p, cx, cy, cz);
  const int z0 = cz > 0 ? cz - 1 : 0, z1 = cz < g.nz - 1 ? cz + 1 : cz;
  const int y0 = cy > 0 ? cy - 1 : 0, y1 = cy < g.ny - 1 ? cy + 1 : cy;
  const int x0 = cx > 0 ? cx - 1 : 0, x1 = cx < g.nx - 1 ? cx + 1 : cx;
  for (int z = z0; z <= z1; ++z) {
    for (int y = y0; y <= y1; ++y) {
      const int row = (z * g.ny + y) * g.nx;
      for (int x = x0; x <= x1; ++x) {
        const int c = row + x;
        const int s = g.cell_start[c];
        if (s < 0) continue;
        const int e = g.cell_end[c];
        for (int j = s; j < e; ++j) f(j);
      }
    }
  }
}

class SpatialHash {
 public:
  /// Defines the grid covering [lo, hi] (padded by one cell) with the given
  /// cell size (normally the kernel support radius).
  void configure(Vec3 lo, Vec3 hi, real cell_size);

  /// Sorts and permutes `ps` into cell order, drops dead particles, builds the
  /// cell lists.  Returns the number of particles that were removed.
  int build(ParticleSystem& ps, stream_t stream);

  GridView view() const;
  int num_cells() const noexcept { return num_cells_; }
  real cell_size() const noexcept { return cell_; }
  Vec3 lower() const noexcept { return origin_; }
  Vec3 upper() const noexcept;

  /// Test / diagnostic helper: for every particle, the number of particles
  /// within `radius` (self included) and the sum of their ids.  Uses the
  /// current cell lists, so call build() first.
  void neighbor_stats(ParticleSystem& ps, real radius, std::vector<int>& counts,
                      std::vector<std::uint64_t>& id_sums, stream_t stream);

 private:
  Vec3 origin_{0, 0, 0};
  real cell_ = 1;
  int nx_ = 1, ny_ = 1, nz_ = 1;
  int num_cells_ = 1;
  DeviceArray<std::uint32_t> keys_, perm_;
  DeviceArray<int> cell_start_, cell_end_;
  std::vector<int> host_counts_;  // CPU counting-sort workspace
};

/// Applies a gather permutation to every persistent particle array:
/// new[i] = old[perm[i]] for i < n_new.
void reorder_particles(ParticleSystem& ps, const std::uint32_t* perm, int n_new, stream_t stream);

}  // namespace cryosph
