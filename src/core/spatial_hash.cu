#include "cryosph/core/spatial_hash.hpp"

#include <cmath>
#include <numeric>

#if defined(CRYOSPH_WITH_CUDA)
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

namespace cryosph {

void SpatialHash::configure(Vec3 lo, Vec3 hi, real cell_size) {
  if (!(cell_size > 0)) throw std::invalid_argument("SpatialHash: cell size must be positive");
  cell_ = cell_size;
  // One padding cell on each side keeps particles that slightly penetrate
  // the outermost wall layer inside the grid.
  origin_ = lo - Vec3{cell_, cell_, cell_};
  const Vec3 ext = hi - lo;
  nx_ = std::max(1, static_cast<int>(std::ceil(ext.x / cell_)) + 2);
  ny_ = std::max(1, static_cast<int>(std::ceil(ext.y / cell_)) + 2);
  nz_ = std::max(1, static_cast<int>(std::ceil(ext.z / cell_)) + 2);
  const long long cells = static_cast<long long>(nx_) * ny_ * nz_;
  if (cells > 400'000'000LL) throw std::runtime_error("SpatialHash: grid too large; increase the cell size");
  num_cells_ = static_cast<int>(cells);
  cell_start_.resize(static_cast<std::size_t>(num_cells_) + 1);
  cell_end_.resize(static_cast<std::size_t>(num_cells_) + 1);
}

Vec3 SpatialHash::upper() const noexcept {
  return origin_ + Vec3{nx_ * cell_, ny_ * cell_, nz_ * cell_};
}

GridView SpatialHash::view() const {
  GridView g;
  g.origin = origin_;
  g.inv_cell = real(1) / cell_;
  g.nx = nx_;
  g.ny = ny_;
  g.nz = nz_;
  g.cell_start = cell_start_.data();
  g.cell_end = cell_end_.data();
  return g;
}

int SpatialHash::build(ParticleSystem& ps, stream_t stream) {
  const int n = static_cast<int>(ps.size());
  if (keys_.size() < ps.capacity()) {
    keys_.resize(ps.capacity());
    perm_.resize(ps.capacity());
  }
  GridView g = view();
  const int sentinel = num_cells_;
  std::uint32_t* keys = keys_.data();
  std::uint32_t* perm = perm_.data();
  const Vec3* x = ps.x.data();
  const std::uint8_t* phase = ps.phase.data();

  parallel_for(
      n,
      [=] CRYOSPH_HD(int i) {
        keys[i] = phase[i] == kDead ? static_cast<std::uint32_t>(sentinel)
                                    : static_cast<std::uint32_t>(grid_cell_index(g, x[i]));
        perm[i] = static_cast<std::uint32_t>(i);
      },
      stream);

  int n_alive = n;
#if defined(CRYOSPH_WITH_CUDA)
  auto policy = thrust::cuda::par.on(stream);
  thrust::device_ptr<std::uint32_t> kp(keys), pp(perm);
  thrust::sort_by_key(policy, kp, kp + n, pp);
  if (n > 0) {
    n_alive = static_cast<int>(
        thrust::lower_bound(policy, kp, kp + n, static_cast<std::uint32_t>(sentinel)) - kp);
  }
  cell_start_.fill_bytes(0xFF);  // -1 marks an empty cell
  cell_end_.fill_bytes(0xFF);
  int* cs = cell_start_.data();
  int* ce = cell_end_.data();
  const int na = n_alive;
  parallel_for(
      na,
      [=] CRYOSPH_HD(int i) {
        const std::uint32_t k = keys[i];
        if (i == 0 || keys[i - 1] != k) cs[k] = i;
        if (i == na - 1 || keys[i + 1] != k) ce[k] = i + 1;
      },
      stream);
#else
  // Stable counting sort: O(N + C), and it yields the cell ranges directly.
  host_counts_.assign(static_cast<std::size_t>(num_cells_) + 2, 0);
  for (int i = 0; i < n; ++i) ++host_counts_[keys[i] + 1];
  for (int c = 0; c <= num_cells_; ++c) host_counts_[c + 1] += host_counts_[c];
  int* cs = cell_start_.data();
  int* ce = cell_end_.data();
  for (int c = 0; c < num_cells_; ++c) {
    const int s = host_counts_[c], e = host_counts_[c + 1];
    cs[c] = (e > s) ? s : -1;
    ce[c] = (e > s) ? e : -1;
  }
  n_alive = host_counts_[num_cells_];
  // Scatter indices into sorted order, using the prefix sums as running offsets.
  std::vector<std::uint32_t> sorted(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) sorted[static_cast<std::size_t>(host_counts_[keys[i]]++)] = static_cast<std::uint32_t>(i);
  std::copy(sorted.begin(), sorted.end(), perm);
#endif

  reorder_particles(ps, perm, n_alive, stream);
  ps.set_size(static_cast<std::size_t>(n_alive));
  return n - n_alive;
}

namespace {
template <class T>
void gather_into(DeviceArray<T>& arr, DeviceArray<T>& spare, const std::uint32_t* perm, int n, stream_t stream) {
  T* dst = spare.data();
  const T* src = arr.data();
  parallel_for(n, [=] CRYOSPH_HD(int i) { dst[i] = src[perm[i]]; }, stream);
  swap(arr, spare);
}
}  // namespace

void reorder_particles(ParticleSystem& ps, const std::uint32_t* perm, int n_new, stream_t stream) {
  gather_into(ps.x, ps.spare_vec3, perm, n_new, stream);
  gather_into(ps.v, ps.spare_vec3, perm, n_new, stream);
  gather_into(ps.vt, ps.spare_vec3, perm, n_new, stream);
  gather_into(ps.nrm, ps.spare_vec3, perm, n_new, stream);
  gather_into(ps.m, ps.spare_real, perm, n_new, stream);
  gather_into(ps.m0, ps.spare_real, perm, n_new, stream);
  gather_into(ps.rho, ps.spare_real, perm, n_new, stream);
  gather_into(ps.T, ps.spare_real, perm, n_new, stream);
  gather_into(ps.phase, ps.spare_u8, perm, n_new, stream);
  gather_into(ps.flags, ps.spare_u8, perm, n_new, stream);
  gather_into(ps.id, ps.spare_u32, perm, n_new, stream);
}

void SpatialHash::neighbor_stats(ParticleSystem& ps, real radius, std::vector<int>& counts,
                                 std::vector<std::uint64_t>& id_sums, stream_t stream) {
  const int n = static_cast<int>(ps.size());
  DeviceArray<int> d_counts(static_cast<std::size_t>(std::max(n, 1)));
  DeviceArray<unsigned long long> d_sums(static_cast<std::size_t>(std::max(n, 1)));
  int* cnt = d_counts.data();
  unsigned long long* sums = d_sums.data();
  const Vec3* x = ps.x.data();
  const std::uint32_t* id = ps.id.data();
  const GridView g = view();
  const real r2max = radius * radius;
  parallel_for(
      n,
      [=] CRYOSPH_HD(int i) {
        const Vec3 xi = x[i];
        int c = 0;
        unsigned long long s = 0;
        for_each_neighbor(g, xi, [&](int j) {
          if (norm2(xi - x[j]) < r2max) {
            ++c;
            s += id[j];
          }
        });
        cnt[i] = c;
        sums[i] = s;
      },
      stream);
  mem::synchronize(stream);
  counts = d_counts.to_host(static_cast<std::size_t>(n));
  auto tmp = d_sums.to_host(static_cast<std::size_t>(n));
  id_sums.assign(tmp.begin(), tmp.end());
}

}  // namespace cryosph
