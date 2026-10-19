// Backend layer shared by the CUDA and CPU builds: real/Vec3, DeviceArray,
// parallel_for and reductions. Physics code is written once as host/device
// lambdas and runs as CUDA kernels or OpenMP loops.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(CRYOSPH_WITH_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__)
#define CRYOSPH_HD __host__ __device__
#define CRYOSPH_INLINE __forceinline__
#else
#define CRYOSPH_HD
#define CRYOSPH_INLINE inline
#endif

#if !defined(CRYOSPH_WITH_CUDA) || defined(__CUDACC__)
#define CRYOSPH_DEVICE_TU 1
#else
#define CRYOSPH_DEVICE_TU 0
#endif

#if CRYOSPH_DEVICE_TU && defined(CRYOSPH_WITH_CUDA)
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cryosph {

// precision
#if defined(CRYOSPH_DOUBLE_PRECISION)
using real = double;
#else
using real = float;
#endif

#if defined(CRYOSPH_WITH_CUDA)
using stream_t = cudaStream_t;
using event_t = cudaEvent_t;
inline constexpr bool kCudaBackend = true;
#else
using stream_t = void*;
using event_t = void*;
inline constexpr bool kCudaBackend = false;
#endif

inline constexpr real kPi = real(3.14159265358979323846);

// error checks
#if defined(CRYOSPH_WITH_CUDA)
#define CRYOSPH_CUDA_CHECK(expr)                                                       \
  do {                                                                                 \
    cudaError_t cryosph_err__ = (expr);                                                \
    if (cryosph_err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error '") +                           \
                               cudaGetErrorString(cryosph_err__) + "' at " __FILE__ \
                               ":" + std::to_string(__LINE__));                        \
    }                                                                                  \
  } while (0)
#endif

// Vec3
/// Plain aggregate 3-vector.  Trivially copyable so it can live in device
/// memory and be memcpy'd between host and device.
struct Vec3 {
  real x, y, z;
};

CRYOSPH_HD CRYOSPH_INLINE Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
CRYOSPH_HD CRYOSPH_INLINE Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
CRYOSPH_HD CRYOSPH_INLINE Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
CRYOSPH_HD CRYOSPH_INLINE Vec3 operator*(Vec3 a, real s) { return {a.x * s, a.y * s, a.z * s}; }
CRYOSPH_HD CRYOSPH_INLINE Vec3 operator*(real s, Vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
CRYOSPH_HD CRYOSPH_INLINE Vec3 operator/(Vec3 a, real s) { return {a.x / s, a.y / s, a.z / s}; }
CRYOSPH_HD CRYOSPH_INLINE Vec3& operator+=(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
CRYOSPH_HD CRYOSPH_INLINE Vec3& operator-=(Vec3& a, Vec3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
CRYOSPH_HD CRYOSPH_INLINE Vec3& operator*=(Vec3& a, real s) { a.x *= s; a.y *= s; a.z *= s; return a; }
CRYOSPH_HD CRYOSPH_INLINE real dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
CRYOSPH_HD CRYOSPH_INLINE Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
CRYOSPH_HD CRYOSPH_INLINE real norm2(Vec3 a) { return dot(a, a); }
CRYOSPH_HD CRYOSPH_INLINE real norm(Vec3 a) { return std::sqrt(dot(a, a)); }
/// Unit vector, or the zero vector when |a| is below `eps`.
CRYOSPH_HD CRYOSPH_INLINE Vec3 normalized(Vec3 a, real eps = real(1e-20)) {
  const real n = norm(a);
  return n > eps ? a / n : Vec3{0, 0, 0};
}

template <class T>
CRYOSPH_HD CRYOSPH_INLINE T clamp_val(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// atomics
/// Atomically increments *addr and returns the previous value (both backends).
CRYOSPH_HD CRYOSPH_INLINE unsigned atomic_inc(unsigned* addr) {
#if defined(__CUDA_ARCH__)
  return atomicAdd(addr, 1u);
#else
  return std::atomic_ref<unsigned>(*addr).fetch_add(1u, std::memory_order_relaxed);
#endif
}

// memory primitives
namespace mem {
void* device_alloc(std::size_t bytes);
void device_free(void* ptr) noexcept;
void copy_h2d(void* dst, const void* src, std::size_t bytes);
void copy_d2h(void* dst, const void* src, std::size_t bytes);
void copy_d2d(void* dst, const void* src, std::size_t bytes);
void copy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t stream);
void memset_device(void* ptr, int value, std::size_t bytes);

/// Page-locked host memory (enables truly asynchronous DMA).  With
/// `mapped = true` the allocation is also mapped into the device address
/// space for zero-copy access (see mapped_device_pointer).
void* pinned_alloc(std::size_t bytes, bool mapped);
void pinned_free(void* ptr) noexcept;
void* mapped_device_pointer(void* host_ptr);

/// ordered = true gives a blocking stream, so the default-stream memset/memcpy
/// helpers stay ordered with its kernels. Only the telemetry copy stream uses false.
stream_t create_stream(bool ordered = true);
void destroy_stream(stream_t s) noexcept;
void synchronize(stream_t s);
void device_synchronize();

event_t create_event();
void destroy_event(event_t e) noexcept;
void record_event(event_t e, stream_t s);
void stream_wait_event(stream_t s, event_t e);
bool event_ready(event_t e);
void event_synchronize(event_t e);

/// Name of the compute device ("CPU (OpenMP, N threads)" for the CPU build).
std::string device_name();
/// Number of CUDA devices visible (0 for the CPU build).
int device_count();
}  // namespace mem

// DeviceArray
/// Owning, move-only buffer that lives in device memory on the CUDA backend
/// and in ordinary host memory on the CPU backend.
template <class T>
class DeviceArray {
  static_assert(std::is_trivially_copyable_v<T>, "DeviceArray requires trivially copyable T");

 public:
  DeviceArray() = default;
  explicit DeviceArray(std::size_t n) { resize(n); }
  ~DeviceArray() { release(); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }
  DeviceArray& operator=(DeviceArray&& o) noexcept {
    if (this != &o) {
      release();
      ptr_ = o.ptr_; size_ = o.size_;
      o.ptr_ = nullptr; o.size_ = 0;
    }
    return *this;
  }

  /// Reallocates to exactly n elements; contents are NOT preserved.
  void resize(std::size_t n) {
    if (n == size_) return;
    release();
    if (n > 0) ptr_ = static_cast<T*>(mem::device_alloc(n * sizeof(T)));
    size_ = n;
  }

  /// Reallocates to n elements keeping the first `keep` elements.
  void resize_preserve(std::size_t n, std::size_t keep) {
    keep = std::min({keep, n, size_});
    T* fresh = n > 0 ? static_cast<T*>(mem::device_alloc(n * sizeof(T))) : nullptr;
    if (keep > 0) mem::copy_d2d(fresh, ptr_, keep * sizeof(T));
    release();
    ptr_ = fresh;
    size_ = n;
  }

  void release() noexcept {
    if (ptr_) mem::device_free(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }

  void zero() { if (size_) mem::memset_device(ptr_, 0, size_ * sizeof(T)); }
  void fill_bytes(int byte) { if (size_) mem::memset_device(ptr_, byte, size_ * sizeof(T)); }

  void upload(const T* host, std::size_t count, std::size_t offset = 0) {
    check_range(count, offset);
    if (count) mem::copy_h2d(ptr_ + offset, host, count * sizeof(T));
  }
  void upload(const std::vector<T>& host, std::size_t offset = 0) { upload(host.data(), host.size(), offset); }

  void download(T* host, std::size_t count, std::size_t offset = 0) const {
    check_range(count, offset);
    if (count) mem::copy_d2h(host, ptr_ + offset, count * sizeof(T));
  }
  std::vector<T> to_host(std::size_t count) const {
    std::vector<T> out(count);
    download(out.data(), count);
    return out;
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return size_; }

  friend void swap(DeviceArray& a, DeviceArray& b) noexcept {
    std::swap(a.ptr_, b.ptr_);
    std::swap(a.size_, b.size_);
  }

 private:
  void check_range(std::size_t count, std::size_t offset) const {
    if (offset + count > size_) throw std::out_of_range("DeviceArray: transfer out of range");
  }
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

// parallel primitives
#if CRYOSPH_DEVICE_TU
#if defined(CRYOSPH_WITH_CUDA)

namespace detail {
template <class F>
__global__ void for_each_kernel(int n, F f) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) f(i);
}
inline constexpr int kBlockSize = 128;
}  // namespace detail

/// Executes f(i) for i in [0, n) as a CUDA kernel on `stream`.
template <class F>
void parallel_for(int n, F f, stream_t stream) {
  if (n <= 0) return;
  const int grid = (n + detail::kBlockSize - 1) / detail::kBlockSize;
  detail::for_each_kernel<<<grid, detail::kBlockSize, 0, stream>>>(n, f);
  CRYOSPH_CUDA_CHECK(cudaGetLastError());
}

/// Returns sum_i f(i) (f must return T).  Synchronous.
template <class T, class F>
T reduce_sum(int n, F f, T init, stream_t stream) {
  if (n <= 0) return init;
  return thrust::transform_reduce(thrust::cuda::par.on(stream), thrust::counting_iterator<int>(0),
                                  thrust::counting_iterator<int>(n), f, init, thrust::plus<T>());
}

/// Returns max(init, max_i f(i)).  Synchronous.
template <class T, class F>
T reduce_max(int n, F f, T init, stream_t stream) {
  if (n <= 0) return init;
  return thrust::transform_reduce(thrust::cuda::par.on(stream), thrust::counting_iterator<int>(0),
                                  thrust::counting_iterator<int>(n), f, init, thrust::maximum<T>());
}

/// out[i] = sum_{k<i} in[k]; returns the total.  Deterministic.  Synchronous.
inline unsigned exclusive_scan(int n, const unsigned* in, unsigned* out, stream_t stream) {
  if (n <= 0) return 0;
  thrust::exclusive_scan(thrust::cuda::par.on(stream), in, in + n, out);
  unsigned last_in = 0, last_out = 0;
  CRYOSPH_CUDA_CHECK(cudaMemcpyAsync(&last_in, in + n - 1, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
  CRYOSPH_CUDA_CHECK(cudaMemcpyAsync(&last_out, out + n - 1, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
  CRYOSPH_CUDA_CHECK(cudaStreamSynchronize(stream));
  return last_out + last_in;
}

#else  // ------------------------------------------------- CPU / OpenMP ----

template <class F>
void parallel_for(int n, F f, stream_t /*stream*/) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) f(i);
}

template <class T, class F>
T reduce_sum(int n, F f, T init, stream_t /*stream*/) {
#if defined(_OPENMP)
  // Fixed per-thread blocks summed in thread order: an OpenMP reduction clause
  // combines partial sums in arbitrary order and broke run-to-run determinism.
  std::vector<T> part(static_cast<std::size_t>(omp_get_max_threads()), T(0));
#pragma omp parallel
  {
    const int t = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    const int lo = static_cast<int>(static_cast<long long>(n) * t / nt);
    const int hi = static_cast<int>(static_cast<long long>(n) * (t + 1) / nt);
    T acc = T(0);
    for (int i = lo; i < hi; ++i) acc += f(i);
    part[static_cast<std::size_t>(t)] = acc;
  }
  T acc = init;
  for (const T& v : part) acc += v;
  return acc;
#else
  T acc = init;
  for (int i = 0; i < n; ++i) acc += f(i);
  return acc;
#endif
}

template <class T, class F>
T reduce_max(int n, F f, T init, stream_t /*stream*/) {
  T acc = init;
#pragma omp parallel for schedule(static) reduction(max : acc)
  for (int i = 0; i < n; ++i) {
    const T v = f(i);
    acc = v > acc ? v : acc;
  }
  return acc;
}

/// out[i] = sum_{k<i} in[k]; returns the total.
inline unsigned exclusive_scan(int n, const unsigned* in, unsigned* out, stream_t /*stream*/) {
  unsigned acc = 0;
  for (int i = 0; i < n; ++i) {
    const unsigned v = in[i];
    out[i] = acc;
    acc += v;
  }
  return acc;
}

#endif
#endif  // CRYOSPH_DEVICE_TU

}  // namespace cryosph
