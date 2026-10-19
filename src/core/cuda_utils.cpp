// Memory, stream and event helpers for both backends.
#include "cryosph/core/cuda_utils.cuh"

#include <cstdlib>
#include <new>
#include <thread>

namespace cryosph::mem {

#if defined(CRYOSPH_WITH_CUDA)

void* device_alloc(std::size_t bytes) {
  void* p = nullptr;
  if (bytes == 0) return nullptr;
  CRYOSPH_CUDA_CHECK(cudaMalloc(&p, bytes));
  return p;
}
void device_free(void* ptr) noexcept { if (ptr) cudaFree(ptr); }
void copy_h2d(void* dst, const void* src, std::size_t bytes) {
  CRYOSPH_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void copy_d2h(void* dst, const void* src, std::size_t bytes) {
  CRYOSPH_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
void copy_d2d(void* dst, const void* src, std::size_t bytes) {
  CRYOSPH_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}
void copy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t stream) {
  CRYOSPH_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}
void memset_device(void* ptr, int value, std::size_t bytes) {
  CRYOSPH_CUDA_CHECK(cudaMemset(ptr, value, bytes));
}

void* pinned_alloc(std::size_t bytes, bool mapped) {
  void* p = nullptr;
  if (bytes == 0) return nullptr;
  CRYOSPH_CUDA_CHECK(cudaHostAlloc(&p, bytes, mapped ? cudaHostAllocMapped : cudaHostAllocDefault));
  return p;
}
void pinned_free(void* ptr) noexcept { if (ptr) cudaFreeHost(ptr); }
void* mapped_device_pointer(void* host_ptr) {
  void* d = nullptr;
  CRYOSPH_CUDA_CHECK(cudaHostGetDevicePointer(&d, host_ptr, 0));
  return d;
}

stream_t create_stream(bool ordered) {
  cudaStream_t s = nullptr;
  CRYOSPH_CUDA_CHECK(cudaStreamCreateWithFlags(&s, ordered ? cudaStreamDefault : cudaStreamNonBlocking));
  return s;
}
void destroy_stream(stream_t s) noexcept { if (s) cudaStreamDestroy(s); }
void synchronize(stream_t s) { CRYOSPH_CUDA_CHECK(cudaStreamSynchronize(s)); }
void device_synchronize() { CRYOSPH_CUDA_CHECK(cudaDeviceSynchronize()); }

event_t create_event() {
  cudaEvent_t e = nullptr;
  CRYOSPH_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  return e;
}
void destroy_event(event_t e) noexcept { if (e) cudaEventDestroy(e); }
void record_event(event_t e, stream_t s) { CRYOSPH_CUDA_CHECK(cudaEventRecord(e, s)); }
void stream_wait_event(stream_t s, event_t e) { CRYOSPH_CUDA_CHECK(cudaStreamWaitEvent(s, e, 0)); }
bool event_ready(event_t e) {
  const cudaError_t st = cudaEventQuery(e);
  if (st == cudaSuccess) return true;
  if (st == cudaErrorNotReady) return false;
  CRYOSPH_CUDA_CHECK(st);
  return false;
}
void event_synchronize(event_t e) { CRYOSPH_CUDA_CHECK(cudaEventSynchronize(e)); }

std::string device_name() {
  int dev = 0;
  CRYOSPH_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CRYOSPH_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ")";
}
int device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

#else  // ---------------------------------------------------- CPU backend ----

namespace {
constexpr std::size_t kAlign = 64;  // cache-line aligned SoA streams
}

void* device_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  return ::operator new(bytes, std::align_val_t{kAlign});
}
void device_free(void* ptr) noexcept {
  if (ptr) ::operator delete(ptr, std::align_val_t{kAlign});
}
void copy_h2d(void* dst, const void* src, std::size_t bytes) { std::memcpy(dst, src, bytes); }
void copy_d2h(void* dst, const void* src, std::size_t bytes) { std::memcpy(dst, src, bytes); }
void copy_d2d(void* dst, const void* src, std::size_t bytes) { std::memmove(dst, src, bytes); }
void copy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t) { std::memcpy(dst, src, bytes); }
void memset_device(void* ptr, int value, std::size_t bytes) { std::memset(ptr, value, bytes); }

void* pinned_alloc(std::size_t bytes, bool) { return device_alloc(bytes); }
void pinned_free(void* ptr) noexcept { device_free(ptr); }
void* mapped_device_pointer(void* host_ptr) { return host_ptr; }

stream_t create_stream(bool) { return nullptr; }
void destroy_stream(stream_t) noexcept {}
void synchronize(stream_t) {}
void device_synchronize() {}

event_t create_event() { return nullptr; }
void destroy_event(event_t) noexcept {}
void record_event(event_t, stream_t) {}
void stream_wait_event(stream_t, event_t) {}
bool event_ready(event_t) { return true; }
void event_synchronize(event_t) {}

std::string device_name() {
#if defined(_OPENMP)
  return "CPU (OpenMP, " + std::to_string(omp_get_max_threads()) + " threads)";
#else
  return "CPU (serial)";
#endif
}
int device_count() { return 0; }

#endif

}  // namespace cryosph::mem
