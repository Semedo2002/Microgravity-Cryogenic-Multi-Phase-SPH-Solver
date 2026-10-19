// Non-blocking frame capture for live visualisation: async DMA through a
// pinned buffer, or zero-copy into mapped host memory.
#pragma once

#include <cstdint>
#include <vector>

#include "cryosph/core/cuda_utils.cuh"
#include "cryosph/core/particle_system.hpp"

namespace cryosph {

enum class TelemetryMode : int { AsyncCopy = 0, ZeroCopy = 1 };

/// One captured frame in host memory (structure of arrays, float precision).
struct TelemetryFrame {
  double time = 0;
  long long step = 0;
  int n = 0;
  std::vector<float> position;  ///< 3n
  std::vector<float> temperature;  ///< n
  std::vector<float> pressure;  ///< n
  std::vector<float> speed;  ///< n
  std::vector<std::uint8_t> phase; ///< n
};

/// Packed per-particle record written by the capture kernel.
struct TelemetryRecord {
  float x, y, z, T, p, speed;
  std::uint32_t phase;
};

class TelemetryStream {
 public:
  TelemetryStream() = default;
  ~TelemetryStream();
  TelemetryStream(const TelemetryStream&) = delete;
  TelemetryStream& operator=(const TelemetryStream&) = delete;

  /// (Re)allocates buffers for up to `capacity` particles.
  void configure(std::size_t capacity, TelemetryMode mode);
  TelemetryMode mode() const noexcept { return mode_; }

  /// Enqueues a capture of the current particle state behind all work already
  /// submitted to `compute`.  Returns false if a previous frame is still
  /// pending (call fetch() first) or the buffers are too small.
  bool request(ParticleSystem& ps, stream_t compute, double time, long long step);
  /// True when no frame is in flight or the in-flight frame has landed.
  bool ready() const;
  bool pending() const noexcept { return pending_; }
  /// Blocks until the pending frame has landed and converts it into `out`.
  /// Returns false when nothing was requested.
  bool fetch(TelemetryFrame& out);

 private:
  void release();

  TelemetryMode mode_ = TelemetryMode::AsyncCopy;
  std::size_t capacity_ = 0;
  TelemetryRecord* host_ = nullptr;  ///< pinned host buffer
  TelemetryRecord* host_dev_ = nullptr;  ///< device alias of host_ (ZeroCopy)
  DeviceArray<TelemetryRecord> staging_;  ///< device staging (AsyncCopy)
  stream_t copy_stream_ = nullptr;
  event_t computed_ = nullptr;
  event_t landed_ = nullptr;
  bool pending_ = false;
  double time_ = 0;
  long long step_ = 0;
  int n_ = 0;
};

}  // namespace cryosph
