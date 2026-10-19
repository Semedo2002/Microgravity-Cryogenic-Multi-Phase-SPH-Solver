#include "cryosph/core/telemetry.hpp"

namespace cryosph {

TelemetryStream::~TelemetryStream() { release(); }

void TelemetryStream::release() {
  if (pending_) {
    try {
      mem::event_synchronize(mode_ == TelemetryMode::ZeroCopy ? computed_ : landed_);
    } catch (...) {
    }
    pending_ = false;
  }
  mem::pinned_free(host_);
  host_ = nullptr;
  host_dev_ = nullptr;
  staging_.release();
  mem::destroy_event(computed_);
  mem::destroy_event(landed_);
  mem::destroy_stream(copy_stream_);
  computed_ = landed_ = nullptr;
  copy_stream_ = nullptr;
  capacity_ = 0;
}

void TelemetryStream::configure(std::size_t capacity, TelemetryMode mode) {
  release();
  mode_ = mode;
  capacity_ = capacity;
  if (capacity == 0) return;
  const std::size_t bytes = capacity * sizeof(TelemetryRecord);
  if (mode_ == TelemetryMode::ZeroCopy) {
    try {
      host_ = static_cast<TelemetryRecord*>(mem::pinned_alloc(bytes, /*mapped=*/true));
      host_dev_ = static_cast<TelemetryRecord*>(mem::mapped_device_pointer(host_));
    } catch (const std::exception&) {
      // Platform without mapped pinned memory (e.g. some WSL2 set-ups).
      mem::pinned_free(host_);
      host_ = nullptr;
      host_dev_ = nullptr;
      mode_ = TelemetryMode::AsyncCopy;
    }
  }
  if (mode_ == TelemetryMode::AsyncCopy) {
    host_ = static_cast<TelemetryRecord*>(mem::pinned_alloc(bytes, /*mapped=*/false));
    if constexpr (kCudaBackend) {
      staging_.resize(capacity);
      copy_stream_ = mem::create_stream(/*ordered=*/false);  // only touches staging_ / host_
    }
  }
  computed_ = mem::create_event();
  landed_ = mem::create_event();
}

bool TelemetryStream::request(ParticleSystem& ps, stream_t compute, double time, long long step) {
  if (pending_ || ps.size() > capacity_) return false;
  const int n = static_cast<int>(ps.size());
  TelemetryRecord* dst = nullptr;
  if (mode_ == TelemetryMode::ZeroCopy) {
    dst = host_dev_;
  } else {
    dst = kCudaBackend ? staging_.data() : host_;
  }
  const Vec3* x = ps.x.data();
  const Vec3* v = ps.v.data();
  const real* T = ps.T.data();
  const real* p = ps.p.data();
  const std::uint8_t* ph = ps.phase.data();
  parallel_for(
      n,
      [=] CRYOSPH_HD(int i) {
        const Vec3 xi = x[i];
        TelemetryRecord r;
        r.x = static_cast<float>(xi.x);
        r.y = static_cast<float>(xi.y);
        r.z = static_cast<float>(xi.z);
        r.T = static_cast<float>(T[i]);
        r.p = static_cast<float>(p[i]);
        r.speed = static_cast<float>(norm(v[i]));
        r.phase = ph[i];
        dst[i] = r;
      },
      compute);
  mem::record_event(computed_, compute);
  if (mode_ == TelemetryMode::AsyncCopy && kCudaBackend) {
    mem::stream_wait_event(copy_stream_, computed_);
    mem::copy_d2h_async(host_, staging_.data(), static_cast<std::size_t>(n) * sizeof(TelemetryRecord), copy_stream_);
    mem::record_event(landed_, copy_stream_);
  }
  pending_ = true;
  time_ = time;
  step_ = step;
  n_ = n;
  return true;
}

bool TelemetryStream::ready() const {
  if (!pending_) return true;
  return mem::event_ready(mode_ == TelemetryMode::ZeroCopy || !kCudaBackend ? computed_ : landed_);
}

bool TelemetryStream::fetch(TelemetryFrame& out) {
  if (!pending_) return false;
  mem::event_synchronize(mode_ == TelemetryMode::ZeroCopy || !kCudaBackend ? computed_ : landed_);
  pending_ = false;
  out.time = time_;
  out.step = step_;
  out.n = n_;
  const auto n = static_cast<std::size_t>(n_);
  out.position.resize(3 * n);
  out.temperature.resize(n);
  out.pressure.resize(n);
  out.speed.resize(n);
  out.phase.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const TelemetryRecord& r = host_[i];
    out.position[3 * i] = r.x;
    out.position[3 * i + 1] = r.y;
    out.position[3 * i + 2] = r.z;
    out.temperature[i] = r.T;
    out.pressure[i] = r.p;
    out.speed[i] = r.speed;
    out.phase[i] = static_cast<std::uint8_t>(r.phase);
  }
  return true;
}

}  // namespace cryosph
