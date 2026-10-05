#pragma once

#include "so3lr/kokkos_filter_block0.hpp"

#include <cublas_v2.h>

#include <cstddef>
#include <vector>

namespace so3lr {

class KokkosFilterBlock0Blas {
 public:
  explicit KokkosFilterBlock0Blas(const NativeModel &model);
  ~KokkosFilterBlock0Blas();

  KokkosFilterBlock0Blas(const KokkosFilterBlock0Blas &) = delete;
  KokkosFilterBlock0Blas &operator=(const KokkosFilterBlock0Blas &) = delete;

  FilterBlock0Results evaluate(const std::vector<double> &radial_basis,
                               const std::vector<double> &ev_invariants,
                               std::size_t edges) const;
  FilterBlock0Benchmark benchmark(std::size_t edges,
                                  std::size_t repetitions) const;
  std::size_t persistent_device_bytes() const { return persistent_device_bytes_; }
  bool aliases_verified() const { return aliases_verified_; }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }

 private:
  FilterDeviceParameters invariant_;
  FilterDeviceParameters equivariant_;
  cublasHandle_t handle_ = nullptr;
  std::size_t persistent_device_bytes_ = 0;
  bool aliases_verified_ = false;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr

