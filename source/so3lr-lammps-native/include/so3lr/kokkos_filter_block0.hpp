#pragma once

#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <vector>

namespace so3lr {

struct FilterDeviceParameters {
  Kokkos::View<double *> rbf_weight_0;
  Kokkos::View<double *> rbf_bias_0;
  Kokkos::View<double *> rbf_weight_1;
  Kokkos::View<double *> rbf_bias_1;
  Kokkos::View<double *> ev_weight_0;
  Kokkos::View<double *> ev_bias_0;
  Kokkos::View<double *> ev_weight_1;
  Kokkos::View<double *> ev_bias_1;
};

struct FilterBlock0Results {
  std::vector<double> invariant_filter;
  std::vector<double> equivariant_filter;
};

struct FilterBlock0Benchmark {
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t workspace_bytes = 0;
  double total_milliseconds = 0.0;
  double milliseconds_per_iteration = 0.0;
  double checksum = 0.0;
};

class KokkosFilterBlock0 {
 public:
  explicit KokkosFilterBlock0(const NativeModel &model);

  FilterBlock0Results evaluate(const std::vector<double> &radial_basis,
                               const std::vector<double> &ev_invariants,
                               std::size_t edges) const;
  FilterBlock0Benchmark benchmark(std::size_t edges,
                                  std::size_t repetitions) const;
  std::size_t persistent_device_bytes() const { return persistent_device_bytes_; }
  bool aliases_verified() const { return aliases_verified_; }

 private:
  FilterDeviceParameters invariant_;
  FilterDeviceParameters equivariant_;
  std::size_t persistent_device_bytes_ = 0;
  bool aliases_verified_ = false;
};

}  // namespace so3lr

