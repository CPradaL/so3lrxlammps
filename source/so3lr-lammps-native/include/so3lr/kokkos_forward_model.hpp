#pragma once

#include "so3lr/kokkos_output_heads.hpp"
#include "so3lr/kokkos_transformer_stack012.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct NativeForwardResults {
  std::vector<double> final_inv;
  std::vector<double> final_ev;
  OutputHeadsResults heads;
};

struct NativeForwardBenchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double transformer_milliseconds = 0.0;
  double output_heads_milliseconds = 0.0;
  double phase_sum_milliseconds = 0.0;
  double integrated_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

struct NativeForwardDeviceWorkspace {
  NativeForwardDeviceWorkspace(std::size_t nodes, std::size_t edges);
  TransformerStack012DeviceWorkspace transformer;
  OutputHeadsDeviceWorkspace heads;
};

class KokkosForwardModel {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosForwardModel(const NativeModel &model);

  void launch_device(const Int64View &atomic_numbers,
                     const DoubleView &distances,
                     const DoubleView &sh_vectors,
                     const IndexView &senders,
                     const IndexView &receivers, double total_charge,
                     const NativeForwardDeviceWorkspace &workspace) const;
  NativeForwardResults evaluate(
      const std::vector<std::int64_t> &atomic_numbers,
      const std::vector<double> &distances,
      const std::vector<double> &sh_vectors,
      const std::vector<std::size_t> &senders,
      const std::vector<std::size_t> &receivers, double total_charge) const;
  NativeForwardBenchmark benchmark(std::size_t nodes, std::size_t edges,
                                   std::size_t repetitions) const;

  bool device_bridge_verified() const { return device_bridge_verified_; }
  std::size_t persistent_device_bytes() const;

 private:
  KokkosTransformerStack012 transformer_;
  KokkosOutputHeads heads_;
  bool device_bridge_verified_ = false;
};

}  // namespace so3lr
