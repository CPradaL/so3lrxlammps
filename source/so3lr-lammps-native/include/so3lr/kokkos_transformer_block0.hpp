#pragma once

#include "so3lr/kokkos_integrated_block0.hpp"
#include "so3lr/kokkos_post_attention_block0.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct TransformerBlock0Results {
  std::vector<double> embedding;
  std::vector<double> attention_update_inv;
  std::vector<double> attention_update_ev;
  std::vector<double> final_inv;
  std::vector<double> final_ev;
};

struct TransformerBlock0Benchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double attention_pipeline_milliseconds = 0.0;
  double post_attention_milliseconds = 0.0;
  double phase_sum_milliseconds = 0.0;
  double complete_block_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

struct TransformerBlock0DeviceWorkspace {
  using DoubleView = Kokkos::View<double *>;

  TransformerBlock0DeviceWorkspace(std::size_t nodes, std::size_t edges);
  DoubleView initial_ev_features;
  DoubleView initial_ev_invariants;
  IntegratedBlock0DeviceWorkspace attention;
  PostAttentionBlock0DeviceWorkspace post_attention;
};

class KokkosTransformerBlock0 {
 public:
  explicit KokkosTransformerBlock0(const NativeModel &model);

  TransformerBlock0Results evaluate(
      const std::vector<std::int64_t> &atomic_numbers,
      const std::vector<double> &distances,
      const std::vector<double> &sh_vectors,
      const std::vector<std::size_t> &senders,
      const std::vector<std::size_t> &receivers) const;
  TransformerBlock0Benchmark benchmark(std::size_t nodes, std::size_t edges,
                                       std::size_t repetitions) const;

  bool device_contract_verified() const { return device_contract_verified_; }
  std::size_t persistent_device_bytes() const;

  // Public device-view API for native stack composition.
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  void launch_device(const Int64View &atomic_numbers,
                     const DoubleView &distances,
                     const DoubleView &sh_vectors,
                     const IndexView &senders,
                     const IndexView &receivers,
                     const TransformerBlock0DeviceWorkspace &workspace) const;

 private:
  KokkosIntegratedBlock0 attention_;
  KokkosPostAttentionBlock0 post_attention_;
  bool device_contract_verified_ = false;
};

}  // namespace so3lr
