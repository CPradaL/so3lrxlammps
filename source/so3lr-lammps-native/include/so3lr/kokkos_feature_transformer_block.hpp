#pragma once

#include "so3lr/kokkos_integrated_block0.hpp"
#include "so3lr/kokkos_post_attention_block0.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <vector>

namespace so3lr {

struct FeatureTransformerResults {
  std::vector<double> edge_ev_invariants;
  std::vector<double> attention_update_inv;
  std::vector<double> attention_update_ev;
  std::vector<double> final_inv;
  std::vector<double> final_ev;
};

struct FeatureTransformerBenchmark {
  std::size_t block_index = 0;
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double complete_block_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

struct FeatureTransformerDeviceWorkspace {
  FeatureTransformerDeviceWorkspace(std::size_t nodes, std::size_t edges);
  IntegratedBlock0DeviceWorkspace attention;
  PostAttentionBlock0DeviceWorkspace post_attention;
};

class KokkosFeatureTransformerBlock {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  KokkosFeatureTransformerBlock(const NativeModel &model,
                                std::size_t block_index);

  void launch_device(
      const DoubleView &inv_features, const DoubleView &ev_features,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const FeatureTransformerDeviceWorkspace &workspace) const;
  FeatureTransformerResults evaluate(
      const std::vector<double> &inv_features,
      const std::vector<double> &ev_features,
      const std::vector<double> &distances,
      const std::vector<double> &sh_vectors,
      const std::vector<std::size_t> &senders,
      const std::vector<std::size_t> &receivers) const;
  FeatureTransformerBenchmark benchmark(std::size_t nodes,
                                         std::size_t edges,
                                         std::size_t repetitions) const;

  std::size_t block_index() const { return block_index_; }
  bool device_contract_verified() const { return device_contract_verified_; }
  std::size_t persistent_device_bytes() const;

 private:
  KokkosIntegratedBlock0 attention_;
  KokkosPostAttentionBlock0 post_attention_;
  std::size_t block_index_ = 0;
  bool device_contract_verified_ = false;
};

}  // namespace so3lr
