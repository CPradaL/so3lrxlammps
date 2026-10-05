#pragma once

#include "so3lr/kokkos_qkv_block0.hpp"

#include <cstddef>
#include <vector>

namespace so3lr {

struct AttentionScatterBlock0Results {
  std::vector<double> invariant_update;
  std::vector<double> equivariant_update;
  std::vector<double> alpha_invariant;
  std::vector<double> alpha_equivariant;
};

struct AttentionScatterBlock0Benchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t workspace_bytes = 0;
  std::size_t avoided_edge_intermediate_bytes = 0;
  double total_milliseconds = 0.0;
  double milliseconds_per_iteration = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

class KokkosAttentionScatterBlock0 {
 public:
  explicit KokkosAttentionScatterBlock0(const NativeModel &model);

  AttentionScatterBlock0Results evaluate(
      const std::vector<double> &q_inv_nodes,
      const std::vector<double> &k_inv_nodes,
      const std::vector<double> &v_inv_nodes,
      const std::vector<double> &q_ev_nodes,
      const std::vector<double> &k_ev_nodes,
      const std::vector<double> &filter_inv,
      const std::vector<double> &filter_ev,
      const std::vector<double> &sh_vectors,
      const std::vector<double> &cutoffs,
      const std::vector<std::size_t> &senders,
      const std::vector<std::size_t> &receivers,
      std::size_t nodes) const;

  AttentionScatterBlock0Benchmark benchmark(std::size_t nodes,
                                             std::size_t edges,
                                             std::size_t repetitions) const;

  double attention_norm_inv() const { return attention_norm_inv_; }
  double attention_norm_ev() const { return attention_norm_ev_; }
  std::size_t persistent_device_bytes() const { return persistent_device_bytes_; }
  bool degree_contract_verified() const { return degree_contract_verified_; }

 private:
  Kokkos::View<std::size_t *> degree_repeats_;
  Kokkos::View<std::size_t *> degree_offsets_;
  double attention_norm_inv_ = 0.0;
  double attention_norm_ev_ = 0.0;
  std::size_t persistent_device_bytes_ = 0;
  bool degree_contract_verified_ = false;
};

}  // namespace so3lr

