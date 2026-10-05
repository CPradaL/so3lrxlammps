#pragma once

#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>

namespace so3lr {

struct AttentionScatterReverseDeviceWorkspace {
  using DoubleView = Kokkos::View<double *>;

  AttentionScatterReverseDeviceWorkspace(
      std::size_t nodes, std::size_t edges,
      const ArchDims &dims = v1_arch_dims());
  AttentionScatterReverseDeviceWorkspace(std::size_t nodes, std::size_t edges,
                                         std::size_t feature_width,
                                         std::size_t equivariant_width,
                                         std::size_t heads);
  AttentionScatterReverseDeviceWorkspace(
      std::size_t nodes, std::size_t edges,
      const AttentionScatterReverseDeviceWorkspace &shared);
  DoubleView invariant_update, equivariant_update;
  std::array<DoubleView, 5> grad_qkv;
  DoubleView grad_filter_inv, grad_filter_ev;
  DoubleView grad_sh, grad_cutoff, grad_cutoff_heads;
};

struct AttentionScatterReverseBenchmark {
  std::size_t block_index = 0;
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double reverse_milliseconds = 0.0;
  double forward_reverse_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

class KokkosAttentionScatterReverse {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosAttentionScatterReverse(const NativeModel &model,
                                          std::size_t block_index = 2);

  void launch_forward_device(
      const std::array<DoubleView, 5> &qkv,
      const DoubleView &filter_inv, const DoubleView &filter_ev,
      const DoubleView &sh, const DoubleView &cutoff,
      const IndexView &senders, const IndexView &receivers,
      const AttentionScatterReverseDeviceWorkspace &workspace) const;
  void launch_reverse_device(
      const std::array<DoubleView, 5> &qkv,
      const DoubleView &filter_inv, const DoubleView &filter_ev,
      const DoubleView &sh, const DoubleView &cutoff,
      const IndexView &senders, const IndexView &receivers,
      const DoubleView &grad_invariant_update,
      const DoubleView &grad_equivariant_update,
      const AttentionScatterReverseDeviceWorkspace &workspace) const;
  AttentionScatterReverseBenchmark benchmark(std::size_t nodes,
                                               std::size_t edges,
                                               std::size_t repetitions) const;

  std::size_t block_index() const { return block_index_; }
  double attention_norm_inv() const { return attention_norm_inv_; }
  double attention_norm_ev() const { return attention_norm_ev_; }
  bool checkpoint_contract_verified() const {
    return checkpoint_contract_verified_;
  }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  ArchDims dims_;
  IndexView degree_repeats_, degree_offsets_;
  double attention_norm_inv_ = 0.0;
  double attention_norm_ev_ = 0.0;
  std::size_t block_index_ = 0;
  std::size_t persistent_device_bytes_ = 0;
  bool checkpoint_contract_verified_ = false;
};

}  // namespace so3lr
