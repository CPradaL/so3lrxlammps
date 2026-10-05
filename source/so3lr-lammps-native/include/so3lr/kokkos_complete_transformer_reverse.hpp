#pragma once

#include "so3lr/kokkos_block2_attention_input_reverse.hpp"
#include "so3lr/kokkos_post_attention_reverse.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace so3lr {

// Forward checkpoints remain block-specific.  The overload taking `shared`
// aliases reverse-only scratch while retaining geometry-gradient outputs.
struct CompleteTransformerReverseWorkspace {
  CompleteTransformerReverseWorkspace(std::size_t nodes, std::size_t edges,
                                      const ArchDims &dims = v1_arch_dims());
  CompleteTransformerReverseWorkspace(
      std::size_t nodes, std::size_t edges,
      const CompleteTransformerReverseWorkspace &shared,
      const ArchDims &dims = v1_arch_dims());
  Block2AttentionInputReverseWorkspace attention;
  PostAttentionBlock0DeviceWorkspace post_forward;
  PostAttentionReverseDeviceWorkspace post_reverse;
};

struct CompleteTransformerReverseBenchmark {
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

class KokkosCompleteTransformerReverseBlock {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosCompleteTransformerReverseBlock(
      const NativeModel &model, std::size_t block_index);

  void launch_forward_device(
      const DoubleView &inv_features, const DoubleView &ev_features,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const CompleteTransformerReverseWorkspace &workspace) const;
  void launch_reverse_device(
      const DoubleView &inv_features, const DoubleView &ev_features,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const DoubleView &grad_final_inv, const DoubleView &grad_final_ev,
      const CompleteTransformerReverseWorkspace &workspace) const;
  CompleteTransformerReverseBenchmark benchmark(
      std::size_t nodes, std::size_t edges,
      std::size_t repetitions) const;

  const DoubleView &final_inv(
      const CompleteTransformerReverseWorkspace &workspace) const {
    return workspace.post_forward.final_inv;
  }
  const DoubleView &final_ev(
      const CompleteTransformerReverseWorkspace &workspace) const {
    return workspace.post_forward.final_ev;
  }
  const DoubleView &grad_inv(
      const CompleteTransformerReverseWorkspace &workspace) const {
    return workspace.attention.grad_inv_features;
  }
  const DoubleView &grad_ev(
      const CompleteTransformerReverseWorkspace &workspace) const {
    return workspace.attention.grad_ev_features;
  }
  const DoubleView &grad_distances(
      const CompleteTransformerReverseWorkspace &workspace) const {
    return workspace.attention.grad_distances;
  }
  const DoubleView &grad_sh(
      const CompleteTransformerReverseWorkspace &workspace) const {
    return workspace.attention.attention_reverse.grad_sh;
  }

  std::size_t block_index() const { return block_index_; }
  bool checkpoint_contract_verified() const {
    return checkpoint_contract_verified_;
  }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  KokkosBlock2AttentionInputReverse attention_;
  KokkosPostAttentionReverseBlock post_;
  std::size_t block_index_ = 0;
  std::size_t persistent_device_bytes_ = 0;
  bool checkpoint_contract_verified_ = false;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr
