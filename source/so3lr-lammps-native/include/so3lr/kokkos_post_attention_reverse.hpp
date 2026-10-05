#pragma once

#include "so3lr/kokkos_post_attention_block0.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>
#include <cublas_v2.h>

#include <cstddef>

namespace so3lr {

struct PostAttentionReverseDeviceWorkspace {
  using DoubleView = Kokkos::View<double *>;

  explicit PostAttentionReverseDeviceWorkspace(
      std::size_t nodes, const ArchDims &dims = v1_arch_dims());
  PostAttentionReverseDeviceWorkspace(std::size_t nodes, std::size_t inv_width,
                                      std::size_t ev_width,
                                      std::size_t interaction_width);
  DoubleView hidden_preact;
  DoubleView grad_pre_norm2, grad_post_mlp;
  DoubleView grad_transformed, grad_concatenated;
  DoubleView grad_attention_ev;
  DoubleView grad_hidden, grad_hidden_preact, grad_activation;
  DoubleView grad_norm1, grad_attention_inv;
};

struct PostAttentionReverseBenchmark {
  std::size_t block_index = 0;
  std::size_t nodes = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double reverse_milliseconds = 0.0;
  double forward_reverse_milliseconds = 0.0;
  double nodes_per_second = 0.0;
  double checksum = 0.0;
};

class KokkosPostAttentionReverseBlock {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosPostAttentionReverseBlock(const NativeModel &model,
                                            std::size_t block_index = 2);
  ~KokkosPostAttentionReverseBlock();
  KokkosPostAttentionReverseBlock(
      const KokkosPostAttentionReverseBlock &) = delete;
  KokkosPostAttentionReverseBlock &operator=(
      const KokkosPostAttentionReverseBlock &) = delete;

  void launch_forward_device(
      const DoubleView &inv_features, const DoubleView &ev_features,
      const DoubleView &d_attention_inv, const DoubleView &d_attention_ev,
      const PostAttentionBlock0DeviceWorkspace &workspace) const;
  void launch_reverse_device(
      const DoubleView &grad_final_inv, const DoubleView &grad_final_ev,
      const PostAttentionBlock0DeviceWorkspace &forward,
      const PostAttentionReverseDeviceWorkspace &reverse) const;
  PostAttentionReverseBenchmark benchmark(std::size_t nodes,
                                           std::size_t repetitions) const;

  bool checkpoint_contract_verified() const {
    return checkpoint_contract_verified_;
  }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t block_index() const { return block_index_; }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  ArchDims dims_;
  KokkosPostAttentionBlock0 forward_;
  DoubleView layer_norm_1_weight_, layer_norm_2_weight_;
  DoubleView mlp_1_weight_1_, mlp_1_bias_1_, mlp_1_weight_2_;
  DoubleView interaction_weight_, interaction_cg_rep_;
  IndexView degree_offsets_;
  cublasHandle_t handle_ = nullptr;
  double layer_norm_epsilon_ = 1.0e-6;
  std::size_t persistent_device_bytes_ = 0;
  std::size_t block_index_ = 0;
  bool checkpoint_contract_verified_ = false;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr
