#pragma once

#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>
#include <cublas_v2.h>

#include <cstddef>
#include <vector>

namespace so3lr {

struct PostAttentionBlock0Results {
  std::vector<double> attention_residual_inv;
  std::vector<double> attention_residual_ev;
  std::vector<double> layer_norm_1;
  std::vector<double> post_mlp_inv;
  std::vector<double> interaction_ev_invariants;
  std::vector<double> interaction_transformed;
  std::vector<double> pre_layer_norm_2_inv;
  std::vector<double> final_inv;
  std::vector<double> final_ev;
};

struct PostAttentionBlock0Benchmark {
  std::size_t nodes = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double residual_norm1_milliseconds = 0.0;
  double residual_mlp1_milliseconds = 0.0;
  double interaction_milliseconds = 0.0;
  double residual_norm2_milliseconds = 0.0;
  double phase_sum_milliseconds = 0.0;
  double pipeline_milliseconds = 0.0;
  double nodes_per_second = 0.0;
  double checksum = 0.0;
};

struct PostAttentionBlock0DeviceWorkspace {
  using DoubleView = Kokkos::View<double *>;

  explicit PostAttentionBlock0DeviceWorkspace(
      std::size_t nodes, const ArchDims &dims = v1_arch_dims());
  PostAttentionBlock0DeviceWorkspace(std::size_t nodes, std::size_t inv_width,
                                     std::size_t ev_width,
                                     std::size_t degree_count);
  DoubleView attention_residual_inv, attention_residual_ev, layer_norm_1;
  DoubleView mlp_activation, mlp_hidden, mlp_output, post_mlp_inv;
  DoubleView interaction_ev_invariants, interaction_concatenated;
  DoubleView interaction_transformed, pre_layer_norm_2_inv;
  DoubleView final_inv, final_ev;
};

class KokkosPostAttentionBlock0 {
 public:
  explicit KokkosPostAttentionBlock0(const NativeModel &model,
                                     std::size_t block_index = 0);
  ~KokkosPostAttentionBlock0();
  KokkosPostAttentionBlock0(const KokkosPostAttentionBlock0 &) = delete;
  KokkosPostAttentionBlock0 &operator=(const KokkosPostAttentionBlock0 &) = delete;

  PostAttentionBlock0Results evaluate(
      const std::vector<double> &inv_features,
      const std::vector<double> &ev_features,
      const std::vector<double> &d_attention_inv,
      const std::vector<double> &d_attention_ev,
      std::size_t nodes) const;
  PostAttentionBlock0Benchmark benchmark(std::size_t nodes,
                                         std::size_t repetitions) const;
  void launch_device(
      const Kokkos::View<double *> &inv_features,
      const Kokkos::View<double *> &ev_features,
      const Kokkos::View<double *> &d_attention_inv,
      const Kokkos::View<double *> &d_attention_ev,
      const PostAttentionBlock0DeviceWorkspace &workspace) const;

  bool checkpoint_contract_verified() const {
    return checkpoint_contract_verified_;
  }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t block_index() const { return block_index_; }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

  NormMode norm_1_mode() const { return norm_1_mode_; }
  NormMode norm_2_mode() const { return norm_2_mode_; }
  double layer_norm_epsilon() const { return layer_norm_epsilon_; }
  const ArchDims &dims() const { return dims_; }

 private:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  ArchDims dims_;
  DoubleView layer_norm_1_weight_, layer_norm_1_bias_;
  DoubleView mlp_1_weight_1_, mlp_1_bias_1_;
  DoubleView mlp_1_weight_2_, mlp_1_bias_2_;
  DoubleView interaction_weight_, interaction_bias_;
  DoubleView interaction_cg_rep_;
  IndexView degree_repeats_, degree_offsets_;
  DoubleView layer_norm_2_weight_, layer_norm_2_bias_;
  cublasHandle_t handle_ = nullptr;
  double layer_norm_epsilon_ = 1.0e-6;
  NormMode norm_1_mode_ = NormMode::LayerAffine;
  NormMode norm_2_mode_ = NormMode::LayerAffine;
  std::size_t persistent_device_bytes_ = 0;
  std::size_t block_index_ = 0;
  bool checkpoint_contract_verified_ = false;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr
