#pragma once

#include "so3lr/kokkos_attention_scatter_reverse.hpp"
#include "so3lr/kokkos_integrated_block0.hpp"

#include <Kokkos_Core.hpp>
#include <cublas_v2.h>

#include <array>
#include <cstddef>

namespace so3lr {

struct Block2AttentionInputReverseWorkspace {
  using DoubleView = Kokkos::View<double *>;

  Block2AttentionInputReverseWorkspace(
      std::size_t nodes, std::size_t edges,
      const ArchDims &dims = v1_arch_dims());
  Block2AttentionInputReverseWorkspace(
      std::size_t nodes, std::size_t edges,
      const Block2AttentionInputReverseWorkspace &shared,
      const ArchDims &dims = v1_arch_dims());
  IntegratedBlock0DeviceWorkspace forward;
  AttentionScatterReverseDeviceWorkspace attention_reverse;
  DoubleView grad_inv_features, grad_ev_features;
  DoubleView grad_distances, grad_radial_basis, grad_edge_ev_invariants;
  DoubleView filter_preactivation, filter_hidden_gradient;
};

struct Block2AttentionInputReverseBenchmark {
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

class KokkosBlock2AttentionInputReverse {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosBlock2AttentionInputReverse(
      const NativeModel &model, std::size_t block_index = 2);
  ~KokkosBlock2AttentionInputReverse();
  KokkosBlock2AttentionInputReverse(
      const KokkosBlock2AttentionInputReverse &) = delete;
  KokkosBlock2AttentionInputReverse &operator=(
      const KokkosBlock2AttentionInputReverse &) = delete;

  void launch_forward_device(
      const DoubleView &inv_features, const DoubleView &ev_features,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const Block2AttentionInputReverseWorkspace &workspace) const;
  void launch_reverse_device(
      const DoubleView &inv_features, const DoubleView &ev_features,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const DoubleView &grad_attention_inv,
      const DoubleView &grad_attention_ev,
      const Block2AttentionInputReverseWorkspace &workspace) const;
  Block2AttentionInputReverseBenchmark benchmark(
      std::size_t nodes, std::size_t edges,
      std::size_t repetitions) const;

  const DoubleView &grad_sh(
      const Block2AttentionInputReverseWorkspace &workspace) const {
    return workspace.attention_reverse.grad_sh;
  }
  bool checkpoint_contract_verified() const {
    return checkpoint_contract_verified_;
  }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t block_index() const { return block_index_; }
  const ArchDims &dims() const { return dims_; }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  using Int64View = Kokkos::View<std::int64_t *>;

  ArchDims dims_;
  bool qk_norm_ = false;

  KokkosIntegratedBlock0 forward_;
  KokkosAttentionScatterReverse attention_reverse_;
  IntegratedFilterParameters filter_inv_, filter_ev_;
  std::array<DoubleView, 5> qkv_weights_;
  DoubleView attention_cg_rep_, bernstein_b_;
  Int64View bernstein_k_, bernstein_k_reverse_;
  IndexView degree_repeats_, degree_offsets_;
  cublasHandle_t handle_ = nullptr;
  double gamma_ = 0.0;
  double cutoff_radius_ = 0.0;
  std::size_t block_index_ = 2;
  std::size_t persistent_device_bytes_ = 0;
  bool shared_kokkos_stream_ = false;
  bool checkpoint_contract_verified_ = false;
};

}  // namespace so3lr
