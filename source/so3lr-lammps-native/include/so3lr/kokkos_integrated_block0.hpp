#pragma once

#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>
#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct IntegratedBlock0Results {
  std::vector<double> embedding;
  std::vector<double> radial_basis;
  std::vector<double> cutoff;
  std::vector<double> invariant_update;
  std::vector<double> equivariant_update;
};

struct IntegratedBlock0Benchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double input_milliseconds = 0.0;
  double filter_milliseconds = 0.0;
  double qkv_milliseconds = 0.0;
  double attention_milliseconds = 0.0;
  double phase_sum_milliseconds = 0.0;
  double pipeline_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

struct IntegratedFilterParameters {
  Kokkos::View<double *> rbf_weight_0, rbf_bias_0, rbf_weight_1, rbf_bias_1;
  Kokkos::View<double *> ev_weight_0, ev_bias_0, ev_weight_1, ev_bias_1;
};

struct IntegratedBlock0DeviceWorkspace {
  using DoubleView = Kokkos::View<double *>;

  IntegratedBlock0DeviceWorkspace(std::size_t nodes, std::size_t edges,
                                  const ArchDims &dims = v1_arch_dims());
  DoubleView embedding, cutoff, radial_basis, edge_ev_invariants;
  DoubleView rbf_hidden, ev_hidden, filter_inv, filter_ev;
  std::array<DoubleView, 5> qkv;
  DoubleView invariant_update, equivariant_update;
  // 1/rms of each (node, head) row of q_inv, k_inv, q_ev, k_ev, kept for the
  // QK-normalisation adjoint. Written only when the model enables qk_norm.
  DoubleView qk_inverse_rms;
};

class KokkosIntegratedBlock0 {
 public:
  explicit KokkosIntegratedBlock0(const NativeModel &model,
                                  std::size_t block_index = 0);
  ~KokkosIntegratedBlock0();
  KokkosIntegratedBlock0(const KokkosIntegratedBlock0 &) = delete;
  KokkosIntegratedBlock0 &operator=(const KokkosIntegratedBlock0 &) = delete;

  IntegratedBlock0Results evaluate(
      const std::vector<std::int64_t> &atomic_numbers,
      const std::vector<double> &distances,
      const std::vector<double> &ev_invariants,
      const std::vector<double> &sh_vectors,
      const std::vector<std::size_t> &senders,
      const std::vector<std::size_t> &receivers) const;
  IntegratedBlock0Benchmark benchmark(std::size_t nodes, std::size_t edges,
                                      std::size_t repetitions) const;
  void launch_device(
      const Kokkos::View<std::int64_t *> &atomic_numbers,
      const Kokkos::View<double *> &distances,
      const Kokkos::View<double *> &ev_invariants,
      const Kokkos::View<double *> &sh_vectors,
      const Kokkos::View<std::size_t *> &senders,
      const Kokkos::View<std::size_t *> &receivers,
      const IntegratedBlock0DeviceWorkspace &workspace) const;
  void launch_features_device(
      const Kokkos::View<double *> &inv_features,
      const Kokkos::View<double *> &ev_features,
      const Kokkos::View<double *> &distances,
      const Kokkos::View<double *> &sh_vectors,
      const Kokkos::View<std::size_t *> &senders,
      const Kokkos::View<std::size_t *> &receivers,
      const IntegratedBlock0DeviceWorkspace &workspace) const;

  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t block_index() const { return block_index_; }

  bool qk_norm() const { return qk_norm_; }
  const ArchDims &dims() const { return dims_; }

 private:
  ArchDims dims_;
  bool qk_norm_ = false;
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  DoubleView embedding_weight_, bernstein_b_, attention_cg_rep_;
  Int64View bernstein_k_, bernstein_k_reverse_;
  IntegratedFilterParameters filter_inv_, filter_ev_;
  std::array<DoubleView, 5> qkv_weights_;
  IndexView degree_repeats_, degree_offsets_;
  cublasHandle_t handle_ = nullptr;
  double gamma_ = 0.0;
  double cutoff_radius_ = 0.0;
  double attention_norm_inv_ = 0.0;
  double attention_norm_ev_ = 0.0;
  std::size_t persistent_device_bytes_ = 0;
  std::size_t block_index_ = 0;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr
