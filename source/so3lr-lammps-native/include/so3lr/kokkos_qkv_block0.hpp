#pragma once

#include "so3lr/kokkos_filter_block0_blas.hpp"

#include <cublas_v2.h>

#include <cstddef>
#include <vector>

namespace so3lr {

struct QkvBlock0Results {
  std::vector<double> q_inv_nodes;
  std::vector<double> k_inv_nodes;
  std::vector<double> v_inv_nodes;
  std::vector<double> q_ev_nodes;
  std::vector<double> k_ev_nodes;
  std::vector<double> q_inv_edges;
  std::vector<double> k_inv_edges;
  std::vector<double> v_inv_edges;
  std::vector<double> q_ev_edges;
  std::vector<double> k_ev_edges;
};

struct QkvBlock0Benchmark {
  std::size_t nodes = 0;
  std::size_t repetitions = 0;
  std::size_t workspace_bytes = 0;
  double total_milliseconds = 0.0;
  double milliseconds_per_iteration = 0.0;
  double checksum = 0.0;
};

class KokkosQkvBlock0 {
 public:
  explicit KokkosQkvBlock0(const NativeModel &model);
  ~KokkosQkvBlock0();

  KokkosQkvBlock0(const KokkosQkvBlock0 &) = delete;
  KokkosQkvBlock0 &operator=(const KokkosQkvBlock0 &) = delete;

  QkvBlock0Results evaluate(const std::vector<double> &features,
                            const std::vector<std::size_t> &senders,
                            const std::vector<std::size_t> &receivers,
                            std::size_t nodes) const;
  QkvBlock0Benchmark benchmark(std::size_t nodes,
                               std::size_t repetitions) const;
  QkvBlock0Benchmark benchmark_baseline(std::size_t nodes,
                                        std::size_t repetitions) const;

  std::size_t persistent_device_bytes() const { return persistent_device_bytes_; }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  bool identity_qk_activation() const { return identity_qk_activation_; }

 private:
  Kokkos::View<double *> w_q_inv_;
  Kokkos::View<double *> w_k_inv_;
  Kokkos::View<double *> w_v_inv_;
  Kokkos::View<double *> w_q_ev_;
  Kokkos::View<double *> w_k_ev_;
  cublasHandle_t handle_ = nullptr;
  std::size_t persistent_device_bytes_ = 0;
  bool shared_kokkos_stream_ = false;
  bool identity_qk_activation_ = false;
};

}  // namespace so3lr

