#pragma once

#include "so3lr/kokkos_feature_transformer_block.hpp"
#include "so3lr/kokkos_transformer_block0.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct TransformerStack01Results {
  std::vector<double> block0_final_inv;
  std::vector<double> block0_final_ev;
  std::vector<double> block1_final_inv;
  std::vector<double> block1_final_ev;
};

struct TransformerStack01Benchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double block0_milliseconds = 0.0;
  double block1_milliseconds = 0.0;
  double phase_sum_milliseconds = 0.0;
  double stack_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

struct TransformerStack01DeviceWorkspace {
  TransformerStack01DeviceWorkspace(std::size_t nodes, std::size_t edges);
  TransformerBlock0DeviceWorkspace block0;
  FeatureTransformerDeviceWorkspace block1;
};

class KokkosTransformerStack01 {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosTransformerStack01(const NativeModel &model);

  void launch_device(
      const Int64View &atomic_numbers, const DoubleView &distances,
      const DoubleView &sh_vectors, const IndexView &senders,
      const IndexView &receivers,
      const TransformerStack01DeviceWorkspace &workspace) const;
  TransformerStack01Results evaluate(
      const std::vector<std::int64_t> &atomic_numbers,
      const std::vector<double> &distances,
      const std::vector<double> &sh_vectors,
      const std::vector<std::size_t> &senders,
      const std::vector<std::size_t> &receivers) const;
  TransformerStack01Benchmark benchmark(std::size_t nodes,
                                        std::size_t edges,
                                        std::size_t repetitions) const;

  bool device_contract_verified() const { return device_contract_verified_; }
  std::size_t persistent_device_bytes() const;

 private:
  KokkosTransformerBlock0 block0_;
  KokkosFeatureTransformerBlock block1_;
  bool device_contract_verified_ = false;
};

}  // namespace so3lr
