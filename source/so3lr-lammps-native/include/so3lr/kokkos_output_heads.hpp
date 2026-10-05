#pragma once

#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct OutputHeadsResults {
  std::vector<double> atomic_energies;
  std::vector<double> raw_charges;
  std::vector<double> partial_charges;
  std::vector<double> hirshfeld_ratios;
  double learned_sr_energy = 0.0;
  double charge_sum = 0.0;
};

struct OutputHeadsBenchmark {
  std::size_t nodes = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double milliseconds = 0.0;
  double nodes_per_second = 0.0;
  double checksum = 0.0;
};

struct EnergyHeadReverseBenchmark {
  std::size_t nodes = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double milliseconds = 0.0;
  double nodes_per_second = 0.0;
  double checksum = 0.0;
};

// One ratio head: |v_shift[Z] + sum_k q[Z,k] key_k / sqrt(K)|, where
// key = W2 silu(W1 x + b1) + b2. With key width K = 64 this is the SO3LR v1
// Hirshfeld head. With K = 1 and q = ones it is exactly an a0-ratio head
// |final(silu(W x)) + q[Z]|, and a C6-ratio head has the same form -- so one
// kernel serves all three.
struct RatioHeadParameters {
  Kokkos::View<double *> v_shift, q_embedding;
  Kokkos::View<double *> weight_1, bias_1, weight_2, bias_2;
  std::size_t hidden_width = 0;  // H: rows of weight_1
  std::size_t key_width = 0;     // K: rows of weight_2
  bool present = false;
};

struct OutputHeadsDeviceWorkspace {
  using DoubleView = Kokkos::View<double *>;

  explicit OutputHeadsDeviceWorkspace(std::size_t nodes,
                                      const ArchDims &dims = v1_arch_dims());
  OutputHeadsDeviceWorkspace(std::size_t nodes, std::size_t inv_width,
                             std::size_t hidden_width);
  DoubleView energy_preact, energy_hidden, atomic_energies;
  DoubleView energy_hidden_grad, energy_input_grad;
  DoubleView charge_preact, charge_hidden, raw_charges, partial_charges;
  DoubleView hirshfeld_preact_1, hirshfeld_hidden_1;
  DoubleView hirshfeld_hidden_2, hirshfeld_ratios;
  DoubleView combined_input_grad, reverse_input_scratch;
  DoubleView reductions, reverse_reductions;
  // Optional C6 head. Allocated lazily on first use, so models without the
  // head never pay for it.
  mutable DoubleView c6_preact_1, c6_hidden_1, c6_hidden_2, c6_ratios;
};

class KokkosOutputHeads {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;

  explicit KokkosOutputHeads(const NativeModel &model);
  ~KokkosOutputHeads();
  KokkosOutputHeads(const KokkosOutputHeads &) = delete;
  KokkosOutputHeads &operator=(const KokkosOutputHeads &) = delete;

  void launch_device(const DoubleView &inv_features,
                     const Int64View &atomic_numbers, double total_charge,
                     const OutputHeadsDeviceWorkspace &workspace) const;
  void launch_energy_reverse_device(
      const Int64View &atomic_numbers,
      const OutputHeadsDeviceWorkspace &workspace) const;
  void launch_multihead_reverse_device(
      const Int64View &atomic_numbers, const DoubleView &energy_seeds,
      const DoubleView &partial_charge_seeds,
      const DoubleView &hirshfeld_seeds,
      const OutputHeadsDeviceWorkspace &workspace,
      double global_charge_seed_mean = 0.0,
      bool use_global_charge_seed_mean = false,
      const DoubleView &c6_seeds = DoubleView()) const;
  bool has_c6_head() const { return c6_head_.present; }
  OutputHeadsResults evaluate(
      const std::vector<double> &inv_features,
      const std::vector<std::int64_t> &atomic_numbers,
      double total_charge) const;
  std::vector<double> evaluate_energy_input_gradient(
      const std::vector<double> &inv_features,
      const std::vector<std::int64_t> &atomic_numbers) const;
  OutputHeadsBenchmark benchmark(std::size_t nodes,
                                 std::size_t repetitions) const;
  EnergyHeadReverseBenchmark benchmark_energy_reverse(
      std::size_t nodes, std::size_t repetitions) const;

  bool checkpoint_contract_verified() const {
    return checkpoint_contract_verified_;
  }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  std::size_t input_width_ = 0;   // F
  std::size_t hidden_width_ = 0;  // energy/charge head hidden width
  DoubleView energy_weight_1_, energy_bias_1_, energy_weight_2_;
  DoubleView energy_bias_2_, energy_scales_, energy_shifts_;
  DoubleView charge_embedding_, charge_weight_1_, charge_bias_1_;
  DoubleView charge_weight_2_, charge_bias_2_;
  RatioHeadParameters hirshfeld_head_;  // Hirshfeld (v1) or a0 ratio
  RatioHeadParameters c6_head_;         // optional
  cublasHandle_t handle_ = nullptr;
  std::size_t persistent_device_bytes_ = 0;
  bool checkpoint_contract_verified_ = false;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr
