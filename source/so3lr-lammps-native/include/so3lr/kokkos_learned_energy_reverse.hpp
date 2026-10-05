#pragma once

#include "so3lr/kokkos_complete_transformer_reverse.hpp"
#include "so3lr/kokkos_output_heads.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace so3lr {

struct LearnedEnergyReverseWorkspace {
  using DoubleView = Kokkos::View<double *>;

  LearnedEnergyReverseWorkspace(std::size_t nodes, std::size_t edges);
  LearnedEnergyReverseWorkspace(std::size_t nodes, std::size_t edges,
                                std::size_t head_nodes);
  using Int64View = Kokkos::View<std::int64_t *>;
  DoubleView embedding, initial_ev, energy_final_ev_gradient;
  DoubleView owned_head_input, expanded_head_input_gradient;
  std::array<DoubleView, 2> synchronized_inv, synchronized_ev;
  DoubleView synchronized_grad_inv, synchronized_grad_ev;
  Int64View owned_head_atomic_numbers;
  std::array<CompleteTransformerReverseWorkspace, 3> blocks;
  OutputHeadsDeviceWorkspace heads;
  DoubleView grad_distances, grad_sh;
  bool shared_reverse_scratch_verified = false;
};

struct LearnedEnergyReverseBenchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double forward_milliseconds = 0.0;
  double reverse_milliseconds = 0.0;
  double forward_reverse_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

class KokkosLearnedEnergyReverseChain {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosLearnedEnergyReverseChain(const NativeModel &model);

  void launch_forward_device(
      const Int64View &atomic_numbers, const DoubleView &distances,
      const DoubleView &sh_vectors, const IndexView &senders,
      const IndexView &receivers,
      const LearnedEnergyReverseWorkspace &workspace) const;
  void launch_forward_owned_device(
      const Int64View &atomic_numbers, std::size_t owned_nodes,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const LearnedEnergyReverseWorkspace &workspace) const;
  void launch_forward_periodic_device(
      const Int64View &atomic_numbers, std::size_t owned_nodes,
      const IndexView &node_owners, const DoubleView &distances,
      const DoubleView &sh_vectors, const IndexView &senders,
      const IndexView &receivers,
      const LearnedEnergyReverseWorkspace &workspace) const;
  void launch_reverse_device(
      const Int64View &atomic_numbers, const DoubleView &distances,
      const DoubleView &sh_vectors, const IndexView &senders,
      const IndexView &receivers,
      const LearnedEnergyReverseWorkspace &workspace) const;
  void launch_multihead_reverse_device(
      const Int64View &atomic_numbers, const DoubleView &distances,
      const DoubleView &sh_vectors, const IndexView &senders,
      const IndexView &receivers, const DoubleView &energy_seeds,
      const DoubleView &partial_charge_seeds,
      const DoubleView &hirshfeld_seeds,
      const LearnedEnergyReverseWorkspace &workspace) const;
  void launch_multihead_reverse_owned_device(
      const Int64View &atomic_numbers, std::size_t owned_nodes,
      const DoubleView &distances, const DoubleView &sh_vectors,
      const IndexView &senders, const IndexView &receivers,
      const DoubleView &energy_seeds,
      const DoubleView &partial_charge_seeds,
      const DoubleView &hirshfeld_seeds,
      const LearnedEnergyReverseWorkspace &workspace) const;
  void launch_multihead_reverse_periodic_device(
      const Int64View &atomic_numbers, std::size_t owned_nodes,
      const IndexView &node_owners, const DoubleView &distances,
      const DoubleView &sh_vectors, const IndexView &senders,
      const IndexView &receivers, const DoubleView &energy_seeds,
      const DoubleView &partial_charge_seeds,
      const DoubleView &hirshfeld_seeds,
      const LearnedEnergyReverseWorkspace &workspace) const;
  LearnedEnergyReverseBenchmark benchmark(
      std::size_t nodes, std::size_t edges,
      std::size_t repetitions) const;

  const DoubleView &grad_embedding(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.blocks[0].attention.grad_inv_features;
  }
  const DoubleView &grad_initial_ev(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.blocks[0].attention.grad_ev_features;
  }
  const DoubleView &grad_distances(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.grad_distances;
  }
  const DoubleView &grad_sh(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.grad_sh;
  }
  const DoubleView &block_grad_distances(
      const LearnedEnergyReverseWorkspace &workspace,
      std::size_t block_index) const;
  const DoubleView &block_grad_sh(
      const LearnedEnergyReverseWorkspace &workspace,
      std::size_t block_index) const;
  const DoubleView &atomic_energies(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.heads.atomic_energies;
  }
  const DoubleView &reductions(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.heads.reductions;
  }
  const DoubleView &partial_charges(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.heads.partial_charges;
  }
  const DoubleView &hirshfeld_ratios(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.heads.hirshfeld_ratios;
  }
  const DoubleView &combined_head_input_gradient(
      const LearnedEnergyReverseWorkspace &workspace) const {
    return workspace.heads.combined_input_grad;
  }

  bool chain_contract_verified() const { return chain_contract_verified_; }
  bool shared_kokkos_stream() const { return shared_kokkos_stream_; }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  std::array<KokkosCompleteTransformerReverseBlock, 3> blocks_;
  KokkosOutputHeads heads_;
  DoubleView embedding_weight_;
  double inverse_embedding_scale_ = 1.0;
  std::size_t persistent_device_bytes_ = 0;
  bool chain_contract_verified_ = false;
  bool shared_kokkos_stream_ = false;
};

}  // namespace so3lr
