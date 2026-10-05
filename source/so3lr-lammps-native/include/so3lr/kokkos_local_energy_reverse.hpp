#pragma once

#include "so3lr/kokkos_complete_transformer_reverse.hpp"
#include "so3lr/kokkos_output_heads.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>
#include <cstdint>

namespace so3lr {

struct LocalEnergyReverseWorkspace {
  using DoubleView = Kokkos::View<double *>;

  // `layers` is the model's interaction depth (3 for SO3LR v1).
  LocalEnergyReverseWorkspace(std::size_t nodes, std::size_t local_edges,
                              std::size_t layers = 3,
                              const ArchDims &dims = v1_arch_dims());
  DoubleView embedding;
  // Optional (nodes x F) charge/spin embedding offset, added to the element
  // embedding before the embedding scale (so3lr_charge_spin.hpp). Left empty
  // for a neutral singlet, where it is exactly zero; the caller fills it.
  mutable DoubleView embedding_offset;
  DoubleView initial_ev;
  DoubleView final_ev_seed;
  std::vector<CompleteTransformerReverseWorkspace> blocks;
  OutputHeadsDeviceWorkspace heads;
  // Block inputs for models with per-layer residual scalars
  // (x_in = a_n x_prev + b_n x0). Allocated lazily on first use, so models
  // without them never pay for the buffers.
  mutable std::vector<DoubleView> scaled_inv;
};

// Staged energy VJP for a receiver-owned rank-local SR graph.  MPI remains in
// the caller: after reverse block 2 and block 1, ghost input adjoints must be
// sent to their owners, accumulated, and then zeroed on the ghost copies.
// offset(node, c) = table(Z(node) - 1, c) for a (118 x width) per-element
// charge/spin table; zero for an unmapped Z. Grows `offset` if needed.
void fill_charge_spin_offset_device(const Kokkos::View<std::int64_t *> &atomic_numbers,
                                    const Kokkos::View<double *> &table,
                                    std::size_t width,
                                    Kokkos::View<double *> &offset);

class KokkosLocalEnergyReverse {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosLocalEnergyReverse(const NativeModel &model);

  // Interaction depth of the loaded model.
  std::size_t layers() const { return blocks_.size(); }
  const ArchDims &dims() const { return dims_; }

  // Forward of block n. Block 0 also computes the atom-type embedding; later
  // blocks read the previous block's output, which the MPI caller has already
  // made consistent on ghost rows.
  void launch_block_forward_device(
      std::size_t n, const Int64View &atomic_numbers,
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;
  // Reverse of block n. The last block is seeded by the output heads
  // (energy-only, or all heads when `multihead`); block n < last by block
  // n+1's input gradient, which the caller has already exchanged.
  void launch_block_reverse_device(
      std::size_t n, const DoubleView &local_distances,
      const DoubleView &local_sh, const IndexView &local_senders,
      const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace,
      bool multihead = true) const;

  // Fixed-depth names kept for the v1 kernel tests; they forward to the
  // indexed API and are only valid for a three-block model.
  void launch_block0_forward_device(
      const Int64View &atomic_numbers, const DoubleView &local_distances,
      const DoubleView &local_sh, const IndexView &local_senders,
      const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_block1_forward_device(
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_block2_forward_device(
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;

  // Runs energy heads for all local+ghost rows, then retains reverse seeds only
  // on rows whose owner mask is one.  Owned rows are the only valid block-2
  // outputs for a rank-local edge graph.
  void launch_owned_energy_head_reverse_device(
      const Int64View &atomic_numbers, const IndexView &owned_mask,
      const LocalEnergyReverseWorkspace &workspace) const;

  // Distributed multi-head sequence. First evaluate uncorrected local output
  // heads, then use a caller-provided global raw-charge correction. The
  // reverse method masks ghost rows and uses the global charge-seed mean.
  void launch_output_heads_device(
      const Int64View &atomic_numbers,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_distributed_charge_correction_device(
      double charge_correction,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_owned_multihead_reverse_device(
      const Int64View &atomic_numbers, const IndexView &owned_mask,
      const DoubleView &energy_seeds, const DoubleView &partial_charge_seeds,
      const DoubleView &hirshfeld_seeds, double global_charge_seed_mean,
      const LocalEnergyReverseWorkspace &workspace,
      const DoubleView &c6_seeds = DoubleView()) const;

  void launch_block2_reverse_device(
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_block2_multihead_reverse_device(
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_block1_reverse_device(
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;
  void launch_block0_reverse_device(
      const DoubleView &local_distances, const DoubleView &local_sh,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalEnergyReverseWorkspace &workspace) const;

  // Used after packing ghost contributions and before adding the received
  // owner contribution.  It prevents a ghost adjoint from becoming a seed for
  // the preceding local reverse block.
  void zero_selected_adjoint_device(const IndexView &nodes,
                                    const DoubleView &grad_inv,
                                    const DoubleView &grad_ev) const;

  const DoubleView &block_final_inv(
      const LocalEnergyReverseWorkspace &workspace,
      std::size_t block) const;
  const DoubleView &block_final_ev(
      const LocalEnergyReverseWorkspace &workspace,
      std::size_t block) const;
  const DoubleView &block_grad_inv(
      const LocalEnergyReverseWorkspace &workspace,
      std::size_t block) const;
  const DoubleView &block_grad_ev(
      const LocalEnergyReverseWorkspace &workspace,
      std::size_t block) const;
  const DoubleView &block_grad_distances(
      const LocalEnergyReverseWorkspace &workspace,
      std::size_t block) const;
  const DoubleView &block_grad_sh(
      const LocalEnergyReverseWorkspace &workspace,
      std::size_t block) const;
  const DoubleView &atomic_energies(
      const LocalEnergyReverseWorkspace &workspace) const {
    return workspace.heads.atomic_energies;
  }
  const DoubleView &raw_charges(
      const LocalEnergyReverseWorkspace &workspace) const {
    return workspace.heads.raw_charges;
  }
  const DoubleView &partial_charges(
      const LocalEnergyReverseWorkspace &workspace) const {
    return workspace.heads.partial_charges;
  }
  const DoubleView &hirshfeld_ratios(
      const LocalEnergyReverseWorkspace &workspace) const {
    return workspace.heads.hirshfeld_ratios;
  }
  // Optional C6 head (absent for v1).
  bool has_c6_head() const { return heads_.has_c6_head(); }
  const DoubleView &c6_ratios(const LocalEnergyReverseWorkspace &workspace) const {
    return workspace.heads.c6_ratios;
  }
  const DoubleView &embedding_gradient(
      const LocalEnergyReverseWorkspace &workspace) const;
  const DoubleView &initial_ev_gradient(
      const LocalEnergyReverseWorkspace &workspace) const;

  bool receiver_owned_edge_contract() const { return true; }
  bool owned_output_seed_contract() const { return true; }
  bool staged_reverse_contract_verified() const {
    return staged_reverse_contract_verified_;
  }

 private:
  // Input to block n in the forward pass: the previous block's output, or the
  // scaled combination when the model has residual scalars.
  DoubleView block_input(std::size_t n,
                         const LocalEnergyReverseWorkspace &workspace) const;
  // The same input, for the reverse pass: reuses the buffer the forward filled.
  DoubleView reverse_block_input(std::size_t n,
                                 const LocalEnergyReverseWorkspace &workspace) const;
  // Turns the gradient w.r.t. block n's scaled input into the gradient w.r.t.
  // block n-1's output, in place. The embedding gradient is not needed: the
  // embedding depends only on atomic numbers, never on positions.
  void scale_input_gradient(std::size_t n, const DoubleView &gradient) const;
  bool residual_scalars_ = false;
  std::vector<double> resid_lambdas_;
  std::vector<double> x0_lambdas_;
  const KokkosCompleteTransformerReverseBlock &block(
      std::size_t index) const;

  ArchDims dims_;
  // unique_ptr: each block owns a cuBLAS handle and is neither copyable nor
  // movable, so it cannot sit in a vector by value.
  std::vector<std::unique_ptr<KokkosCompleteTransformerReverseBlock>> blocks_;
  KokkosOutputHeads heads_;
  DoubleView embedding_weight_;
  double inverse_embedding_scale_ = 1.0;
  bool staged_reverse_contract_verified_ = false;
};

}  // namespace so3lr
