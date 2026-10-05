#include "so3lr/kokkos_local_energy_reverse.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace so3lr {
namespace {

template <class View, class Reader>
void fill_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

}  // namespace

void fill_charge_spin_offset_device(const Kokkos::View<std::int64_t *> &atomic_numbers,
                                    const Kokkos::View<double *> &table,
                                    std::size_t width,
                                    Kokkos::View<double *> &offset) {
  const std::size_t nodes = atomic_numbers.extent(0);
  if (table.extent(0) != 118 * width)
    throw std::runtime_error("SO3LR charge/spin table has the wrong size");
  if (offset.extent(0) < nodes * width)
    offset = Kokkos::View<double *>("so3lr_charge_spin_embedding_offset",
                                    nodes * width);
  const auto out = offset;
  Kokkos::parallel_for(
      "so3lr_fill_charge_spin_offset", Kokkos::RangePolicy<>(0, nodes * width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / width;
        const std::size_t channel = flat % width;
        const std::int64_t z = atomic_numbers(node);
        out(flat) = z > 0 && z <= 118
                        ? table(static_cast<std::size_t>(z - 1) * width + channel)
                        : 0.0;
      });
}

LocalEnergyReverseWorkspace::LocalEnergyReverseWorkspace(
    std::size_t nodes, std::size_t local_edges, std::size_t layers,
    const ArchDims &dims)
    : embedding("so3lr_local_reverse_embedding",
                nodes * dims.invariant_width),
      initial_ev("so3lr_local_reverse_initial_ev",
                 nodes * dims.equivariant_width),
      final_ev_seed("so3lr_local_reverse_final_ev_seed",
                    nodes * dims.equivariant_width),
      heads(nodes, dims),
      scaled_inv(layers) {
  if (nodes == 0 || local_edges == 0 || layers == 0)
    throw std::runtime_error("SO3LR local energy reverse workspace empty");
  // Independent per-block workspaces: this chain does not alias scratch
  // between blocks, so a runtime-sized vector is safe.
  blocks.reserve(layers);
  for (std::size_t n = 0; n < layers; ++n)
    blocks.emplace_back(nodes, local_edges, dims);
}

KokkosLocalEnergyReverse::KokkosLocalEnergyReverse(const NativeModel &model)
    : dims_(model.arch().dims()),
      heads_(model),
      embedding_weight_("so3lr_local_reverse_embedding_weight",
                        dims_.invariant_width * 118),
      inverse_embedding_scale_(
          1.0 / model.model_number("embedding_scale")) {
  const std::size_t layers = model.arch().layers;
  const std::size_t inv_width = dims_.invariant_width;
  blocks_.reserve(layers);
  for (std::size_t n = 0; n < layers; ++n)
    blocks_.push_back(std::make_unique<KokkosCompleteTransformerReverseBlock>(model, n));
  const auto &weight =
      model.tensor("model.inv_feature_embedding.embedding.weight");
  if (weight.dtype != "float64" ||
      weight.shape != std::vector<std::size_t>({inv_width, 118}))
    throw std::runtime_error("SO3LR local reverse embedding contract changed");
  if (!std::isfinite(inverse_embedding_scale_) ||
      inverse_embedding_scale_ <= 0.0)
    throw std::runtime_error(
        "SO3LR local reverse embedding scale must be positive and finite");
  fill_view(embedding_weight_, inv_width * 118,
            [&](std::size_t i) { return model.float64(weight, i); });
  residual_scalars_ = model.arch().use_residual_scalars;
  resid_lambdas_.assign(layers, 1.0);
  x0_lambdas_.assign(layers, 0.0);
  if (residual_scalars_) {
    const auto &resid = model.tensor("model.resid_lambdas");
    const auto &x0 = model.tensor("model.x0_lambdas");
    const std::vector<std::size_t> expected{blocks_.size()};
    if (resid.dtype != "float64" || x0.dtype != "float64" ||
        resid.shape != expected || x0.shape != expected)
      throw std::runtime_error(
          "SO3LR residual scalars must be float64 vectors of length "
          "interaction_blocks");
    for (std::size_t n = 0; n < blocks_.size(); ++n) {
      resid_lambdas_[n] = model.float64(resid, n);
      x0_lambdas_[n] = model.float64(x0, n);
    }
  }
  staged_reverse_contract_verified_ =
      heads_.checkpoint_contract_verified() && heads_.shared_kokkos_stream();
  for (std::size_t index = 0; index < blocks_.size(); ++index)
    staged_reverse_contract_verified_ =
        staged_reverse_contract_verified_ &&
        blocks_[index]->checkpoint_contract_verified() &&
        blocks_[index]->shared_kokkos_stream() &&
        blocks_[index]->block_index() == index;
  if (!staged_reverse_contract_verified_)
    throw std::runtime_error("SO3LR staged local reverse contract failed");
  Kokkos::fence();
}

const KokkosCompleteTransformerReverseBlock &KokkosLocalEnergyReverse::block(
    std::size_t index) const {
  if (index >= blocks_.size())
    throw std::runtime_error("SO3LR local reverse block index out of range");
  return *blocks_[index];
}

namespace {

// Kernels for the residual scalars. They live here rather than in the member
// functions that use them because nvcc does not allow an extended device
// lambda inside a private member function.
void residual_combine(const Kokkos::View<double *> &input,
                      const Kokkos::View<double *> &previous,
                      const Kokkos::View<double *> &x0, double a, double b) {
  Kokkos::parallel_for(
      "so3lr_local_residual_scalars", Kokkos::RangePolicy<>(0, input.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { input(i) = a * previous(i) + b * x0(i); });
}

void scale_in_place(const Kokkos::View<double *> &values, double factor) {
  Kokkos::parallel_for(
      "so3lr_local_residual_scalars_reverse",
      Kokkos::RangePolicy<>(0, values.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { values(i) *= factor; });
}

}  // namespace

KokkosLocalEnergyReverse::DoubleView KokkosLocalEnergyReverse::block_input(
    std::size_t n, const LocalEnergyReverseWorkspace &workspace) const {
  const DoubleView previous =
      n == 0 ? workspace.embedding : blocks_[n - 1]->final_inv(workspace.blocks[n - 1]);
  if (!residual_scalars_) return previous;
  // Optional residual scalars, applied before every block exactly as
  // StackNetSparse does: x = resid[n] * x + x0[n] * x0, with x = x0 for n = 0.
  const std::size_t count = workspace.embedding.extent(0);
  if (workspace.scaled_inv[n].extent(0) != count)
    workspace.scaled_inv[n] =
        DoubleView("so3lr_local_residual_input_" + std::to_string(n), count);
  residual_combine(workspace.scaled_inv[n], previous, workspace.embedding,
                   resid_lambdas_[n], x0_lambdas_[n]);
  return workspace.scaled_inv[n];
}

KokkosLocalEnergyReverse::DoubleView KokkosLocalEnergyReverse::reverse_block_input(
    std::size_t n, const LocalEnergyReverseWorkspace &workspace) const {
  if (residual_scalars_) return workspace.scaled_inv[n];
  return n == 0 ? workspace.embedding : blocks_[n - 1]->final_inv(workspace.blocks[n - 1]);
}

void KokkosLocalEnergyReverse::scale_input_gradient(
    std::size_t n, const DoubleView &gradient) const {
  if (!residual_scalars_) return;
  scale_in_place(gradient, resid_lambdas_[n]);
}




void KokkosLocalEnergyReverse::launch_block_forward_device(
    std::size_t n, const Int64View &atomic_numbers,
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  if (n >= blocks_.size() || workspace.blocks.size() != blocks_.size())
    throw std::runtime_error("SO3LR local reverse block index out of range");
  const std::size_t inv_width = dims_.invariant_width;
  const std::size_t ev_width = dims_.equivariant_width;
  if (n == 0) {
    const std::size_t nodes = atomic_numbers.extent(0);
    if (workspace.embedding.extent(0) != nodes * inv_width ||
        workspace.initial_ev.extent(0) != nodes * ev_width)
      throw std::runtime_error("SO3LR local reverse embedding shape mismatch");
    const auto weight = embedding_weight_;
    const double inverse_embedding_scale = inverse_embedding_scale_;
    const auto embedding = workspace.embedding;
    Kokkos::parallel_for(
        "so3lr_local_reverse_atomic_embedding",
        Kokkos::RangePolicy<>(0, nodes * inv_width),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t node = flat / inv_width;
          const std::size_t channel = flat % inv_width;
          const std::int64_t z = atomic_numbers(node);
          embedding(flat) =
              z > 0 && z <= 118
                  ? weight(channel * 118 + static_cast<std::size_t>(z - 1)) *
                        inverse_embedding_scale
                  : 0.0;
        });
    // Charge/spin embedding: inv = (element + charge + spin) / scale.
    if (workspace.embedding_offset.extent(0) != 0) {
      if (workspace.embedding_offset.extent(0) < nodes * inv_width)
        throw std::runtime_error("SO3LR charge/spin embedding offset too small");
      const auto offset = workspace.embedding_offset;
      Kokkos::parallel_for(
          "so3lr_local_reverse_charge_spin_embedding",
          Kokkos::RangePolicy<>(0, nodes * inv_width),
          KOKKOS_LAMBDA(const std::size_t flat) {
            embedding(flat) += offset(flat) * inverse_embedding_scale;
          });
    }
    Kokkos::deep_copy(workspace.initial_ev, 0.0);
    Kokkos::deep_copy(workspace.final_ev_seed, 0.0);
  }
  const DoubleView ev_input =
      n == 0 ? workspace.initial_ev : blocks_[n - 1]->final_ev(workspace.blocks[n - 1]);
  blocks_[n]->launch_forward_device(block_input(n, workspace), ev_input,
                                    local_distances, local_sh, local_senders,
                                    local_receivers, workspace.blocks[n]);
}

void KokkosLocalEnergyReverse::launch_block_reverse_device(
    std::size_t n, const DoubleView &local_distances,
    const DoubleView &local_sh, const IndexView &local_senders,
    const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace, bool multihead) const {
  if (n >= blocks_.size() || workspace.blocks.size() != blocks_.size())
    throw std::runtime_error("SO3LR local reverse block index out of range");
  const std::size_t last = blocks_.size() - 1;
  DoubleView seed_inv;
  DoubleView seed_ev;
  if (n == last) {
    seed_inv = multihead ? workspace.heads.combined_input_grad
                         : workspace.heads.energy_input_grad;
    seed_ev = workspace.final_ev_seed;
  } else {
    // Block n+1 returned d/d(its input); turn that into d/d(block n output).
    scale_input_gradient(n + 1, blocks_[n + 1]->grad_inv(workspace.blocks[n + 1]));
    seed_inv = blocks_[n + 1]->grad_inv(workspace.blocks[n + 1]);
    seed_ev = blocks_[n + 1]->grad_ev(workspace.blocks[n + 1]);
  }
  const DoubleView ev_input =
      n == 0 ? workspace.initial_ev : blocks_[n - 1]->final_ev(workspace.blocks[n - 1]);
  blocks_[n]->launch_reverse_device(reverse_block_input(n, workspace), ev_input,
                                    local_distances, local_sh, local_senders,
                                    local_receivers, seed_inv, seed_ev,
                                    workspace.blocks[n]);
}

void KokkosLocalEnergyReverse::launch_block0_forward_device(
    const Int64View &atomic_numbers, const DoubleView &local_distances,
    const DoubleView &local_sh, const IndexView &local_senders,
    const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_forward_device(0, atomic_numbers, local_distances, local_sh,
                              local_senders, local_receivers, workspace);
}

void KokkosLocalEnergyReverse::launch_block1_forward_device(
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_forward_device(1, Int64View(), local_distances, local_sh,
                              local_senders, local_receivers, workspace);
}

void KokkosLocalEnergyReverse::launch_block2_forward_device(
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_forward_device(2, Int64View(), local_distances, local_sh,
                              local_senders, local_receivers, workspace);
}

void KokkosLocalEnergyReverse::launch_block2_reverse_device(
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_reverse_device(2, local_distances, local_sh, local_senders,
                              local_receivers, workspace, false);
}

void KokkosLocalEnergyReverse::launch_block2_multihead_reverse_device(
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_reverse_device(2, local_distances, local_sh, local_senders,
                              local_receivers, workspace, true);
}

void KokkosLocalEnergyReverse::launch_block1_reverse_device(
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_reverse_device(1, local_distances, local_sh, local_senders,
                              local_receivers, workspace);
}

void KokkosLocalEnergyReverse::launch_block0_reverse_device(
    const DoubleView &local_distances, const DoubleView &local_sh,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalEnergyReverseWorkspace &workspace) const {
  launch_block_reverse_device(0, local_distances, local_sh, local_senders,
                              local_receivers, workspace);
}

void KokkosLocalEnergyReverse::launch_owned_energy_head_reverse_device(
    const Int64View &atomic_numbers, const IndexView &owned_mask,
    const LocalEnergyReverseWorkspace &workspace) const {
  const std::size_t inv_width = dims_.invariant_width;
  const std::size_t nodes = atomic_numbers.extent(0);
  if (owned_mask.extent(0) != nodes)
    throw std::runtime_error("SO3LR owned energy mask shape mismatch");
  heads_.launch_device(blocks_.back()->final_inv(workspace.blocks.back()),
                       atomic_numbers, 0.0, workspace.heads);
  heads_.launch_energy_reverse_device(atomic_numbers, workspace.heads);
  const auto gradient = workspace.heads.energy_input_grad;
  Kokkos::parallel_for(
      "so3lr_mask_energy_head_to_owned_nodes",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / inv_width;
        if (owned_mask(node) == 0) gradient(flat) = 0.0;
      });
  Kokkos::deep_copy(workspace.final_ev_seed, 0.0);
}

void KokkosLocalEnergyReverse::launch_output_heads_device(
    const Int64View &atomic_numbers,
    const LocalEnergyReverseWorkspace &workspace) const {
  heads_.launch_device(blocks_.back()->final_inv(workspace.blocks.back()),
                       atomic_numbers, 0.0, workspace.heads);
}

void KokkosLocalEnergyReverse::launch_distributed_charge_correction_device(
    double charge_correction,
    const LocalEnergyReverseWorkspace &workspace) const {
  const auto raw = workspace.heads.raw_charges;
  const auto partial = workspace.heads.partial_charges;
  Kokkos::parallel_for(
      "so3lr_local_distributed_charge_correction",
      Kokkos::RangePolicy<>(0, raw.extent(0)),
      KOKKOS_LAMBDA(const std::size_t node) {
        partial(node) = raw(node) + charge_correction;
      });
}

void KokkosLocalEnergyReverse::launch_owned_multihead_reverse_device(
    const Int64View &atomic_numbers, const IndexView &owned_mask,
    const DoubleView &energy_seeds, const DoubleView &partial_charge_seeds,
    const DoubleView &hirshfeld_seeds, double global_charge_seed_mean,
    const LocalEnergyReverseWorkspace &workspace,
    const DoubleView &c6_seeds) const {
  const std::size_t inv_width = dims_.invariant_width;
  const std::size_t nodes = atomic_numbers.extent(0);
  if (owned_mask.extent(0) != nodes || energy_seeds.extent(0) != nodes ||
      partial_charge_seeds.extent(0) != nodes ||
      hirshfeld_seeds.extent(0) != nodes)
    throw std::runtime_error("SO3LR local multihead seed shape mismatch");
  heads_.launch_multihead_reverse_device(
      atomic_numbers, energy_seeds, partial_charge_seeds, hirshfeld_seeds,
      workspace.heads, global_charge_seed_mean, true, c6_seeds);
  const auto gradient = workspace.heads.combined_input_grad;
  Kokkos::parallel_for(
      "so3lr_mask_multihead_to_owned_nodes",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / inv_width;
        if (owned_mask(node) == 0) gradient(flat) = 0.0;
      });
  Kokkos::deep_copy(workspace.final_ev_seed, 0.0);
}





void KokkosLocalEnergyReverse::zero_selected_adjoint_device(
    const IndexView &nodes, const DoubleView &grad_inv,
    const DoubleView &grad_ev) const {
  const std::size_t inv_width = dims_.invariant_width;
  const std::size_t ev_width = dims_.equivariant_width;
  Kokkos::parallel_for(
      "so3lr_zero_selected_adjoint_inv",
      Kokkos::RangePolicy<>(0, nodes.extent(0) * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / inv_width;
        const std::size_t channel = flat % inv_width;
        grad_inv(nodes(item) * inv_width + channel) = 0.0;
      });
  Kokkos::parallel_for(
      "so3lr_zero_selected_adjoint_ev",
      Kokkos::RangePolicy<>(0, nodes.extent(0) * ev_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / ev_width;
        const std::size_t channel = flat % ev_width;
        grad_ev(nodes(item) * ev_width + channel) = 0.0;
      });
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::block_final_inv(
    const LocalEnergyReverseWorkspace &workspace, std::size_t index) const {
  return block(index).final_inv(workspace.blocks[index]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::block_final_ev(
    const LocalEnergyReverseWorkspace &workspace, std::size_t index) const {
  return block(index).final_ev(workspace.blocks[index]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::block_grad_inv(
    const LocalEnergyReverseWorkspace &workspace, std::size_t index) const {
  return block(index).grad_inv(workspace.blocks[index]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::block_grad_ev(
    const LocalEnergyReverseWorkspace &workspace, std::size_t index) const {
  return block(index).grad_ev(workspace.blocks[index]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::block_grad_distances(
    const LocalEnergyReverseWorkspace &workspace, std::size_t index) const {
  return block(index).grad_distances(workspace.blocks[index]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::block_grad_sh(
    const LocalEnergyReverseWorkspace &workspace, std::size_t index) const {
  return block(index).grad_sh(workspace.blocks[index]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::embedding_gradient(
    const LocalEnergyReverseWorkspace &workspace) const {
  return blocks_[0]->grad_inv(workspace.blocks[0]);
}

const KokkosLocalEnergyReverse::DoubleView &
KokkosLocalEnergyReverse::initial_ev_gradient(
    const LocalEnergyReverseWorkspace &workspace) const {
  return blocks_[0]->grad_ev(workspace.blocks[0]);
}

}  // namespace so3lr
