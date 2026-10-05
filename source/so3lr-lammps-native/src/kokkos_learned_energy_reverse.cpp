#include "so3lr/kokkos_learned_energy_reverse.hpp"

#include <Kokkos_Timer.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace so3lr {
namespace {

constexpr std::size_t inv_width = 128;
constexpr std::size_t ev_width = 24;


std::array<CompleteTransformerReverseWorkspace, 3>
make_shared_reverse_blocks(std::size_t nodes, std::size_t edges) {
  CompleteTransformerReverseWorkspace block0(nodes, edges);
  CompleteTransformerReverseWorkspace block1(nodes, edges, block0);
  CompleteTransformerReverseWorkspace block2(nodes, edges, block0);
  return {block0, block1, block2};
}

bool verify_shared_reverse_scratch(
    const std::array<CompleteTransformerReverseWorkspace, 3> &blocks) {
  const auto &owner = blocks[0];
  for (std::size_t block = 1; block < blocks.size(); ++block) {
    const auto &other = blocks[block];
    if (other.post_reverse.grad_attention_inv.data() !=
            owner.post_reverse.grad_attention_inv.data() ||
        other.attention.grad_inv_features.data() !=
            owner.attention.grad_inv_features.data() ||
        other.attention.attention_reverse.grad_qkv[0].data() !=
            owner.attention.attention_reverse.grad_qkv[0].data() ||
        other.attention.grad_radial_basis.data() !=
            owner.attention.grad_radial_basis.data() ||
        other.attention.attention_reverse.grad_filter_inv.data() !=
            owner.attention.attention_reverse.grad_filter_inv.data() ||
        other.attention.grad_distances.data() ==
            owner.attention.grad_distances.data() ||
        other.attention.attention_reverse.grad_sh.data() ==
            owner.attention.attention_reverse.grad_sh.data())
      return false;
  }
  return true;
}

void synchronize_periodic_features(
    const Kokkos::View<double *> &source_inv,
    const Kokkos::View<double *> &source_ev,
    const Kokkos::View<std::size_t *> &owners, std::size_t nodes,
    const Kokkos::View<double *> &destination_inv,
    const Kokkos::View<double *> &destination_ev) {
  Kokkos::parallel_for(
      "so3lr_chain_sync_periodic_inv",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / inv_width;
        const std::size_t channel = flat % inv_width;
        destination_inv(flat) =
            source_inv(owners(node) * inv_width + channel);
      });
  Kokkos::parallel_for(
      "so3lr_chain_sync_periodic_ev",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / ev_width;
        const std::size_t channel = flat % ev_width;
        destination_ev(flat) = source_ev(owners(node) * ev_width + channel);
      });
}

void reverse_periodic_feature_sync(
    const Kokkos::View<double *> &grad_inv,
    const Kokkos::View<double *> &grad_ev,
    const Kokkos::View<std::size_t *> &owners, std::size_t owned_nodes,
    std::size_t nodes, const Kokkos::View<double *> &synced_grad_inv,
    const Kokkos::View<double *> &synced_grad_ev) {
  Kokkos::deep_copy(synced_grad_inv, 0.0);
  Kokkos::deep_copy(synced_grad_ev, 0.0);
  Kokkos::parallel_for(
      "so3lr_chain_reverse_sync_periodic_inv",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / inv_width;
        const std::size_t channel = flat % inv_width;
        const std::size_t owner = owners(node);
        if (owner < owned_nodes)
          Kokkos::atomic_add(&synced_grad_inv(owner * inv_width + channel),
                             grad_inv(flat));
      });
  Kokkos::parallel_for(
      "so3lr_chain_reverse_sync_periodic_ev",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / ev_width;
        const std::size_t channel = flat % ev_width;
        const std::size_t owner = owners(node);
        if (owner < owned_nodes)
          Kokkos::atomic_add(&synced_grad_ev(owner * ev_width + channel),
                             grad_ev(flat));
      });
}

template <class View, class Reader>
void fill_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

std::size_t chain_workspace_bytes(std::size_t nodes, std::size_t edges,
                                  std::size_t persistent_bytes) {
  // dev_4 retains all forward checkpoints and all three geometry-gradient
  // outputs, but owns one reverse-only scratch arena.  Two duplicate arenas
  // are removed: 2256 doubles/node and 549 doubles/edge each.
  constexpr std::size_t block_node_doubles = 4516;
  constexpr std::size_t block_edge_doubles = 1027;
  constexpr std::size_t shared_reverse_node_doubles = 2256;
  constexpr std::size_t shared_reverse_edge_doubles = 549;
  constexpr std::size_t chain_node_doubles =
      3 * block_node_doubles - 2 * shared_reverse_node_doubles +
      128 + 24 + 24 + 128 + 128 + 3 * 128 + 3 * 24 + 1220;
  constexpr std::size_t chain_edge_doubles =
      3 * block_edge_doubles - 2 * shared_reverse_edge_doubles +
      1 + 24 + 1 + 24;
  return persistent_bytes +
         nodes * (chain_node_doubles * sizeof(double) + sizeof(std::int64_t)) +
         edges * (chain_edge_doubles * sizeof(double) +
                  2 * sizeof(std::size_t));
}

}  // namespace

LearnedEnergyReverseWorkspace::LearnedEnergyReverseWorkspace(
    std::size_t nodes, std::size_t edges)
    : LearnedEnergyReverseWorkspace(nodes, edges, nodes) {}

LearnedEnergyReverseWorkspace::LearnedEnergyReverseWorkspace(
    std::size_t nodes, std::size_t edges, std::size_t head_nodes)
    : embedding("so3lr_chain_embedding", nodes * inv_width),
      initial_ev("so3lr_chain_initial_ev", nodes * ev_width),
      energy_final_ev_gradient("so3lr_chain_energy_final_ev_gradient",
                               nodes * ev_width),
      owned_head_input("so3lr_chain_owned_head_input",
                       head_nodes * inv_width),
      expanded_head_input_gradient(
          "so3lr_chain_expanded_head_input_gradient",
          nodes * inv_width),
      synchronized_inv{
          DoubleView("so3lr_chain_synchronized_inv_block0",
                     nodes * inv_width),
          DoubleView("so3lr_chain_synchronized_inv_block1",
                     nodes * inv_width)},
      synchronized_ev{
          DoubleView("so3lr_chain_synchronized_ev_block0",
                     nodes * ev_width),
          DoubleView("so3lr_chain_synchronized_ev_block1",
                     nodes * ev_width)},
      synchronized_grad_inv("so3lr_chain_synchronized_grad_inv",
                            nodes * inv_width),
      synchronized_grad_ev("so3lr_chain_synchronized_grad_ev",
                           nodes * ev_width),
      owned_head_atomic_numbers(
          "so3lr_chain_owned_head_atomic_numbers", head_nodes),
      blocks(make_shared_reverse_blocks(nodes, edges)),
      heads(head_nodes),
      grad_distances("so3lr_chain_grad_distances", edges),
      grad_sh("so3lr_chain_grad_sh", edges * ev_width) {
  if (nodes == 0 || edges == 0 || head_nodes == 0 || head_nodes > nodes)
    throw std::runtime_error(
        "SO3LR learned-energy reverse workspace empty or invalid");
  shared_reverse_scratch_verified = verify_shared_reverse_scratch(blocks);
  if (!shared_reverse_scratch_verified)
    throw std::runtime_error("SO3LR dev_4 shared reverse scratch contract failed");
}

KokkosLearnedEnergyReverseChain::KokkosLearnedEnergyReverseChain(
    const NativeModel &model)
    : blocks_{KokkosCompleteTransformerReverseBlock(model, 0),
              KokkosCompleteTransformerReverseBlock(model, 1),
              KokkosCompleteTransformerReverseBlock(model, 2)},
      heads_(model),
      embedding_weight_("so3lr_chain_embedding_weight", inv_width * 118),
      inverse_embedding_scale_(
          1.0 / model.model_number("embedding_scale")) {
  // Optional operators that live in the chain rather than in the shared
  // per-block kernels are implemented only in the MPI path
  // (KokkosLocalEnergyReverse, used by pair_style so3lr/native/mpi). Refuse
  // here rather than silently computing a different model.
  const So3lrArchitecture &arch = model.arch();
  if (arch.layers != 3)
    throw std::runtime_error(
        "SO3LR: the single-rank chain supports exactly three interaction "
        "blocks; use pair_style so3lr/native/mpi for other depths");
  if (arch.use_residual_scalars || arch.has_head(HeadKind::C6Ratios) ||
      arch.repulsion == RepulsionKind::NlhTable)
    throw std::runtime_error(
        "SO3LR: this model uses residual scalars, a C6 head or NLH repulsion, "
        "which are implemented only in pair_style so3lr/native/mpi");
  if (arch.invariant_width != 128 || arch.rbf_width != 32 ||
      arch.degrees_duplicated || arch.head_hidden_width != 128)
    throw std::runtime_error(
        "SO3LR: the single-rank chain supports only the SO3LR v1 widths "
        "(128 features, 32 radial functions, degrees 1..4); use pair_style "
        "so3lr/native/mpi for " + arch.summary());
  const auto &weight =
      model.tensor("model.inv_feature_embedding.embedding.weight");
  if (weight.dtype != "float64" ||
      weight.shape != std::vector<std::size_t>({inv_width, 118}))
    throw std::runtime_error("SO3LR chain embedding contract changed");
  if (!std::isfinite(inverse_embedding_scale_) ||
      inverse_embedding_scale_ <= 0.0)
    throw std::runtime_error(
        "SO3LR active-chain embedding scale must be positive and finite");
  fill_view(embedding_weight_, inv_width * 118,
            [&](std::size_t i) { return model.float64(weight, i); });
  persistent_device_bytes_ = inv_width * 118 * sizeof(double) +
                             heads_.persistent_device_bytes();
  shared_kokkos_stream_ = heads_.shared_kokkos_stream();
  chain_contract_verified_ = heads_.checkpoint_contract_verified();
  for (std::size_t block = 0; block < blocks_.size(); ++block) {
    persistent_device_bytes_ += blocks_[block].persistent_device_bytes();
    shared_kokkos_stream_ =
        shared_kokkos_stream_ && blocks_[block].shared_kokkos_stream();
    chain_contract_verified_ =
        chain_contract_verified_ &&
        blocks_[block].checkpoint_contract_verified() &&
        blocks_[block].block_index() == block;
  }
  chain_contract_verified_ =
      chain_contract_verified_ && shared_kokkos_stream_;
  if (!chain_contract_verified_)
    throw std::runtime_error("SO3LR learned-energy reverse chain failed");
  Kokkos::fence();
}

void KokkosLearnedEnergyReverseChain::launch_forward_device(
    const Int64View &atomic_numbers, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers,
    const LearnedEnergyReverseWorkspace &workspace) const {
  launch_forward_owned_device(atomic_numbers, atomic_numbers.extent(0),
                              distances, sh_vectors, senders, receivers,
                              workspace);
}

void KokkosLearnedEnergyReverseChain::launch_forward_owned_device(
    const Int64View &atomic_numbers, std::size_t owned_nodes,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const LearnedEnergyReverseWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  IndexView identity("so3lr_chain_identity_periodic_owners", nodes);
  Kokkos::parallel_for(
      "so3lr_chain_fill_identity_periodic_owners",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) { identity(node) = node; });
  launch_forward_periodic_device(
      atomic_numbers, owned_nodes, identity, distances, sh_vectors, senders,
      receivers, workspace);
}

void KokkosLearnedEnergyReverseChain::launch_forward_periodic_device(
    const Int64View &atomic_numbers, std::size_t owned_nodes,
    const IndexView &node_owners, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers,
    const LearnedEnergyReverseWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  const std::size_t edges = distances.extent(0);
  if (nodes == 0 || owned_nodes == 0 || owned_nodes > nodes || edges == 0 ||
      node_owners.extent(0) != nodes ||
      sh_vectors.extent(0) != edges * ev_width ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      workspace.embedding.extent(0) != nodes * inv_width ||
      workspace.initial_ev.extent(0) != nodes * ev_width ||
      workspace.heads.atomic_energies.extent(0) != owned_nodes)
    throw std::runtime_error("SO3LR learned-energy periodic forward mismatch");
  const auto weight = embedding_weight_;
  const double inverse_embedding_scale = inverse_embedding_scale_;
  const auto embedding = workspace.embedding;
  Kokkos::parallel_for(
      "so3lr_chain_atomic_embedding",
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
  Kokkos::deep_copy(workspace.initial_ev, 0.0);
  Kokkos::deep_copy(workspace.energy_final_ev_gradient, 0.0);
  blocks_[0].launch_forward_device(
      workspace.embedding, workspace.initial_ev, distances, sh_vectors,
      senders, receivers, workspace.blocks[0]);
  synchronize_periodic_features(
      blocks_[0].final_inv(workspace.blocks[0]),
      blocks_[0].final_ev(workspace.blocks[0]), node_owners, nodes,
      workspace.synchronized_inv[0], workspace.synchronized_ev[0]);
  blocks_[1].launch_forward_device(
      workspace.synchronized_inv[0], workspace.synchronized_ev[0], distances,
      sh_vectors, senders, receivers, workspace.blocks[1]);
  synchronize_periodic_features(
      blocks_[1].final_inv(workspace.blocks[1]),
      blocks_[1].final_ev(workspace.blocks[1]), node_owners, nodes,
      workspace.synchronized_inv[1], workspace.synchronized_ev[1]);
  blocks_[2].launch_forward_device(
      workspace.synchronized_inv[1], workspace.synchronized_ev[1], distances,
      sh_vectors, senders, receivers, workspace.blocks[2]);
  const auto final_inv = blocks_[2].final_inv(workspace.blocks[2]);
  const auto owned_inv = workspace.owned_head_input;
  const auto owned_z = workspace.owned_head_atomic_numbers;
  Kokkos::parallel_for(
      "so3lr_chain_pack_owned_head_input",
      Kokkos::RangePolicy<>(0, owned_nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) { owned_inv(i) = final_inv(i); });
  Kokkos::parallel_for(
      "so3lr_chain_pack_owned_head_z", Kokkos::RangePolicy<>(0, owned_nodes),
      KOKKOS_LAMBDA(const std::size_t i) { owned_z(i) = atomic_numbers(i); });
  heads_.launch_device(owned_inv, owned_z, 0.0, workspace.heads);
}

void KokkosLearnedEnergyReverseChain::launch_reverse_device(
    const Int64View &atomic_numbers, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers,
    const LearnedEnergyReverseWorkspace &workspace) const {
  heads_.launch_energy_reverse_device(atomic_numbers, workspace.heads);
  blocks_[2].launch_reverse_device(
      blocks_[1].final_inv(workspace.blocks[1]),
      blocks_[1].final_ev(workspace.blocks[1]), distances, sh_vectors,
      senders, receivers, workspace.heads.energy_input_grad,
      workspace.energy_final_ev_gradient, workspace.blocks[2]);
  blocks_[1].launch_reverse_device(
      blocks_[0].final_inv(workspace.blocks[0]),
      blocks_[0].final_ev(workspace.blocks[0]), distances, sh_vectors,
      senders, receivers, blocks_[2].grad_inv(workspace.blocks[2]),
      blocks_[2].grad_ev(workspace.blocks[2]), workspace.blocks[1]);
  blocks_[0].launch_reverse_device(
      workspace.embedding, workspace.initial_ev, distances, sh_vectors,
      senders, receivers, blocks_[1].grad_inv(workspace.blocks[1]),
      blocks_[1].grad_ev(workspace.blocks[1]), workspace.blocks[0]);

  const auto grad_distance = workspace.grad_distances;
  const auto grad_sh = workspace.grad_sh;
  const auto d0 = blocks_[0].grad_distances(workspace.blocks[0]);
  const auto d1 = blocks_[1].grad_distances(workspace.blocks[1]);
  const auto d2 = blocks_[2].grad_distances(workspace.blocks[2]);
  const auto s0 = blocks_[0].grad_sh(workspace.blocks[0]);
  const auto s1 = blocks_[1].grad_sh(workspace.blocks[1]);
  const auto s2 = blocks_[2].grad_sh(workspace.blocks[2]);
  Kokkos::parallel_for(
      "so3lr_chain_accumulate_distance", Kokkos::RangePolicy<>(0, distances.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        grad_distance(i) = d0(i) + d1(i) + d2(i);
      });
  Kokkos::parallel_for(
      "so3lr_chain_accumulate_sh", Kokkos::RangePolicy<>(0, sh_vectors.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        grad_sh(i) = s0(i) + s1(i) + s2(i);
      });
}

void KokkosLearnedEnergyReverseChain::launch_multihead_reverse_device(
    const Int64View &atomic_numbers, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers, const DoubleView &energy_seeds,
    const DoubleView &partial_charge_seeds,
    const DoubleView &hirshfeld_seeds,
    const LearnedEnergyReverseWorkspace &workspace) const {
  launch_multihead_reverse_owned_device(
      atomic_numbers, atomic_numbers.extent(0), distances, sh_vectors,
      senders, receivers, energy_seeds, partial_charge_seeds,
      hirshfeld_seeds, workspace);
}

void KokkosLearnedEnergyReverseChain::launch_multihead_reverse_owned_device(
    const Int64View &atomic_numbers, std::size_t owned_nodes,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const DoubleView &energy_seeds,
    const DoubleView &partial_charge_seeds,
    const DoubleView &hirshfeld_seeds,
    const LearnedEnergyReverseWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  IndexView identity("so3lr_chain_identity_reverse_periodic_owners", nodes);
  Kokkos::parallel_for(
      "so3lr_chain_fill_identity_reverse_periodic_owners",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) { identity(node) = node; });
  launch_multihead_reverse_periodic_device(
      atomic_numbers, owned_nodes, identity, distances, sh_vectors, senders,
      receivers, energy_seeds, partial_charge_seeds, hirshfeld_seeds,
      workspace);
}

void KokkosLearnedEnergyReverseChain::launch_multihead_reverse_periodic_device(
    const Int64View &atomic_numbers, std::size_t owned_nodes,
    const IndexView &node_owners, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers, const DoubleView &energy_seeds,
    const DoubleView &partial_charge_seeds,
    const DoubleView &hirshfeld_seeds,
    const LearnedEnergyReverseWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  if (owned_nodes == 0 || owned_nodes > nodes ||
      node_owners.extent(0) != nodes || energy_seeds.extent(0) != owned_nodes ||
      partial_charge_seeds.extent(0) != owned_nodes ||
      hirshfeld_seeds.extent(0) != owned_nodes ||
      workspace.expanded_head_input_gradient.extent(0) != nodes * inv_width ||
      workspace.synchronized_grad_inv.extent(0) != nodes * inv_width ||
      workspace.synchronized_grad_ev.extent(0) != nodes * ev_width)
    throw std::runtime_error("SO3LR learned-energy periodic reverse mismatch");
  // dev_3 is a diagnostic build.  One call in twenty is fenced at the
  // real reverse-chain boundaries so asynchronous CUDA work is attributed to
  // the correct head, transformer block, periodic sync, or accumulation phase.
  static std::size_t reverse_profile_call_counter = 0;
  const std::size_t reverse_profile_call = ++reverse_profile_call_counter;
  constexpr std::size_t reverse_profile_every = 20;
  const bool reverse_profile_this_call =
      ((reverse_profile_call - 1) % reverse_profile_every) == 0;
  Kokkos::Timer reverse_profile_timer;
  double reverse_heads_ms = 0.0;
  double reverse_expand_ms = 0.0;
  double reverse_block2_ms = 0.0;
  double reverse_sync2_ms = 0.0;
  double reverse_block1_ms = 0.0;
  double reverse_sync1_ms = 0.0;
  double reverse_block0_ms = 0.0;
  double reverse_accumulate_ms = 0.0;
  const auto reverse_profile_mark = [&](double &milliseconds) {
    if (!reverse_profile_this_call) return;
    Kokkos::fence();
    milliseconds = reverse_profile_timer.seconds() * 1000.0;
    reverse_profile_timer.reset();
  };
  if (reverse_profile_this_call) {
    Kokkos::fence();
    reverse_profile_timer.reset();
  }

  heads_.launch_multihead_reverse_device(
      workspace.owned_head_atomic_numbers, energy_seeds,
      partial_charge_seeds, hirshfeld_seeds, workspace.heads);
  reverse_profile_mark(reverse_heads_ms);
  Kokkos::deep_copy(workspace.expanded_head_input_gradient, 0.0);
  const auto expanded = workspace.expanded_head_input_gradient;
  const auto owned_gradient = workspace.heads.combined_input_grad;
  Kokkos::parallel_for(
      "so3lr_chain_expand_owned_head_gradient",
      Kokkos::RangePolicy<>(0, owned_nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) { expanded(i) = owned_gradient(i); });
  reverse_profile_mark(reverse_expand_ms);
  blocks_[2].launch_reverse_device(
      workspace.synchronized_inv[1], workspace.synchronized_ev[1], distances,
      sh_vectors, senders, receivers, workspace.expanded_head_input_gradient,
      workspace.energy_final_ev_gradient, workspace.blocks[2]);
  reverse_profile_mark(reverse_block2_ms);
  reverse_periodic_feature_sync(
      blocks_[2].grad_inv(workspace.blocks[2]),
      blocks_[2].grad_ev(workspace.blocks[2]), node_owners, owned_nodes, nodes,
      workspace.synchronized_grad_inv, workspace.synchronized_grad_ev);
  reverse_profile_mark(reverse_sync2_ms);
  blocks_[1].launch_reverse_device(
      workspace.synchronized_inv[0], workspace.synchronized_ev[0], distances,
      sh_vectors, senders, receivers, workspace.synchronized_grad_inv,
      workspace.synchronized_grad_ev, workspace.blocks[1]);
  reverse_profile_mark(reverse_block1_ms);
  reverse_periodic_feature_sync(
      blocks_[1].grad_inv(workspace.blocks[1]),
      blocks_[1].grad_ev(workspace.blocks[1]), node_owners, owned_nodes, nodes,
      workspace.synchronized_grad_inv, workspace.synchronized_grad_ev);
  reverse_profile_mark(reverse_sync1_ms);
  blocks_[0].launch_reverse_device(
      workspace.embedding, workspace.initial_ev, distances, sh_vectors,
      senders, receivers, workspace.synchronized_grad_inv,
      workspace.synchronized_grad_ev, workspace.blocks[0]);
  reverse_profile_mark(reverse_block0_ms);

  const auto grad_distance = workspace.grad_distances;
  const auto grad_sh = workspace.grad_sh;
  const auto d0 = blocks_[0].grad_distances(workspace.blocks[0]);
  const auto d1 = blocks_[1].grad_distances(workspace.blocks[1]);
  const auto d2 = blocks_[2].grad_distances(workspace.blocks[2]);
  const auto s0 = blocks_[0].grad_sh(workspace.blocks[0]);
  const auto s1 = blocks_[1].grad_sh(workspace.blocks[1]);
  const auto s2 = blocks_[2].grad_sh(workspace.blocks[2]);
  Kokkos::parallel_for(
      "so3lr_chain_multihead_accumulate_distance",
      Kokkos::RangePolicy<>(0, distances.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        grad_distance(i) = d0(i) + d1(i) + d2(i);
      });
  Kokkos::parallel_for(
      "so3lr_chain_multihead_accumulate_sh",
      Kokkos::RangePolicy<>(0, sh_vectors.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        grad_sh(i) = s0(i) + s1(i) + s2(i);
      });
  reverse_profile_mark(reverse_accumulate_ms);
  if (reverse_profile_this_call) {
    const double reverse_total_ms =
        reverse_heads_ms + reverse_expand_ms + reverse_block2_ms +
        reverse_sync2_ms + reverse_block1_ms + reverse_sync1_ms +
        reverse_block0_ms + reverse_accumulate_ms;
    const char *rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if (rank == nullptr) rank = std::getenv("SLURM_PROCID");
    if (rank == nullptr) rank = "0";
    std::fprintf(
        stdout,
        "SO3LR_NATIVE_SR_REVERSE_PROFILE rank=%s call=%zu owned=%zu "
        "nodes=%zu edges=%zu heads_ms=%.9g expand_ms=%.9g "
        "block2_ms=%.9g sync2_ms=%.9g block1_ms=%.9g sync1_ms=%.9g "
        "block0_ms=%.9g accumulate_ms=%.9g total_ms=%.9g\n",
        rank, reverse_profile_call, owned_nodes, nodes, distances.extent(0),
        reverse_heads_ms, reverse_expand_ms, reverse_block2_ms,
        reverse_sync2_ms, reverse_block1_ms, reverse_sync1_ms,
        reverse_block0_ms, reverse_accumulate_ms, reverse_total_ms);
    std::fflush(stdout);
  }
}

const KokkosLearnedEnergyReverseChain::DoubleView &
KokkosLearnedEnergyReverseChain::block_grad_distances(
    const LearnedEnergyReverseWorkspace &workspace,
    std::size_t block_index) const {
  if (block_index >= blocks_.size())
    throw std::runtime_error("SO3LR chain block index out of range");
  return blocks_[block_index].grad_distances(workspace.blocks[block_index]);
}

const KokkosLearnedEnergyReverseChain::DoubleView &
KokkosLearnedEnergyReverseChain::block_grad_sh(
    const LearnedEnergyReverseWorkspace &workspace,
    std::size_t block_index) const {
  if (block_index >= blocks_.size())
    throw std::runtime_error("SO3LR chain block index out of range");
  return blocks_[block_index].grad_sh(workspace.blocks[block_index]);
}

LearnedEnergyReverseBenchmark KokkosLearnedEnergyReverseChain::benchmark(
    std::size_t nodes, std::size_t edges,
    std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR chain benchmark invalid");
  Int64View z("so3lr_chain_bench_z", nodes);
  DoubleView distances("so3lr_chain_bench_distances", edges);
  DoubleView sh("so3lr_chain_bench_sh", edges * ev_width);
  IndexView senders("so3lr_chain_bench_senders", edges);
  IndexView receivers("so3lr_chain_bench_receivers", edges);
  Kokkos::parallel_for(
      "so3lr_chain_bench_nodes", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  Kokkos::parallel_for(
      "so3lr_chain_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        distances(edge) =
            0.15 + 4.29 * static_cast<double>(edge % 8192) / 8191.0;
        receivers(edge) = edge % nodes;
        senders(edge) = (edge * 17 + 11) % nodes;
      });
  Kokkos::parallel_for(
      "so3lr_chain_bench_sh", Kokkos::RangePolicy<>(0, sh.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });
  LearnedEnergyReverseWorkspace workspace(nodes, edges);
  launch_forward_device(z, distances, sh, senders, receivers, workspace);
  launch_reverse_device(z, distances, sh, senders, receivers, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_forward_device(z, distances, sh, senders, receivers, workspace);
  Kokkos::fence();
  const double forward_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  launch_forward_device(z, distances, sh, senders, receivers, workspace);
  Kokkos::fence();
  timer.reset();
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_reverse_device(z, distances, sh, senders, receivers, workspace);
  Kokkos::fence();
  const double reverse_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  timer.reset();
  for (std::size_t i = 0; i < repetitions; ++i) {
    launch_forward_device(z, distances, sh, senders, receivers, workspace);
    launch_reverse_device(z, distances, sh, senders, receivers, workspace);
  }
  Kokkos::fence();
  const double integrated_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto grad_distance = workspace.grad_distances;
  const auto grad_sh = workspace.grad_sh;
  const auto energy = workspace.heads.atomic_energies;
  Kokkos::parallel_reduce(
      "so3lr_chain_bench_checksum", Kokkos::RangePolicy<>(0, 3),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += energy(nodes / 3);
        if (which == 1) update += grad_distance(edges / 2);
        if (which == 2) update += grad_sh((edges / 4) * ev_width + 7);
      },
      checksum);
  Kokkos::fence();
  LearnedEnergyReverseBenchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes =
      chain_workspace_bytes(nodes, edges, persistent_device_bytes_);
  result.host_boundary_bytes_per_iteration = 0;
  result.forward_milliseconds = forward_ms;
  result.reverse_milliseconds = reverse_ms;
  result.forward_reverse_milliseconds = integrated_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (integrated_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
