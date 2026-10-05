#include "so3lr/kokkos_attention_scatter_reverse.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using IndexView = Kokkos::View<std::size_t *>;

// Fixed at 32 for every released model (capability-enforced) so the
// receiver-group kernels can keep a 32-entry register accumulator. The head
// count and the feature widths are taken from the descriptor at run time.
constexpr std::size_t head_width = 32;

template <class View, class Reader>
void fill_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

struct AttentionContract {
  double norm_inv = 0.0;
  double norm_ev = 0.0;
  bool found = false;
};

AttentionContract load_attention_contract(const NativeModel &model,
                                         std::size_t block_index) {
  (void)block_index;
  AttentionContract result;
  result.norm_inv = model.arch().attention_norm_value;
  result.norm_ev = model.arch().attention_norm_value;
  result.found = true;
  return result;
}

void validate_views(const std::array<DoubleView, 5> &qkv,
                    const DoubleView &filter_inv,
                    const DoubleView &filter_ev, const DoubleView &sh,
                    const DoubleView &cutoff, const IndexView &senders,
                    const IndexView &receivers,
                    const DoubleView &grad_invariant_update,
                    const DoubleView &grad_equivariant_update,
                    const AttentionScatterReverseDeviceWorkspace &workspace,
                    const ArchDims &dims) {
  const std::size_t heads = dims.inv_heads;
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t equivariant_width = dims.equivariant_width;
  const std::size_t edges = senders.extent(0);
  if (edges == 0 || receivers.extent(0) != edges ||
      cutoff.extent(0) != edges ||
      filter_inv.extent(0) != edges * feature_width ||
      filter_ev.extent(0) != edges * feature_width ||
      sh.extent(0) != edges * equivariant_width)
    throw std::runtime_error("SO3LR attention/scatter edge view mismatch");
  if (qkv[0].extent(0) == 0 || qkv[0].extent(0) % feature_width != 0)
    throw std::runtime_error("SO3LR attention/scatter node view mismatch");
  const std::size_t nodes = qkv[0].extent(0) / feature_width;
  for (std::size_t i = 1; i < qkv.size(); ++i)
    if (qkv[i].extent(0) != nodes * feature_width)
      throw std::runtime_error("SO3LR attention/scatter QKV mismatch");
  if (grad_invariant_update.extent(0) != nodes * feature_width ||
      grad_equivariant_update.extent(0) != nodes * equivariant_width ||
      workspace.invariant_update.extent(0) != nodes * feature_width ||
      workspace.equivariant_update.extent(0) != nodes * equivariant_width ||
      workspace.grad_filter_inv.extent(0) != edges * feature_width ||
      workspace.grad_filter_ev.extent(0) != edges * feature_width ||
      workspace.grad_sh.extent(0) != edges * equivariant_width ||
      workspace.grad_cutoff.extent(0) != edges ||
      workspace.grad_cutoff_heads.extent(0) != edges * heads)
    throw std::runtime_error("SO3LR attention/scatter workspace mismatch");
  for (const auto &gradient : workspace.grad_qkv)
    if (gradient.extent(0) != nodes * feature_width)
      throw std::runtime_error("SO3LR attention/scatter QKV gradient mismatch");
}

std::size_t complete_workspace_bytes(std::size_t nodes, std::size_t edges,
                                     std::size_t persistent_bytes,
                                     const ArchDims &dims) {
  const std::size_t heads = dims.inv_heads;
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t equivariant_width = dims.equivariant_width;
  // Inputs: 5 node projections; upstream and forward node updates; output
  // gradients: 5 node projections. Edge inputs and their gradients are kept
  // explicitly because dev_19 consumes every one of these device views.
  const std::size_t node_doubles =
      5 * feature_width + 2 * (feature_width + equivariant_width) +
      5 * feature_width;
  const std::size_t edge_doubles =
      2 * (2 * feature_width + equivariant_width + 1) + heads;
  return persistent_bytes + nodes * node_doubles * sizeof(double) +
         edges * edge_doubles * sizeof(double) +
         2 * edges * sizeof(std::size_t);
}

}  // namespace

AttentionScatterReverseDeviceWorkspace::AttentionScatterReverseDeviceWorkspace(
    std::size_t nodes, std::size_t edges, const ArchDims &dims)
    : AttentionScatterReverseDeviceWorkspace(nodes, edges, dims.invariant_width,
                                             dims.equivariant_width,
                                             dims.inv_heads) {}

AttentionScatterReverseDeviceWorkspace::AttentionScatterReverseDeviceWorkspace(
    std::size_t nodes, std::size_t edges, std::size_t feature_width,
    std::size_t equivariant_width, std::size_t heads)
    : invariant_update("so3lr_att_reverse_invariant_update",
                       nodes * feature_width),
      equivariant_update("so3lr_att_reverse_equivariant_update",
                         nodes * equivariant_width),
      grad_qkv{DoubleView("so3lr_att_reverse_grad_q_inv",
                          nodes * feature_width),
               DoubleView("so3lr_att_reverse_grad_k_inv",
                          nodes * feature_width),
               DoubleView("so3lr_att_reverse_grad_v_inv",
                          nodes * feature_width),
               DoubleView("so3lr_att_reverse_grad_q_ev",
                          nodes * feature_width),
               DoubleView("so3lr_att_reverse_grad_k_ev",
                          nodes * feature_width)},
      grad_filter_inv("so3lr_att_reverse_grad_filter_inv",
                      edges * feature_width),
      grad_filter_ev("so3lr_att_reverse_grad_filter_ev",
                     edges * feature_width),
      grad_sh("so3lr_att_reverse_grad_sh", edges * equivariant_width),
      grad_cutoff("so3lr_att_reverse_grad_cutoff", edges),
      grad_cutoff_heads("so3lr_att_reverse_grad_cutoff_heads",
                        edges * heads) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR attention/scatter reverse workspace empty");
}


AttentionScatterReverseDeviceWorkspace::AttentionScatterReverseDeviceWorkspace(
    std::size_t nodes, std::size_t edges,
    const AttentionScatterReverseDeviceWorkspace &shared)
    // Every width is recovered from the shared views, so the sharing
    // constructor needs no descriptor.
    : invariant_update(shared.invariant_update),
      equivariant_update(shared.equivariant_update),
      grad_qkv(shared.grad_qkv),
      grad_filter_inv(shared.grad_filter_inv),
      grad_filter_ev(shared.grad_filter_ev),
      // Geometry gradients remain block-specific because the chain sums all
      // three after reverse traversal.  Everything else is sequential scratch.
      grad_sh("so3lr_att_reverse_block_geometry_grad_sh",
              edges * (shared.grad_sh.extent(0) /
                       std::max<std::size_t>(shared.grad_cutoff.extent(0), 1))),
      grad_cutoff(shared.grad_cutoff),
      grad_cutoff_heads(shared.grad_cutoff_heads) {
  if (nodes == 0 || edges == 0 || shared.grad_cutoff.extent(0) != edges ||
      shared.grad_qkv[0].extent(0) % nodes != 0 ||
      shared.grad_filter_inv.extent(0) !=
          edges * (shared.grad_qkv[0].extent(0) / nodes))
    throw std::runtime_error("SO3LR shared attention reverse scratch mismatch");
}

KokkosAttentionScatterReverse::KokkosAttentionScatterReverse(
    const NativeModel &model, std::size_t block_index)
    : dims_(model.arch().dims()), block_index_(block_index) {
  const std::size_t heads = dims_.inv_heads;
  const auto blocks = static_cast<std::size_t>(
      model.architecture_number("interaction_blocks"));
  if (block_index_ >= blocks || dims_.ev_heads != heads ||
      dims_.inv_head_width != head_width ||
      dims_.ev_head_width != head_width ||
      model.architecture_number("attention_heads") != heads ||
      model.architecture_number("attention_head_width") != head_width ||
      model.architecture_number("euclidean_degree_channels") != heads)
    throw std::runtime_error("SO3LR attention/scatter architecture mismatch");
  const auto contract = load_attention_contract(model, block_index_);
  if (!contract.found || !std::isfinite(contract.norm_inv) ||
      !std::isfinite(contract.norm_ev) || contract.norm_inv <= 0.0 ||
      contract.norm_ev <= 0.0)
    throw std::runtime_error("SO3LR attention normalization contract missing");
  attention_norm_inv_ = contract.norm_inv;
  attention_norm_ev_ = contract.norm_ev;

  const std::string key =
      "model.euclidean_transformers." + std::to_string(block_index_) +
      ".euclidean_attention_block.degree_repeats";
  const auto &tensor = model.tensor(key);
  if (tensor.dtype != "int64" ||
      tensor.shape != std::vector<std::size_t>{heads})
    throw std::runtime_error("SO3LR attention degree-repeat mismatch");
  const auto &arch = model.arch();
  for (std::size_t head = 0; head < heads; ++head) {
    const auto value = model.int64(tensor, head);
    if (value <= 0 ||
        static_cast<std::size_t>(value) != arch.degree_repeats[head])
      throw std::runtime_error("SO3LR attention degree repeats changed");
  }
  degree_repeats_ = IndexView("so3lr_att_reverse_degree_repeats", heads);
  degree_offsets_ = IndexView("so3lr_att_reverse_degree_offsets", heads);
  fill_view(degree_repeats_, heads,
            [&](std::size_t i) { return arch.degree_repeats[i]; });
  fill_view(degree_offsets_, heads,
            [&](std::size_t i) { return arch.degree_offsets[i]; });
  persistent_device_bytes_ = 2 * heads * sizeof(std::size_t);
  checkpoint_contract_verified_ = true;
  Kokkos::fence();
}

void KokkosAttentionScatterReverse::launch_forward_device(
    const std::array<DoubleView, 5> &qkv, const DoubleView &filter_inv,
    const DoubleView &filter_ev, const DoubleView &sh,
    const DoubleView &cutoff, const IndexView &senders,
    const IndexView &receivers,
    const AttentionScatterReverseDeviceWorkspace &workspace) const {
  const std::size_t heads = dims_.inv_heads;
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  validate_views(qkv, filter_inv, filter_ev, sh, cutoff, senders, receivers,
                 workspace.invariant_update, workspace.equivariant_update,
                 workspace, dims_);
  Kokkos::deep_copy(workspace.invariant_update, 0.0);
  Kokkos::deep_copy(workspace.equivariant_update, 0.0);
  const auto q_inv = qkv[0];
  const auto k_inv = qkv[1];
  const auto v_inv = qkv[2];
  const auto q_ev = qkv[3];
  const auto k_ev = qkv[4];
  const auto repeats = degree_repeats_;
  const auto offsets = degree_offsets_;
  const auto invariant_update = workspace.invariant_update;
  const auto equivariant_update = workspace.equivariant_update;
  const double norm_inv = attention_norm_inv_;
  const double norm_ev = attention_norm_ev_;
  const std::size_t edges = senders.extent(0);
  Kokkos::parallel_for(
      "so3lr_attention_scatter_forward_dev18",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / heads;
        const std::size_t head = flat % heads;
        const std::size_t sender = senders(edge);
        const std::size_t receiver = receivers(edge);
        const std::size_t head_offset = head * head_width;
        const std::size_t edge_offset = edge * feature_width + head_offset;
        const std::size_t q_offset =
            receiver * feature_width + head_offset;
        const std::size_t kv_offset = sender * feature_width + head_offset;
        double alpha_inv = 0.0;
        double alpha_ev = 0.0;
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          alpha_inv += q_inv(q_offset + channel) *
                       k_inv(kv_offset + channel) *
                       filter_inv(edge_offset + channel);
          alpha_ev += q_ev(q_offset + channel) *
                      k_ev(kv_offset + channel) *
                      filter_ev(edge_offset + channel);
        }
        alpha_inv /= norm_inv;
        alpha_ev /= norm_ev;
        const double edge_cutoff = cutoff(edge);
        for (std::size_t channel = 0; channel < head_width; ++channel)
          Kokkos::atomic_add(
              &invariant_update(q_offset + channel),
              edge_cutoff * alpha_inv * v_inv(kv_offset + channel));
        for (std::size_t local = 0; local < repeats(head); ++local) {
          const std::size_t channel = offsets(head) + local;
          Kokkos::atomic_add(
              &equivariant_update(
                  receiver * equivariant_width + channel),
              edge_cutoff * alpha_ev *
                  sh(edge * equivariant_width + channel));
        }
      });
}

void KokkosAttentionScatterReverse::launch_reverse_device(
    const std::array<DoubleView, 5> &qkv, const DoubleView &filter_inv,
    const DoubleView &filter_ev, const DoubleView &sh,
    const DoubleView &cutoff, const IndexView &senders,
    const IndexView &receivers, const DoubleView &grad_invariant_update,
    const DoubleView &grad_equivariant_update,
    const AttentionScatterReverseDeviceWorkspace &workspace) const {
  const std::size_t heads = dims_.inv_heads;
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  validate_views(qkv, filter_inv, filter_ev, sh, cutoff, senders, receivers,
                 grad_invariant_update, grad_equivariant_update, workspace,
                 dims_);
  const auto q_inv = qkv[0];
  const auto k_inv = qkv[1];
  const auto v_inv = qkv[2];
  const auto q_ev = qkv[3];
  const auto k_ev = qkv[4];
  const auto grad_q_inv = workspace.grad_qkv[0];
  const auto grad_k_inv = workspace.grad_qkv[1];
  const auto grad_v_inv = workspace.grad_qkv[2];
  const auto grad_q_ev = workspace.grad_qkv[3];
  const auto grad_k_ev = workspace.grad_qkv[4];

  // SO3LR_STAGE3_OPTIM_DEV9_REVERSE_ZERO_ELISION
  // The five Q/K/V node adjoints are true atomic accumulators and must start
  // from zero. Clear them in one coalesced launch instead of five separate
  // device fills. The three large filter/sh edge-gradient outputs below are
  // assigned exactly once for every active edge/channel; dev_11 separately
  // handles the cutoff accumulator and padded tail without full-array clears.
  const std::size_t node_gradient_values = grad_q_inv.extent(0);
  Kokkos::parallel_for(
      "so3lr_attention_reverse_zero_node_adjoints_dev9",
      Kokkos::RangePolicy<>(0, node_gradient_values),
      KOKKOS_LAMBDA(const std::size_t index) {
        grad_q_inv(index) = 0.0;
        grad_k_inv(index) = 0.0;
        grad_v_inv(index) = 0.0;
        grad_q_ev(index) = 0.0;
        grad_k_ev(index) = 0.0;
      });
  const auto grad_filter_inv = workspace.grad_filter_inv;
  const auto grad_filter_ev = workspace.grad_filter_ev;
  const auto grad_sh = workspace.grad_sh;
  const auto grad_cutoff = workspace.grad_cutoff;
  const auto grad_cutoff_heads = workspace.grad_cutoff_heads;
  const auto repeats = degree_repeats_;
  const auto offsets = degree_offsets_;
  const double norm_inv = attention_norm_inv_;
  const double norm_ev = attention_norm_ev_;
  const std::size_t edges = senders.extent(0);

  // SO3LR_STAGE3_OPTIM_DEV11_RECEIVER_GROUPED_REVERSE
  // The LAMMPS full-neighbour graph and its stable exact-cutoff compaction
  // keep all edges of a receiver contiguous.  Accumulate each receiver/head
  // Q adjoint locally across that contiguous group and issue only one atomic
  // add per channel and group.  For a non-grouped generic graph, every
  // contiguous group still contributes atomically and the result remains
  // correct. Sender K/V adjoints retain their original atomic ownership.
  // SO3LR_STAGE3_OPTIM_DEV12_CUTOFF_PARTIAL_REDUCTION
  // Each receiver-group/head work item owns one edge/head cutoff partial.
  // Store the invariant and equivariant contributions in that unique slot,
  // then reduce the four heads once per edge. This removes eight contended
  // cutoff atomics per edge and the full cutoff-array clear.
  Kokkos::parallel_for(
      "so3lr_attention_reverse_receiver_inv_dev11",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t first_edge = flat / heads;
        const std::size_t head = flat % heads;
        const std::size_t receiver = receivers(first_edge);
        if ((first_edge > 0 && receivers(first_edge - 1) == receiver) ||
            cutoff(first_edge) == 0.0)
          return;
        double receiver_gradient[32];
        for (std::size_t channel = 0; channel < head_width; ++channel)
          receiver_gradient[channel] = 0.0;
        const std::size_t head_offset = head * head_width;
        const std::size_t q_offset =
            receiver * feature_width + head_offset;
        for (std::size_t edge = first_edge;
             edge < edges && receivers(edge) == receiver &&
             cutoff(edge) != 0.0;
             ++edge) {
          const std::size_t sender = senders(edge);
          const double edge_cutoff = cutoff(edge);
          const std::size_t edge_offset =
              edge * feature_width + head_offset;
          const std::size_t kv_offset =
              sender * feature_width + head_offset;
          double alpha_inv = 0.0;
          for (std::size_t channel = 0; channel < head_width; ++channel)
            alpha_inv += q_inv(q_offset + channel) *
                         k_inv(kv_offset + channel) *
                         filter_inv(edge_offset + channel);
          alpha_inv /= norm_inv;
          double grad_alpha_inv = 0.0;
          double cutoff_gradient = 0.0;
          for (std::size_t channel = 0; channel < head_width; ++channel) {
            const std::size_t q_index = q_offset + channel;
            const std::size_t kv_index = kv_offset + channel;
            const double upstream = grad_invariant_update(q_index);
            grad_alpha_inv += upstream * edge_cutoff * v_inv(kv_index);
            cutoff_gradient += upstream * alpha_inv * v_inv(kv_index);
            Kokkos::atomic_add(&grad_v_inv(kv_index),
                               upstream * edge_cutoff * alpha_inv);
          }
          const double scaled = grad_alpha_inv / norm_inv;
          for (std::size_t channel = 0; channel < head_width; ++channel) {
            const std::size_t q_index = q_offset + channel;
            const std::size_t kv_index = kv_offset + channel;
            const std::size_t edge_index = edge_offset + channel;
            receiver_gradient[channel] +=
                scaled * k_inv(kv_index) * filter_inv(edge_index);
            Kokkos::atomic_add(
                &grad_k_inv(kv_index),
                scaled * q_inv(q_index) * filter_inv(edge_index));
            grad_filter_inv(edge_index) =
                scaled * q_inv(q_index) * k_inv(kv_index);
          }
          grad_cutoff_heads(edge * heads + head) = cutoff_gradient;
        }
        for (std::size_t channel = 0; channel < head_width; ++channel)
          Kokkos::atomic_add(&grad_q_inv(q_offset + channel),
                             receiver_gradient[channel]);
      });

  Kokkos::parallel_for(
      "so3lr_attention_reverse_receiver_ev_dev11",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t first_edge = flat / heads;
        const std::size_t head = flat % heads;
        const std::size_t receiver = receivers(first_edge);
        if ((first_edge > 0 && receivers(first_edge - 1) == receiver) ||
            cutoff(first_edge) == 0.0)
          return;
        double receiver_gradient[32];
        for (std::size_t channel = 0; channel < head_width; ++channel)
          receiver_gradient[channel] = 0.0;
        const std::size_t head_offset = head * head_width;
        const std::size_t q_offset =
            receiver * feature_width + head_offset;
        for (std::size_t edge = first_edge;
             edge < edges && receivers(edge) == receiver &&
             cutoff(edge) != 0.0;
             ++edge) {
          const std::size_t sender = senders(edge);
          const double edge_cutoff = cutoff(edge);
          const std::size_t edge_offset =
              edge * feature_width + head_offset;
          const std::size_t kv_offset =
              sender * feature_width + head_offset;
          double alpha_ev = 0.0;
          for (std::size_t channel = 0; channel < head_width; ++channel)
            alpha_ev += q_ev(q_offset + channel) *
                        k_ev(kv_offset + channel) *
                        filter_ev(edge_offset + channel);
          alpha_ev /= norm_ev;
          double grad_alpha_ev = 0.0;
          double cutoff_gradient = 0.0;
          for (std::size_t local = 0; local < repeats(head); ++local) {
            const std::size_t channel = offsets(head) + local;
            const std::size_t node_index =
                receiver * equivariant_width + channel;
            const std::size_t edge_index =
                edge * equivariant_width + channel;
            const double upstream = grad_equivariant_update(node_index);
            grad_alpha_ev += upstream * edge_cutoff * sh(edge_index);
            cutoff_gradient += upstream * alpha_ev * sh(edge_index);
            grad_sh(edge_index) = upstream * edge_cutoff * alpha_ev;
          }
          const double scaled = grad_alpha_ev / norm_ev;
          for (std::size_t channel = 0; channel < head_width; ++channel) {
            const std::size_t q_index = q_offset + channel;
            const std::size_t kv_index = kv_offset + channel;
            const std::size_t edge_index = edge_offset + channel;
            receiver_gradient[channel] +=
                scaled * k_ev(kv_index) * filter_ev(edge_index);
            Kokkos::atomic_add(
                &grad_k_ev(kv_index),
                scaled * q_ev(q_index) * filter_ev(edge_index));
            grad_filter_ev(edge_index) =
                scaled * q_ev(q_index) * k_ev(kv_index);
          }
          grad_cutoff_heads(edge * heads + head) += cutoff_gradient;
        }
        for (std::size_t channel = 0; channel < head_width; ++channel)
          Kokkos::atomic_add(&grad_q_ev(q_offset + channel),
                             receiver_gradient[channel]);
      });

  Kokkos::parallel_for(
      "so3lr_attention_reverse_cutoff_reduce_dev12",
      Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        if (cutoff(edge) == 0.0) {
          grad_cutoff(edge) = 0.0;
          return;
        }
        double gradient = 0.0;
        for (std::size_t head = 0; head < heads; ++head)
          gradient += grad_cutoff_heads(edge * heads + head);
        grad_cutoff(edge) = gradient;
      });

  // Group kernels skip the padded zero-cutoff tail. Clear only those dummy
  // edge outputs, without restoring dev_9's three large full-array clears.
  Kokkos::parallel_for(
      "so3lr_attention_reverse_dummy_tail_dev11",
      Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        if (cutoff(edge) != 0.0) return;
        for (std::size_t channel = 0; channel < feature_width; ++channel) {
          grad_filter_inv(edge * feature_width + channel) = 0.0;
          grad_filter_ev(edge * feature_width + channel) = 0.0;
        }
        for (std::size_t channel = 0; channel < equivariant_width; ++channel)
          grad_sh(edge * equivariant_width + channel) = 0.0;
      });
}

AttentionScatterReverseBenchmark KokkosAttentionScatterReverse::benchmark(
    std::size_t nodes, std::size_t edges,
    std::size_t repetitions) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR attention reverse benchmark invalid");
  std::array<DoubleView, 5> qkv = {
      DoubleView("so3lr_dev18_bench_q_inv", nodes * feature_width),
      DoubleView("so3lr_dev18_bench_k_inv", nodes * feature_width),
      DoubleView("so3lr_dev18_bench_v_inv", nodes * feature_width),
      DoubleView("so3lr_dev18_bench_q_ev", nodes * feature_width),
      DoubleView("so3lr_dev18_bench_k_ev", nodes * feature_width)};
  for (std::size_t which = 0; which < qkv.size(); ++which) {
    const auto view = qkv[which];
    Kokkos::parallel_for(
        "so3lr_dev18_bench_qkv", Kokkos::RangePolicy<>(0, view.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          view(i) = (static_cast<double>((i + 17 * which) % 251) - 125.0) /
                    (137.0 + static_cast<double>(which));
        });
  }
  DoubleView filter_inv("so3lr_dev18_bench_filter_inv",
                        edges * feature_width);
  DoubleView filter_ev("so3lr_dev18_bench_filter_ev",
                       edges * feature_width);
  DoubleView sh("so3lr_dev18_bench_sh", edges * equivariant_width);
  DoubleView cutoff("so3lr_dev18_bench_cutoff", edges);
  IndexView senders("so3lr_dev18_bench_senders", edges);
  IndexView receivers("so3lr_dev18_bench_receivers", edges);
  Kokkos::parallel_for(
      "so3lr_dev18_bench_filters",
      Kokkos::RangePolicy<>(0, edges * feature_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            (static_cast<double>(i % 197) - 98.0) / 113.0;
        filter_inv(i) = value;
        filter_ev(i) = 0.71 * value + 0.03;
      });
  Kokkos::parallel_for(
      "so3lr_dev18_bench_sh",
      Kokkos::RangePolicy<>(0, edges * equivariant_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev18_bench_graph", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        receivers(edge) = edge % nodes;
        senders(edge) = (edge * 17 + 11) % nodes;
        cutoff(edge) =
            0.15 + 0.85 * static_cast<double>((edge * 13) % 101) / 100.0;
      });
  DoubleView grad_inv("so3lr_dev18_bench_grad_inv", nodes * feature_width);
  DoubleView grad_ev("so3lr_dev18_bench_grad_ev",
                     nodes * equivariant_width);
  Kokkos::parallel_for(
      "so3lr_dev18_bench_grad_inv_init",
      Kokkos::RangePolicy<>(0, grad_inv.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        grad_inv(i) =
            (static_cast<double>(i % 113) - 56.0) / 127.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev18_bench_grad_ev_init",
      Kokkos::RangePolicy<>(0, grad_ev.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        grad_ev(i) = (static_cast<double>(i % 47) - 23.0) / 59.0;
      });
  AttentionScatterReverseDeviceWorkspace workspace(nodes, edges, dims_);

  for (int warmup = 0; warmup < 2; ++warmup) {
    launch_forward_device(qkv, filter_inv, filter_ev, sh, cutoff, senders,
                          receivers, workspace);
    launch_reverse_device(qkv, filter_inv, filter_ev, sh, cutoff, senders,
                          receivers, grad_inv, grad_ev, workspace);
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch_reverse_device(qkv, filter_inv, filter_ev, sh, cutoff, senders,
                          receivers, grad_inv, grad_ev, workspace);
  Kokkos::fence();
  const double reverse_ms = timer.seconds() * 1000.0 /
                            static_cast<double>(repetitions);
  timer.reset();
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    launch_forward_device(qkv, filter_inv, filter_ev, sh, cutoff, senders,
                          receivers, workspace);
    launch_reverse_device(qkv, filter_inv, filter_ev, sh, cutoff, senders,
                          receivers, grad_inv, grad_ev, workspace);
  }
  Kokkos::fence();
  const double forward_reverse_ms = timer.seconds() * 1000.0 /
                                    static_cast<double>(repetitions);
  const auto checksum_q = workspace.grad_qkv[0];
  const auto checksum_filter = workspace.grad_filter_inv;
  const auto checksum_sh = workspace.grad_sh;
  const auto checksum_cutoff = workspace.grad_cutoff;
  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev18_bench_checksum", Kokkos::RangePolicy<>(0, 4),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += checksum_q(17);
        if (which == 1) update += checksum_filter(29);
        if (which == 2) update += checksum_sh(11);
        if (which == 3) update += checksum_cutoff(7);
      },
      checksum);
  Kokkos::fence();
  AttentionScatterReverseBenchmark result;
  result.block_index = block_index_;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes =
      complete_workspace_bytes(nodes, edges, persistent_device_bytes_, dims_);
  result.host_boundary_bytes_per_iteration = 0;
  result.reverse_milliseconds = reverse_ms;
  result.forward_reverse_milliseconds = forward_reverse_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (reverse_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
