#include "so3lr/kokkos_attention_scatter_block0.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using IndexView = Kokkos::View<std::size_t *>;

constexpr std::size_t heads = 4;
constexpr std::size_t head_width = 32;
constexpr std::size_t feature_width = heads * head_width;
constexpr std::size_t equivariant_width = 24;
constexpr std::size_t avoided_doubles_per_edge =
    5 * feature_width +       // gathered edge Q/K/V
    2 * feature_width +       // filtered invariant/equivariant keys
    heads + equivariant_width +  // invariant and expanded equivariant alpha
    feature_width + equivariant_width;  // scaled edge messages

template <class Reader>
void fill_double_view(const DoubleView &device, std::size_t count,
                      Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

template <class Reader>
void fill_index_view(const IndexView &device, std::size_t count,
                     Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy_host(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

struct AttentionContract {
  double norm_inv = 0.0;
  double norm_ev = 0.0;
  bool found = false;
};

AttentionContract load_attention_contract(const NativeModel &model) {
  // Both attention streams currently share one divisor; the descriptor
  // resolves it, so no manifest module tree is required.
  AttentionContract result;
  result.norm_inv = model.arch().attention_norm_value;
  result.norm_ev = model.arch().attention_norm_value;
  result.found = true;
  return result;
}

void launch_fused_attention(
    const DoubleView &q_inv, const DoubleView &k_inv,
    const DoubleView &v_inv, const DoubleView &q_ev,
    const DoubleView &k_ev, const DoubleView &filter_inv,
    const DoubleView &filter_ev, const DoubleView &sh_vectors,
    const DoubleView &cutoffs, const IndexView &senders,
    const IndexView &receivers, const IndexView &degree_repeats,
    const IndexView &degree_offsets, std::size_t edges,
    double norm_inv, double norm_ev, const DoubleView &invariant_update,
    const DoubleView &equivariant_update, const DoubleView &alpha_inv,
    const DoubleView &alpha_ev, bool store_attention) {
  Kokkos::deep_copy(invariant_update, 0.0);
  Kokkos::deep_copy(equivariant_update, 0.0);
  if (store_attention) {
    Kokkos::deep_copy(alpha_inv, 0.0);
    Kokkos::deep_copy(alpha_ev, 0.0);
  }
  Kokkos::parallel_for(
      "so3lr_fused_attention_scatter_block0",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / heads;
        const std::size_t head = flat % heads;
        const std::size_t sender = senders(edge);
        const std::size_t receiver = receivers(edge);
        const std::size_t node_head_offset = head * head_width;
        const std::size_t edge_head_offset =
            edge * feature_width + node_head_offset;
        const std::size_t q_offset =
            receiver * feature_width + node_head_offset;
        const std::size_t kv_offset =
            sender * feature_width + node_head_offset;
        double invariant_alpha = 0.0;
        double equivariant_alpha = 0.0;
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          invariant_alpha +=
              q_inv(q_offset + channel) * k_inv(kv_offset + channel) *
              filter_inv(edge_head_offset + channel);
          equivariant_alpha +=
              q_ev(q_offset + channel) * k_ev(kv_offset + channel) *
              filter_ev(edge_head_offset + channel);
        }
        invariant_alpha /= norm_inv;
        equivariant_alpha /= norm_ev;
        if (store_attention)
          alpha_inv(edge * heads + head) = invariant_alpha;
        const double cutoff = cutoffs(edge);
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          Kokkos::atomic_add(
              &invariant_update(receiver * feature_width + node_head_offset +
                                channel),
              cutoff * invariant_alpha * v_inv(kv_offset + channel));
        }
        const std::size_t degree_offset = degree_offsets(head);
        const std::size_t degree_repeat = degree_repeats(head);
        for (std::size_t local = 0; local < degree_repeat; ++local) {
          const std::size_t channel = degree_offset + local;
          if (store_attention)
            alpha_ev(edge * equivariant_width + channel) = equivariant_alpha;
          Kokkos::atomic_add(
              &equivariant_update(receiver * equivariant_width + channel),
              cutoff * equivariant_alpha *
                  sh_vectors(edge * equivariant_width + channel));
        }
      });
}

std::size_t workspace_bytes(std::size_t nodes, std::size_t edges,
                            std::size_t persistent_bytes) {
  const std::size_t node_doubles =
      nodes * (5 * feature_width + feature_width + equivariant_width);
  const std::size_t edge_doubles =
      edges * (2 * feature_width + equivariant_width + 1);
  const std::size_t edge_indices = edges * 2 * sizeof(std::size_t);
  return persistent_bytes +
         (node_doubles + edge_doubles) * sizeof(double) + edge_indices;
}

}  // namespace

KokkosAttentionScatterBlock0::KokkosAttentionScatterBlock0(
    const NativeModel &model) {
  if (model.architecture_number("attention_heads") != heads ||
      model.architecture_number("attention_head_width") != head_width ||
      model.architecture_number("euclidean_degree_channels") != heads)
    throw std::runtime_error("SO3LR fused-attention architecture mismatch");
  const auto contract = load_attention_contract(model);
  if (!contract.found || !std::isfinite(contract.norm_inv) ||
      !std::isfinite(contract.norm_ev) || contract.norm_inv <= 0.0 ||
      contract.norm_ev <= 0.0)
    throw std::runtime_error("SO3LR attention normalization contract missing");
  attention_norm_inv_ = contract.norm_inv;
  attention_norm_ev_ = contract.norm_ev;

  const auto &tensor = model.tensor(
      "model.euclidean_transformers.0.euclidean_attention_block."
      "degree_repeats");
  if (tensor.dtype != "int64" ||
      tensor.shape != std::vector<std::size_t>{heads})
    throw std::runtime_error("SO3LR degree-repeat tensor contract mismatch");
  constexpr std::array<std::size_t, heads> expected = {3, 5, 7, 9};
  std::array<std::size_t, heads> offsets = {};
  std::size_t total = 0;
  for (std::size_t head = 0; head < heads; ++head) {
    const auto value = model.int64(tensor, head);
    if (value <= 0 || static_cast<std::size_t>(value) != expected[head])
      throw std::runtime_error("SO3LR degree-repeat values changed");
    offsets[head] = total;
    total += static_cast<std::size_t>(value);
  }
  if (total != equivariant_width)
    throw std::runtime_error("SO3LR equivariant width mismatch");
  degree_repeats_ = IndexView("so3lr_attention_degree_repeats", heads);
  degree_offsets_ = IndexView("so3lr_attention_degree_offsets", heads);
  fill_index_view(degree_repeats_, heads,
                  [&](std::size_t i) { return expected[i]; });
  fill_index_view(degree_offsets_, heads,
                  [&](std::size_t i) { return offsets[i]; });
  persistent_device_bytes_ = 2 * heads * sizeof(std::size_t);
  degree_contract_verified_ = true;
  Kokkos::fence();
}

AttentionScatterBlock0Results KokkosAttentionScatterBlock0::evaluate(
    const std::vector<double> &q_inv_nodes,
    const std::vector<double> &k_inv_nodes,
    const std::vector<double> &v_inv_nodes,
    const std::vector<double> &q_ev_nodes,
    const std::vector<double> &k_ev_nodes,
    const std::vector<double> &filter_inv,
    const std::vector<double> &filter_ev,
    const std::vector<double> &sh_vectors,
    const std::vector<double> &cutoffs,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers, std::size_t nodes) const {
  const std::size_t edges = senders.size();
  const std::size_t node_size = nodes * feature_width;
  if (nodes == 0 || receivers.size() != edges || cutoffs.size() != edges ||
      q_inv_nodes.size() != node_size || k_inv_nodes.size() != node_size ||
      v_inv_nodes.size() != node_size || q_ev_nodes.size() != node_size ||
      k_ev_nodes.size() != node_size ||
      filter_inv.size() != edges * feature_width ||
      filter_ev.size() != edges * feature_width ||
      sh_vectors.size() != edges * equivariant_width)
    throw std::runtime_error("SO3LR fused-attention input contract mismatch");
  if (std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR fused-attention edge index out of range");

  DoubleView q_inv("so3lr_att_q_inv", node_size);
  DoubleView k_inv("so3lr_att_k_inv", node_size);
  DoubleView v_inv("so3lr_att_v_inv", node_size);
  DoubleView q_ev("so3lr_att_q_ev", node_size);
  DoubleView k_ev("so3lr_att_k_ev", node_size);
  DoubleView filter_i("so3lr_att_filter_inv", filter_inv.size());
  DoubleView filter_e("so3lr_att_filter_ev", filter_ev.size());
  DoubleView sh("so3lr_att_sh", sh_vectors.size());
  DoubleView cutoff("so3lr_att_cutoff", edges);
  IndexView sender("so3lr_att_sender", edges);
  IndexView receiver("so3lr_att_receiver", edges);
  fill_double_view(q_inv, node_size,
                   [&](std::size_t i) { return q_inv_nodes[i]; });
  fill_double_view(k_inv, node_size,
                   [&](std::size_t i) { return k_inv_nodes[i]; });
  fill_double_view(v_inv, node_size,
                   [&](std::size_t i) { return v_inv_nodes[i]; });
  fill_double_view(q_ev, node_size,
                   [&](std::size_t i) { return q_ev_nodes[i]; });
  fill_double_view(k_ev, node_size,
                   [&](std::size_t i) { return k_ev_nodes[i]; });
  fill_double_view(filter_i, filter_inv.size(),
                   [&](std::size_t i) { return filter_inv[i]; });
  fill_double_view(filter_e, filter_ev.size(),
                   [&](std::size_t i) { return filter_ev[i]; });
  fill_double_view(sh, sh_vectors.size(),
                   [&](std::size_t i) { return sh_vectors[i]; });
  fill_double_view(cutoff, edges,
                   [&](std::size_t i) { return cutoffs[i]; });
  fill_index_view(sender, edges,
                  [&](std::size_t i) { return senders[i]; });
  fill_index_view(receiver, edges,
                  [&](std::size_t i) { return receivers[i]; });

  DoubleView invariant_update("so3lr_att_d_inv", nodes * feature_width);
  DoubleView equivariant_update("so3lr_att_d_ev",
                                nodes * equivariant_width);
  DoubleView alpha_inv("so3lr_att_alpha_inv", edges * heads);
  DoubleView alpha_ev("so3lr_att_alpha_ev", edges * equivariant_width);
  launch_fused_attention(
      q_inv, k_inv, v_inv, q_ev, k_ev, filter_i, filter_e, sh, cutoff,
      sender, receiver, degree_repeats_, degree_offsets_, edges,
      attention_norm_inv_, attention_norm_ev_, invariant_update,
      equivariant_update, alpha_inv, alpha_ev, true);
  Kokkos::fence();
  AttentionScatterBlock0Results result;
  result.invariant_update = copy_host(invariant_update);
  result.equivariant_update = copy_host(equivariant_update);
  result.alpha_invariant = copy_host(alpha_inv);
  result.alpha_equivariant = copy_host(alpha_ev);
  return result;
}

AttentionScatterBlock0Benchmark KokkosAttentionScatterBlock0::benchmark(
    std::size_t nodes, std::size_t edges, std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR fused-attention benchmark sizes invalid");
  const std::size_t node_size = nodes * feature_width;
  DoubleView q_inv("so3lr_att_bench_q_inv", node_size);
  DoubleView k_inv("so3lr_att_bench_k_inv", node_size);
  DoubleView v_inv("so3lr_att_bench_v_inv", node_size);
  DoubleView q_ev("so3lr_att_bench_q_ev", node_size);
  DoubleView k_ev("so3lr_att_bench_k_ev", node_size);
  Kokkos::parallel_for(
      "so3lr_att_bench_nodes", Kokkos::RangePolicy<>(0, node_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            (static_cast<double>(i % 257) - 128.0) / 131.0;
        q_inv(i) = value;
        k_inv(i) = 0.8 * value + 0.03;
        v_inv(i) = -0.6 * value + 0.02;
        q_ev(i) = 0.7 * value - 0.01;
        k_ev(i) = -0.5 * value + 0.04;
      });
  DoubleView filter_i("so3lr_att_bench_filter_inv", edges * feature_width);
  DoubleView filter_e("so3lr_att_bench_filter_ev", edges * feature_width);
  Kokkos::parallel_for(
      "so3lr_att_bench_filters",
      Kokkos::RangePolicy<>(0, edges * feature_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            (static_cast<double>(i % 127) - 63.0) / 89.0;
        filter_i(i) = value;
        filter_e(i) = 0.75 * value + 0.1;
      });
  DoubleView sh("so3lr_att_bench_sh", edges * equivariant_width);
  Kokkos::parallel_for(
      "so3lr_att_bench_sh_init",
      Kokkos::RangePolicy<>(0, edges * equivariant_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 61) - 30.0) / 37.0;
      });
  DoubleView cutoff("so3lr_att_bench_cutoff", edges);
  IndexView sender("so3lr_att_bench_sender", edges);
  IndexView receiver("so3lr_att_bench_receiver", edges);
  Kokkos::parallel_for(
      "so3lr_att_bench_graph", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        receiver(edge) = edge % nodes;
        sender(edge) = (edge * 17 + 11) % nodes;
        cutoff(edge) = 0.15 + 0.85 *
            static_cast<double>((edge * 13) % 101) / 100.0;
      });
  DoubleView invariant_update("so3lr_att_bench_d_inv",
                              nodes * feature_width);
  DoubleView equivariant_update("so3lr_att_bench_d_ev",
                                nodes * equivariant_width);
  const DoubleView no_alpha_inv;
  const DoubleView no_alpha_ev;
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_fused_attention(
        q_inv, k_inv, v_inv, q_ev, k_ev, filter_i, filter_e, sh, cutoff,
        sender, receiver, degree_repeats_, degree_offsets_, edges,
        attention_norm_inv_, attention_norm_ev_, invariant_update,
        equivariant_update, no_alpha_inv, no_alpha_ev, false);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch_fused_attention(
        q_inv, k_inv, v_inv, q_ev, k_ev, filter_i, filter_e, sh, cutoff,
        sender, receiver, degree_repeats_, degree_offsets_, edges,
        attention_norm_inv_, attention_norm_ev_, invariant_update,
        equivariant_update, no_alpha_inv, no_alpha_ev, false);
  Kokkos::fence();
  const double elapsed_ms = timer.seconds() * 1000.0;
  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_att_bench_checksum", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += invariant_update((nodes / 3) * feature_width + 17);
        if (which == 1)
          update += equivariant_update((nodes / 2) * equivariant_width + 9);
      },
      checksum);
  Kokkos::fence();
  AttentionScatterBlock0Benchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.workspace_bytes = workspace_bytes(nodes, edges,
                                           persistent_device_bytes_);
  result.avoided_edge_intermediate_bytes =
      edges * avoided_doubles_per_edge * sizeof(double);
  result.total_milliseconds = elapsed_ms;
  result.milliseconds_per_iteration =
      elapsed_ms / static_cast<double>(repetitions);
  result.edges_per_second =
      static_cast<double>(edges) /
      (result.milliseconds_per_iteration / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
