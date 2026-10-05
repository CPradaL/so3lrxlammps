#include "so3lr/kokkos_transformer_block0.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

constexpr std::size_t inv_width = 128;
constexpr std::size_t ev_width = 24;
constexpr std::size_t degree_count = 4;

template <class View, class Reader>
void fill_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy_host(const Kokkos::View<double *> &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

std::size_t dynamic_bytes(std::size_t nodes, std::size_t edges) {
  constexpr std::size_t node_doubles =
      ev_width +  // initial equivariant features
      920 +       // attention workspace: embedding, QKV and updates
      1340;       // post-attention workspace
  constexpr std::size_t edge_doubles =
      1 + ev_width + degree_count + 449;  // input geometry + attention work
  return nodes * (sizeof(std::int64_t) + node_doubles * sizeof(double)) +
         edges * (edge_doubles * sizeof(double) +
                  2 * sizeof(std::size_t));
}

}  // namespace

TransformerBlock0DeviceWorkspace::TransformerBlock0DeviceWorkspace(
    std::size_t nodes, std::size_t edges)
    : initial_ev_features("so3lr_full0_initial_ev", nodes * ev_width),
      initial_ev_invariants("so3lr_full0_initial_ev_invariants",
                            edges * degree_count),
      attention(nodes, edges),
      post_attention(nodes) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR complete block-0 workspace is empty");
  Kokkos::deep_copy(initial_ev_features, 0.0);
  Kokkos::deep_copy(initial_ev_invariants, 0.0);
}

KokkosTransformerBlock0::KokkosTransformerBlock0(const NativeModel &model)
    : attention_(model), post_attention_(model) {
  device_contract_verified_ = attention_.shared_kokkos_stream() &&
                              post_attention_.shared_kokkos_stream() &&
                              post_attention_.checkpoint_contract_verified();
  if (!device_contract_verified_)
    throw std::runtime_error("SO3LR complete block-0 device contract failed");
}

std::size_t KokkosTransformerBlock0::persistent_device_bytes() const {
  return attention_.persistent_device_bytes() +
         post_attention_.persistent_device_bytes();
}

void KokkosTransformerBlock0::launch_device(
    const Int64View &atomic_numbers, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers,
    const TransformerBlock0DeviceWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  const std::size_t edges = distances.extent(0);
  if (nodes == 0 || edges == 0 || sh_vectors.extent(0) != edges * ev_width ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      workspace.initial_ev_features.extent(0) != nodes * ev_width ||
      workspace.initial_ev_invariants.extent(0) != edges * degree_count)
    throw std::runtime_error("SO3LR complete block-0 input contract mismatch");
  attention_.launch_device(
      atomic_numbers, distances, workspace.initial_ev_invariants, sh_vectors,
      senders, receivers, workspace.attention);
  post_attention_.launch_device(
      workspace.attention.embedding, workspace.initial_ev_features,
      workspace.attention.invariant_update,
      workspace.attention.equivariant_update, workspace.post_attention);
}

TransformerBlock0Results KokkosTransformerBlock0::evaluate(
    const std::vector<std::int64_t> &atomic_numbers,
    const std::vector<double> &distances,
    const std::vector<double> &sh_vectors,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers) const {
  const std::size_t nodes = atomic_numbers.size();
  const std::size_t edges = distances.size();
  if (nodes == 0 || edges == 0 || sh_vectors.size() != edges * ev_width ||
      senders.size() != edges || receivers.size() != edges)
    throw std::runtime_error("SO3LR complete block-0 host input mismatch");
  if (std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR complete block-0 edge index out of range");

  Int64View z("so3lr_full0_z", nodes);
  DoubleView distance("so3lr_full0_distance", edges);
  DoubleView sh("so3lr_full0_sh", edges * ev_width);
  IndexView sender("so3lr_full0_sender", edges);
  IndexView receiver("so3lr_full0_receiver", edges);
  fill_view(z, nodes, [&](std::size_t i) { return atomic_numbers[i]; });
  fill_view(distance, edges, [&](std::size_t i) { return distances[i]; });
  fill_view(sh, sh_vectors.size(),
            [&](std::size_t i) { return sh_vectors[i]; });
  fill_view(sender, edges, [&](std::size_t i) { return senders[i]; });
  fill_view(receiver, edges, [&](std::size_t i) { return receivers[i]; });
  TransformerBlock0DeviceWorkspace workspace(nodes, edges);
  launch_device(z, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  TransformerBlock0Results result;
  result.embedding = copy_host(workspace.attention.embedding);
  result.attention_update_inv =
      copy_host(workspace.attention.invariant_update);
  result.attention_update_ev =
      copy_host(workspace.attention.equivariant_update);
  result.final_inv = copy_host(workspace.post_attention.final_inv);
  result.final_ev = copy_host(workspace.post_attention.final_ev);
  return result;
}

TransformerBlock0Benchmark KokkosTransformerBlock0::benchmark(
    std::size_t nodes, std::size_t edges, std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR complete block-0 benchmark sizes invalid");
  Int64View z("so3lr_full0_bench_z", nodes);
  DoubleView distance("so3lr_full0_bench_distance", edges);
  DoubleView sh("so3lr_full0_bench_sh", edges * ev_width);
  IndexView sender("so3lr_full0_bench_sender", edges);
  IndexView receiver("so3lr_full0_bench_receiver", edges);
  Kokkos::parallel_for(
      "so3lr_full0_bench_nodes", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  Kokkos::parallel_for(
      "so3lr_full0_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        distance(edge) =
            0.05 + 4.449 * static_cast<double>(edge % 8192) / 8191.0;
        sender(edge) = (edge * 17 + 11) % nodes;
        receiver(edge) = edge % nodes;
      });
  Kokkos::parallel_for(
      "so3lr_full0_bench_sh", Kokkos::RangePolicy<>(0, edges * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 61) - 30.0) / 37.0;
      });
  TransformerBlock0DeviceWorkspace workspace(nodes, edges);

  auto run_attention = [&]() {
    attention_.launch_device(z, distance, workspace.initial_ev_invariants, sh,
                             sender, receiver, workspace.attention);
  };
  auto run_post_attention = [&]() {
    post_attention_.launch_device(
        workspace.attention.embedding, workspace.initial_ev_features,
        workspace.attention.invariant_update,
        workspace.attention.equivariant_update, workspace.post_attention);
  };
  auto time_phase = [&](auto launch) {
    launch();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i) launch();
    Kokkos::fence();
    return timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  };

  run_attention();
  run_post_attention();
  Kokkos::fence();
  const double attention_ms = time_phase(run_attention);
  const double post_attention_ms = time_phase(run_post_attention);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_device(z, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_device(z, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  const double complete_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto final_inv = workspace.post_attention.final_inv;
  const auto final_ev = workspace.post_attention.final_ev;
  Kokkos::parallel_reduce(
      "so3lr_full0_bench_checksum", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += final_inv((nodes / 3) * inv_width + 17);
        if (which == 1) update += final_ev((nodes / 2) * ev_width + 9);
      },
      checksum);
  Kokkos::fence();
  TransformerBlock0Benchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes();
  result.workspace_bytes =
      persistent_device_bytes() + dynamic_bytes(nodes, edges);
  result.host_boundary_bytes_per_iteration = 0;
  result.attention_pipeline_milliseconds = attention_ms;
  result.post_attention_milliseconds = post_attention_ms;
  result.phase_sum_milliseconds = attention_ms + post_attention_ms;
  result.complete_block_milliseconds = complete_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (complete_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
