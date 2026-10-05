#include "so3lr/kokkos_feature_transformer_block.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

constexpr std::size_t inv_width = 128;
constexpr std::size_t ev_width = 24;

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
      inv_width + ev_width +  // input features
      inv_width +             // unused embedding buffer retained by API
      5 * inv_width + inv_width + ev_width +  // QKV and attention updates
      1340;                              // post-attention workspace
  constexpr std::size_t edge_doubles =
      1 + ev_width +  // input distance and spherical harmonics
      1 + 32 + 4 + 128 + 32 + 2 * inv_width;  // attention workspace
  return nodes * node_doubles * sizeof(double) +
         edges * (edge_doubles * sizeof(double) +
                  2 * sizeof(std::size_t));
}

}  // namespace

FeatureTransformerDeviceWorkspace::FeatureTransformerDeviceWorkspace(
    std::size_t nodes, std::size_t edges)
    : attention(nodes, edges), post_attention(nodes) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR feature transformer workspace is empty");
}

KokkosFeatureTransformerBlock::KokkosFeatureTransformerBlock(
    const NativeModel &model, std::size_t block_index)
    : attention_(model, block_index),
      post_attention_(model, block_index),
      block_index_(block_index) {
  device_contract_verified_ =
      attention_.block_index() == block_index_ &&
      post_attention_.block_index() == block_index_ &&
      attention_.shared_kokkos_stream() &&
      post_attention_.shared_kokkos_stream() &&
      post_attention_.checkpoint_contract_verified();
  if (!device_contract_verified_)
    throw std::runtime_error("SO3LR feature transformer contract failed");
}

std::size_t KokkosFeatureTransformerBlock::persistent_device_bytes() const {
  return attention_.persistent_device_bytes() +
         post_attention_.persistent_device_bytes();
}

void KokkosFeatureTransformerBlock::launch_device(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const FeatureTransformerDeviceWorkspace &workspace) const {
  attention_.launch_features_device(
      inv_features, ev_features, distances, sh_vectors, senders, receivers,
      workspace.attention);
  post_attention_.launch_device(
      inv_features, ev_features, workspace.attention.invariant_update,
      workspace.attention.equivariant_update, workspace.post_attention);
}

FeatureTransformerResults KokkosFeatureTransformerBlock::evaluate(
    const std::vector<double> &inv_features,
    const std::vector<double> &ev_features,
    const std::vector<double> &distances,
    const std::vector<double> &sh_vectors,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers) const {
  const std::size_t nodes = inv_features.size() / inv_width;
  const std::size_t edges = distances.size();
  if (nodes == 0 || edges == 0 ||
      inv_features.size() != nodes * inv_width ||
      ev_features.size() != nodes * ev_width ||
      sh_vectors.size() != edges * ev_width || senders.size() != edges ||
      receivers.size() != edges)
    throw std::runtime_error("SO3LR feature transformer host contract mismatch");
  if (std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR feature transformer edge out of range");

  DoubleView inv("so3lr_feature_input_inv", inv_features.size());
  DoubleView ev("so3lr_feature_input_ev", ev_features.size());
  DoubleView distance("so3lr_feature_distance", edges);
  DoubleView sh("so3lr_feature_sh", sh_vectors.size());
  IndexView sender("so3lr_feature_sender", edges);
  IndexView receiver("so3lr_feature_receiver", edges);
  fill_view(inv, inv_features.size(),
            [&](std::size_t i) { return inv_features[i]; });
  fill_view(ev, ev_features.size(),
            [&](std::size_t i) { return ev_features[i]; });
  fill_view(distance, edges, [&](std::size_t i) { return distances[i]; });
  fill_view(sh, sh_vectors.size(),
            [&](std::size_t i) { return sh_vectors[i]; });
  fill_view(sender, edges, [&](std::size_t i) { return senders[i]; });
  fill_view(receiver, edges, [&](std::size_t i) { return receivers[i]; });
  FeatureTransformerDeviceWorkspace workspace(nodes, edges);
  launch_device(inv, ev, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  FeatureTransformerResults result;
  result.edge_ev_invariants =
      copy_host(workspace.attention.edge_ev_invariants);
  result.attention_update_inv =
      copy_host(workspace.attention.invariant_update);
  result.attention_update_ev =
      copy_host(workspace.attention.equivariant_update);
  result.final_inv = copy_host(workspace.post_attention.final_inv);
  result.final_ev = copy_host(workspace.post_attention.final_ev);
  return result;
}

FeatureTransformerBenchmark KokkosFeatureTransformerBlock::benchmark(
    std::size_t nodes, std::size_t edges, std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR feature transformer benchmark invalid");
  DoubleView inv("so3lr_feature_bench_inv", nodes * inv_width);
  DoubleView ev("so3lr_feature_bench_ev", nodes * ev_width);
  DoubleView distance("so3lr_feature_bench_distance", edges);
  DoubleView sh("so3lr_feature_bench_sh", edges * ev_width);
  IndexView sender("so3lr_feature_bench_sender", edges);
  IndexView receiver("so3lr_feature_bench_receiver", edges);
  Kokkos::parallel_for(
      "so3lr_feature_bench_nodes_inv",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        inv(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });
  Kokkos::parallel_for(
      "so3lr_feature_bench_nodes_ev",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        ev(i) = (static_cast<double>(i % 43) - 21.0) / 53.0;
      });
  Kokkos::parallel_for(
      "so3lr_feature_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        distance(edge) =
            0.05 + 4.449 * static_cast<double>(edge % 8192) / 8191.0;
        sender(edge) = (edge * 17 + 11) % nodes;
        receiver(edge) = edge % nodes;
      });
  Kokkos::parallel_for(
      "so3lr_feature_bench_sh",
      Kokkos::RangePolicy<>(0, edges * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 61) - 30.0) / 37.0;
      });
  FeatureTransformerDeviceWorkspace workspace(nodes, edges);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_device(inv, ev, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_device(inv, ev, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  const double elapsed_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto final_inv = workspace.post_attention.final_inv;
  const auto final_ev = workspace.post_attention.final_ev;
  const auto edge_invariants = workspace.attention.edge_ev_invariants;
  Kokkos::parallel_reduce(
      "so3lr_feature_bench_checksum", Kokkos::RangePolicy<>(0, 3),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += final_inv((nodes / 3) * inv_width + 17);
        if (which == 1) update += final_ev((nodes / 2) * ev_width + 9);
        if (which == 2) update += edge_invariants((edges / 2) * 4 + 2);
      },
      checksum);
  Kokkos::fence();
  FeatureTransformerBenchmark result;
  result.block_index = block_index_;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes();
  result.workspace_bytes =
      persistent_device_bytes() + dynamic_bytes(nodes, edges);
  result.host_boundary_bytes_per_iteration = 0;
  result.complete_block_milliseconds = elapsed_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (elapsed_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
