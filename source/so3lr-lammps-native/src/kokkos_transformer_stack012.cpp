#include "so3lr/kokkos_transformer_stack012.hpp"

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
  // Dev_13 retains all three validated block workspaces so every boundary can
  // be inspected. Geometry and edge indices are shared by the complete stack.
  constexpr std::size_t node_doubles = 6804;
  constexpr std::size_t edge_doubles = 1388;
  return nodes * (sizeof(std::int64_t) + node_doubles * sizeof(double)) +
         edges * (edge_doubles * sizeof(double) +
                  2 * sizeof(std::size_t));
}

}  // namespace

TransformerStack012DeviceWorkspace::TransformerStack012DeviceWorkspace(
    std::size_t nodes, std::size_t edges)
    : stack01(nodes, edges), block2(nodes, edges) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR stack 0-1-2 workspace is empty");
}

KokkosTransformerStack012::KokkosTransformerStack012(
    const NativeModel &model)
    : stack01_(model), block2_(model, 2) {
  device_contract_verified_ = stack01_.device_contract_verified() &&
                              block2_.device_contract_verified() &&
                              block2_.block_index() == 2;
  if (!device_contract_verified_)
    throw std::runtime_error("SO3LR stack 0-1-2 device contract failed");
}

std::size_t KokkosTransformerStack012::persistent_device_bytes() const {
  return stack01_.persistent_device_bytes() +
         block2_.persistent_device_bytes();
}

void KokkosTransformerStack012::launch_device(
    const Int64View &atomic_numbers, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers,
    const TransformerStack012DeviceWorkspace &workspace) const {
  stack01_.launch_device(atomic_numbers, distances, sh_vectors, senders,
                         receivers, workspace.stack01);
  block2_.launch_device(
      workspace.stack01.block1.post_attention.final_inv,
      workspace.stack01.block1.post_attention.final_ev, distances, sh_vectors,
      senders, receivers, workspace.block2);
}

TransformerStack012Results KokkosTransformerStack012::evaluate(
    const std::vector<std::int64_t> &atomic_numbers,
    const std::vector<double> &distances,
    const std::vector<double> &sh_vectors,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers) const {
  const std::size_t nodes = atomic_numbers.size();
  const std::size_t edges = distances.size();
  if (nodes == 0 || edges == 0 || sh_vectors.size() != edges * ev_width ||
      senders.size() != edges || receivers.size() != edges)
    throw std::runtime_error("SO3LR stack 0-1-2 host contract mismatch");
  if (std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR stack 0-1-2 edge out of range");

  Int64View z("so3lr_stack012_z", nodes);
  DoubleView distance("so3lr_stack012_distance", edges);
  DoubleView sh("so3lr_stack012_sh", sh_vectors.size());
  IndexView sender("so3lr_stack012_sender", edges);
  IndexView receiver("so3lr_stack012_receiver", edges);
  fill_view(z, nodes, [&](std::size_t i) { return atomic_numbers[i]; });
  fill_view(distance, edges, [&](std::size_t i) { return distances[i]; });
  fill_view(sh, sh_vectors.size(),
            [&](std::size_t i) { return sh_vectors[i]; });
  fill_view(sender, edges, [&](std::size_t i) { return senders[i]; });
  fill_view(receiver, edges, [&](std::size_t i) { return receivers[i]; });
  TransformerStack012DeviceWorkspace workspace(nodes, edges);
  launch_device(z, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  TransformerStack012Results result;
  result.block0_final_inv =
      copy_host(workspace.stack01.block0.post_attention.final_inv);
  result.block0_final_ev =
      copy_host(workspace.stack01.block0.post_attention.final_ev);
  result.block1_final_inv =
      copy_host(workspace.stack01.block1.post_attention.final_inv);
  result.block1_final_ev =
      copy_host(workspace.stack01.block1.post_attention.final_ev);
  result.block2_final_inv =
      copy_host(workspace.block2.post_attention.final_inv);
  result.block2_final_ev =
      copy_host(workspace.block2.post_attention.final_ev);
  return result;
}

TransformerStack012Benchmark KokkosTransformerStack012::benchmark(
    std::size_t nodes, std::size_t edges, std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR stack 0-1-2 benchmark invalid");
  // Measure blocks 0 and 1 with the already validated dev_12 benchmark. Its
  // temporary workspace is released before the three-block workspace below
  // is allocated.
  const auto stack01_benchmark =
      stack01_.benchmark(nodes, edges, repetitions);
  Int64View z("so3lr_stack012_bench_z", nodes);
  DoubleView distance("so3lr_stack012_bench_distance", edges);
  DoubleView sh("so3lr_stack012_bench_sh", edges * ev_width);
  IndexView sender("so3lr_stack012_bench_sender", edges);
  IndexView receiver("so3lr_stack012_bench_receiver", edges);
  Kokkos::parallel_for(
      "so3lr_stack012_bench_nodes", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  Kokkos::parallel_for(
      "so3lr_stack012_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        distance(edge) =
            0.05 + 4.449 * static_cast<double>(edge % 8192) / 8191.0;
        sender(edge) = (edge * 17 + 11) % nodes;
        receiver(edge) = edge % nodes;
      });
  Kokkos::parallel_for(
      "so3lr_stack012_bench_sh",
      Kokkos::RangePolicy<>(0, edges * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 61) - 30.0) / 37.0;
  });
  TransformerStack012DeviceWorkspace workspace(nodes, edges);
  auto run_block2 = [&]() {
    block2_.launch_device(
        workspace.stack01.block1.post_attention.final_inv,
        workspace.stack01.block1.post_attention.final_ev, distance, sh,
        sender, receiver, workspace.block2);
  };
  auto time_phase = [&](auto launch) {
    launch();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i) launch();
    Kokkos::fence();
    return timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  };
  stack01_.launch_device(z, distance, sh, sender, receiver,
                         workspace.stack01);
  Kokkos::fence();
  const double block2_ms = time_phase(run_block2);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_device(z, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_device(z, distance, sh, sender, receiver, workspace);
  Kokkos::fence();
  const double stack_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);

  double checksum = 0.0;
  const auto block0_inv =
      workspace.stack01.block0.post_attention.final_inv;
  const auto block1_inv =
      workspace.stack01.block1.post_attention.final_inv;
  const auto block2_inv = workspace.block2.post_attention.final_inv;
  const auto block2_ev = workspace.block2.post_attention.final_ev;
  Kokkos::parallel_reduce(
      "so3lr_stack012_bench_checksum", Kokkos::RangePolicy<>(0, 4),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += block0_inv((nodes / 3) * inv_width + 17);
        if (which == 1) update += block1_inv((nodes / 2) * inv_width + 9);
        if (which == 2) update += block2_inv((nodes / 4) * inv_width + 5);
        if (which == 3) update += block2_ev((nodes / 5) * ev_width + 3);
      },
      checksum);
  Kokkos::fence();
  TransformerStack012Benchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes();
  result.workspace_bytes =
      persistent_device_bytes() + dynamic_bytes(nodes, edges);
  result.host_boundary_bytes_per_iteration = 0;
  result.block0_milliseconds = stack01_benchmark.block0_milliseconds;
  result.block1_milliseconds = stack01_benchmark.block1_milliseconds;
  result.block2_milliseconds = block2_ms;
  result.phase_sum_milliseconds =
      stack01_benchmark.block0_milliseconds +
      stack01_benchmark.block1_milliseconds + block2_ms;
  result.stack_milliseconds = stack_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (stack_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
