#include "so3lr/kokkos_forward_model.hpp"

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
  // The dev_13 transformer retains inspection workspaces for all three
  // blocks. Dev_15 adds only the head workspace; geometry is shared.
  constexpr std::size_t transformer_node_doubles = 6804;
  constexpr std::size_t transformer_edge_doubles = 1388;
  constexpr std::size_t head_node_doubles = 772;
  return nodes *
             (sizeof(std::int64_t) +
              (transformer_node_doubles + head_node_doubles) *
                  sizeof(double)) +
         edges * (transformer_edge_doubles * sizeof(double) +
                  2 * sizeof(std::size_t)) +
         2 * sizeof(double);
}

}  // namespace

NativeForwardDeviceWorkspace::NativeForwardDeviceWorkspace(
    std::size_t nodes, std::size_t edges)
    : transformer(nodes, edges), heads(nodes) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR native-forward workspace is empty");
}

KokkosForwardModel::KokkosForwardModel(const NativeModel &model)
    : transformer_(model), heads_(model) {
  device_bridge_verified_ = transformer_.device_contract_verified() &&
                            heads_.checkpoint_contract_verified() &&
                            heads_.shared_kokkos_stream();
  if (!device_bridge_verified_)
    throw std::runtime_error("SO3LR transformer-to-heads contract failed");
}

std::size_t KokkosForwardModel::persistent_device_bytes() const {
  return transformer_.persistent_device_bytes() +
         heads_.persistent_device_bytes();
}

void KokkosForwardModel::launch_device(
    const Int64View &atomic_numbers, const DoubleView &distances,
    const DoubleView &sh_vectors, const IndexView &senders,
    const IndexView &receivers, double total_charge,
    const NativeForwardDeviceWorkspace &workspace) const {
  transformer_.launch_device(atomic_numbers, distances, sh_vectors, senders,
                             receivers, workspace.transformer);
  heads_.launch_device(
      workspace.transformer.block2.post_attention.final_inv, atomic_numbers,
      total_charge, workspace.heads);
}

NativeForwardResults KokkosForwardModel::evaluate(
    const std::vector<std::int64_t> &atomic_numbers,
    const std::vector<double> &distances,
    const std::vector<double> &sh_vectors,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers, double total_charge) const {
  const std::size_t nodes = atomic_numbers.size();
  const std::size_t edges = distances.size();
  if (nodes == 0 || edges == 0 || sh_vectors.size() != edges * ev_width ||
      senders.size() != edges || receivers.size() != edges)
    throw std::runtime_error("SO3LR native-forward host contract mismatch");
  if (std::any_of(atomic_numbers.begin(), atomic_numbers.end(),
                  [](std::int64_t z) { return z < 0 || z >= 100; }) ||
      std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR native-forward index out of range");

  Int64View z("so3lr_forward_z", nodes);
  DoubleView distance("so3lr_forward_distance", edges);
  DoubleView sh("so3lr_forward_sh", sh_vectors.size());
  IndexView sender("so3lr_forward_sender", edges);
  IndexView receiver("so3lr_forward_receiver", edges);
  fill_view(z, nodes, [&](std::size_t i) { return atomic_numbers[i]; });
  fill_view(distance, edges, [&](std::size_t i) { return distances[i]; });
  fill_view(sh, sh_vectors.size(),
            [&](std::size_t i) { return sh_vectors[i]; });
  fill_view(sender, edges, [&](std::size_t i) { return senders[i]; });
  fill_view(receiver, edges, [&](std::size_t i) { return receivers[i]; });

  NativeForwardDeviceWorkspace workspace(nodes, edges);
  launch_device(z, distance, sh, sender, receiver, total_charge, workspace);
  Kokkos::fence();
  NativeForwardResults result;
  result.final_inv =
      copy_host(workspace.transformer.block2.post_attention.final_inv);
  result.final_ev =
      copy_host(workspace.transformer.block2.post_attention.final_ev);
  result.heads.atomic_energies = copy_host(workspace.heads.atomic_energies);
  result.heads.raw_charges = copy_host(workspace.heads.raw_charges);
  result.heads.partial_charges = copy_host(workspace.heads.partial_charges);
  result.heads.hirshfeld_ratios =
      copy_host(workspace.heads.hirshfeld_ratios);
  const auto reductions = copy_host(workspace.heads.reductions);
  result.heads.learned_sr_energy = reductions[1];
  for (const double charge : result.heads.partial_charges)
    result.heads.charge_sum += charge;
  return result;
}

NativeForwardBenchmark KokkosForwardModel::benchmark(
    std::size_t nodes, std::size_t edges, std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR native-forward benchmark invalid");
  Int64View z("so3lr_forward_bench_z", nodes);
  DoubleView distance("so3lr_forward_bench_distance", edges);
  DoubleView sh("so3lr_forward_bench_sh", edges * ev_width);
  IndexView sender("so3lr_forward_bench_sender", edges);
  IndexView receiver("so3lr_forward_bench_receiver", edges);
  Kokkos::parallel_for(
      "so3lr_forward_bench_nodes", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  Kokkos::parallel_for(
      "so3lr_forward_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        distance(edge) =
            0.05 + 4.449 * static_cast<double>(edge % 8192) / 8191.0;
        sender(edge) = (edge * 17 + 11) % nodes;
        receiver(edge) = edge % nodes;
      });
  Kokkos::parallel_for(
      "so3lr_forward_bench_sh",
      Kokkos::RangePolicy<>(0, edges * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 61) - 30.0) / 37.0;
      });
  NativeForwardDeviceWorkspace workspace(nodes, edges);
  auto run_transformer = [&]() {
    transformer_.launch_device(z, distance, sh, sender, receiver,
                               workspace.transformer);
  };
  auto run_heads = [&]() {
    heads_.launch_device(
        workspace.transformer.block2.post_attention.final_inv, z, 0.0,
        workspace.heads);
  };
  auto run_integrated = [&]() {
    launch_device(z, distance, sh, sender, receiver, 0.0, workspace);
  };
  auto time_phase = [&](auto launch) {
    launch();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i) launch();
    Kokkos::fence();
    return timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  };

  const double transformer_ms = time_phase(run_transformer);
  run_transformer();
  Kokkos::fence();
  const double heads_ms = time_phase(run_heads);
  for (int warmup = 0; warmup < 2; ++warmup) run_integrated();
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i) run_integrated();
  Kokkos::fence();
  const double integrated_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);

  double checksum = 0.0;
  const auto final_inv =
      workspace.transformer.block2.post_attention.final_inv;
  const auto energy = workspace.heads.atomic_energies;
  const auto charge = workspace.heads.partial_charges;
  const auto hirshfeld = workspace.heads.hirshfeld_ratios;
  Kokkos::parallel_reduce(
      "so3lr_forward_bench_checksum", Kokkos::RangePolicy<>(0, 4),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += final_inv((nodes / 3) * inv_width + 17);
        if (which == 1) update += energy(nodes / 2);
        if (which == 2) update += charge(nodes / 4);
        if (which == 3) update += hirshfeld(nodes / 5);
      },
      checksum);
  Kokkos::fence();

  NativeForwardBenchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes();
  result.workspace_bytes =
      persistent_device_bytes() + dynamic_bytes(nodes, edges);
  result.host_boundary_bytes_per_iteration = 0;
  result.transformer_milliseconds = transformer_ms;
  result.output_heads_milliseconds = heads_ms;
  result.phase_sum_milliseconds = transformer_ms + heads_ms;
  result.integrated_milliseconds = integrated_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (integrated_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
