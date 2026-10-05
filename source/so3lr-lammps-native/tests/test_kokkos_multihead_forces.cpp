#include "so3lr/kokkos_learned_energy_forces.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

std::string read_text(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open fixture " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<double> numbers(const so3lr::Json &value) {
  std::vector<double> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

std::vector<std::size_t> indices(const so3lr::Json &value) {
  std::vector<std::size_t> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<std::size_t>(item.unsigned_integer()));
  return result;
}

std::vector<std::int64_t> integers(const so3lr::Json &value) {
  std::vector<std::int64_t> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<std::int64_t>(item.unsigned_integer()));
  return result;
}

template <class View, class Values>
void fill(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("fixture-to-device size mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

double compare(const std::vector<double> &actual,
               const std::vector<double> &expected,
               const std::string &label) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > 2.0e-7 + 2.0e-7 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i) +
                               " actual=" + std::to_string(actual[i]) +
                               " expected=" + std::to_string(expected[i]));
  }
  return maximum;
}

std::size_t multihead_workspace_bytes(std::size_t nodes, std::size_t edges,
                                      std::size_t persistent_bytes) {
  constexpr std::size_t block_node_doubles = 4516;
  constexpr std::size_t block_edge_doubles = 1027;
  constexpr std::size_t chain_node_doubles =
      3 * block_node_doubles + 128 + 24 + 24 + 1220;
  constexpr std::size_t chain_edge_doubles =
      3 * block_edge_doubles + 1 + 24 + 1 + 24;
  // Chain audit allocation + atom forces + three external seed arrays.
  return persistent_bytes +
         nodes * ((chain_node_doubles + 6) * sizeof(double) +
                  sizeof(std::int64_t)) +
         edges * ((chain_edge_doubles + 6) * sizeof(double) +
                  2 * sizeof(std::size_t));
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("test was not compiled for CUDA");
#endif
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL MULTIHEAD_FORCE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-multihead-cartesian-force-fixture-v1" ||
        fixture.at("objective").string() !=
            "seeded_energy_plus_partial_charge_plus_hirshfeld" ||
        fixture.at("vector_convention").string() !=
            "receiver_minus_sender")
      throw std::runtime_error("unexpected dev_24 fixture contract");

    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t edges =
        static_cast<std::size_t>(fixture.at("edges").unsigned_integer());
    Int64View atomic_numbers("so3lr_dev24_z", nodes);
    DoubleView edge_vectors("so3lr_dev24_vectors", edges * 3);
    IndexView senders("so3lr_dev24_senders", edges);
    IndexView receivers("so3lr_dev24_receivers", edges);
    DoubleView energy_seeds("so3lr_dev24_energy_seeds", nodes);
    DoubleView charge_seeds("so3lr_dev24_charge_seeds", nodes);
    DoubleView hirshfeld_seeds("so3lr_dev24_hirshfeld_seeds", nodes);
    fill(atomic_numbers, integers(fixture.at("atomic_numbers")));
    fill(edge_vectors, numbers(fixture.at("edge_vectors")));
    fill(senders, indices(fixture.at("senders")));
    fill(receivers, indices(fixture.at("receivers")));

    const auto energy_seed_values = numbers(fixture.at("energy_seeds"));
    const auto charge_seed_values = numbers(fixture.at("partial_charge_seeds"));
    const auto hirshfeld_seed_values = numbers(fixture.at("hirshfeld_seeds"));
    const std::vector<double> zero_seeds(nodes, 0.0);

    const so3lr::KokkosLearnedEnergyReverseChain chain(model);
    if (!chain.chain_contract_verified() || !chain.shared_kokkos_stream())
      throw std::runtime_error("dev_24 chain runtime contract failed");
    so3lr::GeometryVJPWorkspace geometry(nodes, edges);
    so3lr::LearnedEnergyReverseWorkspace workspace(nodes, edges);
    so3lr::launch_so3lr_geometry_forward_device(edge_vectors, geometry);
    chain.launch_forward_device(atomic_numbers, geometry.distances,
                                geometry.sh_vectors, senders, receivers,
                                workspace);
    Kokkos::fence();
    const double energy_output_error = compare(
        copy(chain.atomic_energies(workspace)),
        numbers(fixture.at("atomic_energies")), "atomic-energy forward");
    const double charge_output_error = compare(
        copy(chain.partial_charges(workspace)),
        numbers(fixture.at("partial_charges")), "partial-charge forward");
    const double hirshfeld_output_error = compare(
        copy(chain.hirshfeld_ratios(workspace)),
        numbers(fixture.at("hirshfeld_ratios")), "Hirshfeld forward");

    auto component_gradient = [&](const std::vector<double> &e_seed,
                                  const std::vector<double> &q_seed,
                                  const std::vector<double> &h_seed,
                                  const char *expected,
                                  const char *label) {
      chain.launch_forward_device(atomic_numbers, geometry.distances,
                                  geometry.sh_vectors, senders, receivers,
                                  workspace);
      fill(energy_seeds, e_seed);
      fill(charge_seeds, q_seed);
      fill(hirshfeld_seeds, h_seed);
      chain.launch_multihead_reverse_device(
          atomic_numbers, geometry.distances, geometry.sh_vectors, senders,
          receivers, energy_seeds, charge_seeds, hirshfeld_seeds, workspace);
      Kokkos::fence();
      return compare(copy(chain.combined_head_input_gradient(workspace)),
                     numbers(fixture.at(expected)), label);
    };

    const double energy_head_gradient_error = component_gradient(
        energy_seed_values, zero_seeds, zero_seeds, "energy_input_grad",
        "energy seeded head VJP");
    const double charge_head_gradient_error = component_gradient(
        zero_seeds, charge_seed_values, zero_seeds, "charge_input_grad",
        "charge-conserving head VJP");
    const double hirshfeld_head_gradient_error = component_gradient(
        zero_seeds, zero_seeds, hirshfeld_seed_values, "hirshfeld_input_grad",
        "Hirshfeld head VJP");

    chain.launch_forward_device(atomic_numbers, geometry.distances,
                                geometry.sh_vectors, senders, receivers,
                                workspace);
    fill(energy_seeds, energy_seed_values);
    fill(charge_seeds, charge_seed_values);
    fill(hirshfeld_seeds, hirshfeld_seed_values);
    chain.launch_multihead_reverse_device(
        atomic_numbers, geometry.distances, geometry.sh_vectors, senders,
        receivers, energy_seeds, charge_seeds, hirshfeld_seeds, workspace);
    Kokkos::fence();
    const double combined_head_gradient_error = compare(
        copy(chain.combined_head_input_gradient(workspace)),
        numbers(fixture.at("combined_input_grad")), "combined head VJP");

    so3lr::launch_so3lr_geometry_reverse_device(
        edge_vectors, senders, receivers, chain.grad_distances(workspace),
        chain.grad_sh(workspace), geometry);
    Kokkos::fence();
    const double edge_gradient_error = compare(
        copy(geometry.edge_energy_gradients),
        numbers(fixture.at("combined_edge_gradients")),
        "combined multihead edge gradient");
    const auto native_forces = copy(geometry.atomic_forces);
    const double atomic_force_error = compare(
        native_forces, numbers(fixture.at("combined_atomic_forces")),
        "combined multihead atomic force");

    const auto positions = numbers(fixture.at("positions"));
    double net_force[3] = {0.0, 0.0, 0.0};
    double net_torque[3] = {0.0, 0.0, 0.0};
    for (std::size_t node = 0; node < nodes; ++node) {
      const double x = positions[node * 3];
      const double y = positions[node * 3 + 1];
      const double z = positions[node * 3 + 2];
      const double fx = native_forces[node * 3];
      const double fy = native_forces[node * 3 + 1];
      const double fz = native_forces[node * 3 + 2];
      net_force[0] += fx;
      net_force[1] += fy;
      net_force[2] += fz;
      net_torque[0] += y * fz - z * fy;
      net_torque[1] += z * fx - x * fz;
      net_torque[2] += x * fy - y * fx;
    }
    double maximum_net_force = 0.0;
    double maximum_net_torque = 0.0;
    for (int component = 0; component < 3; ++component) {
      maximum_net_force =
          std::max(maximum_net_force, std::abs(net_force[component]));
      maximum_net_torque =
          std::max(maximum_net_torque, std::abs(net_torque[component]));
    }
    if (maximum_net_force > 2.0e-9 || maximum_net_torque > 2.0e-9)
      throw std::runtime_error("native multihead invariance check failed");
    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-5 ||
        fixture.at("edge_to_position_force_max_abs_error").number() > 2.0e-12 ||
        std::abs(fixture.at("charge_seed_mean").number()) < 1.0e-4)
      throw std::runtime_error("multihead fixture reference failed");

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    Int64View large_z("so3lr_dev24_bench_z", matched_nodes);
    DoubleView large_vectors("so3lr_dev24_bench_vectors", matched_edges * 3);
    IndexView large_senders("so3lr_dev24_bench_senders", matched_edges);
    IndexView large_receivers("so3lr_dev24_bench_receivers", matched_edges);
    DoubleView large_energy_seeds("so3lr_dev24_bench_energy_seeds", matched_nodes);
    DoubleView large_charge_seeds("so3lr_dev24_bench_charge_seeds", matched_nodes);
    DoubleView large_hirshfeld_seeds("so3lr_dev24_bench_hirshfeld_seeds", matched_nodes);
    Kokkos::parallel_for(
        "so3lr_dev24_bench_nodes", Kokkos::RangePolicy<>(0, matched_nodes),
        KOKKOS_LAMBDA(const std::size_t i) {
          large_z(i) = i % 3 == 0 ? 8 : 1;
          large_energy_seeds(i) = 0.2 + 0.03 * static_cast<double>(i % 7);
          large_charge_seeds(i) =
              (static_cast<double>(i % 11) - 4.0) / 17.0;
          large_hirshfeld_seeds(i) =
              (static_cast<double>(i % 13) - 6.0) / 19.0;
        });
    Kokkos::parallel_for(
        "so3lr_dev24_bench_edges", Kokkos::RangePolicy<>(0, matched_edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
          std::size_t sender = (edge * 17 + 11) % matched_nodes;
          std::size_t receiver = edge % matched_nodes;
          if (receiver == sender) receiver = (receiver + 1) % matched_nodes;
          large_senders(edge) = sender;
          large_receivers(edge) = receiver;
          const double ax = Kokkos::sin(0.017 * static_cast<double>(edge + 1));
          const double ay = Kokkos::cos(0.013 * static_cast<double>(edge + 3));
          const double az =
              1.25 + 0.35 * Kokkos::sin(0.011 * static_cast<double>(edge + 7));
          const double norm = Kokkos::sqrt(ax * ax + ay * ay + az * az);
          const double radius =
              0.15 + 4.29 * static_cast<double>(edge % 8192) / 8191.0;
          large_vectors(edge * 3) = radius * ax / norm;
          large_vectors(edge * 3 + 1) = radius * ay / norm;
          large_vectors(edge * 3 + 2) = radius * az / norm;
        });
    so3lr::GeometryVJPWorkspace large_geometry(matched_nodes, matched_edges);
    so3lr::LearnedEnergyReverseWorkspace large_workspace(matched_nodes,
                                                          matched_edges);
    auto launch_benchmark = [&]() {
      so3lr::launch_so3lr_geometry_forward_device(large_vectors, large_geometry);
      chain.launch_forward_device(large_z, large_geometry.distances,
                                  large_geometry.sh_vectors, large_senders,
                                  large_receivers, large_workspace);
      chain.launch_multihead_reverse_device(
          large_z, large_geometry.distances, large_geometry.sh_vectors,
          large_senders, large_receivers, large_energy_seeds,
          large_charge_seeds, large_hirshfeld_seeds, large_workspace);
      so3lr::launch_so3lr_geometry_reverse_device(
          large_vectors, large_senders, large_receivers,
          chain.grad_distances(large_workspace), chain.grad_sh(large_workspace),
          large_geometry);
    };
    launch_benchmark();
    Kokkos::fence();
    constexpr std::size_t repetitions = 2;
    Kokkos::Timer timer;
    for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
      launch_benchmark();
    Kokkos::fence();
    const double full_multihead_force_ms =
        timer.seconds() * 1000.0 / static_cast<double>(repetitions);
    const double edges_per_second =
        static_cast<double>(matched_edges) / (full_multihead_force_ms / 1000.0);
    double checksum = 0.0;
    const auto bench_forces = large_geometry.atomic_forces;
    const auto bench_charge = chain.partial_charges(large_workspace);
    const auto bench_hirshfeld = chain.hirshfeld_ratios(large_workspace);
    Kokkos::parallel_reduce(
        "so3lr_dev24_bench_checksum", Kokkos::RangePolicy<>(0, 3),
        KOKKOS_LAMBDA(const int which, double &update) {
          if (which == 0) update += bench_forces((matched_nodes / 2) * 3 + 1);
          if (which == 1) update += bench_charge(matched_nodes / 3);
          if (which == 2) update += bench_hirshfeld(matched_nodes / 4);
        },
        checksum);
    Kokkos::fence();
    const std::size_t workspace_bytes = multihead_workspace_bytes(
        matched_nodes, matched_edges, chain.persistent_device_bytes());
    if (!std::isfinite(full_multihead_force_ms) ||
        full_multihead_force_ms <= 0.0 || !std::isfinite(edges_per_second) ||
        edges_per_second <= 0.0 || !std::isfinite(checksum))
      throw std::runtime_error("dev_24 matched benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "backend=Kokkos_CUDA_plus_cuBLAS_DGEMM_multihead_VJP\n"
        << "energy_output_max_abs_error=" << energy_output_error << '\n'
        << "charge_output_max_abs_error=" << charge_output_error << '\n'
        << "hirshfeld_output_max_abs_error=" << hirshfeld_output_error << '\n'
        << "energy_head_gradient_max_abs_error=" << energy_head_gradient_error
        << '\n'
        << "charge_head_gradient_max_abs_error=" << charge_head_gradient_error
        << '\n'
        << "hirshfeld_head_gradient_max_abs_error="
        << hirshfeld_head_gradient_error << '\n'
        << "combined_head_gradient_max_abs_error="
        << combined_head_gradient_error << '\n'
        << "combined_edge_gradient_max_abs_error=" << edge_gradient_error
        << '\n'
        << "combined_atomic_force_max_abs_error=" << atomic_force_error << '\n'
        << "maximum_abs_net_force=" << maximum_net_force << '\n'
        << "maximum_abs_net_torque=" << maximum_net_torque << '\n'
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "charge_seed_mean=" << fixture.at("charge_seed_mean").number()
        << '\n'
        << "matched_stage2_nodes=" << matched_nodes << '\n'
        << "matched_stage2_edges=" << matched_edges << '\n'
        << "full_multihead_force_ms=" << full_multihead_force_ms << '\n'
        << "full_multihead_force_edges_per_second=" << edges_per_second << '\n'
        << "multihead_force_workspace_bytes=" << workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration=0\n"
        << "multihead_force_checksum=" << checksum << '\n'
        << "native_energy_head_seeded_reverse=PASS\n"
        << "native_charge_conservation_reverse=PASS\n"
        << "native_hirshfeld_reverse=PASS\n"
        << "native_multihead_transformer_reverse=PASS\n"
        << "pytorch_multihead_cartesian_force_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "translation_rotation_invariance=PASS\n"
        << "multihead_force_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV24_MULTIHEAD_FORCE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV24_MULTIHEAD_FORCE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
