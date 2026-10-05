#include "so3lr/kokkos_learned_energy_forces.hpp"

#include <Kokkos_Core.hpp>

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
    if (error > 1.0e-7 + 1.0e-7 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i) + " actual=" +
                               std::to_string(actual[i]) + " expected=" +
                               std::to_string(expected[i]));
  }
  return maximum;
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
      throw std::runtime_error("usage: test MODEL CARTESIAN_FORCE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-learned-energy-cartesian-force-fixture-v1" ||
        fixture.at("geometry").string() !=
            "physical_water_dimer_nonperiodic" ||
        fixture.at("vector_convention").string() !=
            "receiver_minus_sender" ||
        fixture.at("spherical_harmonic_argument").string() !=
            "negative_normalized_edge_vector")
      throw std::runtime_error("unexpected dev_23 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t edges =
        static_cast<std::size_t>(fixture.at("edges").unsigned_integer());

    Int64View atomic_numbers("so3lr_dev23_z", nodes);
    DoubleView edge_vectors("so3lr_dev23_edge_vectors", edges * 3);
    IndexView senders("so3lr_dev23_senders", edges);
    IndexView receivers("so3lr_dev23_receivers", edges);
    fill(atomic_numbers, integers(fixture.at("atomic_numbers")));
    fill(edge_vectors, numbers(fixture.at("edge_vectors")));
    fill(senders, indices(fixture.at("senders")));
    fill(receivers, indices(fixture.at("receivers")));

    so3lr::GeometryVJPWorkspace probe_workspace(nodes, edges);
    so3lr::launch_so3lr_geometry_forward_device(edge_vectors, probe_workspace);
    Kokkos::fence();
    const double distance_forward_error = compare(
        copy(probe_workspace.distances), numbers(fixture.at("distances")),
        "geometry distance forward");
    const double sh_forward_error = compare(
        copy(probe_workspace.sh_vectors), numbers(fixture.at("sh_vectors")),
        "geometry SH forward");
    DoubleView probe_grad_distances("so3lr_dev23_probe_grad_distance", edges);
    DoubleView probe_grad_sh("so3lr_dev23_probe_grad_sh", edges * 24);
    fill(probe_grad_distances, numbers(fixture.at("probe_grad_distances")));
    fill(probe_grad_sh, numbers(fixture.at("probe_grad_sh")));
    so3lr::launch_so3lr_geometry_reverse_device(
        edge_vectors, senders, receivers, probe_grad_distances, probe_grad_sh,
        probe_workspace);
    Kokkos::fence();
    const double probe_edge_error = compare(
        copy(probe_workspace.edge_energy_gradients),
        numbers(fixture.at("probe_edge_energy_gradients")),
        "probe edge geometry gradient");
    const double probe_force_error = compare(
        copy(probe_workspace.atomic_forces),
        numbers(fixture.at("probe_atomic_forces")),
        "probe atomic force scatter");

    const so3lr::KokkosLearnedEnergyForces force_model(model);
    if (!force_model.force_contract_verified() ||
        !force_model.shared_kokkos_stream())
      throw std::runtime_error("dev_23 force-chain runtime contract failed");
    so3lr::LearnedEnergyForcesWorkspace workspace(nodes, edges);
    force_model.launch_device(atomic_numbers, edge_vectors, senders,
                              receivers, workspace);
    Kokkos::fence();
    const double learned_distance_error = compare(
        copy(force_model.distances(workspace)),
        numbers(fixture.at("distances")), "learned-force distances");
    const double learned_sh_error = compare(
        copy(force_model.sh_vectors(workspace)),
        numbers(fixture.at("sh_vectors")), "learned-force SH");
    const double atomic_energy_error = compare(
        copy(force_model.atomic_energies(workspace)),
        numbers(fixture.at("atomic_energies")), "atomic learned energies");
    const auto reductions = copy(force_model.reductions(workspace));
    const double total_energy_error =
        std::abs(reductions.at(1) - fixture.at("total_energy").number());
    if (total_energy_error >
        1.0e-7 + 1.0e-7 * std::abs(fixture.at("total_energy").number()))
      throw std::runtime_error("total learned energy mismatch");
    const double edge_gradient_error = compare(
        copy(force_model.edge_energy_gradients(workspace)),
        numbers(fixture.at("edge_energy_gradients")),
        "learned-energy edge gradient");
    const auto native_forces = copy(force_model.atomic_forces(workspace));
    const double atomic_force_error = compare(
        native_forces, numbers(fixture.at("atomic_forces")),
        "learned-energy Cartesian force");

    double net_force[3] = {0.0, 0.0, 0.0};
    double net_torque[3] = {0.0, 0.0, 0.0};
    const auto positions = numbers(fixture.at("positions"));
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
      throw std::runtime_error("native force invariance check failed");
    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-5 ||
        fixture.at("edge_to_position_force_max_abs_error").number() > 2.0e-12)
      throw std::runtime_error("fixture Cartesian-force reference failed");

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    const auto benchmark = force_model.benchmark(matched_nodes, matched_edges, 2);
    if (benchmark.nodes != matched_nodes || benchmark.edges != matched_edges ||
        benchmark.host_boundary_bytes_per_iteration != 0 ||
        !std::isfinite(benchmark.geometry_forward_milliseconds) ||
        benchmark.geometry_forward_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.geometry_reverse_milliseconds) ||
        benchmark.geometry_reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.full_energy_force_milliseconds) ||
        benchmark.full_energy_force_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("dev_23 matched benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "backend=Kokkos_CUDA_plus_cuBLAS_DGEMM_VJP\n"
        << "vector_convention=receiver_minus_sender\n"
        << "spherical_harmonic_degrees=1,2,3,4\n"
        << "spherical_harmonic_argument=negative_normalized_edge_vector\n"
        << "distance_forward_max_abs_error=" << distance_forward_error << '\n'
        << "sh_forward_max_abs_error=" << sh_forward_error << '\n'
        << "probe_edge_gradient_max_abs_error=" << probe_edge_error << '\n'
        << "probe_atomic_force_max_abs_error=" << probe_force_error << '\n'
        << "learned_distance_max_abs_error=" << learned_distance_error << '\n'
        << "learned_sh_max_abs_error=" << learned_sh_error << '\n'
        << "learned_atomic_energy_max_abs_error=" << atomic_energy_error << '\n'
        << "learned_total_energy_error=" << total_energy_error << '\n'
        << "learned_edge_gradient_max_abs_error=" << edge_gradient_error << '\n'
        << "learned_atomic_force_max_abs_error=" << atomic_force_error << '\n'
        << "maximum_abs_net_force=" << maximum_net_force << '\n'
        << "maximum_abs_net_torque=" << maximum_net_torque << '\n'
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << benchmark.nodes << '\n'
        << "matched_stage2_edges=" << benchmark.edges << '\n'
        << "geometry_forward_ms=" << benchmark.geometry_forward_milliseconds
        << '\n'
        << "geometry_reverse_scatter_ms="
        << benchmark.geometry_reverse_milliseconds << '\n'
        << "full_learned_energy_force_ms="
        << benchmark.full_energy_force_milliseconds << '\n'
        << "full_learned_energy_force_edges_per_second="
        << benchmark.edges_per_second << '\n'
        << "learned_energy_force_workspace_bytes=" << benchmark.workspace_bytes
        << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "force_checksum=" << benchmark.checksum << '\n'
        << "native_so3lr_spherical_harmonics_forward=PASS\n"
        << "native_geometry_vjp_probe=PASS\n"
        << "pytorch_learned_energy_cartesian_force_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "translation_rotation_invariance=PASS\n"
        << "learned_energy_cartesian_force_equivalence=PASS\n"
        << "learned_energy_cartesian_force_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV23_CARTESIAN_FORCE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV23_CARTESIAN_FORCE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
