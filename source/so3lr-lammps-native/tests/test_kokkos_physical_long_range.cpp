#include "so3lr/kokkos_learned_energy_forces.hpp"
#include "so3lr/kokkos_physical_long_range.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
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
               const std::string &label,
               double absolute = 3.0e-7,
               double relative = 3.0e-7) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > absolute + relative * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i) +
                               " actual=" + std::to_string(actual[i]) +
                               " expected=" + std::to_string(expected[i]));
  }
  return maximum;
}

double sum(const DoubleView &view) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev25_sum", Kokkos::RangePolicy<>(0, view.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double &update) {
        update += view(i);
      },
      result);
  Kokkos::fence();
  return result;
}

so3lr::PhysicalLongRangeParameters parameters(const so3lr::Json &fixture) {
  const auto &value = fixture.at("parameters");
  so3lr::PhysicalLongRangeParameters result;
  result.ke = value.at("ke").number();
  result.electrostatic_sigma = value.at("electrostatic_sigma").number();
  result.cutoff = value.at("cutoff").number();
  result.electrostatic_cuton = value.at("electrostatic_cuton").number();
  result.fine_structure = value.at("fine_structure").number();
  result.bohr = value.at("bohr").number();
  result.hartree = value.at("hartree").number();
  result.dispersion_scale = value.at("dispersion_scale").number();
  result.dispersion_cuton = value.at("dispersion_cuton").number();
  result.pair_scale = value.at("pair_scale").number();
  return result;
}

std::size_t full_workspace_bytes(std::size_t nodes, std::size_t sr_edges,
                                 std::size_t lr_pairs,
                                 std::size_t persistent_bytes,
                                 std::size_t reference_elements) {
  constexpr std::size_t block_node_doubles = 4516;
  constexpr std::size_t block_edge_doubles = 1027;
  constexpr std::size_t chain_node_doubles =
      3 * block_node_doubles + 128 + 24 + 24 + 1220;
  constexpr std::size_t chain_edge_doubles =
      3 * block_edge_doubles + 1 + 24 + 1 + 24;
  const std::size_t chain =
      persistent_bytes +
      nodes * ((chain_node_doubles + 6) * sizeof(double) +
               sizeof(std::int64_t)) +
      sr_edges * ((chain_edge_doubles + 6) * sizeof(double) +
                  2 * sizeof(std::size_t));
  const std::size_t lr_inputs =
      lr_pairs * (3 * sizeof(double) + 2 * sizeof(std::size_t));
  return chain + lr_inputs +
         so3lr::physical_long_range_workspace_bytes(nodes, lr_pairs) +
         (3 * nodes + 2 * reference_elements) * sizeof(double);
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
      throw std::runtime_error("usage: test MODEL PHYSICAL_FORCE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-physical-lr-cartesian-force-fixture-v1" ||
        fixture.at("objective").string() !=
            "learned_energy_plus_electrostatics_plus_qdo_dispersion" ||
        fixture.at("vector_convention").string() !=
            "receiver_minus_sender" ||
        fixture.at("lr_topology").string() !=
            "unique_unordered_half_pairs")
      throw std::runtime_error("unexpected dev_25 fixture contract");

    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t sr_edges =
        static_cast<std::size_t>(fixture.at("sr_edges").unsigned_integer());
    const std::size_t lr_pairs =
        static_cast<std::size_t>(fixture.at("lr_pairs").unsigned_integer());
    const auto alpha_values = numbers(fixture.at("reference_alphas"));
    const auto c6_values = numbers(fixture.at("reference_c6"));
    if (alpha_values.size() != c6_values.size())
      throw std::runtime_error("reference dispersion tables differ in size");

    Int64View atomic_numbers("so3lr_dev25_z", nodes);
    DoubleView sr_vectors("so3lr_dev25_sr_vectors", sr_edges * 3);
    IndexView sr_senders("so3lr_dev25_sr_senders", sr_edges);
    IndexView sr_receivers("so3lr_dev25_sr_receivers", sr_edges);
    DoubleView lr_vectors("so3lr_dev25_lr_vectors", lr_pairs * 3);
    IndexView lr_senders("so3lr_dev25_lr_senders", lr_pairs);
    IndexView lr_receivers("so3lr_dev25_lr_receivers", lr_pairs);
    DoubleView reference_alphas("so3lr_dev25_reference_alphas",
                                alpha_values.size());
    DoubleView reference_c6("so3lr_dev25_reference_c6", c6_values.size());
    DoubleView energy_seeds("so3lr_dev25_energy_seeds", nodes);
    fill(atomic_numbers, integers(fixture.at("atomic_numbers")));
    fill(sr_vectors, numbers(fixture.at("sr_edge_vectors")));
    fill(sr_senders, indices(fixture.at("sr_senders")));
    fill(sr_receivers, indices(fixture.at("sr_receivers")));
    fill(lr_vectors, numbers(fixture.at("lr_pair_vectors")));
    fill(lr_senders, indices(fixture.at("lr_senders")));
    fill(lr_receivers, indices(fixture.at("lr_receivers")));
    fill(reference_alphas, alpha_values);
    fill(reference_c6, c6_values);
    Kokkos::deep_copy(energy_seeds, 1.0);

    const so3lr::KokkosLearnedEnergyReverseChain chain(model);
    if (!chain.chain_contract_verified() || !chain.shared_kokkos_stream())
      throw std::runtime_error("dev_25 native chain contract failed");
    so3lr::GeometryVJPWorkspace geometry(nodes, sr_edges);
    so3lr::LearnedEnergyReverseWorkspace chain_workspace(nodes, sr_edges);
    so3lr::PhysicalLongRangeWorkspace lr_workspace(nodes, lr_pairs);
    const auto physical_parameters = parameters(fixture);

    so3lr::launch_so3lr_geometry_forward_device(sr_vectors, geometry);
    chain.launch_forward_device(
        atomic_numbers, geometry.distances, geometry.sh_vectors,
        sr_senders, sr_receivers, chain_workspace);
    so3lr::launch_so3lr_physical_long_range_device(
        atomic_numbers, chain.partial_charges(chain_workspace),
        chain.hirshfeld_ratios(chain_workspace), reference_alphas, reference_c6,
        lr_vectors, lr_senders, lr_receivers, physical_parameters,
        lr_workspace);
    Kokkos::fence();

    const double energy_output_error = compare(
        copy(chain.atomic_energies(chain_workspace)),
        numbers(fixture.at("atomic_energies")), "learned atomic energies");
    const double charge_output_error = compare(
        copy(chain.partial_charges(chain_workspace)),
        numbers(fixture.at("partial_charges")), "partial charges");
    const double hirshfeld_output_error = compare(
        copy(chain.hirshfeld_ratios(chain_workspace)),
        numbers(fixture.at("hirshfeld_ratios")), "Hirshfeld ratios");
    const double charge_gradient_error = compare(
        copy(lr_workspace.charge_gradient),
        numbers(fixture.at("charge_gradient")), "physical dE/dq");
    const double hirshfeld_gradient_error = compare(
        copy(lr_workspace.hirshfeld_gradient),
        numbers(fixture.at("hirshfeld_gradient")), "physical dE/dh");
    const double radial_gradient_error = compare(
        copy(lr_workspace.pair_radial_gradient),
        numbers(fixture.at("lr_pair_radial_gradient")), "physical dE/dr",
        5.0e-7, 5.0e-7);
    const double pair_force_error = compare(
        copy(lr_workspace.pair_force_vectors),
        numbers(fixture.at("direct_lr_pair_force_vectors")),
        "direct LR pair forces", 5.0e-7, 5.0e-7);
    const double direct_force_error = compare(
        copy(lr_workspace.atomic_forces),
        numbers(fixture.at("direct_lr_atomic_forces")),
        "direct LR atomic forces", 5.0e-7, 5.0e-7);

    const double electrostatic_total = sum(lr_workspace.electrostatic_pair_energy);
    const double dispersion_total = sum(lr_workspace.dispersion_pair_energy);
    const double lr_atomic_total = sum(lr_workspace.atomic_energy);
    const double electrostatic_energy_error = std::abs(
        electrostatic_total - fixture.at("electrostatic_energy_total").number());
    const double dispersion_energy_error = std::abs(
        dispersion_total - fixture.at("dispersion_energy_total").number());
    const double lr_atomic_energy_error = std::abs(
        lr_atomic_total - fixture.at("physical_lr_energy_total").number());
    if (electrostatic_energy_error > 3.0e-7 ||
        dispersion_energy_error > 3.0e-7 || lr_atomic_energy_error > 3.0e-7)
      throw std::runtime_error("physical LR energy mismatch");

    chain.launch_multihead_reverse_device(
        atomic_numbers, geometry.distances, geometry.sh_vectors,
        sr_senders, sr_receivers, energy_seeds, lr_workspace.charge_gradient,
        lr_workspace.hirshfeld_gradient, chain_workspace);
    so3lr::launch_so3lr_geometry_reverse_device(
        sr_vectors, sr_senders, sr_receivers,
        chain.grad_distances(chain_workspace), chain.grad_sh(chain_workspace),
        geometry);
    Kokkos::fence();
    const double implicit_edge_error = compare(
        copy(geometry.edge_energy_gradients),
        numbers(fixture.at("implicit_sr_edge_gradients")),
        "implicit SR edge gradient", 8.0e-7, 8.0e-7);
    const double implicit_force_error = compare(
        copy(geometry.atomic_forces),
        numbers(fixture.at("implicit_sr_atomic_forces")),
        "implicit SR atomic force", 8.0e-7, 8.0e-7);

    DoubleView assembled_forces("so3lr_dev25_assembled_forces", nodes * 3);
    const auto implicit = geometry.atomic_forces;
    const auto direct = lr_workspace.atomic_forces;
    Kokkos::parallel_for(
        "so3lr_dev25_assemble_forces", Kokkos::RangePolicy<>(0, nodes * 3),
        KOKKOS_LAMBDA(const std::size_t i) {
          assembled_forces(i) = implicit(i) + direct(i);
        });
    Kokkos::fence();
    const double assembled_force_error = compare(
        copy(assembled_forces), numbers(fixture.at("assembled_atomic_forces")),
        "assembled physical forces", 1.2e-6, 1.2e-6);

    const std::size_t matched_nodes = 31865;
    const std::size_t matched_sr_edges = 954436;
    const std::size_t matched_lr_pairs = 954436;
    Int64View bench_z("so3lr_dev25_bench_z", matched_nodes);
    DoubleView bench_sr_vectors("so3lr_dev25_bench_sr_vectors",
                                matched_sr_edges * 3);
    IndexView bench_sr_senders("so3lr_dev25_bench_sr_senders",
                               matched_sr_edges);
    IndexView bench_sr_receivers("so3lr_dev25_bench_sr_receivers",
                                 matched_sr_edges);
    DoubleView bench_lr_vectors("so3lr_dev25_bench_lr_vectors",
                                matched_lr_pairs * 3);
    IndexView bench_lr_senders("so3lr_dev25_bench_lr_senders",
                               matched_lr_pairs);
    IndexView bench_lr_receivers("so3lr_dev25_bench_lr_receivers",
                                 matched_lr_pairs);
    DoubleView bench_energy_seeds("so3lr_dev25_bench_energy_seeds",
                                  matched_nodes);
    Kokkos::parallel_for(
        "so3lr_dev25_bench_nodes", Kokkos::RangePolicy<>(0, matched_nodes),
        KOKKOS_LAMBDA(const std::size_t i) {
          bench_z(i) = i % 3 == 0 ? 8 : 1;
          bench_energy_seeds(i) = 1.0;
        });
    Kokkos::parallel_for(
        "so3lr_dev25_bench_sr_edges",
        Kokkos::RangePolicy<>(0, matched_sr_edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
          std::size_t s = (edge * 17 + 11) % matched_nodes;
          std::size_t r = edge % matched_nodes;
          if (s == r) r = (r + 1) % matched_nodes;
          bench_sr_senders(edge) = s;
          bench_sr_receivers(edge) = r;
          const double ax = Kokkos::sin(0.017 * static_cast<double>(edge + 1));
          const double ay = Kokkos::cos(0.013 * static_cast<double>(edge + 3));
          const double az = 1.25 + 0.35 *
              Kokkos::sin(0.011 * static_cast<double>(edge + 7));
          const double norm = Kokkos::sqrt(ax * ax + ay * ay + az * az);
          const double radius =
              0.15 + 4.29 * static_cast<double>(edge % 8192) / 8191.0;
          bench_sr_vectors(edge * 3) = radius * ax / norm;
          bench_sr_vectors(edge * 3 + 1) = radius * ay / norm;
          bench_sr_vectors(edge * 3 + 2) = radius * az / norm;
        });
    Kokkos::parallel_for(
        "so3lr_dev25_bench_lr_pairs",
        Kokkos::RangePolicy<>(0, matched_lr_pairs),
        KOKKOS_LAMBDA(const std::size_t pair) {
          std::size_t s = (pair * 29 + 7) % matched_nodes;
          std::size_t r = (pair * 13 + 3) % matched_nodes;
          if (s == r) r = (r + 1) % matched_nodes;
          bench_lr_senders(pair) = s;
          bench_lr_receivers(pair) = r;
          const double ax = Kokkos::sin(0.007 * static_cast<double>(pair + 2));
          const double ay = Kokkos::cos(0.009 * static_cast<double>(pair + 5));
          const double az = 1.1 + 0.25 *
              Kokkos::sin(0.005 * static_cast<double>(pair + 9));
          const double norm = Kokkos::sqrt(ax * ax + ay * ay + az * az);
          const double radius =
              0.55 + 11.35 * static_cast<double>(pair % 16384) / 16383.0;
          bench_lr_vectors(pair * 3) = radius * ax / norm;
          bench_lr_vectors(pair * 3 + 1) = radius * ay / norm;
          bench_lr_vectors(pair * 3 + 2) = radius * az / norm;
        });
    so3lr::GeometryVJPWorkspace bench_geometry(matched_nodes,
                                               matched_sr_edges);
    so3lr::LearnedEnergyReverseWorkspace bench_chain(matched_nodes,
                                                     matched_sr_edges);
    so3lr::PhysicalLongRangeWorkspace bench_lr(matched_nodes,
                                               matched_lr_pairs);
    DoubleView bench_assembled("so3lr_dev25_bench_assembled",
                               matched_nodes * 3);

    auto launch_full = [&]() {
      so3lr::launch_so3lr_geometry_forward_device(bench_sr_vectors,
                                                   bench_geometry);
      chain.launch_forward_device(
          bench_z, bench_geometry.distances, bench_geometry.sh_vectors,
          bench_sr_senders, bench_sr_receivers, bench_chain);
      so3lr::launch_so3lr_physical_long_range_device(
          bench_z, chain.partial_charges(bench_chain),
          chain.hirshfeld_ratios(bench_chain), reference_alphas, reference_c6,
          bench_lr_vectors, bench_lr_senders, bench_lr_receivers,
          physical_parameters, bench_lr);
      chain.launch_multihead_reverse_device(
          bench_z, bench_geometry.distances, bench_geometry.sh_vectors,
          bench_sr_senders, bench_sr_receivers, bench_energy_seeds,
          bench_lr.charge_gradient, bench_lr.hirshfeld_gradient, bench_chain);
      so3lr::launch_so3lr_geometry_reverse_device(
          bench_sr_vectors, bench_sr_senders, bench_sr_receivers,
          chain.grad_distances(bench_chain), chain.grad_sh(bench_chain),
          bench_geometry);
      const auto implicit_force = bench_geometry.atomic_forces;
      const auto direct_force = bench_lr.atomic_forces;
      Kokkos::parallel_for(
          "so3lr_dev25_bench_assemble",
          Kokkos::RangePolicy<>(0, matched_nodes * 3),
          KOKKOS_LAMBDA(const std::size_t i) {
            bench_assembled(i) = implicit_force(i) + direct_force(i);
          });
    };
    launch_full();
    Kokkos::fence();
    constexpr std::size_t repetitions = 3;
    chain.launch_forward_device(
        bench_z, bench_geometry.distances, bench_geometry.sh_vectors,
        bench_sr_senders, bench_sr_receivers, bench_chain);
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i)
      so3lr::launch_so3lr_physical_long_range_device(
          bench_z, chain.partial_charges(bench_chain),
          chain.hirshfeld_ratios(bench_chain), reference_alphas, reference_c6,
          bench_lr_vectors, bench_lr_senders, bench_lr_receivers,
          physical_parameters, bench_lr);
    Kokkos::fence();
    const double lr_ms = timer.seconds() * 1000.0 /
                         static_cast<double>(repetitions);
    timer.reset();
    for (std::size_t i = 0; i < repetitions; ++i) launch_full();
    Kokkos::fence();
    const double full_ms = timer.seconds() * 1000.0 /
                           static_cast<double>(repetitions);
    const std::size_t workspace_bytes = full_workspace_bytes(
        matched_nodes, matched_sr_edges, matched_lr_pairs,
        chain.persistent_device_bytes(), alpha_values.size());
    double checksum = 0.0;
    const auto bench_qg = bench_lr.charge_gradient;
    const auto bench_hg = bench_lr.hirshfeld_gradient;
    const auto bench_force = bench_assembled;
    Kokkos::parallel_reduce(
        "so3lr_dev25_checksum", Kokkos::RangePolicy<>(0, 3),
        KOKKOS_LAMBDA(const int which, double &update) {
          if (which == 0) update += bench_qg(matched_nodes / 3);
          if (which == 1) update += bench_hg(matched_nodes / 2);
          if (which == 2) update += bench_force((matched_nodes / 4) * 3 + 1);
        },
        checksum);
    Kokkos::fence();
    if (!std::isfinite(checksum) || !std::isfinite(full_ms) || full_ms <= 0.0)
      throw std::runtime_error("dev_25 benchmark is not finite");

    std::cout << std::setprecision(17)
              << "learned_energy_output_max_abs_error=" << energy_output_error << '\n'
              << "partial_charge_output_max_abs_error=" << charge_output_error << '\n'
              << "hirshfeld_output_max_abs_error=" << hirshfeld_output_error << '\n'
              << "electrostatic_energy_max_abs_error=" << electrostatic_energy_error << '\n'
              << "dispersion_energy_max_abs_error=" << dispersion_energy_error << '\n'
              << "physical_lr_atomic_energy_max_abs_error=" << lr_atomic_energy_error << '\n'
              << "physical_charge_gradient_max_abs_error=" << charge_gradient_error << '\n'
              << "physical_hirshfeld_gradient_max_abs_error=" << hirshfeld_gradient_error << '\n'
              << "physical_radial_gradient_max_abs_error=" << radial_gradient_error << '\n'
              << "direct_lr_pair_force_max_abs_error=" << pair_force_error << '\n'
              << "direct_lr_atomic_force_max_abs_error=" << direct_force_error << '\n'
              << "implicit_sr_edge_gradient_max_abs_error=" << implicit_edge_error << '\n'
              << "implicit_sr_atomic_force_max_abs_error=" << implicit_force_error << '\n'
              << "assembled_force_max_abs_error=" << assembled_force_error << '\n'
              << "pytorch_finite_difference_max_scaled_error="
              << fixture.at("finite_difference_max_scaled_error").number() << '\n'
              << "pytorch_maximum_abs_net_force="
              << fixture.at("maximum_abs_net_force").number() << '\n'
              << "pytorch_maximum_abs_net_torque="
              << fixture.at("maximum_abs_net_torque").number() << '\n'
              << "matched_stage2_nodes=" << matched_nodes << '\n'
              << "matched_stage2_sr_edges=" << matched_sr_edges << '\n'
              << "synthetic_lr_half_pairs=" << matched_lr_pairs << '\n'
              << "physical_lr_kernel_ms=" << lr_ms << '\n'
              << "full_physical_force_ms=" << full_ms << '\n'
              << "full_physical_force_sr_edges_per_second="
              << static_cast<double>(matched_sr_edges) / (full_ms / 1000.0) << '\n'
              << "physical_force_workspace_bytes=" << workspace_bytes << '\n'
              << "host_boundary_bytes_per_iteration=0\n"
              << "benchmark_checksum=" << checksum << '\n'
              << "native_physical_electrostatic_energy=PASS\n"
              << "native_physical_dispersion_energy=PASS\n"
              << "native_physical_charge_hirshfeld_adjoints=PASS\n"
              << "native_physical_direct_lr_forces=PASS\n"
              << "native_full_so3lr_cartesian_forces=PASS\n"
              << "separate_sr_lr_graph_contract=PASS\n"
              << "pytorch_physical_long_range_reference=PASS\n"
              << "finite_difference_reference=PASS\n"
              << "translation_rotation_invariance=PASS\n"
              << "physical_force_device_residency=PASS\n"
              << "SO3LR_NATIVE_DEV25_PHYSICAL_FORCE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV25_PHYSICAL_FORCE_TEST=FAIL: "
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
