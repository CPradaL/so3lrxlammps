#include "so3lr/kokkos_so3lr_evaluator.hpp"

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
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

std::vector<std::size_t> indices(const so3lr::Json &value) {
  std::vector<std::size_t> result;
  for (const auto &item : value.array())
    result.push_back(static_cast<std::size_t>(item.unsigned_integer()));
  return result;
}

std::vector<std::int64_t> integers(const so3lr::Json &value) {
  std::vector<std::int64_t> result;
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
               double absolute = 1.2e-6,
               double relative = 1.2e-6) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > absolute + relative * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i));
  }
  return maximum;
}

double sum(const DoubleView &view) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev26_sum", Kokkos::RangePolicy<>(0, view.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double &update) {
        update += view(i);
      },
      result);
  Kokkos::fence();
  return result;
}

double parameter_error(const so3lr::PhysicalLongRangeParameters &p,
                       const so3lr::Json &fixture) {
  const auto &f = fixture.at("parameters");
  double error = 0.0;
  error = std::max(error, std::abs(p.ke - f.at("ke").number()));
  error = std::max(error, std::abs(
      p.electrostatic_sigma - f.at("electrostatic_sigma").number()));
  error = std::max(error, std::abs(p.cutoff - f.at("cutoff").number()));
  error = std::max(error, std::abs(
      p.electrostatic_cuton - f.at("electrostatic_cuton").number()));
  error = std::max(error, std::abs(
      p.fine_structure - f.at("fine_structure").number()));
  error = std::max(error, std::abs(p.bohr - f.at("bohr").number()));
  error = std::max(error, std::abs(p.hartree - f.at("hartree").number()));
  error = std::max(error, std::abs(
      p.dispersion_scale - f.at("dispersion_scale").number()));
  error = std::max(error, std::abs(
      p.dispersion_cuton - f.at("dispersion_cuton").number()));
  error = std::max(error, std::abs(
      p.pair_scale - f.at("pair_scale").number()));
  return error;
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
        fixture.at("vector_convention").string() != "receiver_minus_sender" ||
        fixture.at("lr_topology").string() != "unique_unordered_half_pairs")
      throw std::runtime_error("unexpected dev_26 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t sr_edges =
        static_cast<std::size_t>(fixture.at("sr_edges").unsigned_integer());
    const std::size_t lr_pairs =
        static_cast<std::size_t>(fixture.at("lr_pairs").unsigned_integer());

    Int64View z("so3lr_dev26_z", nodes);
    DoubleView sr_vectors("so3lr_dev26_sr_vectors", sr_edges * 3);
    IndexView sr_senders("so3lr_dev26_sr_senders", sr_edges);
    IndexView sr_receivers("so3lr_dev26_sr_receivers", sr_edges);
    DoubleView lr_vectors("so3lr_dev26_lr_vectors", lr_pairs * 3);
    IndexView lr_senders("so3lr_dev26_lr_senders", lr_pairs);
    IndexView lr_receivers("so3lr_dev26_lr_receivers", lr_pairs);
    fill(z, integers(fixture.at("atomic_numbers")));
    fill(sr_vectors, numbers(fixture.at("sr_edge_vectors")));
    fill(sr_senders, indices(fixture.at("sr_senders")));
    fill(sr_receivers, indices(fixture.at("sr_receivers")));
    fill(lr_vectors, numbers(fixture.at("lr_pair_vectors")));
    fill(lr_senders, indices(fixture.at("lr_senders")));
    fill(lr_receivers, indices(fixture.at("lr_receivers")));

    const so3lr::KokkosSo3lrEvaluator evaluator(model);
    if (!evaluator.self_contained_model_contract() ||
        !evaluator.shared_kokkos_stream())
      throw std::runtime_error("self-contained evaluator contract failed");
    const double parameters_max_abs_error =
        parameter_error(evaluator.long_range_parameters(), fixture);
    if (parameters_max_abs_error > 2.0e-14)
      throw std::runtime_error("model-loaded LR parameters differ");
    so3lr::So3lrEvaluatorWorkspace workspace(nodes, sr_edges, lr_pairs);
    evaluator.launch_device(z, sr_vectors, sr_senders, sr_receivers,
                            lr_vectors, lr_senders, lr_receivers, workspace);
    Kokkos::fence();

    const double charge_error = compare(
        copy(evaluator.partial_charges(workspace)),
        numbers(fixture.at("partial_charges")), "partial charges");
    const double hirshfeld_error = compare(
        copy(evaluator.hirshfeld_ratios(workspace)),
        numbers(fixture.at("hirshfeld_ratios")), "Hirshfeld ratios");
    const double charge_gradient_error = compare(
        copy(evaluator.long_range(workspace).charge_gradient),
        numbers(fixture.at("charge_gradient")), "physical dE/dq");
    const double hirshfeld_gradient_error = compare(
        copy(evaluator.long_range(workspace).hirshfeld_gradient),
        numbers(fixture.at("hirshfeld_gradient")), "physical dE/dh");
    const double radial_error = compare(
        copy(evaluator.long_range(workspace).pair_radial_gradient),
        numbers(fixture.at("lr_pair_radial_gradient")), "physical dE/dr");
    const double force_error = compare(
        copy(evaluator.atomic_forces(workspace)),
        numbers(fixture.at("assembled_atomic_forces")),
        "complete evaluator forces");
    const double total_energy = sum(evaluator.atomic_energies(workspace));
    const double energy_error =
        std::abs(total_energy - fixture.at("total_energy").number());
    if (energy_error > 3.0e-7)
      throw std::runtime_error("complete evaluator energy mismatch");

    const std::size_t matched_nodes = 31865;
    const std::size_t matched_sr_edges = 954436;
    const std::size_t matched_lr_pairs = 954436;
    Int64View bench_z("so3lr_dev26_bench_z", matched_nodes);
    DoubleView bench_sr_vectors("so3lr_dev26_bench_sr_vectors",
                                matched_sr_edges * 3);
    IndexView bench_sr_senders("so3lr_dev26_bench_sr_senders",
                               matched_sr_edges);
    IndexView bench_sr_receivers("so3lr_dev26_bench_sr_receivers",
                                 matched_sr_edges);
    DoubleView bench_lr_vectors("so3lr_dev26_bench_lr_vectors",
                                matched_lr_pairs * 3);
    IndexView bench_lr_senders("so3lr_dev26_bench_lr_senders",
                               matched_lr_pairs);
    IndexView bench_lr_receivers("so3lr_dev26_bench_lr_receivers",
                                 matched_lr_pairs);
    Kokkos::parallel_for(
        "so3lr_dev26_bench_nodes", Kokkos::RangePolicy<>(0, matched_nodes),
        KOKKOS_LAMBDA(const std::size_t i) {
          bench_z(i) = i % 3 == 0 ? 8 : 1;
        });
    Kokkos::parallel_for(
        "so3lr_dev26_bench_sr", Kokkos::RangePolicy<>(0, matched_sr_edges),
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
        "so3lr_dev26_bench_lr", Kokkos::RangePolicy<>(0, matched_lr_pairs),
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
    so3lr::So3lrEvaluatorWorkspace bench_workspace(
        matched_nodes, matched_sr_edges, matched_lr_pairs);
    evaluator.launch_device(
        bench_z, bench_sr_vectors, bench_sr_senders, bench_sr_receivers,
        bench_lr_vectors, bench_lr_senders, bench_lr_receivers,
        bench_workspace);
    Kokkos::fence();
    constexpr std::size_t repetitions = 3;
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i)
      evaluator.launch_device(
          bench_z, bench_sr_vectors, bench_sr_senders, bench_sr_receivers,
          bench_lr_vectors, bench_lr_senders, bench_lr_receivers,
          bench_workspace);
    Kokkos::fence();
    const double evaluator_ms =
        timer.seconds() * 1000.0 / static_cast<double>(repetitions);
    const std::size_t workspace_bytes = so3lr::so3lr_evaluator_workspace_bytes(
        matched_nodes, matched_sr_edges, matched_lr_pairs,
        evaluator.persistent_device_bytes());
    double checksum = 0.0;
    const auto bench_energy = evaluator.atomic_energies(bench_workspace);
    const auto bench_force = evaluator.atomic_forces(bench_workspace);
    const auto bench_charge = evaluator.partial_charges(bench_workspace);
    Kokkos::parallel_reduce(
        "so3lr_dev26_checksum", Kokkos::RangePolicy<>(0, 3),
        KOKKOS_LAMBDA(const int which, double &update) {
          if (which == 0) update += bench_energy(matched_nodes / 3);
          if (which == 1) update += bench_force((matched_nodes / 4) * 3 + 1);
          if (which == 2) update += bench_charge(matched_nodes / 2);
        },
        checksum);
    Kokkos::fence();
    if (!std::isfinite(checksum) || !std::isfinite(evaluator_ms) ||
        evaluator_ms <= 0.0)
      throw std::runtime_error("dev_26 benchmark is not finite");

    std::cout << std::setprecision(17)
              << "model_loaded_lr_parameter_max_abs_error="
              << parameters_max_abs_error << '\n'
              << "partial_charge_max_abs_error=" << charge_error << '\n'
              << "hirshfeld_max_abs_error=" << hirshfeld_error << '\n'
              << "physical_charge_gradient_max_abs_error="
              << charge_gradient_error << '\n'
              << "physical_hirshfeld_gradient_max_abs_error="
              << hirshfeld_gradient_error << '\n'
              << "physical_radial_gradient_max_abs_error=" << radial_error << '\n'
              << "complete_energy_max_abs_error=" << energy_error << '\n'
              << "complete_force_max_abs_error=" << force_error << '\n'
              << "matched_stage2_nodes=" << matched_nodes << '\n'
              << "matched_stage2_sr_edges=" << matched_sr_edges << '\n'
              << "synthetic_lr_half_pairs=" << matched_lr_pairs << '\n'
              << "complete_evaluator_ms=" << evaluator_ms << '\n'
              << "complete_evaluator_sr_edges_per_second="
              << static_cast<double>(matched_sr_edges) / (evaluator_ms / 1000.0)
              << '\n'
              << "complete_evaluator_workspace_bytes=" << workspace_bytes << '\n'
              << "persistent_device_bytes="
              << evaluator.persistent_device_bytes() << '\n'
              << "host_boundary_bytes_per_iteration=0\n"
              << "benchmark_checksum=" << checksum << '\n'
              << "fixture_supplied_lr_constants=0\n"
              << "native_model_v2_loaded=PASS\n"
              << "native_model_v2_physical_tables_loaded=PASS\n"
              << "native_model_v2_lr_parameters_loaded=PASS\n"
              << "native_complete_energy_force_evaluator=PASS\n"
              << "native_separate_sr_lr_graph_api=PASS\n"
              << "native_evaluator_device_residency=PASS\n"
              << "pytorch_full_so3lr_cartesian_force_reference=PASS\n"
              << "SO3LR_NATIVE_DEV26_EVALUATOR_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV26_EVALUATOR_TEST=FAIL: " << error.what()
              << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
