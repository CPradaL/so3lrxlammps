#include "so3lr/kokkos_forward_model.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

std::vector<std::int64_t> integers(const so3lr::Json &value) {
  std::vector<std::int64_t> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<std::int64_t>(item.unsigned_integer()));
  return result;
}

std::vector<std::size_t> indices(const so3lr::Json &value) {
  std::vector<std::size_t> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<std::size_t>(item.unsigned_integer()));
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
    if (error > 5.0e-10 + 5.0e-10 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i));
  }
  return maximum;
}

void compare_scalar(double actual, double expected, double tolerance,
                    const std::string &label) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) >
          tolerance + 5.0e-10 * std::abs(expected))
    throw std::runtime_error(label + " mismatch");
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
      throw std::runtime_error("usage: test MODEL FORWARD_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-complete-forward-fixture-v1" ||
        fixture.at("geometry").string() !=
            "physical_water_dimer_nonperiodic")
      throw std::runtime_error("unexpected complete-forward fixture");

    const so3lr::KokkosForwardModel forward(model);
    if (!forward.device_bridge_verified())
      throw std::runtime_error("transformer-to-heads bridge failed");
    const double total_charge = fixture.at("total_charge").number();
    const auto actual = forward.evaluate(
        integers(fixture.at("atomic_numbers")),
        numbers(fixture.at("distances")),
        numbers(fixture.at("sh_vectors")), indices(fixture.at("senders")),
        indices(fixture.at("receivers")), total_charge);
    const double inv_error =
        compare(actual.final_inv, numbers(fixture.at("final_inv")),
                "complete-forward invariant features");
    const double ev_error =
        compare(actual.final_ev, numbers(fixture.at("final_ev")),
                "complete-forward equivariant features");
    const double energy_error =
        compare(actual.heads.atomic_energies,
                numbers(fixture.at("atomic_energies")), "atomic energies");
    const double raw_charge_error =
        compare(actual.heads.raw_charges,
                numbers(fixture.at("raw_charges")), "raw charges");
    const double partial_charge_error =
        compare(actual.heads.partial_charges,
                numbers(fixture.at("partial_charges")), "partial charges");
    const double hirshfeld_error =
        compare(actual.heads.hirshfeld_ratios,
                numbers(fixture.at("hirshfeld_ratios")),
                "Hirshfeld ratios");
    compare_scalar(actual.heads.learned_sr_energy,
                   fixture.at("learned_sr_energy").number(), 5.0e-10,
                   "learned short-range energy");
    compare_scalar(actual.heads.charge_sum, total_charge, 2.0e-12,
                   "charge conservation");

    constexpr std::size_t benchmark_nodes = 24000;
    constexpr std::size_t benchmark_edges = 1000000;
    const auto benchmark =
        forward.benchmark(benchmark_nodes, benchmark_edges, 10);
    if (benchmark.host_boundary_bytes_per_iteration != 0 ||
        benchmark.nodes != benchmark_nodes ||
        benchmark.edges != benchmark_edges || benchmark.repetitions != 10 ||
        benchmark.persistent_device_bytes !=
            forward.persistent_device_bytes() ||
        benchmark.workspace_bytes <= benchmark.persistent_device_bytes ||
        !std::isfinite(benchmark.transformer_milliseconds) ||
        benchmark.transformer_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.output_heads_milliseconds) ||
        benchmark.output_heads_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.integrated_milliseconds) ||
        benchmark.integrated_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("complete-forward benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=three_transformers_to_native_output_heads\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "physical_geometry_fixture=1\n"
        << "stack_to_heads_device_view_bridge=1\n"
        << "final_inv_max_abs_error=" << inv_error << '\n'
        << "final_ev_max_abs_error=" << ev_error << '\n'
        << "atomic_energy_max_abs_error=" << energy_error << '\n'
        << "raw_charge_max_abs_error=" << raw_charge_error << '\n'
        << "partial_charge_max_abs_error=" << partial_charge_error << '\n'
        << "charge_sum=" << actual.heads.charge_sum << '\n'
        << "hirshfeld_max_abs_error=" << hirshfeld_error << '\n'
        << "benchmark_nodes=" << benchmark.nodes << '\n'
        << "benchmark_edges=" << benchmark.edges << '\n'
        << "benchmark_repetitions=" << benchmark.repetitions << '\n'
        << "transformer_ms=" << benchmark.transformer_milliseconds << '\n'
        << "output_heads_ms=" << benchmark.output_heads_milliseconds << '\n'
        << "phase_sum_ms=" << benchmark.phase_sum_milliseconds << '\n'
        << "integrated_forward_ms=" << benchmark.integrated_milliseconds
        << '\n'
        << "integrated_edges_per_second=" << benchmark.edges_per_second
        << '\n'
        << "persistent_device_bytes=" << benchmark.persistent_device_bytes
        << '\n'
        << "integrated_workspace_bytes=" << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "benchmark_checksum=" << benchmark.checksum << '\n'
        << "stack_to_heads_device_bridge=PASS\n"
        << "complete_learned_forward_pytorch_equivalence=PASS\n"
        << "complete_learned_forward_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV15_FORWARD_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV15_FORWARD_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
