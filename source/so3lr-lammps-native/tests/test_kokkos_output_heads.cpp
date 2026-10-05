#include "so3lr/kokkos_output_heads.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
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

void compare_scalar(double actual, double expected, double absolute_tolerance,
                    const std::string &label) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) >
          absolute_tolerance + 5.0e-10 * std::abs(expected))
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
      throw std::runtime_error("usage: test MODEL OUTPUT_HEADS_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-output-heads-fixture-v1" ||
        fixture.at("geometry").string() !=
            "physical_water_dimer_nonperiodic")
      throw std::runtime_error("unexpected output-head fixture contract");

    const so3lr::KokkosOutputHeads heads(model);
    if (!heads.checkpoint_contract_verified() ||
        !heads.shared_kokkos_stream())
      throw std::runtime_error("output-head device contract failed");
    const double total_charge = fixture.at("total_charge").number();
    const auto actual = heads.evaluate(
        numbers(fixture.at("final_inv")),
        integers(fixture.at("atomic_numbers")), total_charge);
    const double energy_error =
        compare(actual.atomic_energies,
                numbers(fixture.at("atomic_energies")), "atomic energies");
    const double raw_charge_error =
        compare(actual.raw_charges, numbers(fixture.at("raw_charges")),
                "raw partial charges");
    const double partial_charge_error =
        compare(actual.partial_charges,
                numbers(fixture.at("partial_charges")), "partial charges");
    const double hirshfeld_error =
        compare(actual.hirshfeld_ratios,
                numbers(fixture.at("hirshfeld_ratios")),
                "Hirshfeld ratios");
    compare_scalar(actual.learned_sr_energy,
                   fixture.at("learned_sr_energy").number(), 5.0e-10,
                   "learned short-range energy");
    compare_scalar(actual.charge_sum, total_charge, 2.0e-12,
                   "charge conservation");

    constexpr std::size_t benchmark_nodes = 24000;
    const auto benchmark = heads.benchmark(benchmark_nodes, 20);
    if (benchmark.host_boundary_bytes_per_iteration != 0 ||
        benchmark.nodes != benchmark_nodes || benchmark.repetitions != 20 ||
        benchmark.persistent_device_bytes !=
            heads.persistent_device_bytes() ||
        benchmark.workspace_bytes <= benchmark.persistent_device_bytes ||
        !std::isfinite(benchmark.milliseconds) ||
        benchmark.milliseconds <= 0.0 ||
        !std::isfinite(benchmark.nodes_per_second) ||
        benchmark.nodes_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("output-head benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=isolated_native_energy_charge_hirshfeld_heads\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "physical_geometry_fixture=1\n"
        << "shared_kokkos_cublas_stream=1\n"
        << "atomic_energy_max_abs_error=" << energy_error << '\n'
        << "raw_charge_max_abs_error=" << raw_charge_error << '\n'
        << "partial_charge_max_abs_error=" << partial_charge_error << '\n'
        << "charge_sum=" << actual.charge_sum << '\n'
        << "hirshfeld_max_abs_error=" << hirshfeld_error << '\n'
        << "benchmark_nodes=" << benchmark.nodes << '\n'
        << "benchmark_repetitions=" << benchmark.repetitions << '\n'
        << "output_heads_ms=" << benchmark.milliseconds << '\n'
        << "output_heads_nodes_per_second=" << benchmark.nodes_per_second
        << '\n'
        << "persistent_device_bytes=" << benchmark.persistent_device_bytes
        << '\n'
        << "output_heads_workspace_bytes=" << benchmark.workspace_bytes
        << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "benchmark_checksum=" << benchmark.checksum << '\n'
        << "energy_head_pytorch_equivalence=PASS\n"
        << "charge_head_pytorch_equivalence=PASS\n"
        << "charge_neutrality_device=PASS\n"
        << "hirshfeld_head_pytorch_equivalence=PASS\n"
        << "output_heads_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV14_OUTPUT_HEADS_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV14_OUTPUT_HEADS_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
