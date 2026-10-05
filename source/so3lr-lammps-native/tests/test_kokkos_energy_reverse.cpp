#include "so3lr/kokkos_forward_model.hpp"
#include "so3lr/kokkos_output_heads.hpp"

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
               const std::string &label, double tolerance = 5.0e-10) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > tolerance + tolerance * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i));
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
      throw std::runtime_error("usage: test MODEL ENERGY_REVERSE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-energy-head-reverse-fixture-v1" ||
        fixture.at("geometry").string() !=
            "physical_water_dimer_nonperiodic")
      throw std::runtime_error("unexpected energy-reverse fixture");

    const so3lr::KokkosOutputHeads heads(model);
    const auto gradient = heads.evaluate_energy_input_gradient(
        numbers(fixture.at("final_inv")),
        integers(fixture.at("atomic_numbers")));
    const double gradient_error =
        compare(gradient, numbers(fixture.at("energy_input_gradient")),
                "energy-input gradient");

    const auto selected = indices(fixture.at("finite_difference_indices"));
    const auto finite = numbers(fixture.at("finite_difference_values"));
    const auto autograd = numbers(
        fixture.at("autograd_values_at_finite_difference_indices"));
    const double finite_difference_error =
        compare(finite, autograd, "finite-difference reference", 2.0e-7);
    if (selected.size() != finite.size())
      throw std::runtime_error("finite-difference index contract mismatch");
    for (std::size_t i = 0; i < selected.size(); ++i) {
      if (selected[i] >= gradient.size() ||
          std::abs(gradient[selected[i]] - finite[i]) > 2.0e-7)
        throw std::runtime_error("native finite-difference mismatch");
    }

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    const auto reverse_benchmark =
        heads.benchmark_energy_reverse(matched_nodes, 50);
    const so3lr::KokkosForwardModel forward(model);
    const auto forward_benchmark =
        forward.benchmark(matched_nodes, matched_edges, 10);
    if (reverse_benchmark.host_boundary_bytes_per_iteration != 0 ||
        reverse_benchmark.nodes != matched_nodes ||
        !std::isfinite(reverse_benchmark.milliseconds) ||
        reverse_benchmark.milliseconds <= 0.0 ||
        !std::isfinite(reverse_benchmark.nodes_per_second) ||
        reverse_benchmark.nodes_per_second <= 0.0 ||
        !std::isfinite(reverse_benchmark.checksum) ||
        forward_benchmark.host_boundary_bytes_per_iteration != 0 ||
        forward_benchmark.nodes != matched_nodes ||
        forward_benchmark.edges != matched_edges ||
        !std::isfinite(forward_benchmark.integrated_milliseconds) ||
        forward_benchmark.integrated_milliseconds <= 0.0)
      throw std::runtime_error("dev_16 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "reverse_scope=learned_energy_head_to_final_invariant_features\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "physical_geometry_fixture=1\n"
        << "energy_input_gradient_max_abs_error=" << gradient_error << '\n'
        << "finite_difference_max_abs_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << forward_benchmark.nodes << '\n'
        << "matched_stage2_edges=" << forward_benchmark.edges << '\n'
        << "matched_forward_ms="
        << forward_benchmark.integrated_milliseconds << '\n'
        << "matched_forward_edges_per_second="
        << forward_benchmark.edges_per_second << '\n'
        << "matched_forward_workspace_bytes="
        << forward_benchmark.workspace_bytes << '\n'
        << "energy_head_reverse_ms=" << reverse_benchmark.milliseconds
        << '\n'
        << "energy_head_reverse_nodes_per_second="
        << reverse_benchmark.nodes_per_second << '\n'
        << "energy_head_reverse_workspace_bytes="
        << reverse_benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << reverse_benchmark.host_boundary_bytes_per_iteration << '\n'
        << "energy_head_reverse_checksum=" << reverse_benchmark.checksum
        << '\n'
        << "pytorch_autograd_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "native_energy_head_reverse_equivalence=PASS\n"
        << "native_energy_head_reverse_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV16_ENERGY_REVERSE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV16_ENERGY_REVERSE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
