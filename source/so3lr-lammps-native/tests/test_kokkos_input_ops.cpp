#include "so3lr/kokkos_input_ops.hpp"

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
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}
std::vector<double> flatten(const so3lr::Json &value) {
  std::vector<double> result;
  for (const auto &row : value.array())
    for (const auto &item : row.array()) result.push_back(item.number());
  return result;
}
void compare(const std::vector<double> &actual,
             const std::vector<double> &expected, double atol, double rtol,
             const std::string &label, double &maximum) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > atol + rtol * std::abs(expected[i]))
      throw std::runtime_error(label + " numerical mismatch at " +
                               std::to_string(i));
  }
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("test was not compiled with KOKKOS_ENABLE_CUDA");
#endif
    if (argc != 3) throw std::runtime_error("usage: test MODEL FIXTURE.json");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    std::vector<std::int64_t> atomic_numbers;
    for (const auto &item : fixture.at("atomic_numbers").array())
      atomic_numbers.push_back(static_cast<std::int64_t>(item.unsigned_integer()));
    const auto distances = numbers(fixture.at("distances"));
    const so3lr::KokkosInputOps runtime(model);
    const auto outputs = runtime.evaluate(atomic_numbers, distances);
    double embedding_error = 0.0, cutoff_error = 0.0, rbf_error = 0.0;
    compare(outputs.embedding, flatten(fixture.at("invariant_embedding")),
            0.0, 0.0, "embedding", embedding_error);
    compare(outputs.cutoff, numbers(fixture.at("physnet_cutoff")),
            3.0e-13, 3.0e-13, "cutoff", cutoff_error);
    compare(outputs.radial_basis, flatten(fixture.at("bernstein_rbf")),
            3.0e-13, 3.0e-13, "radial basis", rbf_error);
    const auto benchmark = runtime.benchmark(24000, 1000000, 20);
    if (!std::isfinite(benchmark.checksum) ||
        benchmark.milliseconds_per_iteration <= 0.0)
      throw std::runtime_error("invalid GPU benchmark result");

    std::cout << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "kokkos_cuda_enabled=1\n"
              << "persistent_device_bytes=" << runtime.persistent_device_bytes() << '\n'
              << "embedding_max_abs_error=" << embedding_error << '\n'
              << "cutoff_max_abs_error=" << cutoff_error << '\n'
              << "rbf_max_abs_error=" << rbf_error << '\n'
              << "benchmark_atoms=" << benchmark.atoms << '\n'
              << "benchmark_edges=" << benchmark.edges << '\n'
              << "benchmark_repetitions=" << benchmark.repetitions << '\n'
              << "benchmark_workspace_bytes=" << benchmark.workspace_bytes << '\n'
              << "benchmark_total_ms=" << benchmark.total_milliseconds << '\n'
              << "benchmark_ms_per_iteration=" << benchmark.milliseconds_per_iteration << '\n'
              << "benchmark_checksum=" << benchmark.checksum << '\n'
              << "kokkos_input_operator_equivalence=PASS\n"
              << "SO3LR_NATIVE_DEV3_GPU_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV3_GPU_TEST=FAIL reason=" << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}

