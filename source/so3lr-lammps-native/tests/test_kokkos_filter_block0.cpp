#include "so3lr/kokkos_filter_block0.hpp"

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
std::vector<double> flatten(const so3lr::Json &value) {
  std::vector<double> result;
  for (const auto &row : value.array())
    for (const auto &item : row.array()) result.push_back(item.number());
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
    if (error > 2.0e-11 + 2.0e-11 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i));
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
    if (argc != 3) throw std::runtime_error("usage: test MODEL FILTER_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const auto edges = static_cast<std::size_t>(fixture.at("edges").unsigned_integer());
    const auto radial_basis = flatten(fixture.at("radial_basis"));
    const auto ev_invariants = flatten(fixture.at("ev_invariants"));
    const auto zero_ev = flatten(fixture.at("zero_ev_invariants"));
    const so3lr::KokkosFilterBlock0 runtime(model);
    if (!runtime.aliases_verified())
      throw std::runtime_error("filter tensor aliases were not verified");
    const auto nonzero = runtime.evaluate(radial_basis, ev_invariants, edges);
    const auto physical = runtime.evaluate(radial_basis, zero_ev, edges);
    const double inv_error = compare(
        nonzero.invariant_filter, flatten(fixture.at("invariant_filter")),
        "nonzero invariant filter");
    const double ev_error = compare(
        nonzero.equivariant_filter, flatten(fixture.at("equivariant_filter")),
        "nonzero equivariant filter");
    const double inv_zero_error = compare(
        physical.invariant_filter,
        flatten(fixture.at("invariant_filter_zero_ev")),
        "zero-ev invariant filter");
    const double ev_zero_error = compare(
        physical.equivariant_filter,
        flatten(fixture.at("equivariant_filter_zero_ev")),
        "zero-ev equivariant filter");
    const auto benchmark = runtime.benchmark(1000000, 5);
    if (!std::isfinite(benchmark.checksum) ||
        benchmark.milliseconds_per_iteration <= 0.0)
      throw std::runtime_error("invalid block-0 filter benchmark");
    std::cout << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "filter_aliases_verified=1\n"
              << "persistent_device_bytes=" << runtime.persistent_device_bytes() << '\n'
              << "nonzero_inv_max_abs_error=" << inv_error << '\n'
              << "nonzero_ev_max_abs_error=" << ev_error << '\n'
              << "zero_ev_inv_max_abs_error=" << inv_zero_error << '\n'
              << "zero_ev_ev_max_abs_error=" << ev_zero_error << '\n'
              << "benchmark_edges=" << benchmark.edges << '\n'
              << "benchmark_repetitions=" << benchmark.repetitions << '\n'
              << "benchmark_workspace_bytes=" << benchmark.workspace_bytes << '\n'
              << "benchmark_total_ms=" << benchmark.total_milliseconds << '\n'
              << "benchmark_ms_per_iteration=" << benchmark.milliseconds_per_iteration << '\n'
              << "benchmark_checksum=" << benchmark.checksum << '\n'
              << "block0_filter_equivalence=PASS\n"
              << "SO3LR_NATIVE_DEV4_FILTER_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV4_FILTER_TEST=FAIL reason=" << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}

