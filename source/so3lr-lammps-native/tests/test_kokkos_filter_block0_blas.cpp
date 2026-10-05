#include "so3lr/kokkos_filter_block0_blas.hpp"

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
    if (error > 3.0e-11 + 3.0e-11 * std::abs(expected[i]))
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
    const so3lr::KokkosFilterBlock0 baseline(model);
    const so3lr::KokkosFilterBlock0Blas optimized(model);
    if (!optimized.aliases_verified() || !optimized.shared_kokkos_stream())
      throw std::runtime_error("optimized filter runtime contract failed");

    const auto baseline_nonzero = baseline.evaluate(radial_basis, ev_invariants, edges);
    const auto optimized_nonzero = optimized.evaluate(radial_basis, ev_invariants, edges);
    const auto optimized_zero = optimized.evaluate(radial_basis, zero_ev, edges);
    const double inv_reference_error = compare(
        optimized_nonzero.invariant_filter,
        flatten(fixture.at("invariant_filter")), "BLAS invariant/PyTorch");
    const double ev_reference_error = compare(
        optimized_nonzero.equivariant_filter,
        flatten(fixture.at("equivariant_filter")), "BLAS equivariant/PyTorch");
    const double inv_zero_error = compare(
        optimized_zero.invariant_filter,
        flatten(fixture.at("invariant_filter_zero_ev")), "BLAS zero invariant");
    const double ev_zero_error = compare(
        optimized_zero.equivariant_filter,
        flatten(fixture.at("equivariant_filter_zero_ev")), "BLAS zero equivariant");
    const double inv_ab_error = compare(
        optimized_nonzero.invariant_filter, baseline_nonzero.invariant_filter,
        "BLAS/baseline invariant");
    const double ev_ab_error = compare(
        optimized_nonzero.equivariant_filter, baseline_nonzero.equivariant_filter,
        "BLAS/baseline equivariant");

    const auto baseline_benchmark = baseline.benchmark(1000000, 3);
    const auto optimized_benchmark = optimized.benchmark(1000000, 10);
    const double speedup = baseline_benchmark.milliseconds_per_iteration /
                           optimized_benchmark.milliseconds_per_iteration;
    if (!std::isfinite(speedup) || speedup <= 1.05)
      throw std::runtime_error("optimized dense backend did not improve performance");
    if (std::abs(baseline_benchmark.checksum - optimized_benchmark.checksum) > 3.0e-11)
      throw std::runtime_error("A/B benchmark checksum mismatch");

    std::cout << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "backend=cuBLAS_DGEMM\n"
              << "shared_kokkos_stream=1\n"
              << "filter_aliases_verified=1\n"
              << "persistent_device_bytes=" << optimized.persistent_device_bytes() << '\n'
              << "inv_pytorch_max_abs_error=" << inv_reference_error << '\n'
              << "ev_pytorch_max_abs_error=" << ev_reference_error << '\n'
              << "inv_zero_ev_max_abs_error=" << inv_zero_error << '\n'
              << "ev_zero_ev_max_abs_error=" << ev_zero_error << '\n'
              << "inv_ab_max_abs_error=" << inv_ab_error << '\n'
              << "ev_ab_max_abs_error=" << ev_ab_error << '\n'
              << "benchmark_edges=1000000\n"
              << "baseline_ms_per_iteration=" << baseline_benchmark.milliseconds_per_iteration << '\n'
              << "optimized_ms_per_iteration=" << optimized_benchmark.milliseconds_per_iteration << '\n'
              << "dense_backend_speedup=" << speedup << '\n'
              << "benchmark_workspace_bytes=" << optimized_benchmark.workspace_bytes << '\n'
              << "benchmark_checksum=" << optimized_benchmark.checksum << '\n'
              << "cublas_filter_equivalence=PASS\n"
              << "cublas_performance_gate=PASS\n"
              << "SO3LR_NATIVE_DEV5_BLAS_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV5_BLAS_TEST=FAIL reason=" << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}

