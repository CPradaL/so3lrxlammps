#include "so3lr/kokkos_transformer_stack012.hpp"

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

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("test was not compiled for CUDA");
#endif
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL STACK012_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const so3lr::KokkosTransformerStack012 stack(model);
    if (!stack.device_contract_verified())
      throw std::runtime_error("stack 0-1-2 device contract failed");
    const auto actual = stack.evaluate(
        integers(fixture.at("atomic_numbers")),
        numbers(fixture.at("distances")),
        numbers(fixture.at("sh_vectors")), indices(fixture.at("senders")),
        indices(fixture.at("receivers")));
    const double block0_inv_error = compare(
        actual.block0_final_inv, numbers(fixture.at("block0_final_inv")),
        "stack block-0 invariant output");
    const double block0_ev_error = compare(
        actual.block0_final_ev, numbers(fixture.at("block0_final_ev")),
        "stack block-0 equivariant output");
    const double block1_inv_error = compare(
        actual.block1_final_inv, numbers(fixture.at("block1_final_inv")),
        "stack block-1 invariant output");
    const double block1_ev_error = compare(
        actual.block1_final_ev, numbers(fixture.at("block1_final_ev")),
        "stack block-1 equivariant output");
    const double block2_inv_error = compare(
        actual.block2_final_inv, numbers(fixture.at("block2_final_inv")),
        "stack block-2 invariant output");
    const double block2_ev_error = compare(
        actual.block2_final_ev, numbers(fixture.at("block2_final_ev")),
        "stack block-2 equivariant output");

    constexpr std::size_t benchmark_nodes = 24000;
    constexpr std::size_t benchmark_edges = 1000000;
    const auto benchmark =
        stack.benchmark(benchmark_nodes, benchmark_edges, 10);
    if (benchmark.host_boundary_bytes_per_iteration != 0)
      throw std::runtime_error("stack 0-1-2 has a host boundary");
    if (!std::isfinite(benchmark.block0_milliseconds) ||
        benchmark.block0_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.block1_milliseconds) ||
        benchmark.block1_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.block2_milliseconds) ||
        benchmark.block2_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.stack_milliseconds) ||
        benchmark.stack_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("stack 0-1-2 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=complete_transformer_stack_0_1_2\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "device_view_bridge_0_to_1=1\n"
        << "device_view_bridge_1_to_2=1\n"
        << "shared_geometry_buffers=1\n"
        << "block0_final_inv_max_abs_error=" << block0_inv_error << '\n'
        << "block0_final_ev_max_abs_error=" << block0_ev_error << '\n'
        << "block1_final_inv_max_abs_error=" << block1_inv_error << '\n'
        << "block1_final_ev_max_abs_error=" << block1_ev_error << '\n'
        << "block2_final_inv_max_abs_error=" << block2_inv_error << '\n'
        << "block2_final_ev_max_abs_error=" << block2_ev_error << '\n'
        << "benchmark_nodes=" << benchmark.nodes << '\n'
        << "benchmark_edges=" << benchmark.edges << '\n'
        << "benchmark_repetitions=" << benchmark.repetitions << '\n'
        << "block0_ms=" << benchmark.block0_milliseconds << '\n'
        << "block1_ms=" << benchmark.block1_milliseconds << '\n'
        << "block2_ms=" << benchmark.block2_milliseconds << '\n'
        << "phase_sum_ms=" << benchmark.phase_sum_milliseconds << '\n'
        << "stack012_ms=" << benchmark.stack_milliseconds << '\n'
        << "stack012_edges_per_second=" << benchmark.edges_per_second << '\n'
        << "persistent_device_bytes=" << benchmark.persistent_device_bytes
        << '\n'
        << "stack012_workspace_bytes=" << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "benchmark_checksum=" << benchmark.checksum << '\n'
        << "transformer_stack012_pytorch_equivalence=PASS\n"
        << "transformer_stack012_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV13_STACK012_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV13_STACK012_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
