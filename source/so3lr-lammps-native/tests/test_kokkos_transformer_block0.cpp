#include "so3lr/kokkos_transformer_block0.hpp"

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
    if (error > 2.0e-10 + 2.0e-10 * std::abs(expected[i]))
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
      throw std::runtime_error("usage: test MODEL FULL_BLOCK0_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const so3lr::KokkosTransformerBlock0 transformer(model);
    if (!transformer.device_contract_verified())
      throw std::runtime_error("complete block-0 device contract failed");
    const auto actual = transformer.evaluate(
        integers(fixture.at("atomic_numbers")),
        numbers(fixture.at("distances")),
        numbers(fixture.at("sh_vectors")), indices(fixture.at("senders")),
        indices(fixture.at("receivers")));
    const double embedding_error = compare(
        actual.embedding, numbers(fixture.at("embedding")), "embedding");
    const double attention_inv_error = compare(
        actual.attention_update_inv,
        numbers(fixture.at("attention_update_inv")),
        "attention invariant update");
    const double attention_ev_error = compare(
        actual.attention_update_ev,
        numbers(fixture.at("attention_update_ev")),
        "attention equivariant update");
    const double final_inv_error = compare(
        actual.final_inv, numbers(fixture.at("final_inv")),
        "complete block invariant output");
    const double final_ev_error = compare(
        actual.final_ev, numbers(fixture.at("final_ev")),
        "complete block equivariant output");

    constexpr std::size_t benchmark_nodes = 24000;
    constexpr std::size_t benchmark_edges = 1000000;
    const auto benchmark =
        transformer.benchmark(benchmark_nodes, benchmark_edges, 10);
    if (benchmark.host_boundary_bytes_per_iteration != 0)
      throw std::runtime_error("complete block has a host boundary");
    if (!std::isfinite(benchmark.attention_pipeline_milliseconds) ||
        benchmark.attention_pipeline_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.post_attention_milliseconds) ||
        benchmark.post_attention_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.phase_sum_milliseconds) ||
        benchmark.phase_sum_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.complete_block_milliseconds) ||
        benchmark.complete_block_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("complete block benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=complete_transformer_block0\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "device_view_bridge_verified=1\n"
        << "embedding_max_abs_error=" << embedding_error << '\n'
        << "attention_update_inv_max_abs_error=" << attention_inv_error
        << '\n'
        << "attention_update_ev_max_abs_error=" << attention_ev_error << '\n'
        << "final_inv_max_abs_error=" << final_inv_error << '\n'
        << "final_ev_max_abs_error=" << final_ev_error << '\n'
        << "benchmark_nodes=" << benchmark.nodes << '\n'
        << "benchmark_edges=" << benchmark.edges << '\n'
        << "benchmark_repetitions=" << benchmark.repetitions << '\n'
        << "attention_pipeline_ms="
        << benchmark.attention_pipeline_milliseconds << '\n'
        << "post_attention_ms=" << benchmark.post_attention_milliseconds
        << '\n'
        << "phase_sum_ms=" << benchmark.phase_sum_milliseconds << '\n'
        << "complete_block_ms=" << benchmark.complete_block_milliseconds
        << '\n'
        << "complete_block_edges_per_second=" << benchmark.edges_per_second
        << '\n'
        << "persistent_device_bytes=" << benchmark.persistent_device_bytes
        << '\n'
        << "complete_block_workspace_bytes=" << benchmark.workspace_bytes
        << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "benchmark_checksum=" << benchmark.checksum << '\n'
        << "complete_block0_pytorch_equivalence=PASS\n"
        << "complete_block0_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV10_TRANSFORMER_BLOCK0_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV10_TRANSFORMER_BLOCK0_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
