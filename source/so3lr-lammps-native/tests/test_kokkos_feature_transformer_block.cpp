#include "so3lr/kokkos_feature_transformer_block.hpp"

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
    if (error > 2.5e-10 + 2.5e-10 * std::abs(expected[i]))
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
      throw std::runtime_error("usage: test MODEL BLOCK1_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    constexpr std::size_t block_index = 1;
    if (fixture.at("block_index").unsigned_integer() != block_index)
      throw std::runtime_error("fixture block index mismatch");
    const so3lr::KokkosFeatureTransformerBlock block(model, block_index);
    if (!block.device_contract_verified() || block.block_index() != block_index)
      throw std::runtime_error("block-1 device contract failed");
    const auto actual = block.evaluate(
        numbers(fixture.at("input_inv")), numbers(fixture.at("input_ev")),
        numbers(fixture.at("distances")),
        numbers(fixture.at("sh_vectors")), indices(fixture.at("senders")),
        indices(fixture.at("receivers")));
    const double edge_invariant_error = compare(
        actual.edge_ev_invariants,
        numbers(fixture.at("edge_ev_invariants")),
        "nonzero equivariant edge invariants");
    const double attention_inv_error = compare(
        actual.attention_update_inv,
        numbers(fixture.at("attention_update_inv")),
        "block-1 attention invariant update");
    const double attention_ev_error = compare(
        actual.attention_update_ev,
        numbers(fixture.at("attention_update_ev")),
        "block-1 attention equivariant update");
    const double final_inv_error = compare(
        actual.final_inv, numbers(fixture.at("final_inv")),
        "block-1 invariant output");
    const double final_ev_error = compare(
        actual.final_ev, numbers(fixture.at("final_ev")),
        "block-1 equivariant output");

    constexpr std::size_t benchmark_nodes = 24000;
    constexpr std::size_t benchmark_edges = 1000000;
    const auto benchmark =
        block.benchmark(benchmark_nodes, benchmark_edges, 10);
    if (benchmark.block_index != block_index ||
        benchmark.host_boundary_bytes_per_iteration != 0)
      throw std::runtime_error("block-1 benchmark contract failed");
    if (!std::isfinite(benchmark.complete_block_milliseconds) ||
        benchmark.complete_block_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("block-1 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=feature_input_transformer_block\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "block_index=" << block_index << '\n'
        << "block_indexed_parameters_verified=1\n"
        << "nonzero_equivariant_input=1\n"
        << "edge_ev_invariants_max_abs_error=" << edge_invariant_error << '\n'
        << "attention_update_inv_max_abs_error=" << attention_inv_error
        << '\n'
        << "attention_update_ev_max_abs_error=" << attention_ev_error << '\n'
        << "final_inv_max_abs_error=" << final_inv_error << '\n'
        << "final_ev_max_abs_error=" << final_ev_error << '\n'
        << "benchmark_nodes=" << benchmark.nodes << '\n'
        << "benchmark_edges=" << benchmark.edges << '\n'
        << "benchmark_repetitions=" << benchmark.repetitions << '\n'
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
        << "transformer_block1_pytorch_equivalence=PASS\n"
        << "transformer_block1_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV11_FEATURE_TRANSFORMER_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV11_FEATURE_TRANSFORMER_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
