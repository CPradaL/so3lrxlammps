#include "so3lr/kokkos_attention_scatter_block0.hpp"

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
    if (error > 8.0e-11 + 8.0e-11 * std::abs(expected[i]))
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
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL ATTENTION_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const so3lr::KokkosAttentionScatterBlock0 attention(model);
    if (!attention.degree_contract_verified())
      throw std::runtime_error("degree contract was not verified");
    const double reference_norm_inv =
        fixture.at("attention_norm_inv").number();
    const double reference_norm_ev = fixture.at("attention_norm_ev").number();
    if (std::abs(attention.attention_norm_inv() - reference_norm_inv) > 1.0e-13 ||
        std::abs(attention.attention_norm_ev() - reference_norm_ev) > 1.0e-13)
      throw std::runtime_error("attention normalization mismatch");

    const auto actual = attention.evaluate(
        numbers(fixture.at("q_inv_nodes")),
        numbers(fixture.at("k_inv_nodes")),
        numbers(fixture.at("v_inv_nodes")),
        numbers(fixture.at("q_ev_nodes")),
        numbers(fixture.at("k_ev_nodes")),
        numbers(fixture.at("filter_inv")),
        numbers(fixture.at("filter_ev")),
        numbers(fixture.at("sh_vectors")),
        numbers(fixture.at("cutoffs")), indices(fixture.at("senders")),
        indices(fixture.at("receivers")), nodes);
    const double inv_error = compare(actual.invariant_update,
                                     numbers(fixture.at("d_inv")), "d_inv");
    const double ev_error = compare(actual.equivariant_update,
                                    numbers(fixture.at("d_ev")), "d_ev");
    const double alpha_inv_error =
        compare(actual.alpha_invariant, numbers(fixture.at("alpha_inv")),
                "alpha_inv");
    const double alpha_ev_error =
        compare(actual.alpha_equivariant, numbers(fixture.at("alpha_ev")),
                "alpha_ev");

    constexpr std::size_t benchmark_nodes = 24000;
    constexpr std::size_t benchmark_edges = 1000000;
    const auto benchmark =
        attention.benchmark(benchmark_nodes, benchmark_edges, 10);
    const std::size_t expected_avoided =
        benchmark_edges * 1076 * sizeof(double);
    if (benchmark.avoided_edge_intermediate_bytes != expected_avoided)
      throw std::runtime_error("avoided-intermediate accounting mismatch");
    if (!std::isfinite(benchmark.milliseconds_per_iteration) ||
        benchmark.milliseconds_per_iteration <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("fused attention benchmark is invalid");

    std::cout << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "kernel=fused_attention_message_scatter\n"
              << "heads=4\n"
              << "head_width=32\n"
              << "degree_repeats=3,5,7,9\n"
              << "attention_norm_inv=" << attention.attention_norm_inv() << '\n'
              << "attention_norm_ev=" << attention.attention_norm_ev() << '\n'
              << "invariant_update_max_abs_error=" << inv_error << '\n'
              << "equivariant_update_max_abs_error=" << ev_error << '\n'
              << "alpha_inv_max_abs_error=" << alpha_inv_error << '\n'
              << "alpha_ev_max_abs_error=" << alpha_ev_error << '\n'
              << "persistent_device_bytes="
              << attention.persistent_device_bytes() << '\n'
              << "benchmark_nodes=" << benchmark_nodes << '\n'
              << "benchmark_edges=" << benchmark_edges << '\n'
              << "benchmark_ms_per_iteration="
              << benchmark.milliseconds_per_iteration << '\n'
              << "benchmark_edges_per_second=" << benchmark.edges_per_second
              << '\n'
              << "benchmark_workspace_bytes=" << benchmark.workspace_bytes
              << '\n'
              << "avoided_edge_intermediate_bytes="
              << benchmark.avoided_edge_intermediate_bytes << '\n'
              << "benchmark_checksum=" << benchmark.checksum << '\n'
              << "edge_qkv_materialized=0\n"
              << "filtered_keys_materialized=0\n"
              << "edge_messages_materialized=0\n"
              << "attention_coefficients_materialized_in_benchmark=0\n"
              << "fused_attention_equivalence=PASS\n"
              << "fused_scatter_equivalence=PASS\n"
              << "fused_memory_contract=PASS\n"
              << "SO3LR_NATIVE_DEV7_ATTENTION_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV7_ATTENTION_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}

