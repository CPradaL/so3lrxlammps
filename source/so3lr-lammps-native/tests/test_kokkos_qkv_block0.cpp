#include "so3lr/kokkos_qkv_block0.hpp"

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
    if (error > 4.0e-11 + 4.0e-11 * std::abs(expected[i]))
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
    if (argc != 3) throw std::runtime_error("usage: test MODEL QKV_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const auto nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const auto features = numbers(fixture.at("features"));
    const auto senders = indices(fixture.at("senders"));
    const auto receivers = indices(fixture.at("receivers"));
    const so3lr::KokkosQkvBlock0 qkv(model);
    if (!qkv.shared_kokkos_stream() || !qkv.identity_qk_activation())
      throw std::runtime_error("Q/K/V runtime contract failed");
    const auto actual = qkv.evaluate(features, senders, receivers, nodes);

    double maximum_error = 0.0;
    const auto check = [&](const std::vector<double> &values,
                           const char *key) {
      maximum_error = std::max(
          maximum_error, compare(values, numbers(fixture.at(key)), key));
    };
    check(actual.q_inv_nodes, "q_inv_nodes");
    check(actual.k_inv_nodes, "k_inv_nodes");
    check(actual.v_inv_nodes, "v_inv_nodes");
    check(actual.q_ev_nodes, "q_ev_nodes");
    check(actual.k_ev_nodes, "k_ev_nodes");
    check(actual.q_inv_edges, "q_inv_edges");
    check(actual.k_inv_edges, "k_inv_edges");
    check(actual.v_inv_edges, "v_inv_edges");
    check(actual.q_ev_edges, "q_ev_edges");
    check(actual.k_ev_edges, "k_ev_edges");

    constexpr std::size_t benchmark_nodes = 24000;
    const auto baseline = qkv.benchmark_baseline(benchmark_nodes, 3);
    const auto optimized = qkv.benchmark(benchmark_nodes, 20);
    const double speedup = baseline.milliseconds_per_iteration /
                           optimized.milliseconds_per_iteration;
    if (!std::isfinite(speedup) || speedup <= 1.05)
      throw std::runtime_error("cuBLAS Q/K/V backend did not improve performance");
    if (std::abs(baseline.checksum - optimized.checksum) > 4.0e-11)
      throw std::runtime_error("Q/K/V benchmark checksum mismatch");

    std::cout << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "backend=cuBLAS_DGEMM\n"
              << "qk_activation=identity\n"
              << "shared_kokkos_stream=1\n"
              << "heads=4\n"
              << "head_width=32\n"
              << "projections=5\n"
              << "pytorch_max_abs_error=" << maximum_error << '\n'
              << "persistent_device_bytes=" << qkv.persistent_device_bytes() << '\n'
              << "benchmark_nodes=" << benchmark_nodes << '\n'
              << "baseline_ms_per_iteration="
              << baseline.milliseconds_per_iteration << '\n'
              << "optimized_ms_per_iteration="
              << optimized.milliseconds_per_iteration << '\n'
              << "qkv_backend_speedup=" << speedup << '\n'
              << "benchmark_workspace_bytes=" << optimized.workspace_bytes << '\n'
              << "benchmark_checksum=" << optimized.checksum << '\n'
              << "edge_qkv_materialized_in_benchmark=0\n"
              << "node_projection_equivalence=PASS\n"
              << "receiver_sender_gather_equivalence=PASS\n"
              << "qkv_performance_gate=PASS\n"
              << "SO3LR_NATIVE_DEV6_QKV_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV6_QKV_TEST=FAIL reason=" << error.what()
              << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}

