#include "so3lr/kokkos_integrated_block0.hpp"

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
    if (error > 1.0e-10 + 1.0e-10 * std::abs(expected[i]))
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
      throw std::runtime_error("usage: test MODEL INTEGRATED_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const so3lr::KokkosIntegratedBlock0 pipeline(model);
    if (!pipeline.shared_kokkos_stream())
      throw std::runtime_error("cuBLAS and Kokkos streams differ");
    const auto actual = pipeline.evaluate(
        integers(fixture.at("atomic_numbers")),
        numbers(fixture.at("distances")),
        numbers(fixture.at("ev_invariants")),
        numbers(fixture.at("sh_vectors")), indices(fixture.at("senders")),
        indices(fixture.at("receivers")));
    const double embedding_error =
        compare(actual.embedding, numbers(fixture.at("embedding")),
                "embedding");
    const double rbf_error =
        compare(actual.radial_basis, numbers(fixture.at("radial_basis")),
                "radial basis");
    const double cutoff_error =
        compare(actual.cutoff, numbers(fixture.at("cutoffs")), "cutoff");
    const double inv_error =
        compare(actual.invariant_update, numbers(fixture.at("d_inv")),
                "integrated invariant update");
    const double ev_error =
        compare(actual.equivariant_update, numbers(fixture.at("d_ev")),
                "integrated equivariant update");

    constexpr std::size_t benchmark_nodes = 24000;
    constexpr std::size_t benchmark_edges = 1000000;
    const auto benchmark =
        pipeline.benchmark(benchmark_nodes, benchmark_edges, 10);
    if (benchmark.host_boundary_bytes_per_iteration != 0)
      throw std::runtime_error("pipeline retained a host-facing boundary");
    if (!std::isfinite(benchmark.pipeline_milliseconds) ||
        benchmark.pipeline_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.phase_sum_milliseconds) ||
        benchmark.phase_sum_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("integrated benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=embedding_rbf_filter_qkv_attention_scatter\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "shared_kokkos_stream=1\n"
        << "embedding_max_abs_error=" << embedding_error << '\n'
        << "rbf_max_abs_error=" << rbf_error << '\n'
        << "cutoff_max_abs_error=" << cutoff_error << '\n'
        << "invariant_update_max_abs_error=" << inv_error << '\n'
        << "equivariant_update_max_abs_error=" << ev_error << '\n'
        << "benchmark_nodes=" << benchmark_nodes << '\n'
        << "benchmark_edges=" << benchmark_edges << '\n'
        << "input_ms=" << benchmark.input_milliseconds << '\n'
        << "filter_ms=" << benchmark.filter_milliseconds << '\n'
        << "qkv_ms=" << benchmark.qkv_milliseconds << '\n'
        << "attention_ms=" << benchmark.attention_milliseconds << '\n'
        << "phase_sum_ms=" << benchmark.phase_sum_milliseconds << '\n'
        << "pipeline_ms=" << benchmark.pipeline_milliseconds << '\n'
        << "pipeline_edges_per_second=" << benchmark.edges_per_second << '\n'
        << "persistent_device_bytes="
        << benchmark.persistent_device_bytes << '\n'
        << "pipeline_workspace_bytes=" << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "benchmark_checksum=" << benchmark.checksum << '\n'
        << "radial_basis_host_roundtrip=0\n"
        << "filter_host_roundtrip=0\n"
        << "qkv_host_roundtrip=0\n"
        << "attention_host_roundtrip=0\n"
        << "integrated_pytorch_equivalence=PASS\n"
        << "device_residency_contract=PASS\n"
        << "SO3LR_NATIVE_DEV8_PIPELINE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV8_PIPELINE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
