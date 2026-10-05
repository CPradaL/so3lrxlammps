#include "so3lr/kokkos_post_attention_block0.hpp"

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

double compare(const std::vector<double> &actual,
               const std::vector<double> &expected,
               const std::string &label) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > 1.5e-10 + 1.5e-10 * std::abs(expected[i]))
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
      throw std::runtime_error("usage: test MODEL POST_ATTENTION_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const so3lr::KokkosPostAttentionBlock0 post(model);
    if (!post.checkpoint_contract_verified() || !post.shared_kokkos_stream())
      throw std::runtime_error("post-attention runtime contract failed");
    const auto actual = post.evaluate(
        numbers(fixture.at("inv_features")),
        numbers(fixture.at("ev_features")),
        numbers(fixture.at("d_att_inv")),
        numbers(fixture.at("d_att_ev")), nodes);
    const double residual_inv_error = compare(
        actual.attention_residual_inv,
        numbers(fixture.at("attention_residual_inv")),
        "attention invariant residual");
    const double residual_ev_error = compare(
        actual.attention_residual_ev,
        numbers(fixture.at("attention_residual_ev")),
        "attention equivariant residual");
    const double norm1_error = compare(
        actual.layer_norm_1, numbers(fixture.at("layer_norm_1")),
        "layer norm 1");
    const double mlp_error = compare(
        actual.post_mlp_inv, numbers(fixture.at("post_mlp_inv")),
        "residual MLP 1");
    const double l0_error = compare(
        actual.interaction_ev_invariants,
        numbers(fixture.at("interaction_ev_invariants")), "L0 contraction");
    const double interaction_error = compare(
        actual.interaction_transformed,
        numbers(fixture.at("interaction_transformed")),
        "interaction transform");
    const double pre_norm2_error = compare(
        actual.pre_layer_norm_2_inv,
        numbers(fixture.at("pre_layer_norm_2_inv")),
        "interaction invariant residual");
    const double final_inv_error = compare(
        actual.final_inv, numbers(fixture.at("final_inv")),
        "final invariant features");
    const double final_ev_error = compare(
        actual.final_ev, numbers(fixture.at("final_ev")),
        "final equivariant features");

    constexpr std::size_t benchmark_nodes = 24000;
    const auto benchmark = post.benchmark(benchmark_nodes, 50);
    if (benchmark.host_boundary_bytes_per_iteration != 0)
      throw std::runtime_error("post-attention path has a host boundary");
    if (!std::isfinite(benchmark.pipeline_milliseconds) ||
        benchmark.pipeline_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.phase_sum_milliseconds) ||
        benchmark.phase_sum_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.nodes_per_second) ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("post-attention benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "pipeline=attention_residual_layernorm_mlp_interaction_layernorm\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "shared_kokkos_stream=1\n"
        << "checkpoint_contract_verified=1\n"
        << "attention_residual_inv_max_abs_error=" << residual_inv_error << '\n'
        << "attention_residual_ev_max_abs_error=" << residual_ev_error << '\n'
        << "layer_norm_1_max_abs_error=" << norm1_error << '\n'
        << "residual_mlp_1_max_abs_error=" << mlp_error << '\n'
        << "l0_contraction_max_abs_error=" << l0_error << '\n'
        << "interaction_transform_max_abs_error=" << interaction_error << '\n'
        << "pre_layer_norm_2_max_abs_error=" << pre_norm2_error << '\n'
        << "final_inv_max_abs_error=" << final_inv_error << '\n'
        << "final_ev_max_abs_error=" << final_ev_error << '\n'
        << "benchmark_nodes=" << benchmark_nodes << '\n'
        << "residual_norm1_ms="
        << benchmark.residual_norm1_milliseconds << '\n'
        << "residual_mlp1_ms="
        << benchmark.residual_mlp1_milliseconds << '\n'
        << "interaction_ms=" << benchmark.interaction_milliseconds << '\n'
        << "residual_norm2_ms="
        << benchmark.residual_norm2_milliseconds << '\n'
        << "phase_sum_ms=" << benchmark.phase_sum_milliseconds << '\n'
        << "pipeline_ms=" << benchmark.pipeline_milliseconds << '\n'
        << "pipeline_nodes_per_second=" << benchmark.nodes_per_second << '\n'
        << "persistent_device_bytes="
        << benchmark.persistent_device_bytes << '\n'
        << "pipeline_workspace_bytes=" << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "benchmark_checksum=" << benchmark.checksum << '\n'
        << "post_attention_pytorch_equivalence=PASS\n"
        << "post_attention_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV9_POST_ATTENTION_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV9_POST_ATTENTION_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
