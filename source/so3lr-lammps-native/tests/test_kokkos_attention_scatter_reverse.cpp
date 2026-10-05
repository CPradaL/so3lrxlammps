#include "so3lr/kokkos_attention_scatter_reverse.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using IndexView = Kokkos::View<std::size_t *>;

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

template <class View, class Values>
void fill(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("fixture-to-device size mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
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
    if (error > 5.0e-9 + 5.0e-9 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i) + " actual=" +
                               std::to_string(actual[i]) + " expected=" +
                               std::to_string(expected[i]));
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
      throw std::runtime_error("usage: test MODEL ATTENTION_REVERSE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-block2-attention-scatter-reverse-fixture-v1" ||
        fixture.at("block_index").unsigned_integer() != 2)
      throw std::runtime_error("unexpected dev_18 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t edges =
        static_cast<std::size_t>(fixture.at("edges").unsigned_integer());
    constexpr std::size_t feature_width = 128;
    constexpr std::size_t ev_width = 24;

    std::array<DoubleView, 5> qkv = {
        DoubleView("so3lr_dev18_q_inv", nodes * feature_width),
        DoubleView("so3lr_dev18_k_inv", nodes * feature_width),
        DoubleView("so3lr_dev18_v_inv", nodes * feature_width),
        DoubleView("so3lr_dev18_q_ev", nodes * feature_width),
        DoubleView("so3lr_dev18_k_ev", nodes * feature_width)};
    constexpr std::array<const char *, 5> qkv_names = {
        "q_inv", "k_inv", "v_inv", "q_ev", "k_ev"};
    for (std::size_t i = 0; i < qkv.size(); ++i)
      fill(qkv[i], numbers(fixture.at(qkv_names[i])));
    DoubleView filter_inv("so3lr_dev18_filter_inv", edges * feature_width);
    DoubleView filter_ev("so3lr_dev18_filter_ev", edges * feature_width);
    DoubleView sh("so3lr_dev18_sh", edges * ev_width);
    DoubleView cutoff("so3lr_dev18_cutoff", edges);
    IndexView senders("so3lr_dev18_senders", edges);
    IndexView receivers("so3lr_dev18_receivers", edges);
    DoubleView grad_d_inv("so3lr_dev18_grad_d_inv", nodes * feature_width);
    DoubleView grad_d_ev("so3lr_dev18_grad_d_ev", nodes * ev_width);
    fill(filter_inv, numbers(fixture.at("filter_inv")));
    fill(filter_ev, numbers(fixture.at("filter_ev")));
    fill(sh, numbers(fixture.at("sh")));
    fill(cutoff, numbers(fixture.at("cutoff")));
    fill(senders, indices(fixture.at("senders")));
    fill(receivers, indices(fixture.at("receivers")));
    fill(grad_d_inv, numbers(fixture.at("grad_d_inv")));
    fill(grad_d_ev, numbers(fixture.at("grad_d_ev")));

    const so3lr::KokkosAttentionScatterReverse reverse(model, 2);
    if (!reverse.checkpoint_contract_verified() ||
        reverse.block_index() != 2)
      throw std::runtime_error("dev_18 runtime contract failed");
    if (std::abs(reverse.attention_norm_inv() -
                 fixture.at("attention_norm_inv").number()) > 1.0e-13 ||
        std::abs(reverse.attention_norm_ev() -
                 fixture.at("attention_norm_ev").number()) > 1.0e-13)
      throw std::runtime_error("dev_18 normalization contract mismatch");
    so3lr::AttentionScatterReverseDeviceWorkspace workspace(nodes, edges);
    reverse.launch_forward_device(qkv, filter_inv, filter_ev, sh, cutoff,
                                  senders, receivers, workspace);
    reverse.launch_reverse_device(qkv, filter_inv, filter_ev, sh, cutoff,
                                  senders, receivers, grad_d_inv, grad_d_ev,
                                  workspace);
    Kokkos::fence();

    const double d_inv_error = compare(
        copy(workspace.invariant_update), numbers(fixture.at("d_inv")),
        "attention invariant forward");
    const double d_ev_error = compare(
        copy(workspace.equivariant_update), numbers(fixture.at("d_ev")),
        "attention equivariant forward");
    std::array<double, 5> qkv_errors = {};
    for (std::size_t i = 0; i < qkv.size(); ++i) {
      const std::string key = "grad_" + std::string(qkv_names[i]);
      qkv_errors[i] = compare(copy(workspace.grad_qkv[i]),
                              numbers(fixture.at(key)), key);
    }
    const double filter_inv_error = compare(
        copy(workspace.grad_filter_inv),
        numbers(fixture.at("grad_filter_inv")), "grad_filter_inv");
    const double filter_ev_error = compare(
        copy(workspace.grad_filter_ev),
        numbers(fixture.at("grad_filter_ev")), "grad_filter_ev");
    const double sh_error = compare(copy(workspace.grad_sh),
                                    numbers(fixture.at("grad_sh")), "grad_sh");
    const double cutoff_error =
        compare(copy(workspace.grad_cutoff),
                numbers(fixture.at("grad_cutoff")), "grad_cutoff");
    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-6)
      throw std::runtime_error("fixture finite-difference validation failed");

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    const auto benchmark = reverse.benchmark(matched_nodes, matched_edges, 3);
    if (benchmark.block_index != 2 || benchmark.nodes != matched_nodes ||
        benchmark.edges != matched_edges ||
        benchmark.host_boundary_bytes_per_iteration != 0 ||
        !std::isfinite(benchmark.reverse_milliseconds) ||
        benchmark.reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.forward_reverse_milliseconds) ||
        benchmark.forward_reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("dev_18 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "block_index=2\n"
        << "reverse_scope=fused_attention_scatter\n"
        << "backend=Kokkos_CUDA_edge_fused_VJP\n"
        << "attention_norm_inv=" << reverse.attention_norm_inv() << '\n'
        << "attention_norm_ev=" << reverse.attention_norm_ev() << '\n'
        << "d_inv_forward_max_abs_error=" << d_inv_error << '\n'
        << "d_ev_forward_max_abs_error=" << d_ev_error << '\n'
        << "q_inv_gradient_max_abs_error=" << qkv_errors[0] << '\n'
        << "k_inv_gradient_max_abs_error=" << qkv_errors[1] << '\n'
        << "v_inv_gradient_max_abs_error=" << qkv_errors[2] << '\n'
        << "q_ev_gradient_max_abs_error=" << qkv_errors[3] << '\n'
        << "k_ev_gradient_max_abs_error=" << qkv_errors[4] << '\n'
        << "filter_inv_gradient_max_abs_error=" << filter_inv_error << '\n'
        << "filter_ev_gradient_max_abs_error=" << filter_ev_error << '\n'
        << "sh_gradient_max_abs_error=" << sh_error << '\n'
        << "cutoff_gradient_max_abs_error=" << cutoff_error << '\n'
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << benchmark.nodes << '\n'
        << "matched_stage2_edges=" << benchmark.edges << '\n'
        << "block2_attention_scatter_reverse_ms="
        << benchmark.reverse_milliseconds << '\n'
        << "block2_attention_scatter_forward_reverse_ms="
        << benchmark.forward_reverse_milliseconds << '\n'
        << "block2_attention_scatter_reverse_edges_per_second="
        << benchmark.edges_per_second << '\n'
        << "block2_attention_scatter_reverse_workspace_bytes="
        << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "reverse_checksum=" << benchmark.checksum << '\n'
        << "pytorch_attention_scatter_vjp_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "block2_attention_scatter_reverse_equivalence=PASS\n"
        << "block2_attention_scatter_reverse_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV18_ATTENTION_SCATTER_REVERSE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV18_ATTENTION_SCATTER_REVERSE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
