#include "so3lr/kokkos_block2_attention_input_reverse.hpp"

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
    if (error > 2.0e-8 + 2.0e-8 * std::abs(expected[i]))
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
      throw std::runtime_error("usage: test MODEL BLOCK2_REVERSE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-block2-attention-input-reverse-fixture-v1" ||
        fixture.at("block_index").unsigned_integer() != 2)
      throw std::runtime_error("unexpected dev_19 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t edges =
        static_cast<std::size_t>(fixture.at("edges").unsigned_integer());
    constexpr std::size_t feature_width = 128;
    constexpr std::size_t ev_width = 24;

    DoubleView inv("so3lr_dev19_inv", nodes * feature_width);
    DoubleView ev("so3lr_dev19_ev", nodes * ev_width);
    DoubleView distances("so3lr_dev19_distances", edges);
    DoubleView sh("so3lr_dev19_sh", edges * ev_width);
    IndexView senders("so3lr_dev19_senders", edges);
    IndexView receivers("so3lr_dev19_receivers", edges);
    DoubleView grad_attention_inv("so3lr_dev19_grad_attention_inv",
                                  nodes * feature_width);
    DoubleView grad_attention_ev("so3lr_dev19_grad_attention_ev",
                                 nodes * ev_width);
    fill(inv, numbers(fixture.at("inv_features")));
    fill(ev, numbers(fixture.at("ev_features")));
    fill(distances, numbers(fixture.at("distances")));
    fill(sh, numbers(fixture.at("sh_vectors")));
    fill(senders, indices(fixture.at("senders")));
    fill(receivers, indices(fixture.at("receivers")));
    fill(grad_attention_inv, numbers(fixture.at("grad_attention_inv")));
    fill(grad_attention_ev, numbers(fixture.at("grad_attention_ev")));

    const so3lr::KokkosBlock2AttentionInputReverse reverse(model);
    if (!reverse.checkpoint_contract_verified() ||
        !reverse.shared_kokkos_stream())
      throw std::runtime_error("dev_19 runtime contract failed");
    so3lr::Block2AttentionInputReverseWorkspace workspace(nodes, edges);
    reverse.launch_forward_device(inv, ev, distances, sh, senders, receivers,
                                  workspace);
    reverse.launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                                  grad_attention_inv, grad_attention_ev,
                                  workspace);
    Kokkos::fence();

    const double d_inv_error = compare(
        copy(workspace.forward.invariant_update),
        numbers(fixture.at("d_inv")), "block2 attention invariant forward");
    const double d_ev_error = compare(
        copy(workspace.forward.equivariant_update),
        numbers(fixture.at("d_ev")), "block2 attention equivariant forward");
    const double grad_inv_error = compare(
        copy(workspace.grad_inv_features),
        numbers(fixture.at("grad_inv_features")),
        "block2 invariant input gradient");
    const double grad_ev_error = compare(
        copy(workspace.grad_ev_features),
        numbers(fixture.at("grad_ev_features")),
        "block2 equivariant input gradient");
    const double grad_distance_error = compare(
        copy(workspace.grad_distances),
        numbers(fixture.at("grad_distances")), "block2 distance gradient");
    const double grad_sh_error = compare(
        copy(reverse.grad_sh(workspace)),
        numbers(fixture.at("grad_sh_vectors")), "block2 SH gradient");
    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-5)
      throw std::runtime_error("fixture finite-difference validation failed");

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    const auto benchmark = reverse.benchmark(matched_nodes, matched_edges, 3);
    if (benchmark.nodes != matched_nodes || benchmark.edges != matched_edges ||
        benchmark.host_boundary_bytes_per_iteration != 0 ||
        !std::isfinite(benchmark.reverse_milliseconds) ||
        benchmark.reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.forward_reverse_milliseconds) ||
        benchmark.forward_reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("dev_19 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "block_index=2\n"
        << "reverse_scope=attention_residual_to_features_distance_sh\n"
        << "backend=Kokkos_CUDA_plus_cuBLAS_DGEMM_VJP\n"
        << "d_inv_forward_max_abs_error=" << d_inv_error << '\n'
        << "d_ev_forward_max_abs_error=" << d_ev_error << '\n'
        << "inv_feature_gradient_max_abs_error=" << grad_inv_error << '\n'
        << "ev_feature_gradient_max_abs_error=" << grad_ev_error << '\n'
        << "distance_gradient_max_abs_error=" << grad_distance_error << '\n'
        << "sh_gradient_max_abs_error=" << grad_sh_error << '\n'
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << benchmark.nodes << '\n'
        << "matched_stage2_edges=" << benchmark.edges << '\n'
        << "block2_attention_input_reverse_ms="
        << benchmark.reverse_milliseconds << '\n'
        << "block2_attention_input_forward_reverse_ms="
        << benchmark.forward_reverse_milliseconds << '\n'
        << "block2_attention_input_reverse_edges_per_second="
        << benchmark.edges_per_second << '\n'
        << "block2_attention_input_reverse_workspace_bytes="
        << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "reverse_checksum=" << benchmark.checksum << '\n'
        << "pytorch_block2_attention_input_vjp_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "block2_attention_input_reverse_equivalence=PASS\n"
        << "block2_attention_input_reverse_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV19_BLOCK2_ATTENTION_INPUT_REVERSE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr
        << "SO3LR_NATIVE_DEV19_BLOCK2_ATTENTION_INPUT_REVERSE_TEST=FAIL reason="
        << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
