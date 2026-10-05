#include "so3lr/kokkos_complete_transformer_reverse.hpp"

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
      throw std::runtime_error("usage: test MODEL BLOCK0_REVERSE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-complete-block0-reverse-fixture-v1" ||
        fixture.at("block_index").unsigned_integer() != 0 ||
        fixture.at("input_contract").string() !=
            "atomic_embedding_plus_zero_equivariants")
      throw std::runtime_error("unexpected dev_21 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t edges =
        static_cast<std::size_t>(fixture.at("edges").unsigned_integer());
    constexpr std::size_t feature_width = 128;
    constexpr std::size_t ev_width = 24;

    DoubleView inv("so3lr_dev21_inv", nodes * feature_width);
    DoubleView ev("so3lr_dev21_ev", nodes * ev_width);
    DoubleView distances("so3lr_dev21_distances", edges);
    DoubleView sh("so3lr_dev21_sh", edges * ev_width);
    IndexView senders("so3lr_dev21_senders", edges);
    IndexView receivers("so3lr_dev21_receivers", edges);
    DoubleView grad_final_inv("so3lr_dev21_grad_final_inv",
                              nodes * feature_width);
    DoubleView grad_final_ev("so3lr_dev21_grad_final_ev", nodes * ev_width);
    fill(inv, numbers(fixture.at("inv_features")));
    fill(ev, numbers(fixture.at("ev_features")));
    fill(distances, numbers(fixture.at("distances")));
    fill(sh, numbers(fixture.at("sh_vectors")));
    fill(senders, indices(fixture.at("senders")));
    fill(receivers, indices(fixture.at("receivers")));
    fill(grad_final_inv, numbers(fixture.at("grad_final_inv")));
    fill(grad_final_ev, numbers(fixture.at("grad_final_ev")));
    if (fixture.at("maximum_abs_initial_ev").number() != 0.0)
      throw std::runtime_error("block-0 input is not production zero-EV state");

    const so3lr::KokkosCompleteTransformerReverseBlock reverse(model, 0);
    if (reverse.block_index() != 0 ||
        !reverse.checkpoint_contract_verified() ||
        !reverse.shared_kokkos_stream())
      throw std::runtime_error("dev_21 runtime contract failed");
    so3lr::CompleteTransformerReverseWorkspace workspace(nodes, edges);
    reverse.launch_forward_device(inv, ev, distances, sh, senders, receivers,
                                  workspace);
    reverse.launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                                  grad_final_inv, grad_final_ev, workspace);
    Kokkos::fence();

    const double final_inv_error = compare(
        copy(reverse.final_inv(workspace)), numbers(fixture.at("final_inv")),
        "block0 final invariant forward");
    const double final_ev_error = compare(
        copy(reverse.final_ev(workspace)), numbers(fixture.at("final_ev")),
        "block0 final equivariant forward");
    const double grad_inv_error = compare(
        copy(reverse.grad_inv(workspace)),
        numbers(fixture.at("grad_inv_features")),
        "block0 invariant input gradient");
    const double grad_ev_error = compare(
        copy(reverse.grad_ev(workspace)),
        numbers(fixture.at("grad_ev_features")),
        "block0 equivariant input gradient");
    const double grad_distance_error = compare(
        copy(reverse.grad_distances(workspace)),
        numbers(fixture.at("grad_distances")), "block0 distance gradient");
    const double grad_sh_error = compare(
        copy(reverse.grad_sh(workspace)),
        numbers(fixture.at("grad_sh_vectors")), "block0 SH gradient");
    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-5 ||
        fixture.at("maximum_abs_distance_gradient").number() <= 1.0e-12 ||
        fixture.at("maximum_abs_sh_gradient").number() <= 1.0e-12)
      throw std::runtime_error("fixture geometric sensitivity failed");

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    const auto benchmark = reverse.benchmark(matched_nodes, matched_edges, 3);
    if (benchmark.block_index != 0 || benchmark.nodes != matched_nodes ||
        benchmark.edges != matched_edges ||
        benchmark.host_boundary_bytes_per_iteration != 0 ||
        !std::isfinite(benchmark.reverse_milliseconds) ||
        benchmark.reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.forward_reverse_milliseconds) ||
        benchmark.forward_reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("dev_21 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "block_index=0\n"
        << "input_contract=atomic_embedding_plus_zero_equivariants\n"
        << "reverse_scope=complete_transformer_to_features_distance_sh\n"
        << "backend=Kokkos_CUDA_plus_cuBLAS_DGEMM_VJP\n"
        << "final_inv_forward_max_abs_error=" << final_inv_error << '\n'
        << "final_ev_forward_max_abs_error=" << final_ev_error << '\n'
        << "inv_feature_gradient_max_abs_error=" << grad_inv_error << '\n'
        << "ev_feature_gradient_max_abs_error=" << grad_ev_error << '\n'
        << "distance_gradient_max_abs_error=" << grad_distance_error << '\n'
        << "sh_gradient_max_abs_error=" << grad_sh_error << '\n'
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << benchmark.nodes << '\n'
        << "matched_stage2_edges=" << benchmark.edges << '\n'
        << "block0_reverse_ms=" << benchmark.reverse_milliseconds << '\n'
        << "block0_forward_reverse_ms="
        << benchmark.forward_reverse_milliseconds << '\n'
        << "block0_reverse_edges_per_second=" << benchmark.edges_per_second
        << '\n'
        << "block0_reverse_workspace_bytes=" << benchmark.workspace_bytes
        << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "reverse_checksum=" << benchmark.checksum << '\n'
        << "pytorch_complete_block0_vjp_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "complete_block0_reverse_equivalence=PASS\n"
        << "complete_block0_reverse_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV21_BLOCK0_REVERSE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV21_BLOCK0_REVERSE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
