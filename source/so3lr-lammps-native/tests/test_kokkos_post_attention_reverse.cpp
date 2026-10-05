#include "so3lr/kokkos_output_heads.hpp"
#include "so3lr/kokkos_post_attention_reverse.hpp"

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
using Int64View = Kokkos::View<std::int64_t *>;

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
      throw std::runtime_error("usage: test MODEL POST_REVERSE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-block2-post-attention-reverse-fixture-v1" ||
        fixture.at("block_index").unsigned_integer() != 2)
      throw std::runtime_error("unexpected dev_17 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    constexpr std::size_t inv_width = 128;
    constexpr std::size_t ev_width = 24;

    const auto inv_values = numbers(fixture.at("inv_features"));
    const auto ev_values = numbers(fixture.at("ev_features"));
    const auto d_inv_values = numbers(fixture.at("d_att_inv"));
    const auto d_ev_values = numbers(fixture.at("d_att_ev"));
    const auto atomic_numbers = integers(fixture.at("atomic_numbers"));
    DoubleView inv("so3lr_dev17_inv", nodes * inv_width);
    DoubleView ev("so3lr_dev17_ev", nodes * ev_width);
    DoubleView d_inv("so3lr_dev17_d_inv", nodes * inv_width);
    DoubleView d_ev("so3lr_dev17_d_ev", nodes * ev_width);
    Int64View z("so3lr_dev17_z", nodes);
    fill(inv, inv_values);
    fill(ev, ev_values);
    fill(d_inv, d_inv_values);
    fill(d_ev, d_ev_values);
    fill(z, atomic_numbers);

    const so3lr::KokkosPostAttentionReverseBlock block(model, 2);
    const so3lr::KokkosOutputHeads heads(model);
    if (!block.checkpoint_contract_verified() ||
        !block.shared_kokkos_stream() || block.block_index() != 2)
      throw std::runtime_error("dev_17 runtime contract failed");
    so3lr::PostAttentionBlock0DeviceWorkspace forward(nodes);
    so3lr::PostAttentionReverseDeviceWorkspace reverse(nodes);
    so3lr::OutputHeadsDeviceWorkspace head_workspace(nodes);
    DoubleView zero_final_ev_gradient("so3lr_dev17_zero_final_ev_gradient",
                                      nodes * ev_width);
    Kokkos::deep_copy(zero_final_ev_gradient, 0.0);

    // The actual energy-head seed from dev_16 is passed directly on-device to
    // the new block-2 post-attention reverse implementation.
    block.launch_forward_device(inv, ev, d_inv, d_ev, forward);
    heads.launch_device(forward.final_inv, z, 0.0, head_workspace);
    heads.launch_energy_reverse_device(z, head_workspace);
    block.launch_reverse_device(head_workspace.energy_input_grad,
                                zero_final_ev_gradient, forward, reverse);
    Kokkos::fence();

    const double final_inv_error = compare(
        copy(forward.final_inv), numbers(fixture.at("final_inv")),
        "block-2 final invariant forward");
    const double final_ev_error = compare(
        copy(forward.final_ev), numbers(fixture.at("final_ev")),
        "block-2 final equivariant forward");
    const double energy_seed_error = compare(
        copy(head_workspace.energy_input_grad),
        numbers(fixture.at("energy_final_inv_gradient")),
        "energy-head seed");
    const auto energy_grad_inv = copy(reverse.grad_attention_inv);
    const auto energy_grad_ev = copy(reverse.grad_attention_ev);
    const double energy_inv_error = compare(
        energy_grad_inv, numbers(fixture.at("energy_grad_inv_features")),
        "energy gradient invariant input");
    const double energy_d_inv_error = compare(
        energy_grad_inv, numbers(fixture.at("energy_grad_d_att_inv")),
        "energy gradient attention invariant update");
    const double energy_ev_error = compare(
        energy_grad_ev, numbers(fixture.at("energy_grad_ev_features")),
        "energy gradient equivariant input");
    const double energy_d_ev_error = compare(
        energy_grad_ev, numbers(fixture.at("energy_grad_d_att_ev")),
        "energy gradient attention equivariant update");

    // A generic two-output VJP exercises the final-equivariant gate branch,
    // which has zero direct seed for the learned atomic-energy head.
    DoubleView generic_inv_seed("so3lr_dev17_generic_inv_seed",
                                nodes * inv_width);
    DoubleView generic_ev_seed("so3lr_dev17_generic_ev_seed",
                               nodes * ev_width);
    fill(generic_inv_seed, numbers(fixture.at("generic_final_inv_seed")));
    fill(generic_ev_seed, numbers(fixture.at("generic_final_ev_seed")));
    block.launch_reverse_device(generic_inv_seed, generic_ev_seed, forward,
                                reverse);
    Kokkos::fence();
    const auto generic_grad_inv = copy(reverse.grad_attention_inv);
    const auto generic_grad_ev = copy(reverse.grad_attention_ev);
    const double generic_inv_error = compare(
        generic_grad_inv, numbers(fixture.at("generic_grad_inv_features")),
        "generic gradient invariant input");
    const double generic_d_inv_error = compare(
        generic_grad_inv, numbers(fixture.at("generic_grad_d_att_inv")),
        "generic gradient attention invariant update");
    const double generic_ev_error = compare(
        generic_grad_ev, numbers(fixture.at("generic_grad_ev_features")),
        "generic gradient equivariant input");
    const double generic_d_ev_error = compare(
        generic_grad_ev, numbers(fixture.at("generic_grad_d_att_ev")),
        "generic gradient attention equivariant update");

    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-5)
      throw std::runtime_error("fixture finite-difference validation failed");

    constexpr std::size_t matched_nodes = 31865;
    const auto benchmark = block.benchmark(matched_nodes, 20);
    if (benchmark.block_index != 2 || benchmark.nodes != matched_nodes ||
        benchmark.host_boundary_bytes_per_iteration != 0 ||
        !std::isfinite(benchmark.reverse_milliseconds) ||
        benchmark.reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.forward_reverse_milliseconds) ||
        benchmark.forward_reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.nodes_per_second) ||
        benchmark.nodes_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("dev_17 benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "block_index=2\n"
        << "reverse_scope=complete_post_attention_boundary\n"
        << "backend=Kokkos_plus_cuBLAS_DGEMM\n"
        << "energy_to_reverse_device_bridge=1\n"
        << "final_inv_forward_max_abs_error=" << final_inv_error << '\n'
        << "final_ev_forward_max_abs_error=" << final_ev_error << '\n'
        << "energy_seed_max_abs_error=" << energy_seed_error << '\n'
        << "energy_inv_input_gradient_max_abs_error=" << energy_inv_error
        << '\n'
        << "energy_d_att_inv_gradient_max_abs_error=" << energy_d_inv_error
        << '\n'
        << "energy_ev_input_gradient_max_abs_error=" << energy_ev_error
        << '\n'
        << "energy_d_att_ev_gradient_max_abs_error=" << energy_d_ev_error
        << '\n'
        << "generic_inv_input_gradient_max_abs_error=" << generic_inv_error
        << '\n'
        << "generic_d_att_inv_gradient_max_abs_error="
        << generic_d_inv_error << '\n'
        << "generic_ev_input_gradient_max_abs_error=" << generic_ev_error
        << '\n'
        << "generic_d_att_ev_gradient_max_abs_error=" << generic_d_ev_error
        << '\n'
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << benchmark.nodes << '\n'
        << "block2_post_attention_reverse_ms="
        << benchmark.reverse_milliseconds << '\n'
        << "block2_post_attention_forward_reverse_ms="
        << benchmark.forward_reverse_milliseconds << '\n'
        << "block2_post_attention_reverse_nodes_per_second="
        << benchmark.nodes_per_second << '\n'
        << "block2_post_attention_reverse_workspace_bytes="
        << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "reverse_checksum=" << benchmark.checksum << '\n'
        << "pytorch_energy_vjp_reference=PASS\n"
        << "pytorch_generic_vjp_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "block2_post_attention_reverse_equivalence=PASS\n"
        << "block2_post_attention_reverse_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV17_POST_ATTENTION_REVERSE_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV17_POST_ATTENTION_REVERSE_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
