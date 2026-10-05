#include "so3lr/kokkos_learned_energy_reverse.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
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
    if (error > 5.0e-8 + 5.0e-8 * std::abs(expected[i]))
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
      throw std::runtime_error("usage: test MODEL ENERGY_CHAIN_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-learned-energy-reverse-chain-fixture-v1" ||
        fixture.at("geometry").string() !=
            "physical_water_dimer_nonperiodic" ||
        fixture.at("reverse_scope").string() !=
            "energy_head_through_transformer_blocks_2_1_0")
      throw std::runtime_error("unexpected dev_22 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t edges =
        static_cast<std::size_t>(fixture.at("edges").unsigned_integer());
    constexpr std::size_t ev_width = 24;

    Int64View atomic_numbers("so3lr_dev22_z", nodes);
    DoubleView distances("so3lr_dev22_distances", edges);
    DoubleView sh("so3lr_dev22_sh", edges * ev_width);
    IndexView senders("so3lr_dev22_senders", edges);
    IndexView receivers("so3lr_dev22_receivers", edges);
    fill(atomic_numbers, integers(fixture.at("atomic_numbers")));
    fill(distances, numbers(fixture.at("distances")));
    fill(sh, numbers(fixture.at("sh_vectors")));
    fill(senders, indices(fixture.at("senders")));
    fill(receivers, indices(fixture.at("receivers")));

    const so3lr::KokkosLearnedEnergyReverseChain chain(model);
    if (!chain.chain_contract_verified() || !chain.shared_kokkos_stream())
      throw std::runtime_error("dev_22 runtime chain contract failed");
    so3lr::LearnedEnergyReverseWorkspace workspace(nodes, edges);
    chain.launch_forward_device(atomic_numbers, distances, sh, senders,
                                receivers, workspace);
    chain.launch_reverse_device(atomic_numbers, distances, sh, senders,
                                receivers, workspace);
    Kokkos::fence();

    const double energy_error = compare(
        copy(chain.atomic_energies(workspace)),
        numbers(fixture.at("atomic_energies")), "atomic learned energies");
    const auto reduction_values = copy(chain.reductions(workspace));
    const double total_energy_error =
        std::abs(reduction_values.at(1) - fixture.at("total_energy").number());
    if (total_energy_error >
        5.0e-8 + 5.0e-8 * std::abs(fixture.at("total_energy").number()))
      throw std::runtime_error("total learned energy mismatch");
    const double embedding_error = compare(
        copy(chain.grad_embedding(workspace)),
        numbers(fixture.at("grad_embedding")), "embedding gradient");
    const double initial_ev_error = compare(
        copy(chain.grad_initial_ev(workspace)),
        numbers(fixture.at("grad_initial_ev")), "initial EV gradient");
    const double distance_error = compare(
        copy(chain.grad_distances(workspace)),
        numbers(fixture.at("grad_distances")), "accumulated distance gradient");
    const double sh_error = compare(
        copy(chain.grad_sh(workspace)),
        numbers(fixture.at("grad_sh_vectors")), "accumulated SH gradient");

    double block_distance_error[3] = {0.0, 0.0, 0.0};
    double block_sh_error[3] = {0.0, 0.0, 0.0};
    for (std::size_t block = 0; block < 3; ++block) {
      block_distance_error[block] = compare(
          copy(chain.block_grad_distances(workspace, block)),
          numbers(fixture.at("grad_distances_block" +
                             std::to_string(block))),
          "block distance contribution " + std::to_string(block));
      block_sh_error[block] = compare(
          copy(chain.block_grad_sh(workspace, block)),
          numbers(fixture.at("grad_sh_vectors_block" +
                             std::to_string(block))),
          "block SH contribution " + std::to_string(block));
      if (fixture.at("maximum_abs_distance_gradient_block" +
                     std::to_string(block)).number() <= 1.0e-12 ||
          fixture.at("maximum_abs_sh_gradient_block" +
                     std::to_string(block)).number() <= 1.0e-12)
        throw std::runtime_error("missing per-block geometric sensitivity");
    }
    const double finite_difference_error =
        fixture.at("finite_difference_max_scaled_error").number();
    if (!std::isfinite(finite_difference_error) ||
        finite_difference_error > 2.0e-5)
      throw std::runtime_error("fixture finite-difference reference failed");

    constexpr std::size_t matched_nodes = 31865;
    constexpr std::size_t matched_edges = 954436;
    const auto benchmark = chain.benchmark(matched_nodes, matched_edges, 2);
    if (benchmark.nodes != matched_nodes || benchmark.edges != matched_edges ||
        benchmark.host_boundary_bytes_per_iteration != 0 ||
        !std::isfinite(benchmark.forward_milliseconds) ||
        benchmark.forward_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.reverse_milliseconds) ||
        benchmark.reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.forward_reverse_milliseconds) ||
        benchmark.forward_reverse_milliseconds <= 0.0 ||
        !std::isfinite(benchmark.edges_per_second) ||
        benchmark.edges_per_second <= 0.0 ||
        !std::isfinite(benchmark.checksum))
      throw std::runtime_error("dev_22 matched benchmark is invalid");

    std::cout
        << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
        << "backend=Kokkos_CUDA_plus_cuBLAS_DGEMM_VJP\n"
        << "reverse_scope=energy_head_through_transformer_blocks_2_1_0\n"
        << "native_reverse_transformer_blocks=3/3\n"
        << "learned_energy_atomic_max_abs_error=" << energy_error << '\n'
        << "learned_energy_total_error=" << total_energy_error << '\n'
        << "embedding_gradient_max_abs_error=" << embedding_error << '\n'
        << "initial_ev_gradient_max_abs_error=" << initial_ev_error << '\n'
        << "distance_gradient_max_abs_error=" << distance_error << '\n'
        << "sh_gradient_max_abs_error=" << sh_error << '\n';
    for (std::size_t block = 0; block < 3; ++block) {
      std::cout << "block" << block
                << "_distance_contribution_error="
                << block_distance_error[block] << '\n'
                << "block" << block << "_sh_contribution_error="
                << block_sh_error[block] << '\n';
    }
    std::cout
        << "finite_difference_max_scaled_error=" << finite_difference_error
        << '\n'
        << "matched_stage2_nodes=" << benchmark.nodes << '\n'
        << "matched_stage2_edges=" << benchmark.edges << '\n'
        << "learned_energy_forward_ms=" << benchmark.forward_milliseconds
        << '\n'
        << "learned_energy_reverse_ms=" << benchmark.reverse_milliseconds
        << '\n'
        << "learned_energy_forward_reverse_ms="
        << benchmark.forward_reverse_milliseconds << '\n'
        << "learned_energy_reverse_edges_per_second="
        << benchmark.edges_per_second << '\n'
        << "learned_energy_reverse_workspace_bytes="
        << benchmark.workspace_bytes << '\n'
        << "host_boundary_bytes_per_iteration="
        << benchmark.host_boundary_bytes_per_iteration << '\n'
        << "reverse_checksum=" << benchmark.checksum << '\n'
        << "pytorch_learned_energy_reverse_chain_reference=PASS\n"
        << "finite_difference_reference=PASS\n"
        << "per_block_geometry_accumulation=PASS\n"
        << "learned_energy_reverse_chain_equivalence=PASS\n"
        << "learned_energy_reverse_chain_device_residency=PASS\n"
        << "SO3LR_NATIVE_DEV22_ENERGY_REVERSE_CHAIN_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV22_ENERGY_REVERSE_CHAIN_TEST=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
