#include "so3lr/kokkos_feature_transformer_block.hpp"
#include "so3lr/kokkos_node_feature_exchange.hpp"
#include "so3lr/kokkos_transformer_block0.hpp"

#include <Kokkos_Core.hpp>
#include <cuda_runtime_api.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;
using Exchange = so3lr::KokkosNodeFeatureExchange;

void mpi_check(int code, const char *operation) {
  if (code == MPI_SUCCESS) return;
  char text[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(code, text, &length);
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(text, static_cast<std::size_t>(length)));
}

void cuda_check(cudaError_t code, const char *operation) {
  if (code == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(code));
}

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

double compare(const DoubleView &actual, const std::vector<double> &expected,
               const std::string &label) {
  const auto host = copy(actual);
  if (host.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    const double error = std::abs(host[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > 5.0e-10 + 5.0e-10 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i));
  }
  return maximum;
}

std::vector<std::size_t> owned_nodes(int rank, std::size_t nodes) {
  const std::size_t split = (nodes + 1) / 2;
  std::vector<std::size_t> result;
  const std::size_t begin = rank == 0 ? 0 : split;
  const std::size_t end = rank == 0 ? split : nodes;
  for (std::size_t node = begin; node < end; ++node) result.push_back(node);
  return result;
}

std::vector<std::size_t> ghost_nodes(int rank, std::size_t nodes) {
  return owned_nodes(1 - rank, nodes);
}

void corrupt_nodes(const DoubleView &invariant, const DoubleView &equivariant,
                   const IndexView &nodes) {
  Kokkos::parallel_for(
      "so3lr_dev29_corrupt_ghost_inv",
      Kokkos::RangePolicy<>(0, nodes.extent(0) * Exchange::invariant_width),
      KOKKOS_LAMBDA(const std::size_t linear) {
        const std::size_t item = linear / Exchange::invariant_width;
        const std::size_t channel = linear % Exchange::invariant_width;
        invariant(nodes(item) * Exchange::invariant_width + channel) =
            -9000.0 - static_cast<double>(linear);
      });
  Kokkos::parallel_for(
      "so3lr_dev29_corrupt_ghost_ev",
      Kokkos::RangePolicy<>(0, nodes.extent(0) * Exchange::equivariant_width),
      KOKKOS_LAMBDA(const std::size_t linear) {
        const std::size_t item = linear / Exchange::equivariant_width;
        const std::size_t channel = linear % Exchange::equivariant_width;
        equivariant(nodes(item) * Exchange::equivariant_width + channel) =
            -19000.0 - static_cast<double>(linear);
      });
}

void device_sendrecv(const DoubleView &send, const DoubleView &receive,
                     int peer, int tag) {
  if (send.extent(0) > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      receive.extent(0) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("MPI feature message exceeds INT_MAX");
  Kokkos::fence();
  mpi_check(MPI_Sendrecv(send.data(), static_cast<int>(send.extent(0)),
                         MPI_DOUBLE, peer, tag, receive.data(),
                         static_cast<int>(receive.extent(0)), MPI_DOUBLE, peer,
                         tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
            "MPI_Sendrecv node features");
}

void exchange_owner_to_ghost(const Exchange &exchange,
                             const IndexView &owners,
                             const IndexView &ghosts,
                             const DoubleView &invariant,
                             const DoubleView &equivariant,
                             so3lr::NodeFeatureExchangeWorkspace &workspace,
                             int peer, int tag) {
  exchange.pack_device(invariant, equivariant, owners, workspace.send_buffer);
  device_sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag);
  exchange.unpack_overwrite_device(workspace.receive_buffer, ghosts, invariant,
                                   equivariant);
  Kokkos::fence();
}

double contribution(int rank, std::size_t node, std::size_t channel) {
  return 0.01 * static_cast<double>(rank + 1) +
         0.001 * static_cast<double>(node + 1) +
         0.000001 * static_cast<double>(channel + 1);
}

double validate_owned_adjoint(const DoubleView &invariant,
                              const DoubleView &equivariant,
                              const std::vector<std::size_t> &owners,
                              int rank) {
  const auto host_inv = copy(invariant);
  const auto host_ev = copy(equivariant);
  double maximum = 0.0;
  for (const auto node : owners) {
    for (std::size_t channel = 0; channel < Exchange::invariant_width;
         ++channel) {
      const double expected = contribution(rank, node, channel) +
                              contribution(1 - rank, node, channel);
      maximum = std::max(
          maximum,
          std::abs(host_inv[node * Exchange::invariant_width + channel] -
                   expected));
    }
    for (std::size_t channel = 0; channel < Exchange::equivariant_width;
         ++channel) {
      const std::size_t packed_channel = Exchange::invariant_width + channel;
      const double expected = contribution(rank, node, packed_channel) +
                              contribution(1 - rank, node, packed_channel);
      maximum = std::max(
          maximum,
          std::abs(host_ev[node * Exchange::equivariant_width + channel] -
                   expected));
    }
  }
  if (maximum > 2.0e-15)
    throw std::runtime_error("reverse owner adjoint mismatch");
  return maximum;
}

std::string uuid_string(const std::array<unsigned char, 16> &uuid) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(32);
  for (const auto value : uuid) {
    result.push_back(digits[value >> 4]);
    result.push_back(digits[value & 0x0f]);
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  mpi_check(MPI_Init(&argc, &argv), "MPI_Init");
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  int visible_devices = 0;
  cuda_check(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
  if (visible_devices <= 0) {
    std::cerr << "rank=" << rank << " no visible CUDA devices\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  int local_rank = rank;
  if (const char *value = std::getenv("SLURM_LOCALID"))
    local_rank = std::atoi(value);
  cuda_check(cudaSetDevice(local_rank % visible_devices), "cudaSetDevice");
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("test was not compiled for CUDA");
#endif
    if (ranks != 2)
      throw std::runtime_error("dev_29 requires exactly two MPI ranks");
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL STACK012_FIXTURE");
    const int peer = 1 - rank;

    int cuda_device = -1;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&cuda_device), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&properties, cuda_device),
               "cudaGetDeviceProperties");
    std::array<unsigned char, 16> local_uuid{};
    std::copy(std::begin(properties.uuid.bytes), std::end(properties.uuid.bytes),
              local_uuid.begin());
    std::array<unsigned char, 32> gathered_uuids{};
    mpi_check(MPI_Allgather(local_uuid.data(), 16, MPI_UNSIGNED_CHAR,
                            gathered_uuids.data(), 16, MPI_UNSIGNED_CHAR,
                            MPI_COMM_WORLD),
              "MPI_Allgather GPU UUIDs");
    bool distinct_gpus = false;
    for (std::size_t i = 0; i < 16; ++i)
      distinct_gpus = distinct_gpus || gathered_uuids[i] != gathered_uuids[16 + i];
    if (!distinct_gpus)
      throw std::runtime_error("MPI ranks selected the same physical GPU");

    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
        "so3lr-native-transformer-stack012-fixture-v1")
      throw std::runtime_error("unexpected transformer fixture contract");
    const auto host_z = integers(fixture.at("atomic_numbers"));
    const auto host_distances = numbers(fixture.at("distances"));
    const auto host_sh = numbers(fixture.at("sh_vectors"));
    const auto host_senders = indices(fixture.at("senders"));
    const auto host_receivers = indices(fixture.at("receivers"));
    const std::size_t nodes = host_z.size();
    const std::size_t edges = host_distances.size();
    if (nodes != 7 || edges != 11)
      throw std::runtime_error("unexpected dev_29 fixture dimensions");

    const auto host_owners = owned_nodes(rank, nodes);
    const auto host_ghosts = ghost_nodes(rank, nodes);
    Int64View z("so3lr_dev29_z", nodes);
    DoubleView distances("so3lr_dev29_distances", edges);
    DoubleView sh("so3lr_dev29_sh", host_sh.size());
    IndexView senders("so3lr_dev29_senders", edges);
    IndexView receivers("so3lr_dev29_receivers", edges);
    IndexView owners("so3lr_dev29_owners", host_owners.size());
    IndexView ghosts("so3lr_dev29_ghosts", host_ghosts.size());
    fill(z, host_z);
    fill(distances, host_distances);
    fill(sh, host_sh);
    fill(senders, host_senders);
    fill(receivers, host_receivers);
    fill(owners, host_owners);
    fill(ghosts, host_ghosts);

    const Exchange exchange;
    so3lr::NodeFeatureExchangeWorkspace boundary(host_owners.size(),
                                                 host_ghosts.size());
    so3lr::KokkosTransformerBlock0 block0(model);
    so3lr::KokkosFeatureTransformerBlock block1(model, 1);
    so3lr::KokkosFeatureTransformerBlock block2(model, 2);
    so3lr::TransformerBlock0DeviceWorkspace workspace0(nodes, edges);
    so3lr::FeatureTransformerDeviceWorkspace workspace1(nodes, edges);
    so3lr::FeatureTransformerDeviceWorkspace workspace2(nodes, edges);

    // The graph remains replicated in dev_29 as a numerical oracle.  Ghost
    // boundary values are deliberately destroyed, then reconstructed solely
    // by owner-to-ghost device-buffer exchange before the next block runs.
    block0.launch_device(z, distances, sh, senders, receivers, workspace0);
    Kokkos::fence();
    corrupt_nodes(workspace0.post_attention.final_inv,
                  workspace0.post_attention.final_ev, ghosts);
    exchange_owner_to_ghost(exchange, owners, ghosts,
                            workspace0.post_attention.final_inv,
                            workspace0.post_attention.final_ev, boundary, peer,
                            2900);
    const double block0_inv_error = compare(
        workspace0.post_attention.final_inv,
        numbers(fixture.at("block0_final_inv")), "block-0 exchanged invariant");
    const double block0_ev_error = compare(
        workspace0.post_attention.final_ev,
        numbers(fixture.at("block0_final_ev")), "block-0 exchanged equivariant");

    block1.launch_device(workspace0.post_attention.final_inv,
                         workspace0.post_attention.final_ev, distances, sh,
                         senders, receivers, workspace1);
    Kokkos::fence();
    corrupt_nodes(workspace1.post_attention.final_inv,
                  workspace1.post_attention.final_ev, ghosts);
    exchange_owner_to_ghost(exchange, owners, ghosts,
                            workspace1.post_attention.final_inv,
                            workspace1.post_attention.final_ev, boundary, peer,
                            2901);
    const double block1_inv_error = compare(
        workspace1.post_attention.final_inv,
        numbers(fixture.at("block1_final_inv")), "block-1 exchanged invariant");
    const double block1_ev_error = compare(
        workspace1.post_attention.final_ev,
        numbers(fixture.at("block1_final_ev")), "block-1 exchanged equivariant");

    block2.launch_device(workspace1.post_attention.final_inv,
                         workspace1.post_attention.final_ev, distances, sh,
                         senders, receivers, workspace2);
    Kokkos::fence();
    const double block2_inv_error = compare(
        workspace2.post_attention.final_inv,
        numbers(fixture.at("block2_final_inv")), "block-2 staged invariant");
    const double block2_ev_error = compare(
        workspace2.post_attention.final_ev,
        numbers(fixture.at("block2_final_ev")), "block-2 staged equivariant");

    // Validate the reverse communication rule independently of the block VJP:
    // local ghost adjoints are sent to the owner and added, never overwritten.
    DoubleView grad_inv("so3lr_dev29_grad_inv",
                        nodes * Exchange::invariant_width);
    DoubleView grad_ev("so3lr_dev29_grad_ev",
                       nodes * Exchange::equivariant_width);
    Kokkos::parallel_for(
        "so3lr_dev29_init_grad_inv",
        Kokkos::RangePolicy<>(0, grad_inv.extent(0)),
        KOKKOS_LAMBDA(const std::size_t linear) {
          const std::size_t node = linear / Exchange::invariant_width;
          const std::size_t channel = linear % Exchange::invariant_width;
          grad_inv(linear) = 0.01 * static_cast<double>(rank + 1) +
                             0.001 * static_cast<double>(node + 1) +
                             0.000001 * static_cast<double>(channel + 1);
        });
    Kokkos::parallel_for(
        "so3lr_dev29_init_grad_ev",
        Kokkos::RangePolicy<>(0, grad_ev.extent(0)),
        KOKKOS_LAMBDA(const std::size_t linear) {
          const std::size_t node = linear / Exchange::equivariant_width;
          const std::size_t channel = linear % Exchange::equivariant_width;
          const std::size_t packed_channel =
              Exchange::invariant_width + channel;
          grad_ev(linear) = 0.01 * static_cast<double>(rank + 1) +
                            0.001 * static_cast<double>(node + 1) +
                            0.000001 * static_cast<double>(packed_channel + 1);
        });
    so3lr::NodeFeatureExchangeWorkspace reverse_boundary(host_ghosts.size(),
                                                         host_owners.size());
    exchange.pack_device(grad_inv, grad_ev, ghosts,
                         reverse_boundary.send_buffer);
    device_sendrecv(reverse_boundary.send_buffer,
                    reverse_boundary.receive_buffer, peer, 2902);
    exchange.unpack_accumulate_device(reverse_boundary.receive_buffer, owners,
                                      grad_inv, grad_ev);
    Kokkos::fence();
    const double adjoint_error =
        validate_owned_adjoint(grad_inv, grad_ev, host_owners, rank);

    // Production-size transport probe.  It measures pack + direct GPU MPI +
    // overwrite-unpack for an 8000-node halo, not the GNN computation itself.
    constexpr std::size_t benchmark_ghost_nodes = 8000;
    constexpr std::size_t warmups = 3;
    constexpr std::size_t repetitions = 20;
    DoubleView bench_inv("so3lr_dev29_bench_inv",
                         benchmark_ghost_nodes * Exchange::invariant_width);
    DoubleView bench_ev("so3lr_dev29_bench_ev",
                        benchmark_ghost_nodes * Exchange::equivariant_width);
    IndexView bench_indices("so3lr_dev29_bench_indices",
                            benchmark_ghost_nodes);
    Kokkos::parallel_for(
        "so3lr_dev29_bench_indices",
        Kokkos::RangePolicy<>(0, benchmark_ghost_nodes),
        KOKKOS_LAMBDA(const std::size_t i) { bench_indices(i) = i; });
    Kokkos::parallel_for(
        "so3lr_dev29_bench_inv_init",
        Kokkos::RangePolicy<>(0, bench_inv.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          bench_inv(i) = static_cast<double>(rank * 10) +
                         static_cast<double>(i % 103) / 103.0;
        });
    Kokkos::parallel_for(
        "so3lr_dev29_bench_ev_init",
        Kokkos::RangePolicy<>(0, bench_ev.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          bench_ev(i) = static_cast<double>(rank * 20) +
                        static_cast<double>(i % 59) / 59.0;
        });
    so3lr::NodeFeatureExchangeWorkspace bench_workspace(
        benchmark_ghost_nodes, benchmark_ghost_nodes);
    auto benchmark_once = [&](int tag) {
      exchange.pack_device(bench_inv, bench_ev, bench_indices,
                           bench_workspace.send_buffer);
      device_sendrecv(bench_workspace.send_buffer,
                      bench_workspace.receive_buffer, peer, tag);
      exchange.unpack_overwrite_device(bench_workspace.receive_buffer,
                                       bench_indices, bench_inv, bench_ev);
      Kokkos::fence();
    };
    for (std::size_t i = 0; i < warmups; ++i) benchmark_once(2910);
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier benchmark start");
    const double start = MPI_Wtime();
    for (std::size_t i = 0; i < repetitions; ++i) benchmark_once(2911);
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier benchmark finish");
    const double local_seconds = MPI_Wtime() - start;
    double transport_seconds = 0.0;
    mpi_check(MPI_Reduce(&local_seconds, &transport_seconds, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce feature exchange time");

    double local_max_error = std::max(
        {block0_inv_error, block0_ev_error, block1_inv_error, block1_ev_error,
         block2_inv_error, block2_ev_error, adjoint_error});
    double global_max_error = 0.0;
    mpi_check(MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce feature error");

    if (rank == 0) {
      std::array<unsigned char, 16> rank0_uuid{};
      std::array<unsigned char, 16> rank1_uuid{};
      std::copy_n(gathered_uuids.begin(), 16, rank0_uuid.begin());
      std::copy_n(gathered_uuids.begin() + 16, 16, rank1_uuid.begin());
      const double exchange_ms =
          transport_seconds * 1000.0 / static_cast<double>(repetitions);
      const std::size_t message_bytes =
          Exchange::packed_bytes(benchmark_ghost_nodes);
      const double one_way_gb_per_second =
          static_cast<double>(message_bytes) / (exchange_ms / 1000.0) / 1.0e9;
      std::cout << std::setprecision(17)
                << "mpi_ranks=2\n"
                << "rank0_gpu_uuid=" << uuid_string(rank0_uuid) << '\n'
                << "rank1_gpu_uuid=" << uuid_string(rank1_uuid) << '\n'
                << "distinct_physical_gpus=1\n"
                << "feature_invariant_width=" << Exchange::invariant_width
                << '\n'
                << "feature_equivariant_width=" << Exchange::equivariant_width
                << '\n'
                << "feature_packed_width=" << Exchange::packed_width << '\n'
                << "forward_feature_exchange_boundaries=2\n"
                << "reverse_adjoint_exchange_boundaries_required=2\n"
                << "block0_feature_max_abs_error="
                << std::max(block0_inv_error, block0_ev_error) << '\n'
                << "block1_feature_max_abs_error="
                << std::max(block1_inv_error, block1_ev_error) << '\n'
                << "block2_feature_max_abs_error="
                << std::max(block2_inv_error, block2_ev_error) << '\n'
                << "reverse_owned_adjoint_max_abs_error=" << adjoint_error
                << '\n'
                << "global_feature_contract_max_abs_error=" << global_max_error
                << '\n'
                << "benchmark_ghost_nodes_per_rank=" << benchmark_ghost_nodes
                << '\n'
                << "feature_message_bytes_per_rank=" << message_bytes << '\n'
                << "pack_mpi_unpack_ms=" << exchange_ms << '\n'
                << "effective_one_way_GB_per_s=" << one_way_gb_per_second
                << '\n'
                << "transport_repetitions=" << repetitions << '\n'
                << "application_host_staging_bytes=0\n"
                << "cuda_aware_mpi_feature_transport=PASS\n"
                << "forward_owner_to_ghost_overwrite=PASS\n"
                << "reverse_ghost_to_owner_accumulation=PASS\n"
                << "staged_transformer_forward_equivalence=PASS\n"
                << "pytorch_transformer_stack012_reference=PASS\n"
                << "distinct_rank_gpu_binding=PASS\n"
                << "SO3LR_NATIVE_DEV29_FEATURE_EXCHANGE_TEST=PASS\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank
              << " SO3LR_NATIVE_DEV29_FEATURE_EXCHANGE_TEST=FAIL: "
              << error.what() << '\n';
    result_code = 1;
  }
  if (result_code != 0) {
    Kokkos::finalize();
    MPI_Abort(MPI_COMM_WORLD, result_code);
    return result_code;
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
