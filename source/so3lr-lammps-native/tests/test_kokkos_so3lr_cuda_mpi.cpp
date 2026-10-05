#include "so3lr/kokkos_so3lr_ownership.hpp"

#include <Kokkos_Core.hpp>
#include <cuda_runtime_api.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

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
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

std::vector<std::size_t> indices(const so3lr::Json &value) {
  std::vector<std::size_t> result;
  for (const auto &item : value.array())
    result.push_back(static_cast<std::size_t>(item.unsigned_integer()));
  return result;
}

std::vector<std::int64_t> integers(const so3lr::Json &value) {
  std::vector<std::int64_t> result;
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

double sum(const DoubleView &view) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev28_sum", Kokkos::RangePolicy<>(0, view.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double &update) {
        update += view(i);
      },
      result);
  Kokkos::fence();
  return result;
}

struct PairPartition {
  std::vector<double> vectors;
  std::vector<std::size_t> senders;
  std::vector<std::size_t> receivers;
};

PairPartition partition_pairs(const std::vector<double> &vectors,
                              const std::vector<std::size_t> &senders,
                              const std::vector<std::size_t> &receivers,
                              std::size_t parity) {
  if (vectors.size() != senders.size() * 3 ||
      receivers.size() != senders.size())
    throw std::runtime_error("LR fixture shape mismatch");
  PairPartition result;
  for (std::size_t pair = parity; pair < senders.size(); pair += 2) {
    result.senders.push_back(senders[pair]);
    result.receivers.push_back(receivers[pair]);
    for (std::size_t component = 0; component < 3; ++component)
      result.vectors.push_back(vectors[pair * 3 + component]);
  }
  return result;
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
      throw std::runtime_error("dev_28 requires exactly two MPI ranks");
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL PHYSICAL_FORCE_FIXTURE");

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
            "so3lr-native-physical-lr-cartesian-force-fixture-v1" ||
        fixture.at("lr_topology").string() !=
            "unique_unordered_half_pairs")
      throw std::runtime_error("unexpected dev_28 fixture contract");
    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t sr_edges =
        static_cast<std::size_t>(fixture.at("sr_edges").unsigned_integer());
    const std::size_t lr_pairs =
        static_cast<std::size_t>(fixture.at("lr_pairs").unsigned_integer());
    if (nodes != 6 || lr_pairs != 15)
      throw std::runtime_error("dev_28 requires the six-node ownership fixture");

    const auto host_lr_vectors = numbers(fixture.at("lr_pair_vectors"));
    const auto host_lr_senders = indices(fixture.at("lr_senders"));
    const auto host_lr_receivers = indices(fixture.at("lr_receivers"));
    const auto local_pairs = partition_pairs(
        host_lr_vectors, host_lr_senders, host_lr_receivers,
        static_cast<std::size_t>(rank));

    Int64View z("so3lr_dev28_z", nodes);
    DoubleView sr_vectors("so3lr_dev28_sr_vectors", sr_edges * 3);
    IndexView sr_senders("so3lr_dev28_sr_senders", sr_edges);
    IndexView sr_receivers("so3lr_dev28_sr_receivers", sr_edges);
    DoubleView lr_vectors("so3lr_dev28_lr_vectors", local_pairs.vectors.size());
    IndexView lr_senders("so3lr_dev28_lr_senders", local_pairs.senders.size());
    IndexView lr_receivers("so3lr_dev28_lr_receivers", local_pairs.receivers.size());
    IndexView owned_nodes("so3lr_dev28_owned_nodes", 3);
    IndexView ghost_nodes("so3lr_dev28_ghost_nodes", 3);
    fill(z, integers(fixture.at("atomic_numbers")));
    fill(sr_vectors, numbers(fixture.at("sr_edge_vectors")));
    fill(sr_senders, indices(fixture.at("sr_senders")));
    fill(sr_receivers, indices(fixture.at("sr_receivers")));
    fill(lr_vectors, local_pairs.vectors);
    fill(lr_senders, local_pairs.senders);
    fill(lr_receivers, local_pairs.receivers);
    if (rank == 0) {
      fill(owned_nodes, std::vector<std::size_t>{0, 1, 2});
      fill(ghost_nodes, std::vector<std::size_t>{3, 4, 5});
    } else {
      fill(owned_nodes, std::vector<std::size_t>{3, 4, 5});
      fill(ghost_nodes, std::vector<std::size_t>{0, 1, 2});
    }

    const so3lr::KokkosSo3lrOwnedEvaluator evaluator(model);
    so3lr::So3lrEvaluatorWorkspace workspace(
        nodes, sr_edges, local_pairs.senders.size());
    evaluator.launch_rank_device(
        z, sr_vectors, sr_senders, sr_receivers,
        lr_vectors, lr_senders, lr_receivers, owned_nodes, workspace);
    Kokkos::fence();

    const double local_energy =
        sum(evaluator.evaluator().atomic_energies(workspace));
    double distributed_energy = 0.0;
    mpi_check(MPI_Allreduce(&local_energy, &distributed_energy, 1, MPI_DOUBLE,
                            MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce energy");

    DoubleView send_forces("so3lr_dev28_send_forces", 9);
    DoubleView receive_forces("so3lr_dev28_receive_forces", 9);
    so3lr::launch_pack_reverse_forces_device(
        evaluator.evaluator().atomic_forces(workspace), ghost_nodes,
        send_forces);
    Kokkos::fence();
    const int peer = 1 - rank;
    mpi_check(MPI_Sendrecv(
                  send_forces.data(), 9, MPI_DOUBLE, peer, 2801,
                  receive_forces.data(), 9, MPI_DOUBLE, peer, 2801,
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE),
              "MPI_Sendrecv device force buffer");
    so3lr::launch_unpack_reverse_forces_device(
        receive_forces, owned_nodes,
        evaluator.evaluator().atomic_forces(workspace));
    Kokkos::fence();

    const auto forces = copy(evaluator.evaluator().atomic_forces(workspace));
    const auto charges = copy(evaluator.evaluator().partial_charges(workspace));
    const auto hirshfeld =
        copy(evaluator.evaluator().hirshfeld_ratios(workspace));
    const auto expected_forces = numbers(fixture.at("assembled_atomic_forces"));
    const auto expected_charges = numbers(fixture.at("partial_charges"));
    const auto expected_hirshfeld = numbers(fixture.at("hirshfeld_ratios"));
    const auto positions = numbers(fixture.at("positions"));
    const std::size_t first_owned = rank == 0 ? 0 : 3;
    double local_force_error = 0.0;
    double local_charge_error = 0.0;
    double local_hirshfeld_error = 0.0;
    std::array<double, 3> local_net_force{};
    std::array<double, 3> local_net_torque{};
    for (std::size_t offset = 0; offset < 3; ++offset) {
      const std::size_t node = first_owned + offset;
      local_charge_error = std::max(
          local_charge_error, std::abs(charges[node] - expected_charges[node]));
      local_hirshfeld_error = std::max(
          local_hirshfeld_error,
          std::abs(hirshfeld[node] - expected_hirshfeld[node]));
      const double x = positions[node * 3];
      const double y = positions[node * 3 + 1];
      const double z_coordinate = positions[node * 3 + 2];
      const double fx = forces[node * 3];
      const double fy = forces[node * 3 + 1];
      const double fz = forces[node * 3 + 2];
      for (std::size_t component = 0; component < 3; ++component)
        local_force_error = std::max(
            local_force_error,
            std::abs(forces[node * 3 + component] -
                     expected_forces[node * 3 + component]));
      local_net_force[0] += fx;
      local_net_force[1] += fy;
      local_net_force[2] += fz;
      local_net_torque[0] += y * fz - z_coordinate * fy;
      local_net_torque[1] += z_coordinate * fx - x * fz;
      local_net_torque[2] += x * fy - y * fx;
    }

    double force_error = 0.0;
    double charge_error = 0.0;
    double hirshfeld_error = 0.0;
    std::array<double, 3> net_force{};
    std::array<double, 3> net_torque{};
    mpi_check(MPI_Allreduce(&local_force_error, &force_error, 1, MPI_DOUBLE,
                            MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce force error");
    mpi_check(MPI_Allreduce(&local_charge_error, &charge_error, 1, MPI_DOUBLE,
                            MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce charge error");
    mpi_check(MPI_Allreduce(&local_hirshfeld_error, &hirshfeld_error, 1,
                            MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce Hirshfeld error");
    mpi_check(MPI_Allreduce(local_net_force.data(), net_force.data(), 3,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce net force");
    mpi_check(MPI_Allreduce(local_net_torque.data(), net_torque.data(), 3,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce net torque");
    const double energy_error =
        std::abs(distributed_energy - fixture.at("total_energy").number());
    double maximum_net_force = 0.0;
    double maximum_net_torque = 0.0;
    for (std::size_t component = 0; component < 3; ++component) {
      maximum_net_force = std::max(maximum_net_force,
                                   std::abs(net_force[component]));
      maximum_net_torque = std::max(maximum_net_torque,
                                    std::abs(net_torque[component]));
    }
    if (energy_error > 3.0e-7 || force_error > 3.0e-7 ||
        charge_error > 3.0e-7 || hirshfeld_error > 3.0e-7 ||
        maximum_net_force > 3.0e-7 || maximum_net_torque > 3.0e-7)
      throw std::runtime_error("real two-rank energy/force equivalence failed");

    // Direct device-pointer transport benchmark.  No application-level host
    // staging is used.  The selected OpenMPI transport may stage internally.
    constexpr std::size_t transport_values = 24000;
    constexpr std::size_t warmups = 10;
    constexpr std::size_t repetitions = 200;
    DoubleView transport_send("so3lr_dev28_transport_send", transport_values);
    DoubleView transport_receive("so3lr_dev28_transport_receive",
                                 transport_values);
    Kokkos::parallel_for(
        "so3lr_dev28_transport_init",
        Kokkos::RangePolicy<>(0, transport_values),
        KOKKOS_LAMBDA(const std::size_t i) {
          transport_send(i) = static_cast<double>(rank * 1000) +
                              static_cast<double>(i % 997);
        });
    Kokkos::fence();
    for (std::size_t i = 0; i < warmups; ++i)
      mpi_check(MPI_Sendrecv(
                    transport_send.data(), static_cast<int>(transport_values),
                    MPI_DOUBLE, peer, 2802, transport_receive.data(),
                    static_cast<int>(transport_values), MPI_DOUBLE, peer, 2802,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                "MPI_Sendrecv warmup");
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier benchmark start");
    const double start = MPI_Wtime();
    for (std::size_t i = 0; i < repetitions; ++i)
      mpi_check(MPI_Sendrecv(
                    transport_send.data(), static_cast<int>(transport_values),
                    MPI_DOUBLE, peer, 2803, transport_receive.data(),
                    static_cast<int>(transport_values), MPI_DOUBLE, peer, 2803,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                "MPI_Sendrecv benchmark");
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier benchmark finish");
    const double local_transport_seconds = MPI_Wtime() - start;
    double transport_seconds = 0.0;
    mpi_check(MPI_Reduce(&local_transport_seconds, &transport_seconds, 1,
                         MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce transport time");
    const auto received_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), transport_receive);
    const double expected_first = static_cast<double>(peer * 1000);
    const double expected_last = static_cast<double>(peer * 1000) +
                                 static_cast<double>((transport_values - 1) % 997);
    if (received_host(0) != expected_first ||
        received_host(transport_values - 1) != expected_last)
      throw std::runtime_error("device-pointer transport payload mismatch");

    if (rank == 0) {
      std::array<unsigned char, 16> rank0_uuid{};
      std::array<unsigned char, 16> rank1_uuid{};
      std::copy_n(gathered_uuids.begin(), 16, rank0_uuid.begin());
      std::copy_n(gathered_uuids.begin() + 16, 16, rank1_uuid.begin());
      const double transport_ms =
          transport_seconds * 1000.0 / static_cast<double>(repetitions);
      const std::size_t transport_bytes = transport_values * sizeof(double);
      const double transport_gb_per_second =
          static_cast<double>(transport_bytes) / (transport_ms / 1000.0) / 1.0e9;
      std::cout << std::setprecision(17)
                << "mpi_ranks=2\n"
                << "rank0_gpu_uuid=" << uuid_string(rank0_uuid) << '\n'
                << "rank1_gpu_uuid=" << uuid_string(rank1_uuid) << '\n'
                << "distinct_physical_gpus=1\n"
                << "rank0_lr_half_pairs=8\n"
                << "rank1_lr_half_pairs=7\n"
                << "distributed_total_energy=" << distributed_energy << '\n'
                << "mpi_energy_max_abs_error=" << energy_error << '\n'
                << "mpi_force_max_abs_error=" << force_error << '\n'
                << "mpi_charge_max_abs_error=" << charge_error << '\n'
                << "mpi_hirshfeld_max_abs_error=" << hirshfeld_error << '\n'
                << "mpi_maximum_abs_net_force=" << maximum_net_force << '\n'
                << "mpi_maximum_abs_net_torque=" << maximum_net_torque << '\n'
                << "force_message_values_per_rank=9\n"
                << "force_message_bytes_per_rank=72\n"
                << "transport_message_bytes=" << transport_bytes << '\n'
                << "transport_sendrecv_ms=" << transport_ms << '\n'
                << "transport_effective_one_way_GB_per_s="
                << transport_gb_per_second << '\n'
                << "transport_repetitions=" << repetitions << '\n'
                << "application_host_staging_bytes=0\n"
                << "cuda_aware_mpi_device_sendrecv=PASS\n"
                << "real_two_rank_owned_energy=PASS\n"
                << "real_two_rank_reverse_force_exchange=PASS\n"
                << "real_two_rank_energy_force_equivalence=PASS\n"
                << "distinct_rank_gpu_binding=PASS\n"
                << "pytorch_full_so3lr_cartesian_force_reference=PASS\n"
                << "SO3LR_NATIVE_DEV28_CUDA_MPI_TEST=PASS\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank
              << " SO3LR_NATIVE_DEV28_CUDA_MPI_TEST=FAIL: "
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
