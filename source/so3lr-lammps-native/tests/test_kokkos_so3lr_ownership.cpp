#include "so3lr/kokkos_so3lr_ownership.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
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

double max_error(const std::vector<double> &actual,
                 const std::vector<double> &expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("comparison size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
  return maximum;
}

double sum(const DoubleView &view) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev27_sum", Kokkos::RangePolicy<>(0, view.extent(0)),
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

struct DevicePairs {
  DoubleView vectors;
  IndexView senders;
  IndexView receivers;

  DevicePairs(const std::string &label, const PairPartition &host)
      : vectors(label + "_vectors", host.vectors.size()),
        senders(label + "_senders", host.senders.size()),
        receivers(label + "_receivers", host.receivers.size()) {
    fill(vectors, host.vectors);
    fill(senders, host.senders);
    fill(receivers, host.receivers);
  }
};

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("test was not compiled for CUDA");
#endif
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL PHYSICAL_FORCE_FIXTURE");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-physical-lr-cartesian-force-fixture-v1" ||
        fixture.at("lr_topology").string() !=
            "unique_unordered_half_pairs")
      throw std::runtime_error("unexpected dev_27 fixture contract");

    const std::size_t nodes =
        static_cast<std::size_t>(fixture.at("nodes").unsigned_integer());
    const std::size_t sr_edges =
        static_cast<std::size_t>(fixture.at("sr_edges").unsigned_integer());
    const std::size_t lr_pairs =
        static_cast<std::size_t>(fixture.at("lr_pairs").unsigned_integer());
    if (nodes != 6 || lr_pairs < 2)
      throw std::runtime_error("dev_27 requires the six-node ownership fixture");

    const auto host_lr_vectors = numbers(fixture.at("lr_pair_vectors"));
    const auto host_lr_senders = indices(fixture.at("lr_senders"));
    const auto host_lr_receivers = indices(fixture.at("lr_receivers"));
    const auto rank0_pairs = partition_pairs(
        host_lr_vectors, host_lr_senders, host_lr_receivers, 0);
    const auto rank1_pairs = partition_pairs(
        host_lr_vectors, host_lr_senders, host_lr_receivers, 1);
    if (rank0_pairs.senders.size() + rank1_pairs.senders.size() != lr_pairs)
      throw std::runtime_error("LR half-pair partition is incomplete");

    Int64View z("so3lr_dev27_z", nodes);
    DoubleView sr_vectors("so3lr_dev27_sr_vectors", sr_edges * 3);
    IndexView sr_senders("so3lr_dev27_sr_senders", sr_edges);
    IndexView sr_receivers("so3lr_dev27_sr_receivers", sr_edges);
    DoubleView lr_vectors("so3lr_dev27_lr_vectors", lr_pairs * 3);
    IndexView lr_senders("so3lr_dev27_lr_senders", lr_pairs);
    IndexView lr_receivers("so3lr_dev27_lr_receivers", lr_pairs);
    fill(z, integers(fixture.at("atomic_numbers")));
    fill(sr_vectors, numbers(fixture.at("sr_edge_vectors")));
    fill(sr_senders, indices(fixture.at("sr_senders")));
    fill(sr_receivers, indices(fixture.at("sr_receivers")));
    fill(lr_vectors, host_lr_vectors);
    fill(lr_senders, host_lr_senders);
    fill(lr_receivers, host_lr_receivers);
    DevicePairs rank0_lr("so3lr_dev27_rank0_lr", rank0_pairs);
    DevicePairs rank1_lr("so3lr_dev27_rank1_lr", rank1_pairs);

    // Serial dev_26 oracle.
    const so3lr::KokkosSo3lrEvaluator serial(model);
    so3lr::So3lrEvaluatorWorkspace serial_workspace(nodes, sr_edges, lr_pairs);
    serial.launch_device(z, sr_vectors, sr_senders, sr_receivers,
                         lr_vectors, lr_senders, lr_receivers,
                         serial_workspace);
    Kokkos::fence();
    const auto serial_forces = copy(serial.atomic_forces(serial_workspace));
    const double serial_energy = sum(serial.atomic_energies(serial_workspace));

    // Synthetic rank ownership: rank 0 owns atoms 0..2 and rank 1 owns 3..5.
    IndexView rank0_owned("so3lr_dev27_rank0_owned", 3);
    IndexView rank1_owned("so3lr_dev27_rank1_owned", 3);
    IndexView rank0_ghosts("so3lr_dev27_rank0_ghosts", 3);
    IndexView rank1_ghosts("so3lr_dev27_rank1_ghosts", 3);
    fill(rank0_owned, std::vector<std::size_t>{0, 1, 2});
    fill(rank1_owned, std::vector<std::size_t>{3, 4, 5});
    fill(rank0_ghosts, std::vector<std::size_t>{3, 4, 5});
    fill(rank1_ghosts, std::vector<std::size_t>{0, 1, 2});

    const so3lr::KokkosSo3lrOwnedEvaluator distributed(model);
    so3lr::So3lrEvaluatorWorkspace rank0_workspace(
        nodes, sr_edges, rank0_pairs.senders.size());
    so3lr::So3lrEvaluatorWorkspace rank1_workspace(
        nodes, sr_edges, rank1_pairs.senders.size());
    distributed.launch_rank_device(
        z, sr_vectors, sr_senders, sr_receivers,
        rank0_lr.vectors, rank0_lr.senders, rank0_lr.receivers,
        rank0_owned, rank0_workspace);
    distributed.launch_rank_device(
        z, sr_vectors, sr_senders, sr_receivers,
        rank1_lr.vectors, rank1_lr.senders, rank1_lr.receivers,
        rank1_owned, rank1_workspace);
    Kokkos::fence();

    const double partitioned_energy =
        sum(distributed.evaluator().atomic_energies(rank0_workspace)) +
        sum(distributed.evaluator().atomic_energies(rank1_workspace));
    const double ownership_energy_error =
        std::abs(partitioned_energy - serial_energy);

    // Pack both messages before unpacking either one.  This mirrors an MPI
    // reverse exchange and prevents received contributions from being packed
    // into the opposite message.
    DoubleView rank0_to_rank1("so3lr_dev27_rank0_to_rank1", 9);
    DoubleView rank1_to_rank0("so3lr_dev27_rank1_to_rank0", 9);
    so3lr::launch_pack_reverse_forces_device(
        distributed.evaluator().atomic_forces(rank0_workspace),
        rank0_ghosts, rank0_to_rank1);
    so3lr::launch_pack_reverse_forces_device(
        distributed.evaluator().atomic_forces(rank1_workspace),
        rank1_ghosts, rank1_to_rank0);
    so3lr::launch_unpack_reverse_forces_device(
        rank0_to_rank1, rank0_ghosts,
        distributed.evaluator().atomic_forces(rank1_workspace));
    so3lr::launch_unpack_reverse_forces_device(
        rank1_to_rank0, rank1_ghosts,
        distributed.evaluator().atomic_forces(rank0_workspace));
    Kokkos::fence();

    const auto rank0_after =
        copy(distributed.evaluator().atomic_forces(rank0_workspace));
    const auto rank1_after =
        copy(distributed.evaluator().atomic_forces(rank1_workspace));
    std::vector<double> assembled(serial_forces.size(), 0.0);
    for (std::size_t node = 0; node < nodes; ++node) {
      const auto &source = node < 3 ? rank0_after : rank1_after;
      for (std::size_t component = 0; component < 3; ++component)
        assembled[node * 3 + component] = source[node * 3 + component];
    }
    const double ownership_force_error = max_error(assembled, serial_forces);
    const double pytorch_force_error = max_error(
        assembled, numbers(fixture.at("assembled_atomic_forces")));
    const double charge_replica_error = max_error(
        copy(distributed.evaluator().partial_charges(rank0_workspace)),
        copy(distributed.evaluator().partial_charges(rank1_workspace)));
    const double hirshfeld_replica_error = max_error(
        copy(distributed.evaluator().hirshfeld_ratios(rank0_workspace)),
        copy(distributed.evaluator().hirshfeld_ratios(rank1_workspace)));
    if (ownership_energy_error > 3.0e-7 ||
        ownership_force_error > 3.0e-7 || pytorch_force_error > 3.0e-7 ||
        charge_replica_error > 3.0e-7 || hirshfeld_replica_error > 3.0e-7)
      throw std::runtime_error("synthetic two-rank ownership equivalence failed");

    // Measure only ownership bookkeeping; evaluator performance remains the
    // dev_26 result.  This is not an MPI latency benchmark.
    constexpr std::size_t benchmark_nodes = 31865;
    constexpr std::size_t benchmark_owned = 8000;
    constexpr std::size_t repetitions = 100;
    DoubleView benchmark_forces("so3lr_dev27_benchmark_forces",
                                benchmark_nodes * 3);
    DoubleView benchmark_seeds("so3lr_dev27_benchmark_seeds",
                               benchmark_nodes);
    IndexView benchmark_indices("so3lr_dev27_benchmark_indices",
                                benchmark_owned);
    DoubleView benchmark_buffer("so3lr_dev27_benchmark_buffer",
                                benchmark_owned * 3);
    Kokkos::parallel_for(
        "so3lr_dev27_benchmark_init",
        Kokkos::RangePolicy<>(0, benchmark_nodes * 3),
        KOKKOS_LAMBDA(const std::size_t i) {
          benchmark_forces(i) = 1.0e-6 * static_cast<double>(i % 97);
        });
    Kokkos::parallel_for(
        "so3lr_dev27_benchmark_indices",
        Kokkos::RangePolicy<>(0, benchmark_owned),
        KOKKOS_LAMBDA(const std::size_t i) {
          benchmark_indices(i) = i;
        });
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
      Kokkos::deep_copy(benchmark_seeds, 0.0);
      so3lr::launch_pack_reverse_forces_device(
          benchmark_forces, benchmark_indices, benchmark_buffer);
      so3lr::launch_unpack_reverse_forces_device(
          benchmark_buffer, benchmark_indices, benchmark_forces);
    }
    Kokkos::fence();
    const double bookkeeping_ms =
        timer.seconds() * 1000.0 / static_cast<double>(repetitions);
    if (!std::isfinite(bookkeeping_ms) || bookkeeping_ms <= 0.0)
      throw std::runtime_error("ownership benchmark is not finite");

    std::cout << std::setprecision(17)
              << "serial_total_energy=" << serial_energy << '\n'
              << "partitioned_total_energy=" << partitioned_energy << '\n'
              << "ownership_energy_max_abs_error="
              << ownership_energy_error << '\n'
              << "ownership_force_max_abs_error="
              << ownership_force_error << '\n'
              << "pytorch_force_max_abs_error=" << pytorch_force_error << '\n'
              << "replicated_charge_max_abs_error="
              << charge_replica_error << '\n'
              << "replicated_hirshfeld_max_abs_error="
              << hirshfeld_replica_error << '\n'
              << "synthetic_ranks=2\n"
              << "rank0_owned_nodes=3\n"
              << "rank1_owned_nodes=3\n"
              << "rank0_lr_half_pairs=" << rank0_pairs.senders.size() << '\n'
              << "rank1_lr_half_pairs=" << rank1_pairs.senders.size() << '\n'
              << "reverse_force_values_per_message=9\n"
              << "ownership_bookkeeping_ms=" << bookkeeping_ms << '\n'
              << "benchmark_reverse_atoms=" << benchmark_owned << '\n'
              << "benchmark_reverse_buffer_bytes="
              << so3lr::so3lr_reverse_force_buffer_bytes(benchmark_owned)
              << '\n'
              << "host_boundary_bytes_per_evaluator_iteration=0\n"
              << "native_owned_energy_seeding=PASS\n"
              << "native_partitioned_lr_half_pairs=PASS\n"
              << "native_reverse_force_pack_unpack=PASS\n"
              << "synthetic_two_rank_energy_force_equivalence=PASS\n"
              << "replicated_full_sr_reference_graph=PASS\n"
              << "pytorch_full_so3lr_cartesian_force_reference=PASS\n"
              << "SO3LR_NATIVE_DEV27_OWNERSHIP_TEST=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_NATIVE_DEV27_OWNERSHIP_TEST=FAIL: "
              << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
