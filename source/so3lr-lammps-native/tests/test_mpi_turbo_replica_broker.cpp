#include "so3lr/kokkos_so3lr_evaluator.hpp"
#include "so3lr/mpi_turbo_replica_broker.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

struct State {
  std::vector<std::int64_t> z;
  std::vector<double> positions, velocities;
};

void mpi_check(int code, const char *name) {
  if (code != MPI_SUCCESS)
    throw std::runtime_error(std::string("dev_7 MPI failure in ") + name);
}

template <class View, class Values>
void fill(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("dev_7 device fill mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}

template <class View>
std::vector<typename View::non_const_value_type> copy(const View &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<typename View::non_const_value_type> values(device.extent(0));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i);
  return values;
}

State make_state(std::size_t waters, std::size_t image) {
  State state;
  state.z.resize(waters * 3);
  state.positions.resize(waters * 9, 0.0);
  state.velocities.resize(waters * 9, 0.0);
  const double spacing = 4.47 + 0.003 * static_cast<double>(image);
  for (std::size_t molecule = 0; molecule < waters; ++molecule) {
    const std::size_t o = molecule * 3;
    const double ox = spacing * static_cast<double>(molecule);
    const double oy = 0.11 * static_cast<double>((molecule + image) % 2);
    state.z[o] = 8;
    state.z[o + 1] = 1;
    state.z[o + 2] = 1;
    state.positions[o * 3] = ox;
    state.positions[o * 3 + 1] = oy;
    state.positions[(o + 1) * 3] = ox + 0.9572;
    state.positions[(o + 1) * 3 + 1] = oy;
    state.positions[(o + 2) * 3] = ox - 0.2399872;
    state.positions[(o + 2) * 3 + 1] = oy + 0.927297;
    const double vx = molecule % 2 == 0
                          ? -0.23 - 0.01 * static_cast<double>(image)
                          : 0.27 + 0.01 * static_cast<double>(image);
    for (std::size_t local = 0; local < 3; ++local) {
      const std::size_t atom = o + local;
      state.velocities[atom * 3] = vx;
      state.velocities[atom * 3 + 1] =
          0.017 * (static_cast<double>((atom + image) % 3) - 1.0);
      state.velocities[atom * 3 + 2] =
          -0.013 * (static_cast<double>((2 * atom + image) % 3) - 1.0);
    }
  }
  return state;
}

so3lr::TurboReplicaGraph build_graph(const State &state, double sr_cutoff,
                                     double lr_cutoff) {
  so3lr::TurboReplicaGraph graph;
  graph.atomic_numbers = state.z;
  for (std::size_t sender = 0; sender < state.z.size(); ++sender) {
    for (std::size_t receiver = 0; receiver < state.z.size(); ++receiver) {
      if (sender == receiver) continue;
      const double dx = state.positions[receiver * 3] -
                        state.positions[sender * 3];
      const double dy = state.positions[receiver * 3 + 1] -
                        state.positions[sender * 3 + 1];
      const double dz = state.positions[receiver * 3 + 2] -
                        state.positions[sender * 3 + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r < sr_cutoff) {
        graph.sr_senders.push_back(sender);
        graph.sr_receivers.push_back(receiver);
        graph.sr_edge_vectors.insert(graph.sr_edge_vectors.end(),
                                     {dx, dy, dz});
      }
      if (sender < receiver && r < lr_cutoff) {
        graph.lr_senders.push_back(sender);
        graph.lr_receivers.push_back(receiver);
        graph.lr_pair_vectors.insert(graph.lr_pair_vectors.end(),
                                     {dx, dy, dz});
      }
    }
  }
  return graph;
}

so3lr::TurboReplicaResult ordinary_evaluate(
    const so3lr::KokkosSo3lrEvaluator &evaluator,
    const so3lr::TurboReplicaGraph &graph) {
  Int64View z("dev7_reference_z", graph.atomic_numbers.size());
  DoubleView sr_vectors("dev7_reference_sr_vectors",
                        graph.sr_edge_vectors.size());
  IndexView sr_senders("dev7_reference_sr_senders", graph.sr_senders.size());
  IndexView sr_receivers("dev7_reference_sr_receivers",
                         graph.sr_receivers.size());
  DoubleView lr_vectors("dev7_reference_lr_vectors",
                        graph.lr_pair_vectors.size());
  IndexView lr_senders("dev7_reference_lr_senders", graph.lr_senders.size());
  IndexView lr_receivers("dev7_reference_lr_receivers",
                         graph.lr_receivers.size());
  fill(z, graph.atomic_numbers);
  fill(sr_vectors, graph.sr_edge_vectors);
  fill(sr_senders, graph.sr_senders);
  fill(sr_receivers, graph.sr_receivers);
  fill(lr_vectors, graph.lr_pair_vectors);
  fill(lr_senders, graph.lr_senders);
  fill(lr_receivers, graph.lr_receivers);
  so3lr::So3lrEvaluatorWorkspace workspace(
      graph.atomic_numbers.size(), graph.sr_senders.size(),
      graph.lr_senders.size());
  evaluator.launch_device(z, sr_vectors, sr_senders, sr_receivers,
                          lr_vectors, lr_senders, lr_receivers, workspace);
  Kokkos::fence();
  so3lr::TurboReplicaResult result;
  result.atomic_energies = copy(evaluator.atomic_energies(workspace));
  result.partial_charges = copy(evaluator.partial_charges(workspace));
  result.hirshfeld_ratios = copy(evaluator.hirshfeld_ratios(workspace));
  result.atomic_forces = copy(evaluator.atomic_forces(workspace));
  for (const double value : result.atomic_energies)
    result.total_energy += value;
  return result;
}

void advance(State &state, const std::vector<double> &forces) {
  constexpr double dt = 0.02;
  constexpr double force_scale = 1.0e-5;
  for (std::size_t atom = 0; atom < state.z.size(); ++atom) {
    const double mass = state.z[atom] == 8 ? 15.999 : 1.008;
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t index = atom * 3 + component;
      state.velocities[index] +=
          dt * force_scale * forces[index] / mass;
      state.positions[index] += dt * state.velocities[index];
    }
  }
}

double compare(const std::vector<double> &actual,
               const std::vector<double> &expected,
               const std::string &label, double tolerance,
               double &maximum) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > tolerance * (1.0 + std::abs(expected[i])))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i));
  }
  return maximum;
}

std::vector<double> flatten_member(
    const std::vector<so3lr::TurboReplicaResult> &replicas,
    const std::vector<double> so3lr::TurboReplicaResult::*member) {
  std::vector<double> result;
  for (const auto &replica : replicas) {
    const auto &values = replica.*member;
    result.insert(result.end(), values.begin(), values.end());
  }
  return result;
}

std::vector<double> gather_local(const std::vector<double> &local,
                                 const std::vector<int> &counts,
                                 const std::vector<int> &offsets, int rank) {
  std::vector<double> result;
  if (rank == 0) result.resize(offsets.back() + counts.back());
  mpi_check(MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                        rank == 0 ? result.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? offsets.data() : nullptr, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD),
            "MPI_Gatherv result audit");
  return result;
}

std::vector<int> offsets(const std::vector<int> &counts) {
  std::vector<int> result(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i)
    result[i] = result[i - 1] + counts[i - 1];
  return result;
}

std::size_t topology_signature(const so3lr::TurboReplicaGraph &graph) {
  return graph.sr_senders.size() * 1000003ULL + graph.lr_senders.size();
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int result_code = 0;
  bool kokkos_initialized = false;
  try {
    if (size != 4)
      throw std::runtime_error("dev_7 requires exactly four MPI ranks");
    if (argc != 2)
      throw std::runtime_error("usage: mpi_turbo_broker_test MODEL");
    if (rank == 0) {
      Kokkos::initialize(argc, argv);
      kokkos_initialized = true;
    }
    {
      std::unique_ptr<so3lr::NativeModel> model;
      std::unique_ptr<so3lr::KokkosSo3lrEvaluator> ordinary;
      if (rank == 0) {
        model = std::make_unique<so3lr::NativeModel>(
            so3lr::NativeModel::load(argv[1]));
        ordinary =
            std::make_unique<so3lr::KokkosSo3lrEvaluator>(*model);
      }
      so3lr::MpiTurboReplicaBroker broker(
          MPI_COMM_WORLD, 0, rank == 0 ? model.get() : nullptr);

      double sr_cutoff = 0.0, lr_cutoff = 0.0;
      if (rank == 0) {
        sr_cutoff = model->architecture_number(
            "short_range_cutoff_angstrom");
        lr_cutoff = model->architecture_number(
            "long_range_cutoff_angstrom");
      }
      mpi_check(MPI_Bcast(&sr_cutoff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                "MPI_Bcast SR cutoff");
      mpi_check(MPI_Bcast(&lr_cutoff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                "MPI_Bcast LR cutoff");

      State local_state = make_state(static_cast<std::size_t>(rank + 1),
                                     static_cast<std::size_t>(rank));
      std::vector<State> reference_states;
      if (rank == 0)
        for (int image = 0; image < size; ++image)
          reference_states.push_back(make_state(
              static_cast<std::size_t>(image + 1),
              static_cast<std::size_t>(image)));

      const std::vector<int> node_counts = {3, 6, 9, 12};
      const std::vector<int> node_offsets = offsets(node_counts);
      const std::vector<int> force_counts = {9, 18, 27, 36};
      const std::vector<int> force_offsets = offsets(force_counts);
      constexpr std::size_t steps = 40;
      double max_atomic_energy_error = 0.0, max_image_energy_error = 0.0;
      double max_charge_error = 0.0, max_hirshfeld_error = 0.0;
      double max_force_error = 0.0, max_scatter_error = 0.0;
      std::size_t local_topology_changes = 0;
      std::size_t previous_topology = 0;

      for (std::size_t step = 0; step < steps; ++step) {
        const auto local_graph = build_graph(local_state, sr_cutoff, lr_cutoff);
        const std::size_t topology = topology_signature(local_graph);
        if (step > 0 && topology != previous_topology)
          ++local_topology_changes;
        previous_topology = topology;
        const auto local_result = broker.evaluate(local_graph);

        const auto gathered_energies = gather_local(
            local_result.atomic_energies, node_counts, node_offsets, rank);
        const auto gathered_charges = gather_local(
            local_result.partial_charges, node_counts, node_offsets, rank);
        const auto gathered_hirshfeld = gather_local(
            local_result.hirshfeld_ratios, node_counts, node_offsets, rank);
        const auto gathered_forces = gather_local(
            local_result.atomic_forces, force_counts, force_offsets, rank);
        std::vector<double> gathered_total_energy(rank == 0 ? size : 0);
        mpi_check(MPI_Gather(&local_result.total_energy, 1, MPI_DOUBLE,
                             rank == 0 ? gathered_total_energy.data() : nullptr,
                             1, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  "MPI_Gather returned image energy");

        int step_ok = 1;
        if (rank == 0) {
          try {
            const auto &batch = broker.coordinator_batch();
            compare(gathered_energies,
                    flatten_member(batch.replicas,
                                   &so3lr::TurboReplicaResult::atomic_energies),
                    "energy scatter", 0.0, max_scatter_error);
            compare(gathered_charges,
                    flatten_member(batch.replicas,
                                   &so3lr::TurboReplicaResult::partial_charges),
                    "charge scatter", 0.0, max_scatter_error);
            compare(gathered_hirshfeld,
                    flatten_member(batch.replicas,
                                   &so3lr::TurboReplicaResult::hirshfeld_ratios),
                    "Hirshfeld scatter", 0.0, max_scatter_error);
            compare(gathered_forces,
                    flatten_member(batch.replicas,
                                   &so3lr::TurboReplicaResult::atomic_forces),
                    "force scatter", 0.0, max_scatter_error);
            for (int image = 0; image < size; ++image)
              if (gathered_total_energy[image] !=
                  batch.replicas[image].total_energy)
                throw std::runtime_error("total-energy scatter mismatch");

            for (int image = 0; image < size; ++image) {
              const auto graph = build_graph(reference_states[image],
                                             sr_cutoff, lr_cutoff);
              const auto reference = ordinary_evaluate(*ordinary, graph);
              const auto &packed = batch.replicas[image];
              compare(packed.atomic_energies, reference.atomic_energies,
                      "MPI dynamic atomic energy", 2.0e-8,
                      max_atomic_energy_error);
              compare(packed.partial_charges, reference.partial_charges,
                      "MPI dynamic charge", 2.0e-9, max_charge_error);
              compare(packed.hirshfeld_ratios, reference.hirshfeld_ratios,
                      "MPI dynamic Hirshfeld", 2.0e-9,
                      max_hirshfeld_error);
              compare(packed.atomic_forces, reference.atomic_forces,
                      "MPI dynamic force", 2.0e-8, max_force_error);
              max_image_energy_error = std::max(
                  max_image_energy_error,
                  std::abs(packed.total_energy - reference.total_energy));
              if (std::abs(packed.total_energy - reference.total_energy) >
                  2.0e-8 * (1.0 + std::abs(reference.total_energy)))
                throw std::runtime_error(
                    "MPI dynamic total-energy mismatch for image " +
                    std::to_string(image));
              advance(reference_states[image], reference.atomic_forces);
            }
          } catch (const std::exception &error) {
            std::cerr << "dev_7 step=" << step
                      << " comparison failure=" << error.what() << '\n';
            step_ok = 0;
          }
        }
        mpi_check(MPI_Bcast(&step_ok, 1, MPI_INT, 0, MPI_COMM_WORLD),
                  "MPI_Bcast step status");
        if (!step_ok)
          throw std::runtime_error("dev_7 distributed comparison failed");
        advance(local_state, local_result.atomic_forces);
      }

      int topology_changes = 0;
      const int local_changes = static_cast<int>(local_topology_changes);
      mpi_check(MPI_Reduce(&local_changes, &topology_changes, 1, MPI_INT,
                           MPI_SUM, 0, MPI_COMM_WORLD),
                "MPI_Reduce topology changes");
      const auto all_positions = gather_local(
          local_state.positions, force_counts, force_offsets, rank);
      const auto all_velocities = gather_local(
          local_state.velocities, force_counts, force_offsets, rank);
      if (rank == 0) {
        std::vector<double> reference_positions, reference_velocities;
        for (const auto &state : reference_states) {
          reference_positions.insert(reference_positions.end(),
                                     state.positions.begin(),
                                     state.positions.end());
          reference_velocities.insert(reference_velocities.end(),
                                      state.velocities.begin(),
                                      state.velocities.end());
        }
        double position_error = 0.0, velocity_error = 0.0;
        compare(all_positions, reference_positions, "MPI trajectory position",
                2.0e-10, position_error);
        compare(all_velocities, reference_velocities, "MPI trajectory velocity",
                2.0e-10, velocity_error);
        if (topology_changes == 0)
          throw std::runtime_error("dev_7 did not cross a cutoff");
        std::cout << std::setprecision(12)
                  << "mpi_replica_ranks=" << size << '\n'
                  << "cuda_coordinator_ranks=1\n"
                  << "cuda_worker_ranks=0\n"
                  << "shared_gpu_batch_evaluations=" << steps << '\n'
                  << "ordinary_reference_evaluations=" << steps * size
                  << '\n'
                  << "distributed_images_per_batch=" << size << '\n'
                  << "replica_atom_counts=3,6,9,12\n"
                  << "topology_rebuilds_per_rank=" << steps << '\n'
                  << "observed_distributed_topology_changes="
                  << topology_changes << '\n'
                  << "mpi_scatter_max_abs_error=" << max_scatter_error
                  << '\n'
                  << "mpi_atomic_energy_max_abs_error="
                  << max_atomic_energy_error << '\n'
                  << "mpi_image_energy_max_abs_error="
                  << max_image_energy_error << '\n'
                  << "mpi_charge_max_abs_error=" << max_charge_error << '\n'
                  << "mpi_hirshfeld_max_abs_error=" << max_hirshfeld_error
                  << '\n'
                  << "mpi_force_max_abs_error=" << max_force_error << '\n'
                  << "mpi_trajectory_position_max_abs_error="
                  << position_error << '\n'
                  << "mpi_trajectory_velocity_max_abs_error="
                  << velocity_error << '\n'
                  << "broker_topology=mpi_gather,one_cuda_evaluation,mpi_scatter\n"
                  << "SO3LR_STAGE4_DEV7=PASS\n";
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank << " SO3LR_STAGE4_DEV7=FAIL reason="
              << error.what() << '\n';
    result_code = 1;
  }
  if (kokkos_initialized) Kokkos::finalize();
  int global_result = 0;
  MPI_Allreduce(&result_code, &global_result, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Finalize();
  return global_result;
}
