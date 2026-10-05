#include "so3lr/kokkos_so3lr_evaluator.hpp"
#include "so3lr/kokkos_turbo_replica_bridge.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
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

template <class View, class Values>
void fill(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("dev_6 fill mismatch");
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
    const double molecular_vx =
        molecule % 2 == 0 ? -0.23 - 0.01 * image : 0.27 + 0.01 * image;
    for (std::size_t local = 0; local < 3; ++local) {
      const std::size_t atom = o + local;
      state.velocities[atom * 3] = molecular_vx;
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
  const std::size_t nodes = state.z.size();
  for (std::size_t sender = 0; sender < nodes; ++sender) {
    for (std::size_t receiver = 0; receiver < nodes; ++receiver) {
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

so3lr::TurboReplicaResult evaluate_ordinary(
    const so3lr::KokkosSo3lrEvaluator &evaluator,
    const so3lr::TurboReplicaGraph &graph) {
  const std::size_t nodes = graph.atomic_numbers.size();
  Int64View z("dev6_reference_z", nodes);
  DoubleView sr_vectors("dev6_reference_sr_vectors",
                        graph.sr_edge_vectors.size());
  IndexView sr_senders("dev6_reference_sr_senders", graph.sr_senders.size());
  IndexView sr_receivers("dev6_reference_sr_receivers",
                         graph.sr_receivers.size());
  DoubleView lr_vectors("dev6_reference_lr_vectors",
                        graph.lr_pair_vectors.size());
  IndexView lr_senders("dev6_reference_lr_senders", graph.lr_senders.size());
  IndexView lr_receivers("dev6_reference_lr_receivers",
                         graph.lr_receivers.size());
  fill(z, graph.atomic_numbers);
  fill(sr_vectors, graph.sr_edge_vectors);
  fill(sr_senders, graph.sr_senders);
  fill(sr_receivers, graph.sr_receivers);
  fill(lr_vectors, graph.lr_pair_vectors);
  fill(lr_senders, graph.lr_senders);
  fill(lr_receivers, graph.lr_receivers);
  so3lr::So3lrEvaluatorWorkspace workspace(
      nodes, graph.sr_senders.size(), graph.lr_senders.size());
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

void advance(State &state, const std::vector<double> &forces,
             double dt, double force_scale) {
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

std::size_t topology_signature(
    const std::vector<so3lr::TurboReplicaGraph> &graphs) {
  std::size_t value = 1469598103934665603ULL;
  for (const auto &graph : graphs) {
    value ^= graph.sr_senders.size() + 0x9e3779b97f4a7c15ULL;
    value *= 1099511628211ULL;
    value ^= graph.lr_senders.size() + 0x517cc1b727220a95ULL;
    value *= 1099511628211ULL;
  }
  return value;
}

double state_difference(const std::vector<State> &a,
                        const std::vector<State> &b,
                        bool velocities) {
  double maximum = 0.0;
  for (std::size_t image = 0; image < a.size(); ++image) {
    const auto &left = velocities ? a[image].velocities : a[image].positions;
    const auto &right = velocities ? b[image].velocities : b[image].positions;
    for (std::size_t i = 0; i < left.size(); ++i)
      maximum = std::max(maximum, std::abs(left[i] - right[i]));
  }
  return maximum;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("dev_6 test was not compiled for CUDA");
#endif
    if (argc != 2)
      throw std::runtime_error("usage: turbo_replica_bridge_test MODEL");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const so3lr::KokkosTurboReplicaBridge bridge(model);
    const so3lr::KokkosSo3lrEvaluator ordinary(model);
    if (!bridge.contract_verified() ||
        !ordinary.self_contained_model_contract())
      throw std::runtime_error("dev_6 evaluator contract failed");
    const double sr_cutoff =
        model.architecture_number("short_range_cutoff_angstrom");
    const double lr_cutoff =
        model.architecture_number("long_range_cutoff_angstrom");

    std::vector<State> packed_states, reference_states;
    for (const std::size_t waters : {std::size_t(1), std::size_t(2),
                                     std::size_t(3), std::size_t(4)})
      packed_states.push_back(make_state(waters, packed_states.size()));
    reference_states = packed_states;

    constexpr std::size_t steps = 40;
    constexpr double dt = 0.02;
    constexpr double force_scale = 1.0e-5;
    double max_energy_error = 0.0, max_force_error = 0.0;
    double max_charge_error = 0.0, max_hirshfeld_error = 0.0;
    double max_image_energy_error = 0.0;
    std::size_t topology_changes = 0;
    std::size_t previous_topology = 0;
    std::size_t total_packed_evaluations = 0;

    for (std::size_t step = 0; step < steps; ++step) {
      std::vector<so3lr::TurboReplicaGraph> packed_graphs;
      std::vector<so3lr::TurboReplicaGraph> reference_graphs;
      for (std::size_t image = 0; image < packed_states.size(); ++image) {
        packed_graphs.push_back(
            build_graph(packed_states[image], sr_cutoff, lr_cutoff));
        reference_graphs.push_back(
            build_graph(reference_states[image], sr_cutoff, lr_cutoff));
      }
      const std::size_t signature = topology_signature(packed_graphs);
      if (step > 0 && signature != previous_topology) ++topology_changes;
      previous_topology = signature;
      const auto packed_result = bridge.evaluate(packed_graphs);
      ++total_packed_evaluations;
      if (packed_result.replicas.size() != packed_states.size())
        throw std::runtime_error("dev_6 scatter image count mismatch");
      for (std::size_t image = 0; image < packed_states.size(); ++image) {
        const auto reference = evaluate_ordinary(ordinary,
                                                  reference_graphs[image]);
        const auto &packed = packed_result.replicas[image];
        compare(packed.atomic_energies, reference.atomic_energies,
                "dynamic atomic energy", 2.0e-8, max_energy_error);
        compare(packed.partial_charges, reference.partial_charges,
                "dynamic charge", 2.0e-9, max_charge_error);
        compare(packed.hirshfeld_ratios, reference.hirshfeld_ratios,
                "dynamic Hirshfeld", 2.0e-9, max_hirshfeld_error);
        compare(packed.atomic_forces, reference.atomic_forces,
                "dynamic force", 2.0e-8, max_force_error);
        max_image_energy_error = std::max(
            max_image_energy_error,
            std::abs(packed.total_energy - reference.total_energy));
        if (max_image_energy_error >
            2.0e-8 * (1.0 + std::abs(reference.total_energy)))
          throw std::runtime_error("dynamic image energy mismatch");
        advance(packed_states[image], packed.atomic_forces, dt, force_scale);
        advance(reference_states[image], reference.atomic_forces, dt,
                force_scale);
      }
    }

    const double position_error =
        state_difference(packed_states, reference_states, false);
    const double velocity_error =
        state_difference(packed_states, reference_states, true);
    if (position_error > 2.0e-10 || velocity_error > 2.0e-10)
      throw std::runtime_error("dev_6 trajectory parity failed");
    if (topology_changes == 0)
      throw std::runtime_error("dev_6 did not exercise a topology change");

    std::cout << std::setprecision(12)
              << "execution_space=" << Kokkos::DefaultExecutionSpace::name()
              << '\n'
              << "replica_images=" << packed_states.size() << '\n'
              << "replica_atom_counts=3,6,9,12\n"
              << "independent_velocity_fields=4\n"
              << "dynamic_steps=" << steps << '\n'
              << "packed_device_evaluations=" << total_packed_evaluations
              << '\n'
              << "ordinary_reference_evaluations="
              << steps * packed_states.size() << '\n'
              << "topology_rebuilds=" << steps << '\n'
              << "observed_topology_changes=" << topology_changes << '\n'
              << "dynamic_atomic_energy_max_abs_error=" << max_energy_error
              << '\n'
              << "dynamic_image_energy_max_abs_error="
              << max_image_energy_error << '\n'
              << "dynamic_charge_max_abs_error=" << max_charge_error << '\n'
              << "dynamic_hirshfeld_max_abs_error=" << max_hirshfeld_error
              << '\n'
              << "dynamic_force_max_abs_error=" << max_force_error << '\n'
              << "trajectory_position_max_abs_error=" << position_error
              << '\n'
              << "trajectory_velocity_max_abs_error=" << velocity_error
              << '\n'
              << "bridge_contract=gather,packed_full_physics,scatter\n"
              << "SO3LR_STAGE4_DEV6=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_STAGE4_DEV6=FAIL reason=" << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
