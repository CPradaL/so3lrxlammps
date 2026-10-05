#include "so3lr/kokkos_packed_so3lr_evaluator.hpp"
#include "so3lr/kokkos_so3lr_evaluator.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

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

struct Graph {
  std::vector<std::int64_t> z;
  std::vector<double> sr_vectors, lr_vectors;
  std::vector<std::size_t> sr_senders, sr_receivers;
  std::vector<std::size_t> lr_senders, lr_receivers;
};

struct Result {
  std::vector<double> energies, charges, hirshfeld, forces, image_energies;
  std::vector<double> lr_energies, lr_charge_gradients;
  std::vector<double> lr_hirshfeld_gradients, lr_radial_gradients;
  double milliseconds = 0.0;
  std::size_t nodes = 0, sr_edges = 0, lr_pairs = 0;
};

template <class View, class Values>
void fill(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("dev_5 fill size mismatch");
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

// Graph cutoffs: slightly inside the model's own cutoffs, as LAMMPS lists are.
double g_sr_cutoff = 4.45;
double g_lr_cutoff = 11.95;

Graph make_water_cluster(std::size_t waters, std::size_t image) {
  Graph graph;
  const std::size_t atoms = waters * 3;
  graph.z.resize(atoms);
  std::vector<double> x(atoms * 3, 0.0);
  const double deformation = 1.0 + 0.002 * static_cast<double>(image % 11);
  for (std::size_t molecule = 0; molecule < waters; ++molecule) {
    const std::size_t o = molecule * 3;
    const double ox = 2.72 * static_cast<double>(molecule);
    const double oy = 0.19 * static_cast<double>((molecule + image) % 3);
    const double oz = 0.13 * static_cast<double>((2 * molecule + image) % 2);
    graph.z[o] = 8;
    graph.z[o + 1] = 1;
    graph.z[o + 2] = 1;
    x[o * 3] = ox;
    x[o * 3 + 1] = oy;
    x[o * 3 + 2] = oz;
    x[(o + 1) * 3] = ox + 0.9572 * deformation;
    x[(o + 1) * 3 + 1] = oy;
    x[(o + 1) * 3 + 2] = oz;
    x[(o + 2) * 3] = ox - 0.2399872 * deformation;
    x[(o + 2) * 3 + 1] = oy + 0.927297 * deformation;
    x[(o + 2) * 3 + 2] = oz + 0.01 * static_cast<double>(image % 5);
  }
  for (std::size_t i = 0; i < atoms; ++i) {
    for (std::size_t j = 0; j < atoms; ++j) {
      if (i == j) continue;
      const double dx = x[j * 3] - x[i * 3];
      const double dy = x[j * 3 + 1] - x[i * 3 + 1];
      const double dz = x[j * 3 + 2] - x[i * 3 + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r < g_sr_cutoff) {
        graph.sr_senders.push_back(i);
        graph.sr_receivers.push_back(j);
        graph.sr_vectors.insert(graph.sr_vectors.end(), {dx, dy, dz});
      }
      if (i < j && r < g_lr_cutoff) {
        graph.lr_senders.push_back(i);
        graph.lr_receivers.push_back(j);
        graph.lr_vectors.insert(graph.lr_vectors.end(), {dx, dy, dz});
      }
    }
  }
  return graph;
}

Graph pack(const std::vector<Graph> &graphs,
           std::vector<std::int64_t> &image_ids) {
  Graph packed;
  std::size_t offset = 0;
  for (std::size_t image = 0; image < graphs.size(); ++image) {
    const auto &g = graphs[image];
    packed.z.insert(packed.z.end(), g.z.begin(), g.z.end());
    packed.sr_vectors.insert(packed.sr_vectors.end(), g.sr_vectors.begin(),
                             g.sr_vectors.end());
    packed.lr_vectors.insert(packed.lr_vectors.end(), g.lr_vectors.begin(),
                             g.lr_vectors.end());
    for (const auto value : g.sr_senders)
      packed.sr_senders.push_back(offset + value);
    for (const auto value : g.sr_receivers)
      packed.sr_receivers.push_back(offset + value);
    for (const auto value : g.lr_senders)
      packed.lr_senders.push_back(offset + value);
    for (const auto value : g.lr_receivers)
      packed.lr_receivers.push_back(offset + value);
    image_ids.insert(image_ids.end(), g.z.size(),
                     static_cast<std::int64_t>(image));
    offset += g.z.size();
  }
  return packed;
}

Result run_packed(const so3lr::KokkosPackedSo3lrEvaluator &evaluator,
                  const std::vector<Graph> &graphs,
                  std::size_t repetitions) {
  std::vector<std::int64_t> image_ids_host;
  const Graph graph = pack(graphs, image_ids_host);
  Int64View z("dev5_packed_z", graph.z.size());
  DoubleView sr_vectors("dev5_packed_sr_vectors", graph.sr_vectors.size());
  IndexView sr_senders("dev5_packed_sr_senders", graph.sr_senders.size());
  IndexView sr_receivers("dev5_packed_sr_receivers", graph.sr_receivers.size());
  DoubleView lr_vectors("dev5_packed_lr_vectors", graph.lr_vectors.size());
  IndexView lr_senders("dev5_packed_lr_senders", graph.lr_senders.size());
  IndexView lr_receivers("dev5_packed_lr_receivers", graph.lr_receivers.size());
  Int64View image_ids("dev5_packed_image_ids", graph.z.size());
  DoubleView total_charges("dev5_packed_total_charges", graphs.size());
  fill(z, graph.z);
  fill(sr_vectors, graph.sr_vectors);
  fill(sr_senders, graph.sr_senders);
  fill(sr_receivers, graph.sr_receivers);
  fill(lr_vectors, graph.lr_vectors);
  fill(lr_senders, graph.lr_senders);
  fill(lr_receivers, graph.lr_receivers);
  fill(image_ids, image_ids_host);
  fill(total_charges, std::vector<double>(graphs.size(), 0.0));
  for (std::size_t edge = 0; edge < graph.sr_senders.size(); ++edge)
    if (image_ids_host[graph.sr_senders[edge]] !=
        image_ids_host[graph.sr_receivers[edge]])
      throw std::runtime_error("dev_5 cross-image SR edge");
  for (std::size_t pair = 0; pair < graph.lr_senders.size(); ++pair)
    if (image_ids_host[graph.lr_senders[pair]] !=
        image_ids_host[graph.lr_receivers[pair]])
      throw std::runtime_error("dev_5 cross-image LR pair");

  so3lr::PackedSo3lrWorkspace workspace(
      graph.z.size(), graph.sr_senders.size(), graph.lr_senders.size(),
      graphs.size(), evaluator.arch());
  evaluator.launch_device(z, sr_vectors, sr_senders, sr_receivers,
                          lr_vectors, lr_senders, lr_receivers, image_ids,
                          total_charges, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    evaluator.launch_device(z, sr_vectors, sr_senders, sr_receivers,
                            lr_vectors, lr_senders, lr_receivers, image_ids,
                            total_charges, workspace);
  Kokkos::fence();
  Result result;
  result.milliseconds = timer.seconds() * 1000.0 /
                        static_cast<double>(repetitions);
  result.nodes = graph.z.size();
  result.sr_edges = graph.sr_senders.size();
  result.lr_pairs = graph.lr_senders.size();
  result.energies = copy(evaluator.atomic_energies(workspace));
  result.charges = copy(evaluator.partial_charges(workspace));
  result.hirshfeld = copy(evaluator.hirshfeld_ratios(workspace));
  result.forces = copy(evaluator.atomic_forces(workspace));
  result.image_energies = copy(evaluator.image_energies(workspace));
  result.lr_energies = copy(workspace.long_range.atomic_energy);
  result.lr_charge_gradients = copy(workspace.long_range.charge_gradient);
  result.lr_hirshfeld_gradients =
      copy(workspace.long_range.hirshfeld_gradient);
  result.lr_radial_gradients =
      copy(workspace.long_range.pair_radial_gradient);
  return result;
}

Result run_independent(const so3lr::KokkosSo3lrEvaluator &evaluator,
                       const std::vector<Graph> &graphs) {
  Result result;
  for (const auto &graph : graphs) {
    Int64View z("dev5_reference_z", graph.z.size());
    DoubleView sr_vectors("dev5_reference_sr_vectors", graph.sr_vectors.size());
    IndexView sr_senders("dev5_reference_sr_senders", graph.sr_senders.size());
    IndexView sr_receivers("dev5_reference_sr_receivers", graph.sr_receivers.size());
    DoubleView lr_vectors("dev5_reference_lr_vectors", graph.lr_vectors.size());
    IndexView lr_senders("dev5_reference_lr_senders", graph.lr_senders.size());
    IndexView lr_receivers("dev5_reference_lr_receivers", graph.lr_receivers.size());
    fill(z, graph.z);
    fill(sr_vectors, graph.sr_vectors);
    fill(sr_senders, graph.sr_senders);
    fill(sr_receivers, graph.sr_receivers);
    fill(lr_vectors, graph.lr_vectors);
    fill(lr_senders, graph.lr_senders);
    fill(lr_receivers, graph.lr_receivers);
    so3lr::So3lrEvaluatorWorkspace workspace(
        graph.z.size(), graph.sr_senders.size(), graph.lr_senders.size());
    evaluator.launch_device(z, sr_vectors, sr_senders, sr_receivers,
                            lr_vectors, lr_senders, lr_receivers, workspace);
    Kokkos::fence();
    const auto append = [](std::vector<double> &to,
                           const std::vector<double> &from) {
      to.insert(to.end(), from.begin(), from.end());
    };
    const auto energy = copy(evaluator.atomic_energies(workspace));
    append(result.energies, energy);
    append(result.charges, copy(evaluator.partial_charges(workspace)));
    append(result.hirshfeld, copy(evaluator.hirshfeld_ratios(workspace)));
    append(result.forces, copy(evaluator.atomic_forces(workspace)));
    append(result.lr_energies, copy(workspace.long_range.atomic_energy));
    append(result.lr_charge_gradients,
           copy(workspace.long_range.charge_gradient));
    append(result.lr_hirshfeld_gradients,
           copy(workspace.long_range.hirshfeld_gradient));
    append(result.lr_radial_gradients,
           copy(workspace.long_range.pair_radial_gradient));
    double image_energy = 0.0;
    for (const double value : energy) image_energy += value;
    result.image_energies.push_back(image_energy);
    result.nodes += graph.z.size();
    result.sr_edges += graph.sr_senders.size();
    result.lr_pairs += graph.lr_senders.size();
  }
  return result;
}

// Reference for models the single-rank v1 evaluator does not support (SO3LR
// beyond v1): the packed evaluator itself, one image per launch. This checks that
// packing and per-image segmentation do not change any image's result.
Result run_one_image_at_a_time(
    const so3lr::KokkosPackedSo3lrEvaluator &evaluator,
    const std::vector<Graph> &graphs);

double compare(const std::vector<double> &actual,
               const std::vector<double> &expected,
               const std::string &label, double tolerance = 2.0e-9) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > tolerance * (1.0 + std::abs(expected[i])))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i));
  }
  return maximum;
}

double image_charge_error(const Result &result,
                          const std::vector<Graph> &graphs) {
  std::size_t offset = 0;
  double maximum = 0.0;
  for (const auto &graph : graphs) {
    double sum = 0.0;
    for (std::size_t node = 0; node < graph.z.size(); ++node)
      sum += result.charges[offset + node];
    maximum = std::max(maximum, std::abs(sum));
    offset += graph.z.size();
  }
  return maximum;
}

}  // namespace

namespace {
Result run_one_image_at_a_time(
    const so3lr::KokkosPackedSo3lrEvaluator &evaluator,
    const std::vector<Graph> &graphs) {
  Result result;
  const auto append = [](std::vector<double> &to,
                         const std::vector<double> &from) {
    to.insert(to.end(), from.begin(), from.end());
  };
  for (const auto &graph : graphs) {
    const Result one = run_packed(evaluator, {graph}, 1);
    append(result.energies, one.energies);
    append(result.charges, one.charges);
    append(result.hirshfeld, one.hirshfeld);
    append(result.forces, one.forces);
    append(result.image_energies, one.image_energies);
    append(result.lr_energies, one.lr_energies);
    append(result.lr_charge_gradients, one.lr_charge_gradients);
    append(result.lr_hirshfeld_gradients, one.lr_hirshfeld_gradients);
    append(result.lr_radial_gradients, one.lr_radial_gradients);
    result.nodes += one.nodes;
    result.sr_edges += one.sr_edges;
    result.lr_pairs += one.lr_pairs;
  }
  return result;
}
}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("dev_5 test was not compiled for CUDA");
#endif
    if (argc != 2)
      throw std::runtime_error("usage: packed_physical_test MODEL");
    const auto model = so3lr::NativeModel::load(argv[1]);
    const so3lr::KokkosPackedSo3lrEvaluator packed_evaluator(model);
    if (!packed_evaluator.contract_verified())
      throw std::runtime_error("dev_5 model contract failed");
    g_sr_cutoff = model.architecture_number("short_range_cutoff_angstrom") - 0.05;
    g_lr_cutoff = model.architecture_number("long_range_cutoff_angstrom") - 0.05;

    std::vector<Graph> mixed;
    for (const std::size_t waters : {std::size_t(1), std::size_t(2),
                                     std::size_t(3), std::size_t(4)})
      mixed.push_back(make_water_cluster(waters, mixed.size()));
    Result reference;
    std::string reference_kind = "single_rank_v1_evaluator";
    try {
      const so3lr::KokkosSo3lrEvaluator ordinary_evaluator(model);
      if (!ordinary_evaluator.self_contained_model_contract())
        throw std::runtime_error("dev_5 ordinary model contract failed");
      reference = run_independent(ordinary_evaluator, mixed);
    } catch (const std::exception &unsupported) {
      reference_kind = "packed_one_image_at_a_time";
      std::cout << "reference_note=" << unsupported.what() << '\n';
      reference = run_one_image_at_a_time(packed_evaluator, mixed);
    }
    std::cout << "reference_kind=" << reference_kind << '\n';
    const Result packed = run_packed(packed_evaluator, mixed, 2);
    const double energy_error = compare(packed.energies, reference.energies,
                                        "complete atomic energy");
    const double charge_error = compare(packed.charges, reference.charges,
                                        "partial charge");
    const double hirshfeld_error = compare(packed.hirshfeld,
                                           reference.hirshfeld,
                                           "Hirshfeld ratio");
    const double lr_energy_error = compare(packed.lr_energies,
                                           reference.lr_energies,
                                           "physical LR atomic energy");
    const double q_gradient_error = compare(
        packed.lr_charge_gradients, reference.lr_charge_gradients,
        "physical dE/dq");
    const double h_gradient_error = compare(
        packed.lr_hirshfeld_gradients, reference.lr_hirshfeld_gradients,
        "physical dE/dh");
    const double radial_error = compare(packed.lr_radial_gradients,
                                        reference.lr_radial_gradients,
                                        "physical LR dE/dr");
    const double force_error = compare(packed.forces, reference.forces,
                                       "complete Cartesian force", 2.0e-8);
    const double image_energy_error = compare(
        packed.image_energies, reference.image_energies,
        "complete image energy", 2.0e-8);
    const double conservation_error = image_charge_error(packed, mixed);
    if (conservation_error > 2.0e-11)
      throw std::runtime_error("dev_5 per-image charge conservation failed");

    std::size_t lr_only_pairs = 0;
    for (const auto &graph : mixed)
      lr_only_pairs += graph.lr_senders.size() - graph.sr_senders.size() / 2;
    std::cout << std::setprecision(12)
              << "execution_space=" << Kokkos::DefaultExecutionSpace::name()
              << '\n'
              << "physical_correctness_images=" << mixed.size() << '\n'
              << "physical_correctness_nodes=" << packed.nodes << '\n'
              << "physical_correctness_sr_edges=" << packed.sr_edges << '\n'
              << "physical_correctness_lr_half_pairs=" << packed.lr_pairs
              << '\n'
              << "physical_correctness_lr_only_pairs=" << lr_only_pairs
              << '\n'
              << "complete_atomic_energy_max_abs_error=" << energy_error
              << '\n'
              << "partial_charge_max_abs_error=" << charge_error << '\n'
              << "hirshfeld_max_abs_error=" << hirshfeld_error << '\n'
              << "physical_lr_atomic_energy_max_abs_error=" << lr_energy_error
              << '\n'
              << "physical_charge_gradient_max_abs_error=" << q_gradient_error
              << '\n'
              << "physical_hirshfeld_gradient_max_abs_error="
              << h_gradient_error << '\n'
              << "physical_lr_radial_gradient_max_abs_error=" << radial_error
              << '\n'
              << "complete_cartesian_force_max_abs_error=" << force_error
              << '\n'
              << "complete_image_energy_max_abs_error=" << image_energy_error
              << '\n'
              << "image_charge_max_abs_error=" << conservation_error << '\n'
              << "reference_mode=ordinary_independent_full_so3lr\n"
              << "physical_terms=gnn,zbl,electrostatics,dispersion\n"
              << "benchmark_columns=images,nodes,sr_edges,lr_half_pairs,complete_ms,throughput_vs_m1\n";

    double m1 = 0.0;
    for (const std::size_t images : {std::size_t(1), std::size_t(8),
                                     std::size_t(32), std::size_t(64)}) {
      std::vector<Graph> graphs;
      for (std::size_t image = 0; image < images; ++image)
        graphs.push_back(make_water_cluster(4, image));
      const Result outcome = run_packed(packed_evaluator, graphs, 5);
      if (images == 1) m1 = outcome.milliseconds;
      const double throughput =
          m1 * static_cast<double>(images) / outcome.milliseconds;
      std::cout << "physical_benchmark=" << images << ',' << outcome.nodes
                << ',' << outcome.sr_edges << ',' << outcome.lr_pairs << ','
                << outcome.milliseconds << ',' << throughput << '\n';
    }
    std::cout << "SO3LR_STAGE4_DEV5=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_STAGE4_DEV5=FAIL reason=" << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
