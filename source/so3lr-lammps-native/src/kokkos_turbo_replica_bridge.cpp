#include "so3lr/kokkos_turbo_replica_bridge.hpp"

#include <algorithm>

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

template <class View, class Values>
void fill_device(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("SO3LR turbo bridge upload size mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy_device(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> values(device.extent(0));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i);
  return values;
}

void validate_graph(const TurboReplicaGraph &graph, std::size_t image) {
  const std::size_t nodes = graph.atomic_numbers.size();
  if (nodes == 0 || graph.sr_senders.empty() || graph.lr_senders.empty() ||
      graph.sr_senders.size() != graph.sr_receivers.size() ||
      graph.sr_edge_vectors.size() != graph.sr_senders.size() * 3 ||
      graph.lr_senders.size() != graph.lr_receivers.size() ||
      graph.lr_pair_vectors.size() != graph.lr_senders.size() * 3)
    throw std::runtime_error("SO3LR invalid turbo graph for image " +
                             std::to_string(image));
  for (std::size_t edge = 0; edge < graph.sr_senders.size(); ++edge)
    if (graph.sr_senders[edge] >= nodes ||
        graph.sr_receivers[edge] >= nodes ||
        graph.sr_senders[edge] == graph.sr_receivers[edge])
      throw std::runtime_error("SO3LR invalid turbo SR index");
  for (std::size_t pair = 0; pair < graph.lr_senders.size(); ++pair)
    if (graph.lr_senders[pair] >= nodes ||
        graph.lr_receivers[pair] >= nodes ||
        graph.lr_senders[pair] == graph.lr_receivers[pair])
      throw std::runtime_error("SO3LR invalid turbo LR index");
}

}  // namespace

KokkosTurboReplicaBridge::KokkosTurboReplicaBridge(const NativeModel &model)
    : evaluator_(model), charge_spin_(load_charge_spin_embedding(model)) {
  if (!evaluator_.contract_verified())
    throw std::runtime_error("SO3LR turbo replica bridge contract failed");
}

TurboReplicaBatchResult KokkosTurboReplicaBridge::evaluate(
    const std::vector<TurboReplicaGraph> &replicas) const {
  if (replicas.empty())
    throw std::runtime_error("SO3LR turbo batch is empty");

  std::vector<std::int64_t> z, image_ids;
  std::vector<double> sr_vectors, lr_vectors, total_charges;
  std::vector<std::size_t> sr_senders, sr_receivers;
  std::vector<std::size_t> lr_senders, lr_receivers, node_offsets{0};
  for (std::size_t image = 0; image < replicas.size(); ++image) {
    const auto &graph = replicas[image];
    validate_graph(graph, image);
    const std::size_t offset = z.size();
    z.insert(z.end(), graph.atomic_numbers.begin(), graph.atomic_numbers.end());
    image_ids.insert(image_ids.end(), graph.atomic_numbers.size(),
                     static_cast<std::int64_t>(image));
    sr_vectors.insert(sr_vectors.end(), graph.sr_edge_vectors.begin(),
                      graph.sr_edge_vectors.end());
    lr_vectors.insert(lr_vectors.end(), graph.lr_pair_vectors.begin(),
                      graph.lr_pair_vectors.end());
    for (const auto index : graph.sr_senders)
      sr_senders.push_back(offset + index);
    for (const auto index : graph.sr_receivers)
      sr_receivers.push_back(offset + index);
    for (const auto index : graph.lr_senders)
      lr_senders.push_back(offset + index);
    for (const auto index : graph.lr_receivers)
      lr_receivers.push_back(offset + index);
    total_charges.push_back(graph.total_charge);
    node_offsets.push_back(z.size());
  }

  Int64View device_z("so3lr_turbo_z", z.size());
  Int64View device_image_ids("so3lr_turbo_image_ids", image_ids.size());
  DoubleView device_sr_vectors("so3lr_turbo_sr_vectors", sr_vectors.size());
  IndexView device_sr_senders("so3lr_turbo_sr_senders", sr_senders.size());
  IndexView device_sr_receivers("so3lr_turbo_sr_receivers", sr_receivers.size());
  DoubleView device_lr_vectors("so3lr_turbo_lr_vectors", lr_vectors.size());
  IndexView device_lr_senders("so3lr_turbo_lr_senders", lr_senders.size());
  IndexView device_lr_receivers("so3lr_turbo_lr_receivers", lr_receivers.size());
  DoubleView device_total_charges("so3lr_turbo_total_charges",
                                  total_charges.size());
  fill_device(device_z, z);
  fill_device(device_image_ids, image_ids);
  fill_device(device_sr_vectors, sr_vectors);
  fill_device(device_sr_senders, sr_senders);
  fill_device(device_sr_receivers, sr_receivers);
  fill_device(device_lr_vectors, lr_vectors);
  fill_device(device_lr_senders, lr_senders);
  fill_device(device_lr_receivers, lr_receivers);
  fill_device(device_total_charges, total_charges);

  PackedSo3lrWorkspace workspace(z.size(), sr_senders.size(),
                                 lr_senders.size(), replicas.size(),
                                 evaluator_.arch());
  // Charge/spin embedding, one table per charged or open-shell image. The
  // tables depend only on composition and (Q, psi), so this is host work of
  // a few F x F products per element present.
  bool any_charge_spin = false;
  for (const auto &graph : replicas)
    any_charge_spin |= graph.total_charge != 0.0 || graph.unpaired_electrons != 0.0;
  if (any_charge_spin) {
    const std::size_t width = charge_spin_.width;
    std::vector<double> offsets(z.size() * width, 0.0);
    for (std::size_t image = 0; image < replicas.size(); ++image) {
      const auto &graph = replicas[image];
      if (graph.total_charge == 0.0 && graph.unpaired_electrons == 0.0) continue;
      ElementCounts counts{};
      for (const auto zi : graph.atomic_numbers)
        if (zi > 0 && zi <= 118) ++counts[static_cast<std::size_t>(zi)];
      const auto table = charge_spin_offset_table(
          charge_spin_, counts, graph.total_charge, graph.unpaired_electrons);
      for (std::size_t node = node_offsets[image]; node < node_offsets[image + 1]; ++node) {
        const std::int64_t zi = z[node];
        if (zi <= 0 || zi > 118) continue;
        std::copy_n(table.begin() + static_cast<std::ptrdiff_t>((zi - 1) * width),
                    width, offsets.begin() + static_cast<std::ptrdiff_t>(node * width));
      }
    }
    workspace.forces.reverse.embedding_offset =
        DoubleView("so3lr_turbo_charge_spin_offset", offsets.size());
    fill_device(workspace.forces.reverse.embedding_offset, offsets);
  }
  evaluator_.launch_device(
      device_z, device_sr_vectors, device_sr_senders, device_sr_receivers,
      device_lr_vectors, device_lr_senders, device_lr_receivers,
      device_image_ids, device_total_charges, workspace);
  Kokkos::fence();

  const auto energies = copy_device(evaluator_.atomic_energies(workspace));
  const auto charges = copy_device(evaluator_.partial_charges(workspace));
  const auto hirshfeld = copy_device(evaluator_.hirshfeld_ratios(workspace));
  const auto forces = copy_device(evaluator_.atomic_forces(workspace));
  const auto image_energies = copy_device(evaluator_.image_energies(workspace));

  TurboReplicaBatchResult result;
  result.packed_nodes = z.size();
  result.packed_sr_edges = sr_senders.size();
  result.packed_lr_pairs = lr_senders.size();
  result.replicas.resize(replicas.size());
  for (std::size_t image = 0; image < replicas.size(); ++image) {
    const std::size_t begin = node_offsets[image];
    const std::size_t end = node_offsets[image + 1];
    auto &output = result.replicas[image];
    output.atomic_energies.assign(energies.begin() + begin,
                                  energies.begin() + end);
    output.partial_charges.assign(charges.begin() + begin,
                                  charges.begin() + end);
    output.hirshfeld_ratios.assign(hirshfeld.begin() + begin,
                                   hirshfeld.begin() + end);
    output.atomic_forces.assign(forces.begin() + begin * 3,
                                forces.begin() + end * 3);
    output.total_energy = image_energies[image];
  }
  return result;
}

}  // namespace so3lr
