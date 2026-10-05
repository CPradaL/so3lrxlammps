#include "pair_so3lr_native.h"
#include "so3lr_element_map.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "utils.h"
#include "update.h"

#include "so3lr/lammps_neighbor_adapter.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

so3lr::LammpsNeighborSnapshot snapshot_neighbor_list(NeighList *list)
{
  if (list == nullptr) throw std::runtime_error("missing LAMMPS neighbor list");
  so3lr::LammpsNeighborSnapshot result;
  result.ilist.reserve(static_cast<std::size_t>(list->inum));
  result.offsets.reserve(static_cast<std::size_t>(list->inum) + 1);
  result.offsets.push_back(0);
  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    result.ilist.push_back(static_cast<std::size_t>(i));
    const int count = list->numneigh[i];
    int *neighbors = list->firstneigh[i];
    for (int jj = 0; jj < count; ++jj)
      result.neighbors.push_back(
          static_cast<std::size_t>(neighbors[jj] & NEIGHMASK));
    result.offsets.push_back(result.neighbors.size());
  }
  return result;
}

template <class View, class Values>
void copy_to_device(const View &device, const Values &values)
{
  if (device.extent(0) != values.size())
    throw std::runtime_error("SO3LR host/device graph size mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}



std::vector<std::size_t> single_rank_owner_map(
    const so3lr::LammpsCompactGraph &graph, std::size_t nlocal,
    const char *label)
{
  if (graph.nodes() < nlocal || graph.owned_local.size() != nlocal)
    throw std::runtime_error(std::string(label) +
                             " has an invalid owned-node prefix");
  std::unordered_map<std::int64_t, std::size_t> owner_by_tag;
  owner_by_tag.reserve(nlocal);
  for (std::size_t local = 0; local < nlocal; ++local) {
    if (graph.source_atom_rows[local] != local ||
        graph.owned_local[local] != local)
      throw std::runtime_error(std::string(label) +
                               " changed the owned atom ordering");
    if (!owner_by_tag.emplace(graph.tags[local], local).second)
      throw std::runtime_error(std::string(label) +
                               " contains duplicate owned atom tags");
  }
  std::vector<std::size_t> owner(graph.nodes());
  for (std::size_t local = 0; local < nlocal; ++local) owner[local] = local;
  for (std::size_t local = nlocal; local < graph.nodes(); ++local) {
    const auto found = owner_by_tag.find(graph.tags[local]);
    if (found == owner_by_tag.end())
      throw std::runtime_error(std::string(label) +
                               " periodic image has no local owner");
    if (graph.atomic_numbers[local] != graph.atomic_numbers[found->second])
      throw std::runtime_error(std::string(label) +
                               " periodic image element differs from owner");
    owner[local] = found->second;
  }
  for (const auto receiver : graph.receivers)
    if (receiver >= nlocal)
      throw std::runtime_error(std::string(label) +
                               " has a non-owned receiver");
  return owner;
}

std::size_t collapse_single_rank_periodic_images(
    so3lr::LammpsCompactGraph &graph, std::size_t nlocal,
    bool require_unique_half_pairs, const char *label)
{
  const auto owner = single_rank_owner_map(graph, nlocal, label);
  const std::size_t image_nodes = graph.nodes() - nlocal;
  std::unordered_set<std::uint64_t> seen_half_pairs;
  for (std::size_t edge = 0; edge < graph.interactions(); ++edge) {
    graph.senders[edge] = owner.at(graph.senders[edge]);
    graph.receivers[edge] = owner.at(graph.receivers[edge]);
    if (graph.senders[edge] == graph.receivers[edge])
      throw std::runtime_error(std::string(label) +
                               " contains a self-image interaction");
    if (require_unique_half_pairs) {
      const auto low = std::min(graph.senders[edge], graph.receivers[edge]);
      const auto high = std::max(graph.senders[edge], graph.receivers[edge]);
      const std::uint64_t key =
          (static_cast<std::uint64_t>(low) << 32) |
          static_cast<std::uint64_t>(high);
      if (!seen_half_pairs.insert(key).second)
        throw std::runtime_error(std::string(label) +
                                 " contains duplicate owner half-pairs");
    }
  }
  graph.source_atom_rows.resize(nlocal);
  graph.tags.resize(nlocal);
  graph.atomic_numbers.resize(nlocal);
  graph.owned_local.resize(nlocal);
  graph.ghost_local.clear();
  return image_nodes;
}


struct VirialReductionValue {
  double component[6];
};

struct ImageSafeVirialReduction {
  using value_type = VirialReductionValue;

  DoubleView vectors;
  DoubleView gradients;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &value) const
  {
    for (int component = 0; component < 6; ++component)
      value.component[component] = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &destination, const value_type &source) const
  {
    for (int component = 0; component < 6; ++component)
      destination.component[component] += source.component[component];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t edge, value_type &value) const
  {
    const double dx = vectors(edge * 3);
    const double dy = vectors(edge * 3 + 1);
    const double dz = vectors(edge * 3 + 2);
    const double gx = gradients(edge * 3);
    const double gy = gradients(edge * 3 + 1);
    const double gz = gradients(edge * 3 + 2);
    value.component[0] -= dx * gx;
    value.component[1] -= dy * gy;
    value.component[2] -= dz * gz;
    value.component[3] -= 0.5 * (dx * gy + dy * gx);
    value.component[4] -= 0.5 * (dx * gz + dz * gx);
    value.component[5] -= 0.5 * (dy * gz + dz * gy);
  }
};

std::array<double, 6> image_safe_global_virial_device(
    const DoubleView &vectors, const DoubleView &gradients,
    const char *kernel_label)
{
  if (vectors.extent(0) != gradients.extent(0) ||
      vectors.extent(0) % 3 != 0)
    throw std::runtime_error(
        "SO3LR device virial vector/gradient shape mismatch");
  VirialReductionValue reduced{};
  Kokkos::parallel_reduce(
      kernel_label, Kokkos::RangePolicy<>(0, vectors.extent(0) / 3),
      ImageSafeVirialReduction{vectors, gradients}, reduced);
  return {reduced.component[0], reduced.component[1], reduced.component[2],
          reduced.component[3], reduced.component[4], reduced.component[5]};
}

void require_identity_owned_graph(const so3lr::LammpsCompactGraph &graph,
                                  std::size_t nlocal,
                                  const char *label)
{
  if (graph.nodes() != nlocal || graph.owned_local.size() != nlocal ||
      !graph.ghost_local.empty())
    throw std::runtime_error(std::string(label) +
                             " still contains a non-owned node after dev_40 periodic "
                             "image-to-owner collapse");
  for (std::size_t i = 0; i < nlocal; ++i)
    if (graph.source_atom_rows[i] != i || graph.owned_local[i] != i)
      throw std::runtime_error(std::string(label) +
                               " changed the owned atom ordering");
}

}  // namespace

PairSO3LRNative::PairSO3LRNative(LAMMPS *lmp) : Pair(lmp)
{
  manybody_flag = 1;
  one_coeff = 1;
  restartinfo = 0;
  single_enable = 0;
  // The periodic global virial is tallied explicitly from minimum-image
  // edge vectors. Wrapped-coordinate F dot r is not valid here.
  no_virial_fdotr_compute = 1;
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
    initialized_kokkos_here_ = true;
  }
}

PairSO3LRNative::~PairSO3LRNative()
{
  evaluator_.reset();
  model_.reset();
  if (initialized_kokkos_here_ && Kokkos::is_initialized()) Kokkos::finalize();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSO3LRNative::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "so3lr/native:setflag");
  memory->create(cutsq, n + 1, n + 1, "so3lr/native:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairSO3LRNative::settings(int narg, char **arg)
{
  if (narg != 1)
    error->all(FLERR, "Pair style so3lr/native requires one native model path");
  model_path_ = arg[0];
  try {
    model_ = std::make_unique<so3lr::NativeModel>(
        so3lr::NativeModel::load(model_path_));
    short_range_cutoff_ =
        model_->architecture_number("short_range_cutoff_angstrom");
    long_range_cutoff_ =
        model_->architecture_number("long_range_cutoff_angstrom");
    if (!(short_range_cutoff_ > 0.0) ||
        !(long_range_cutoff_ >= short_range_cutoff_))
      throw std::runtime_error("invalid cutoffs in native model");
    evaluator_ = std::make_unique<so3lr::KokkosSo3lrEvaluator>(*model_);
    if (!evaluator_->self_contained_model_contract())
      throw std::runtime_error("self-contained evaluator contract failed");
    const double evaluator_lr = evaluator_->long_range_parameters().cutoff;
    if (std::abs(evaluator_lr - long_range_cutoff_) > 1.0e-12)
      throw std::runtime_error("model and evaluator LR cutoffs disagree");
  } catch (const std::exception &exc) {
    error->all(FLERR, "Cannot initialize native SO3LR model: {}", exc.what());
  }
}

void PairSO3LRNative::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != atom->ntypes + 2 || std::strcmp(arg[0], "*") != 0 ||
      std::strcmp(arg[1], "*") != 0)
    error->all(FLERR,
               "Pair coeff for so3lr/native must be '* * E1 ... Entypes' "
               "where each E is an element symbol or atomic number");
  type_to_atomic_number_.assign(static_cast<std::size_t>(atom->ntypes) + 1,
                                0);
  for (int type = 1; type <= atom->ntypes; ++type) {
    const int z = so3lr_lammps::atomic_number_from_token(arg[type + 1]);
    if (z == 0)
      error->all(FLERR,
                 "Unknown SO3LR element '{}'; use a symbol or atomic number "
                 "from H/1 through Es/99",
                 arg[type + 1]);
    type_to_atomic_number_[static_cast<std::size_t>(type)] = z;
  }
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

double PairSO3LRNative::init_one(int i, int j)
{
  if (!setflag[i][j])
    error->all(FLERR, "All so3lr/native coefficients are not set");
  return long_range_cutoff_;
}

void PairSO3LRNative::init_style()
{
  if (comm->nprocs != 1)
    error->all(FLERR,
               "Dev_48 pair style so3lr/native is restricted to one MPI rank");
  if (!force->newton_pair)
    error->all(FLERR, "Pair style so3lr/native requires newton pair on");
  if (!model_ || !evaluator_)
    error->all(FLERR, "Pair style so3lr/native model was not initialized");
  if (type_to_atomic_number_.size() !=
      static_cast<std::size_t>(atom->ntypes) + 1)
    error->all(FLERR, "Pair style so3lr/native coefficients are missing");

  auto *sr = neighbor->add_request(this, NeighConst::REQ_FULL);
  sr->set_id(0);
  sr->set_cutoff(short_range_cutoff_);
  auto *lr = neighbor->add_request(this);
  lr->set_id(1);
  lr->set_cutoff(long_range_cutoff_);
}

void PairSO3LRNative::init_list(int which, NeighList *ptr)
{
  if (which == 0)
    short_range_list_ = ptr;
  else if (which == 1)
    long_range_list_ = ptr;
  else
    error->all(FLERR, "Unexpected so3lr/native neighbor-list ID");
}

void PairSO3LRNative::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom)
    error->all(FLERR,
               "Dev_48 so3lr/native does not yet provide per-atom virial/stress");
  if (short_range_list_ == nullptr || long_range_list_ == nullptr)
    error->all(FLERR, "SO3LR live neighbor lists were not initialized");

  try {
    using ProfileClock = std::chrono::steady_clock;
    const auto profile_start = ProfileClock::now();
    so3lr::LammpsAtomSnapshot atoms;
    atoms.nlocal = static_cast<std::size_t>(atom->nlocal);
    const std::size_t rows =
        static_cast<std::size_t>(atom->nlocal + atom->nghost);
    atoms.tags.reserve(rows);
    atoms.types.reserve(rows);
    atoms.positions.reserve(rows * 3);
    for (std::size_t row = 0; row < rows; ++row) {
      atoms.tags.push_back(static_cast<std::int64_t>(atom->tag[row]));
      atoms.types.push_back(atom->type[row]);
      atoms.positions.push_back(atom->x[row][0]);
      atoms.positions.push_back(atom->x[row][1]);
      atoms.positions.push_back(atom->x[row][2]);
    }
    const auto atom_snapshot_done = ProfileClock::now();

    auto graphs = so3lr::build_lammps_native_graphs(
        atoms, snapshot_neighbor_list(short_range_list_),
        snapshot_neighbor_list(long_range_list_), type_to_atomic_number_,
        short_range_cutoff_, long_range_cutoff_, force->newton_pair != 0);
    const std::size_t nodes = static_cast<std::size_t>(atom->nlocal);
    const std::size_t sr_nodes = graphs.short_range.nodes();
    const auto sr_node_owner_host =
        single_rank_owner_map(graphs.short_range, nodes, "SR graph");
    const std::size_t sr_image_nodes = sr_nodes - nodes;
    const char *graph_dump_prefix =
        std::getenv("SO3_NATIVE_GRAPH_DUMP_PREFIX");
    if (evaluation_count_ == 0 && graph_dump_prefix != nullptr &&
        graph_dump_prefix[0] != '\0') {
      const std::string prefix(graph_dump_prefix);
      std::ofstream nodes_out(prefix + ".nodes.csv");
      std::ofstream edges_out(prefix + ".edges.csv");
      if (!nodes_out || !edges_out)
        throw std::runtime_error("cannot create native SR graph dump");
      nodes_out << std::setprecision(17);
      edges_out << std::setprecision(17);
      nodes_out << "node,source_row,tag,atomic_number,owner_node,owned,x,y,z\n";
      for (std::size_t node = 0; node < sr_nodes; ++node) {
        const std::size_t row = graphs.short_range.source_atom_rows[node];
        nodes_out << node << ',' << row << ',' << graphs.short_range.tags[node]
                  << ',' << graphs.short_range.atomic_numbers[node] << ','
                  << sr_node_owner_host[node] << ',' << (node < nodes ? 1 : 0)
                  << ',' << atoms.positions[row * 3] << ','
                  << atoms.positions[row * 3 + 1] << ','
                  << atoms.positions[row * 3 + 2] << '\n';
      }
      edges_out << "edge,receiver_node,sender_node,receiver_source_row,"
                   "sender_source_row,receiver_tag,sender_tag,receiver_owner,"
                   "sender_owner,dx,dy,dz,distance\n";
      for (std::size_t edge = 0; edge < graphs.short_range.interactions(); ++edge) {
        const std::size_t receiver = graphs.short_range.receivers[edge];
        const std::size_t sender = graphs.short_range.senders[edge];
        const double dx = graphs.short_range.vectors[edge * 3];
        const double dy = graphs.short_range.vectors[edge * 3 + 1];
        const double dz = graphs.short_range.vectors[edge * 3 + 2];
        edges_out << edge << ',' << receiver << ',' << sender << ','
                  << graphs.short_range.source_atom_rows[receiver] << ','
                  << graphs.short_range.source_atom_rows[sender] << ','
                  << graphs.short_range.tags[receiver] << ','
                  << graphs.short_range.tags[sender] << ','
                  << sr_node_owner_host[receiver] << ','
                  << sr_node_owner_host[sender] << ',' << dx << ',' << dy
                  << ',' << dz << ',' << std::sqrt(dx * dx + dy * dy + dz * dz)
                  << '\n';
      }
      nodes_out.close();
      edges_out.close();
      const auto report_dump = [&](FILE *stream) {
        if (stream != nullptr)
          std::fprintf(stream,
                       "SO3LR_NATIVE_SR_GRAPH_DUMP=PASS prefix=%s nodes=%zu "
                       "edges=%zu\n",
                       graph_dump_prefix, sr_nodes,
                       graphs.short_range.interactions());
      };
      report_dump(screen);
      if (logfile != screen) report_dump(logfile);
    }
    const std::size_t lr_image_nodes =
        collapse_single_rank_periodic_images(
            graphs.long_range, nodes, true, "LR graph");
    require_identity_owned_graph(graphs.long_range, nodes, "LR graph");
    const char *lr_graph_dump_prefix =
        std::getenv("SO3_NATIVE_LR_GRAPH_DUMP_PREFIX");
    if (evaluation_count_ == 0 && lr_graph_dump_prefix != nullptr &&
        lr_graph_dump_prefix[0] != '\0') {
      const std::string prefix(lr_graph_dump_prefix);
      std::ofstream edges_out(prefix + ".edges.csv");
      if (!edges_out)
        throw std::runtime_error("cannot create native LR graph dump");
      edges_out << std::setprecision(17);
      edges_out << "edge,receiver_node,sender_node,receiver_tag,sender_tag,"
                   "receiver_atomic_number,sender_atomic_number,dx,dy,dz,distance\n";
      for (std::size_t edge = 0; edge < graphs.long_range.interactions(); ++edge) {
        const std::size_t receiver = graphs.long_range.receivers[edge];
        const std::size_t sender = graphs.long_range.senders[edge];
        const double dx = graphs.long_range.vectors[edge * 3];
        const double dy = graphs.long_range.vectors[edge * 3 + 1];
        const double dz = graphs.long_range.vectors[edge * 3 + 2];
        edges_out << edge << ',' << receiver << ',' << sender << ','
                  << graphs.long_range.tags[receiver] << ','
                  << graphs.long_range.tags[sender] << ','
                  << graphs.long_range.atomic_numbers[receiver] << ','
                  << graphs.long_range.atomic_numbers[sender] << ','
                  << dx << ',' << dy << ',' << dz << ','
                  << std::sqrt(dx * dx + dy * dy + dz * dz) << '\n';
      }
      edges_out.close();
      const auto report_dump = [&](FILE *stream) {
        if (stream != nullptr)
          std::fprintf(stream,
                       "SO3LR_NATIVE_LR_GRAPH_DUMP=PASS prefix=%s pairs=%zu\n",
                       lr_graph_dump_prefix, graphs.long_range.interactions());
      };
      report_dump(screen);
      if (logfile != screen) report_dump(logfile);
    }
    for (std::size_t node = 0; node < nodes; ++node)
      if (graphs.short_range.atomic_numbers[node] !=
          graphs.long_range.atomic_numbers[node])
        throw std::runtime_error(
            "SR and LR owned atomic-number ordering differs");
    const auto host_graph_done = ProfileClock::now();

    const std::size_t sr_edges = graphs.short_range.interactions();
    const std::size_t lr_pairs = graphs.long_range.interactions();
    Int64View z("so3lr_live_z", sr_nodes);
    IndexView sr_node_owners("so3lr_live_sr_node_owners", sr_nodes);
    DoubleView sr_vectors("so3lr_live_sr_vectors", sr_edges * 3);
    IndexView sr_senders("so3lr_live_sr_senders", sr_edges);
    IndexView sr_receivers("so3lr_live_sr_receivers", sr_edges);
    DoubleView lr_vectors("so3lr_live_lr_vectors", lr_pairs * 3);
    IndexView lr_senders("so3lr_live_lr_senders", lr_pairs);
    IndexView lr_receivers("so3lr_live_lr_receivers", lr_pairs);
    copy_to_device(z, graphs.short_range.atomic_numbers);
    copy_to_device(sr_node_owners, sr_node_owner_host);
    copy_to_device(sr_vectors, graphs.short_range.vectors);
    copy_to_device(sr_senders, graphs.short_range.senders);
    copy_to_device(sr_receivers, graphs.short_range.receivers);
    copy_to_device(lr_vectors, graphs.long_range.vectors);
    copy_to_device(lr_senders, graphs.long_range.senders);
    copy_to_device(lr_receivers, graphs.long_range.receivers);
    Kokkos::fence();
    const auto graph_upload_done = ProfileClock::now();

    const std::size_t persistent_device_bytes =
        evaluator_->persistent_device_bytes();
    const std::size_t workspace_bytes =
        so3lr::so3lr_evaluator_workspace_bytes(
            sr_nodes, nodes, sr_edges, lr_pairs,
            persistent_device_bytes);
    so3lr::So3lrEvaluatorWorkspace workspace(
        sr_nodes, nodes, sr_edges, lr_pairs);
    const auto workspace_done = ProfileClock::now();
    evaluator_->launch_owned_device(
        z, nodes, sr_node_owners, sr_vectors, sr_senders, sr_receivers,
        lr_vectors, lr_senders, lr_receivers, workspace);
    Kokkos::fence();
    const auto evaluator_done = ProfileClock::now();
    const auto host_energy = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), evaluator_->atomic_energies(workspace));
    const auto host_force = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), evaluator_->atomic_forces(workspace));
    const auto host_zbl_energy = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workspace.zbl.atomic_energy);
    const auto host_physical_energy = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workspace.long_range.atomic_energy);
    const auto result_download_done = ProfileClock::now();
    const auto sr_virial = image_safe_global_virial_device(
        sr_vectors, workspace.sr_geometry.edge_energy_gradients,
        "so3lr_sr_global_virial");
    const auto lr_virial = image_safe_global_virial_device(
        lr_vectors, workspace.long_range.pair_force_vectors,
        "so3lr_lr_global_virial");
    constexpr std::size_t virial_host_transfer_bytes =
        2 * sizeof(VirialReductionValue);
    std::array<double, 6> global_virial{};
    for (std::size_t component = 0; component < 6; ++component)
      global_virial[component] =
          sr_virial[component] + lr_virial[component];
    const double sr_virial_trace =
        sr_virial[0] + sr_virial[1] + sr_virial[2];
    const double lr_virial_trace =
        lr_virial[0] + lr_virial[1] + lr_virial[2];
    const double global_virial_trace =
        global_virial[0] + global_virial[1] + global_virial[2];
    const auto virial_done = ProfileClock::now();

    double total_energy = 0.0;
    double zbl_energy = 0.0;
    double physical_energy = 0.0;
    for (std::size_t node = 0; node < nodes; ++node) {
      total_energy += host_energy(node);
      zbl_energy += host_zbl_energy(node);
      physical_energy += host_physical_energy(node);
      const std::size_t row = graphs.short_range.source_atom_rows[node];
      for (std::size_t component = 0; component < 3; ++component)
        atom->f[row][component] += host_force(node * 3 + component);
      if (eflag_atom) eatom[row] += host_energy(node);
    }
    if (eflag_global) eng_vdwl += total_energy;
    if (vflag_global)
      for (std::size_t component = 0; component < 6; ++component)
        virial[component] += global_virial[component];
    const auto lammps_update_done = ProfileClock::now();

    ++evaluation_count_;
    const auto milliseconds = [](const auto begin, const auto end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    const auto report_profile = [&](FILE *stream) {
      if (stream == nullptr) return;
      std::fprintf(
          stream,
          "SO3LR_NATIVE_PHASE_PROFILE call=%zu step=%lld nodes=%zu "
          "sr_nodes=%zu sr_edges=%zu lr_pairs=%zu atom_snapshot_ms=%.9g "
          "host_graph_ms=%.9g graph_upload_ms=%.9g "
          "workspace_allocation_ms=%.9g evaluator_ms=%.9g "
          "result_download_ms=%.9g virial_ms=%.9g "
          "lammps_update_ms=%.9g total_ms=%.9g "
          "workspace_bytes=%zu persistent_device_bytes=%zu\n",
          evaluation_count_, static_cast<long long>(update->ntimestep),
          nodes, sr_nodes, sr_edges, lr_pairs,
          milliseconds(profile_start, atom_snapshot_done),
          milliseconds(atom_snapshot_done, host_graph_done),
          milliseconds(host_graph_done, graph_upload_done),
          milliseconds(graph_upload_done, workspace_done),
          milliseconds(workspace_done, evaluator_done),
          milliseconds(evaluator_done, result_download_done),
          milliseconds(result_download_done, virial_done),
          milliseconds(virial_done, lammps_update_done),
          milliseconds(profile_start, lammps_update_done),
          workspace_bytes, persistent_device_bytes);
    };
    report_profile(screen);
    if (logfile != screen) report_profile(logfile);
    if (screen)
      std::fprintf(screen,
                   "SO3LR_NATIVE_MD_EVALUATION call=%zu step=%lld "
                   "nodes=%zu sr_nodes=%zu sr_edges=%zu lr_pairs=%zu "
                   "sr_image_nodes=%zu lr_image_nodes=%zu "
                   "potential_energy=%.17g learned_energy=%.17g zbl_energy=%.17g "
                   "physical_lr_energy=%.17g sr_virial_trace=%.17g "
                   "lr_virial_trace=%.17g virial_trace=%.17g "
                   "virial_xx=%.17g virial_yy=%.17g "
                   "virial_zz=%.17g virial_xy=%.17g "
                   "virial_xz=%.17g virial_yz=%.17g "
                   "virial_host_transfer_bytes=%zu\n",
                   evaluation_count_,
                   static_cast<long long>(update->ntimestep), nodes,
                   sr_nodes, sr_edges, lr_pairs, sr_image_nodes,
                   lr_image_nodes,
                   total_energy, total_energy - zbl_energy - physical_energy,
                   zbl_energy, physical_energy, sr_virial_trace, lr_virial_trace,
                   global_virial_trace, global_virial[0],
                   global_virial[1], global_virial[2],
                   global_virial[3], global_virial[4],
                   global_virial[5], virial_host_transfer_bytes);
    if (logfile)
      std::fprintf(logfile,
                   "SO3LR_NATIVE_MD_EVALUATION call=%zu step=%lld "
                   "nodes=%zu sr_nodes=%zu sr_edges=%zu lr_pairs=%zu "
                   "sr_image_nodes=%zu lr_image_nodes=%zu "
                   "potential_energy=%.17g learned_energy=%.17g zbl_energy=%.17g "
                   "physical_lr_energy=%.17g sr_virial_trace=%.17g "
                   "lr_virial_trace=%.17g virial_trace=%.17g "
                   "virial_xx=%.17g virial_yy=%.17g "
                   "virial_zz=%.17g virial_xy=%.17g "
                   "virial_xz=%.17g virial_yz=%.17g "
                   "virial_host_transfer_bytes=%zu\n",
                   evaluation_count_,
                   static_cast<long long>(update->ntimestep), nodes,
                   sr_nodes, sr_edges, lr_pairs, sr_image_nodes,
                   lr_image_nodes,
                   total_energy, total_energy - zbl_energy - physical_energy,
                   zbl_energy, physical_energy, sr_virial_trace, lr_virial_trace,
                   global_virial_trace, global_virial[0],
                   global_virial[1], global_virial[2],
                   global_virial[3], global_virial[4],
                   global_virial[5], virial_host_transfer_bytes);

    if (!reported_) {
      if (screen) {
        std::fprintf(screen,
                     "SO3LR_NATIVE_EVALUATION rank=0 nodes=%zu sr_nodes=%zu sr_edges=%zu "
                     "lr_pairs=%zu total_energy=%.17g sr_cutoff=%.17g "
                     "lr_cutoff=%.17g pair_compute_vflag_fdotr=%d\n",
                     nodes, sr_nodes, sr_edges, lr_pairs, total_energy,
                     short_range_cutoff_, long_range_cutoff_,
                     vflag_fdotr ? 1 : 0);
        std::fprintf(screen,
                     "SO3LR_NATIVE_FORCE_SUMMARY nodes=%zu "
                     "force_values=%zu atomic_energies=%zu\n",
                     nodes, nodes * 3, nodes);
        std::fprintf(screen, "SO3LR_LIVE_FORCE_PAIR=PASS\n");
      }
      if (logfile) {
        std::fprintf(logfile,
                     "SO3LR_NATIVE_EVALUATION rank=0 nodes=%zu sr_nodes=%zu sr_edges=%zu "
                     "lr_pairs=%zu total_energy=%.17g sr_cutoff=%.17g "
                     "lr_cutoff=%.17g pair_compute_vflag_fdotr=%d\n",
                     nodes, sr_nodes, sr_edges, lr_pairs, total_energy,
                     short_range_cutoff_, long_range_cutoff_,
                     vflag_fdotr ? 1 : 0);
        std::fprintf(logfile, "SO3LR_LIVE_FORCE_PAIR=PASS\n");
      }
      reported_ = true;
    }
  } catch (const std::exception &exc) {
    error->all(FLERR, "SO3LR native LAMMPS evaluation failed: {}", exc.what());
  }
}
