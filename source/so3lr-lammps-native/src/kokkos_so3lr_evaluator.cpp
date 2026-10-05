#include "so3lr/kokkos_so3lr_evaluator.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace so3lr {
namespace {

void dump_forward_view(const std::string &path,
                       const Kokkos::View<double *> &device) {
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), device);
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot create native forward tensor dump");
  stream.write(reinterpret_cast<const char *>(host.data()),
               static_cast<std::streamsize>(host.extent(0) * sizeof(double)));
  if (!stream)
    throw std::runtime_error("failed to write native forward tensor dump");
}

void dump_lr_int64_view(const std::string &path,
                          const Kokkos::View<std::int64_t *> &device) {
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), device);
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot create native LR integer tensor dump");
  stream.write(reinterpret_cast<const char *>(host.data()),
               static_cast<std::streamsize>(host.extent(0) * sizeof(std::int64_t)));
  if (!stream)
    throw std::runtime_error("failed to write native LR integer tensor dump");
}

template <class Reader>
void fill_device(const Kokkos::View<double *> &device, std::size_t count,
                 Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

void require_finite_positive(double value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0))
    throw std::runtime_error(std::string("SO3LR invalid physical parameter ") +
                             name);
}

}  // namespace

So3lrEvaluatorWorkspace::So3lrEvaluatorWorkspace(
    std::size_t nodes, std::size_t sr_edges, std::size_t lr_pairs)
    : So3lrEvaluatorWorkspace(nodes, nodes, sr_edges, lr_pairs) {}

So3lrEvaluatorWorkspace::So3lrEvaluatorWorkspace(
    std::size_t sr_nodes, std::size_t owned_nodes,
    std::size_t sr_edges, std::size_t lr_pairs)
    : sr_geometry(sr_nodes, sr_edges),
      chain(sr_nodes, sr_edges, owned_nodes),
      zbl(owned_nodes, sr_edges),
      long_range(owned_nodes, lr_pairs),
      energy_seeds("so3lr_evaluator_energy_seeds", owned_nodes),
      partial_charge_seeds("so3lr_evaluator_charge_seeds", owned_nodes),
      hirshfeld_seeds("so3lr_evaluator_hirshfeld_seeds", owned_nodes),
      owned_atomic_numbers("so3lr_evaluator_owned_z", owned_nodes),
      owned_partial_charges("so3lr_evaluator_owned_q", owned_nodes),
      owned_hirshfeld_ratios("so3lr_evaluator_owned_h", owned_nodes),
      atomic_energies("so3lr_evaluator_atomic_energies", owned_nodes),
      atomic_forces("so3lr_evaluator_atomic_forces", owned_nodes * 3) {
  if (sr_nodes == 0 || owned_nodes == 0 || owned_nodes > sr_nodes ||
      sr_edges == 0 || lr_pairs == 0)
    throw std::runtime_error("SO3LR ownership-aware evaluator workspace is empty");
  Kokkos::deep_copy(energy_seeds, 1.0);
  Kokkos::deep_copy(partial_charge_seeds, 0.0);
  Kokkos::deep_copy(hirshfeld_seeds, 0.0);
}

std::size_t so3lr_evaluator_workspace_bytes(
    std::size_t nodes, std::size_t sr_edges, std::size_t lr_pairs,
    std::size_t persistent_bytes) {
  return so3lr_evaluator_workspace_bytes(
      nodes, nodes, sr_edges, lr_pairs, persistent_bytes);
}

std::size_t so3lr_evaluator_workspace_bytes(
    std::size_t sr_nodes, std::size_t owned_nodes,
    std::size_t sr_edges, std::size_t lr_pairs,
    std::size_t persistent_bytes) {
  constexpr std::size_t block_node_doubles = 4516;
  constexpr std::size_t block_edge_doubles = 1027;
  constexpr std::size_t shared_reverse_node_doubles = 2256;
  constexpr std::size_t shared_reverse_edge_doubles = 549;
  constexpr std::size_t chain_sr_node_doubles =
      3 * block_node_doubles - 2 * shared_reverse_node_doubles +
      128 + 24 + 24 + 128 + 3 * 128 + 3 * 24;
  constexpr std::size_t chain_edge_doubles =
      3 * block_edge_doubles - 2 * shared_reverse_edge_doubles +
      1 + 24 + 1 + 24;
  const std::size_t chain_and_sr_geometry =
      persistent_bytes +
      sr_nodes * ((chain_sr_node_doubles + 9) * sizeof(double) +
                  sizeof(std::int64_t) + sizeof(std::size_t)) +
      sr_edges * ((chain_edge_doubles + 6) * sizeof(double) +
                  2 * sizeof(std::size_t));
  const std::size_t lr_graph =
      lr_pairs * (3 * sizeof(double) + 2 * sizeof(std::size_t));
  constexpr std::size_t output_head_node_doubles = 1220;
  const std::size_t owned_bridge =
      owned_nodes * ((output_head_node_doubles + 128 + 9) * sizeof(double) +
                     2 * sizeof(std::int64_t));
  return chain_and_sr_geometry + lr_graph + owned_bridge +
         physical_zbl_workspace_bytes(owned_nodes, sr_edges) +
         physical_long_range_workspace_bytes(owned_nodes, lr_pairs);
}

KokkosSo3lrEvaluator::KokkosSo3lrEvaluator(const NativeModel &model)
    : chain_(model) {
  if (model.manifest().at("schema").string() != "so3lr-native-model-v2")
    throw std::runtime_error(
        "SO3LR evaluator requires self-contained native model v2");
  const auto &alpha_tensor = model.tensor("physical.reference_alphas");
  const auto &c6_tensor = model.tensor("physical.reference_c6");
  if (alpha_tensor.dtype != "float64" || c6_tensor.dtype != "float64" ||
      alpha_tensor.shape.size() != 1 || c6_tensor.shape != alpha_tensor.shape ||
      alpha_tensor.shape[0] < 8)
    throw std::runtime_error("SO3LR physical reference-table contract changed");
  const std::size_t table_size = alpha_tensor.shape[0];
  reference_alphas_ =
      DoubleView("so3lr_evaluator_reference_alphas", table_size);
  reference_c6_ = DoubleView("so3lr_evaluator_reference_c6", table_size);
  fill_device(reference_alphas_, table_size,
              [&](std::size_t i) { return model.float64(alpha_tensor, i); });
  fill_device(reference_c6_, table_size,
              [&](std::size_t i) { return model.float64(c6_tensor, i); });

  if (model.architecture_number("zbl_enabled") != 1.0 ||
      model.manifest().at("architecture").at("zbl_cutoff_function").string() !=
          "phys")
    throw std::runtime_error("SO3LR native ZBL contract changed");
  zbl_parameters_.ke = model.architecture_number("zbl_ke");
  zbl_parameters_.cutoff =
      model.architecture_number("short_range_cutoff_angstrom");
  zbl_parameters_.switch_off =
      model.architecture_number("zbl_switch_off_angstrom");
  zbl_parameters_.p = model.architecture_number("zbl_p");
  zbl_parameters_.d = model.architecture_number("zbl_d");
  for (int k = 0; k < 4; ++k) {
    zbl_parameters_.a[k] =
        model.architecture_number("zbl_a" + std::to_string(k + 1));
    zbl_parameters_.c[k] =
        model.architecture_number("zbl_c" + std::to_string(k + 1));
  }
  parameters_.ke = model.architecture_number("lr_electrostatic_ke");
  parameters_.electrostatic_sigma =
      model.architecture_number("lr_electrostatic_sigma");
  parameters_.cutoff =
      model.architecture_number("long_range_cutoff_angstrom");
  parameters_.electrostatic_cuton =
      model.architecture_number("lr_electrostatic_cuton_angstrom");
  parameters_.fine_structure =
      model.architecture_number("lr_fine_structure");
  parameters_.bohr = model.architecture_number("lr_bohr_angstrom");
  parameters_.hartree = model.architecture_number("lr_hartree_ev");
  parameters_.dispersion_scale =
      model.architecture_number("lr_dispersion_scale");
  parameters_.dispersion_cuton =
      model.architecture_number("lr_dispersion_cuton_angstrom");
  parameters_.pair_scale = model.architecture_number("lr_pair_scale");
  require_finite_positive(parameters_.ke, "ke");
  require_finite_positive(parameters_.electrostatic_sigma, "sigma");
  require_finite_positive(parameters_.cutoff, "cutoff");
  require_finite_positive(parameters_.fine_structure, "fine_structure");
  require_finite_positive(parameters_.bohr, "bohr");
  require_finite_positive(parameters_.hartree, "hartree");
  require_finite_positive(parameters_.dispersion_scale, "dispersion_scale");
  require_finite_positive(parameters_.pair_scale, "pair_scale");
  if (!(parameters_.cutoff > parameters_.electrostatic_cuton) ||
      !(parameters_.electrostatic_cuton > 0.0) ||
      !(parameters_.cutoff > parameters_.dispersion_cuton) ||
      !(parameters_.dispersion_cuton > 0.0) ||
      std::abs(parameters_.pair_scale - 0.5) > 1.0e-12)
    throw std::runtime_error("SO3LR LR cutoff/half-pair contract changed");

  persistent_device_bytes_ = chain_.persistent_device_bytes() +
                             2 * table_size * sizeof(double);
  self_contained_model_contract_ =
      chain_.chain_contract_verified() && chain_.shared_kokkos_stream();
  if (!self_contained_model_contract_)
    throw std::runtime_error("SO3LR self-contained evaluator contract failed");
  Kokkos::fence();
}

void KokkosSo3lrEvaluator::launch_device(
    const Int64View &atomic_numbers,
    const DoubleView &sr_edge_vectors,
    const IndexView &sr_senders,
    const IndexView &sr_receivers,
    const DoubleView &lr_pair_vectors,
    const IndexView &lr_senders,
    const IndexView &lr_receivers,
    const So3lrEvaluatorWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  IndexView identity("so3lr_evaluator_identity_owners", nodes);
  Kokkos::parallel_for(
      "so3lr_evaluator_fill_identity_owners", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) { identity(node) = node; });
  launch_owned_device(
      atomic_numbers, nodes, identity, sr_edge_vectors, sr_senders,
      sr_receivers, lr_pair_vectors, lr_senders, lr_receivers, workspace);
}

void KokkosSo3lrEvaluator::launch_owned_device(
    const Int64View &atomic_numbers, std::size_t owned_nodes,
    const IndexView &sr_node_owners,
    const DoubleView &sr_edge_vectors,
    const IndexView &sr_senders,
    const IndexView &sr_receivers,
    const DoubleView &lr_pair_vectors,
    const IndexView &lr_senders,
    const IndexView &lr_receivers,
    const So3lrEvaluatorWorkspace &workspace) const {
  const std::size_t sr_nodes = atomic_numbers.extent(0);
  const std::size_t sr_edges = sr_senders.extent(0);
  const std::size_t lr_pairs = lr_senders.extent(0);
  if (sr_nodes == 0 || owned_nodes == 0 || owned_nodes > sr_nodes ||
      sr_edges == 0 || lr_pairs == 0 ||
      sr_node_owners.extent(0) != sr_nodes ||
      sr_receivers.extent(0) != sr_edges ||
      sr_edge_vectors.extent(0) != sr_edges * 3 ||
      lr_receivers.extent(0) != lr_pairs ||
      lr_pair_vectors.extent(0) != lr_pairs * 3 ||
      workspace.energy_seeds.extent(0) != owned_nodes ||
      workspace.partial_charge_seeds.extent(0) != owned_nodes ||
      workspace.hirshfeld_seeds.extent(0) != owned_nodes ||
      workspace.owned_atomic_numbers.extent(0) != owned_nodes ||
      workspace.atomic_energies.extent(0) != owned_nodes ||
      workspace.atomic_forces.extent(0) != owned_nodes * 3)
    throw std::runtime_error("SO3LR ownership-aware graph/workspace mismatch");

  // Opt-in profiling only. The boundary fences make asynchronous CUDA phases
  // individually measurable; the normal path retains its original ordering.
  using InternalProfileClock = std::chrono::steady_clock;
  static std::size_t internal_profile_seen = 0;
  ++internal_profile_seen;
  const char *profile_value = std::getenv("SO3_NATIVE_INTERNAL_PROFILE");
  const char *profile_limit_value =
      std::getenv("SO3_NATIVE_INTERNAL_PROFILE_LIMIT");
  const char *profile_every_value =
      std::getenv("SO3_NATIVE_INTERNAL_PROFILE_EVERY");
  const std::size_t profile_limit = profile_limit_value == nullptr
      ? 0
      : static_cast<std::size_t>(std::strtoull(profile_limit_value, nullptr, 10));
  const std::size_t parsed_profile_every = profile_every_value == nullptr
      ? 1
      : static_cast<std::size_t>(std::strtoull(profile_every_value, nullptr, 10));
  const std::size_t profile_every =
      parsed_profile_every == 0 ? 1 : parsed_profile_every;
  const bool internal_profile_requested =
      profile_value != nullptr && profile_value[0] != '\0' &&
      std::string(profile_value) != "0";
  const bool internal_profile = internal_profile_requested &&
      (profile_limit == 0 || internal_profile_seen <= profile_limit) &&
      ((internal_profile_seen - 1) % profile_every == 0);
  const auto profile_mark = [&]() {
    if (internal_profile) Kokkos::fence();
    return InternalProfileClock::now();
  };
  const auto internal_start = profile_mark();

  launch_so3lr_geometry_forward_device(sr_edge_vectors,
                                       workspace.sr_geometry);
  const auto geometry_forward_done = profile_mark();
  launch_so3lr_zbl_device(
      atomic_numbers, owned_nodes, workspace.sr_geometry.distances,
      sr_senders, sr_receivers, zbl_parameters_, workspace.zbl);
  const auto zbl_forward_done = profile_mark();
  chain_.launch_forward_periodic_device(
      atomic_numbers, owned_nodes, sr_node_owners,
      workspace.sr_geometry.distances,
      workspace.sr_geometry.sh_vectors, sr_senders, sr_receivers,
      workspace.chain);
  const auto sr_forward_done = profile_mark();

  const char *forward_dump_prefix =
      std::getenv("SO3_NATIVE_FORWARD_DUMP_PREFIX");
  if (forward_dump_prefix != nullptr && forward_dump_prefix[0] != '\0') {
    const std::string prefix(forward_dump_prefix);
    dump_forward_view(prefix + ".embedding_inv.f64", workspace.chain.embedding);
    dump_forward_view(prefix + ".embedding_ev.f64", workspace.chain.initial_ev);
    dump_forward_view(prefix + ".block0_inv.f64",
                      workspace.chain.blocks[0].post_forward.final_inv);
    dump_forward_view(prefix + ".block0_ev.f64",
                      workspace.chain.blocks[0].post_forward.final_ev);
    dump_forward_view(prefix + ".sync0_inv.f64",
                      workspace.chain.synchronized_inv[0]);
    dump_forward_view(prefix + ".sync0_ev.f64",
                      workspace.chain.synchronized_ev[0]);
    dump_forward_view(prefix + ".block1_inv.f64",
                      workspace.chain.blocks[1].post_forward.final_inv);
    dump_forward_view(prefix + ".block1_ev.f64",
                      workspace.chain.blocks[1].post_forward.final_ev);
    dump_forward_view(prefix + ".sync1_inv.f64",
                      workspace.chain.synchronized_inv[1]);
    dump_forward_view(prefix + ".sync1_ev.f64",
                      workspace.chain.synchronized_ev[1]);
    dump_forward_view(prefix + ".block2_inv.f64",
                      workspace.chain.blocks[2].post_forward.final_inv);
    dump_forward_view(prefix + ".block2_ev.f64",
                      workspace.chain.blocks[2].post_forward.final_ev);
    dump_forward_view(prefix + ".atomic_energies.f64",
                      chain_.atomic_energies(workspace.chain));
    dump_forward_view(prefix + ".partial_charges.f64",
                      chain_.partial_charges(workspace.chain));
    dump_forward_view(prefix + ".hirshfeld_ratios.f64",
                      chain_.hirshfeld_ratios(workspace.chain));
    std::ofstream manifest(prefix + ".manifest.txt");
    if (!manifest)
      throw std::runtime_error("cannot create native forward manifest");
    manifest << "nodes=" << sr_nodes << '\n'
             << "owned_nodes=" << owned_nodes << '\n'
             << "head_nodes=" << owned_nodes << '\n'
             << "inv_width=128\n"
             << "ev_width=24\n"
             << "dtype=float64\n"
             << "status=PASS\n";
  }

  const auto chain_charge = chain_.partial_charges(workspace.chain);
  const auto chain_hirshfeld = chain_.hirshfeld_ratios(workspace.chain);
  const auto owned_z = workspace.owned_atomic_numbers;
  const auto owned_charge = workspace.owned_partial_charges;
  const auto owned_hirshfeld = workspace.owned_hirshfeld_ratios;
  Kokkos::parallel_for(
      "so3lr_evaluator_pack_owned_heads",
      Kokkos::RangePolicy<>(0, owned_nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        owned_z(node) = atomic_numbers(node);
        owned_charge(node) = chain_charge(node);
        owned_hirshfeld(node) = chain_hirshfeld(node);
      });
  const auto owned_head_pack_done = profile_mark();
  launch_so3lr_physical_long_range_device(
      owned_z, owned_charge, owned_hirshfeld, reference_alphas_,
      reference_c6_, lr_pair_vectors, lr_senders, lr_receivers, parameters_,
      workspace.long_range);
  const auto long_range_done = profile_mark();


  const char *zbl_dump_prefix = std::getenv("SO3_NATIVE_ZBL_DUMP_PREFIX");
  if (zbl_dump_prefix != nullptr && zbl_dump_prefix[0] != '\0') {
    Kokkos::fence();
    const std::string prefix(zbl_dump_prefix);
    dump_forward_view(prefix + ".atomic_energy.f64", workspace.zbl.atomic_energy);
    dump_forward_view(prefix + ".edge_energy.f64", workspace.zbl.edge_energy);
    dump_forward_view(prefix + ".edge_radial_gradient.f64",
                      workspace.zbl.edge_radial_gradient);
    std::ofstream manifest(prefix + ".manifest.txt");
    if (!manifest) throw std::runtime_error("cannot create native ZBL manifest");
    manifest << std::setprecision(17)
             << "owned_nodes=" << owned_nodes << '\n'
             << "edges=" << sr_edges << '\n'
             << "ke=" << zbl_parameters_.ke << '\n'
             << "cutoff=" << zbl_parameters_.cutoff << '\n'
             << "switch_off=" << zbl_parameters_.switch_off << '\n'
             << "status=PASS\n";
  }

  const char *lr_dump_prefix = std::getenv("SO3_NATIVE_LR_DUMP_PREFIX");
  if (lr_dump_prefix != nullptr && lr_dump_prefix[0] != '\0') {
    Kokkos::fence();
    const std::string prefix(lr_dump_prefix);
    dump_lr_int64_view(prefix + ".owned_atomic_numbers.i64", owned_z);
    dump_forward_view(prefix + ".owned_partial_charges.f64", owned_charge);
    dump_forward_view(prefix + ".owned_hirshfeld_ratios.f64", owned_hirshfeld);
    dump_forward_view(prefix + ".reference_alphas.f64", reference_alphas_);
    dump_forward_view(prefix + ".reference_c6.f64", reference_c6_);
    dump_forward_view(prefix + ".atomic_energy.f64", workspace.long_range.atomic_energy);
    dump_forward_view(prefix + ".charge_gradient.f64", workspace.long_range.charge_gradient);
    dump_forward_view(prefix + ".hirshfeld_gradient.f64", workspace.long_range.hirshfeld_gradient);
    dump_forward_view(prefix + ".electrostatic_pair_energy.f64", workspace.long_range.electrostatic_pair_energy);
    dump_forward_view(prefix + ".dispersion_pair_energy.f64", workspace.long_range.dispersion_pair_energy);
    dump_forward_view(prefix + ".pair_radial_gradient.f64", workspace.long_range.pair_radial_gradient);
    dump_forward_view(prefix + ".pair_force_vectors.f64", workspace.long_range.pair_force_vectors);
    dump_forward_view(prefix + ".atomic_forces.f64", workspace.long_range.atomic_forces);
    std::ofstream manifest(prefix + ".manifest.txt");
    if (!manifest)
      throw std::runtime_error("cannot create native LR manifest");
    manifest << std::setprecision(17)
             << "nodes=" << owned_nodes << '\n'
             << "pairs=" << lr_senders.extent(0) << '\n'
             << "ke=" << parameters_.ke << '\n'
             << "electrostatic_sigma=" << parameters_.electrostatic_sigma << '\n'
             << "cutoff=" << parameters_.cutoff << '\n'
             << "electrostatic_cuton=" << parameters_.electrostatic_cuton << '\n'
             << "fine_structure=" << parameters_.fine_structure << '\n'
             << "bohr=" << parameters_.bohr << '\n'
             << "hartree=" << parameters_.hartree << '\n'
             << "dispersion_scale=" << parameters_.dispersion_scale << '\n'
             << "dispersion_cuton=" << parameters_.dispersion_cuton << '\n'
             << "pair_scale=" << parameters_.pair_scale << '\n'
             << "dtype=float64\nstatus=PASS\n";
  }

  const auto charge_seed = workspace.partial_charge_seeds;
  const auto hirshfeld_seed = workspace.hirshfeld_seeds;
  const auto physical_charge_gradient = workspace.long_range.charge_gradient;
  const auto physical_hirshfeld_gradient =
      workspace.long_range.hirshfeld_gradient;
  Kokkos::parallel_for(
      "so3lr_evaluator_unpack_owned_head_gradients",
      Kokkos::RangePolicy<>(0, owned_nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        charge_seed(node) = physical_charge_gradient(node);
        hirshfeld_seed(node) = physical_hirshfeld_gradient(node);
      });
  const auto long_range_seed_done = profile_mark();
  chain_.launch_multihead_reverse_periodic_device(
      atomic_numbers, owned_nodes, sr_node_owners,
      workspace.sr_geometry.distances,
      workspace.sr_geometry.sh_vectors, sr_senders, sr_receivers,
      workspace.energy_seeds, workspace.partial_charge_seeds,
      workspace.hirshfeld_seeds, workspace.chain);
  const auto sr_reverse_done = profile_mark();
  const auto combined_sr_radial = chain_.grad_distances(workspace.chain);
  const auto zbl_radial = workspace.zbl.edge_radial_gradient;
  Kokkos::parallel_for(
      "so3lr_evaluator_add_zbl_radial_gradient",
      Kokkos::RangePolicy<>(0, sr_edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        combined_sr_radial(edge) += zbl_radial(edge);
      });
  const auto zbl_merge_done = profile_mark();
  launch_so3lr_geometry_reverse_device(
      sr_edge_vectors, sr_senders, sr_receivers,
      chain_.grad_distances(workspace.chain), chain_.grad_sh(workspace.chain),
      workspace.sr_geometry);
  const auto geometry_reverse_done = profile_mark();

  const auto learned_energy = chain_.atomic_energies(workspace.chain);
  const auto zbl_energy = workspace.zbl.atomic_energy;
  const auto physical_energy = workspace.long_range.atomic_energy;
  const auto implicit_force = workspace.sr_geometry.atomic_forces;
  const auto direct_force = workspace.long_range.atomic_forces;
  const auto total_energy = workspace.atomic_energies;
  const auto total_force = workspace.atomic_forces;
  Kokkos::parallel_for(
      "so3lr_evaluator_assemble_owned_energy_force",
      Kokkos::RangePolicy<>(0, owned_nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        total_energy(node) = learned_energy(node) + zbl_energy(node) + physical_energy(node);
        for (std::size_t component = 0; component < 3; ++component)
          total_force(node * 3 + component) =
              direct_force(node * 3 + component);
      });
  Kokkos::parallel_for(
      "so3lr_evaluator_reduce_sr_image_forces",
      Kokkos::RangePolicy<>(0, sr_nodes * 3),
      KOKKOS_LAMBDA(const std::size_t index) {
        const std::size_t node = index / 3;
        const std::size_t component = index % 3;
        const std::size_t owner = sr_node_owners(node);
        if (owner < owned_nodes)
          Kokkos::atomic_add(&total_force(owner * 3 + component),
                             implicit_force(index));
      });
  const auto assembly_done = profile_mark();

  if (internal_profile) {
    const auto milliseconds = [](const auto begin, const auto end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    const std::size_t persistent_bytes = persistent_device_bytes_;
    const std::size_t zbl_bytes =
        physical_zbl_workspace_bytes(owned_nodes, sr_edges);
    const std::size_t long_range_workspace_bytes =
        physical_long_range_workspace_bytes(owned_nodes, lr_pairs);
    const std::size_t long_range_graph_bytes =
        lr_pairs * (3 * sizeof(double) + 2 * sizeof(std::size_t));
    const std::size_t all_estimated_bytes = so3lr_evaluator_workspace_bytes(
        sr_nodes, owned_nodes, sr_edges, lr_pairs, persistent_bytes);
    const std::size_t sr_gnn_bridge_bytes =
        all_estimated_bytes - persistent_bytes - zbl_bytes -
        long_range_workspace_bytes - long_range_graph_bytes;
    std::fprintf(
        stdout,
        "SO3LR_NATIVE_INTERNAL_PROFILE call=%zu nodes=%zu sr_nodes=%zu "
        "sr_edges=%zu lr_pairs=%zu geometry_forward_ms=%.9g "
        "zbl_forward_ms=%.9g sr_forward_ms=%.9g "
        "owned_head_pack_ms=%.9g long_range_ms=%.9g "
        "long_range_seed_ms=%.9g sr_reverse_ms=%.9g "
        "zbl_merge_ms=%.9g geometry_reverse_ms=%.9g "
        "assembly_ms=%.9g evaluator_internal_total_ms=%.9g "
        "sr_gnn_bridge_bytes=%zu zbl_workspace_bytes=%zu "
        "long_range_workspace_bytes=%zu long_range_graph_bytes=%zu "
        "persistent_device_bytes=%zu estimated_total_device_bytes=%zu\n",
        internal_profile_seen, owned_nodes, sr_nodes, sr_edges, lr_pairs,
        milliseconds(internal_start, geometry_forward_done),
        milliseconds(geometry_forward_done, zbl_forward_done),
        milliseconds(zbl_forward_done, sr_forward_done),
        milliseconds(sr_forward_done, owned_head_pack_done),
        milliseconds(owned_head_pack_done, long_range_done),
        milliseconds(long_range_done, long_range_seed_done),
        milliseconds(long_range_seed_done, sr_reverse_done),
        milliseconds(sr_reverse_done, zbl_merge_done),
        milliseconds(zbl_merge_done, geometry_reverse_done),
        milliseconds(geometry_reverse_done, assembly_done),
        milliseconds(internal_start, assembly_done),
        sr_gnn_bridge_bytes, zbl_bytes, long_range_workspace_bytes,
        long_range_graph_bytes, persistent_bytes, all_estimated_bytes);
    std::fflush(stdout);
  }
}

}  // namespace so3lr
