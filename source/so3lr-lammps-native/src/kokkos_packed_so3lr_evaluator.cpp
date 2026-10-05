#include "so3lr/kokkos_packed_so3lr_evaluator.hpp"

#include <cmath>
#include <stdexcept>

namespace so3lr {

PackedSo3lrWorkspace::PackedSo3lrWorkspace(
    std::size_t nodes, std::size_t sr_edges, std::size_t lr_pairs,
    std::size_t images, const So3lrArchitecture &arch)
    : forces(nodes, sr_edges, arch),
      zbl(nodes, sr_edges),
      long_range(nodes, lr_pairs),
      segments(nodes, images),
      owned_mask("so3lr_packed_owned_mask", nodes),
      energy_seeds("so3lr_packed_energy_seeds", nodes),
      atomic_energies("so3lr_packed_atomic_energy", nodes),
      atomic_forces("so3lr_packed_atomic_forces", nodes * 3),
      image_energies("so3lr_packed_image_energy", images) {
  if (nodes == 0 || sr_edges == 0 || lr_pairs == 0 || images == 0)
    throw std::runtime_error("SO3LR packed physical workspace is empty");
  // Every packed node is owned: no ghosts, so every node seeds its energy.
  Kokkos::deep_copy(owned_mask, static_cast<std::size_t>(1));
  Kokkos::deep_copy(energy_seeds, 1.0);
}

KokkosPackedSo3lrEvaluator::KokkosPackedSo3lrEvaluator(
    const NativeModel &model)
    : arch_(model.arch()),
      forces_(model),
      long_range_(model),
      repulsion_(load_so3lr_repulsion(model)) {
  contract_verified_ =
      forces_.distributed_cartesian_force_contract() &&
      forces_.receiver_owned_edge_contract() &&
      std::abs(long_range_.parameters().cutoff -
               model.architecture_number("long_range_cutoff_angstrom")) <=
          1.0e-12;
  if (!contract_verified_)
    throw std::runtime_error("SO3LR packed physical contract failed");
}

void KokkosPackedSo3lrEvaluator::launch_device(
    const Int64View &atomic_numbers,
    const DoubleView &sr_edge_vectors,
    const IndexView &sr_senders, const IndexView &sr_receivers,
    const DoubleView &lr_pair_vectors,
    const IndexView &lr_senders, const IndexView &lr_receivers,
    const Int64View &image_ids, const DoubleView &total_charges,
    const PackedSo3lrWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  const std::size_t sr_edges = sr_senders.extent(0);
  const std::size_t lr_pairs = lr_senders.extent(0);
  if (nodes == 0 || sr_edges == 0 || lr_pairs == 0 ||
      image_ids.extent(0) != nodes || workspace.segments.nodes != nodes ||
      workspace.owned_mask.extent(0) != nodes ||
      total_charges.extent(0) != workspace.segments.images ||
      sr_receivers.extent(0) != sr_edges ||
      sr_edge_vectors.extent(0) != sr_edges * 3 ||
      lr_receivers.extent(0) != lr_pairs ||
      lr_pair_vectors.extent(0) != lr_pairs * 3)
    throw std::runtime_error("SO3LR packed physical graph mismatch");

  const auto &reverse = forces_.reverse();
  const auto &sr = workspace.forces;
  const auto distances = forces_.distances(sr);
  const auto sh = forces_.sh_vectors(sr);

  // Forward: geometry, repulsion, interaction blocks, output heads. This is
  // the so3lr/native/mpi sequence without the inter-block halo exchanges.
  forces_.launch_geometry_forward_device(sr_edge_vectors, sr);
  launch_so3lr_zbl_device(atomic_numbers, nodes, distances, sr_senders,
                          sr_receivers, repulsion_.parameters, workspace.zbl,
                          repulsion_.nlh_a, repulsion_.nlh_b);
  const std::size_t layers = reverse.layers();
  for (std::size_t b = 0; b < layers; ++b)
    reverse.launch_block_forward_device(b, atomic_numbers, distances, sh,
                                        sr_senders, sr_receivers,
                                        sr.reverse);
  reverse.launch_output_heads_device(atomic_numbers, sr.reverse);

  // Charge conservation per image: q_i = raw_i + (Q_img - sum_img raw) / n_img.
  launch_image_segment_forward(reverse.raw_charges(sr.reverse),
                               reverse.atomic_energies(sr.reverse), image_ids,
                               total_charges, workspace.segments);
  {
    const auto partial = reverse.partial_charges(sr.reverse);
    const auto corrected = workspace.segments.corrected_charges;
    Kokkos::parallel_for(
        "so3lr_packed_publish_image_charges", Kokkos::RangePolicy<>(0, nodes),
        KOKKOS_LAMBDA(const std::size_t node) {
          partial(node) = corrected(node);
        });
  }

  // Long range (electrostatics + dispersion; models with a C6 head also read C6 ratios).
  const bool has_c6 = reverse.has_c6_head();
  long_range_.launch_device(atomic_numbers, reverse.partial_charges(sr.reverse),
                            reverse.hirshfeld_ratios(sr.reverse),
                            lr_pair_vectors, lr_senders, lr_receivers,
                            workspace.long_range,
                            has_c6 ? reverse.c6_ratios(sr.reverse)
                                   : DoubleView());

  // Reverse. The long-range gradients seed the charge, ratio and C6 heads.
  // The charge seeds are projected inside each image (the adjoint of the
  // per-image correction), so the head reverse is told the mean is zero.
  launch_image_segment_reverse(workspace.long_range.charge_gradient, image_ids,
                               workspace.segments);
  reverse.launch_owned_multihead_reverse_device(
      atomic_numbers, workspace.owned_mask, workspace.energy_seeds,
      workspace.segments.raw_charge_seeds,
      workspace.long_range.hirshfeld_gradient, 0.0, sr.reverse,
      has_c6 ? workspace.long_range.c6_gradient : DoubleView());
  for (std::size_t b = layers; b-- > 0;)
    reverse.launch_block_reverse_device(b, distances, sh, sr_senders,
                                        sr_receivers, sr.reverse, true);
  forces_.launch_geometry_reverse_with_extra_radial_device(
      sr_edge_vectors, sr_senders, sr_receivers,
      workspace.zbl.edge_radial_gradient, sr);

  // Assemble per-atom energy and force and the per-image total energy.
  const auto learned_energy = reverse.atomic_energies(sr.reverse);
  const auto zbl_energy = workspace.zbl.atomic_energy;
  const auto lr_energy = workspace.long_range.atomic_energy;
  const auto implicit_force = forces_.atomic_forces(sr);
  const auto direct_force = workspace.long_range.atomic_forces;
  const auto total_energy = workspace.atomic_energies;
  const auto total_force = workspace.atomic_forces;
  const auto image_energy = workspace.image_energies;
  Kokkos::deep_copy(image_energy, 0.0);
  Kokkos::parallel_for(
      "so3lr_packed_physical_assemble",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const double energy = learned_energy(node) + zbl_energy(node) +
                              lr_energy(node);
        total_energy(node) = energy;
        for (std::size_t component = 0; component < 3; ++component)
          total_force(node * 3 + component) =
              implicit_force(node * 3 + component) +
              direct_force(node * 3 + component);
        const std::int64_t image = image_ids(node);
        if (image >= 0 &&
            static_cast<std::size_t>(image) < image_energy.extent(0))
          Kokkos::atomic_add(&image_energy(static_cast<std::size_t>(image)),
                             energy);
      });
}

}  // namespace so3lr
