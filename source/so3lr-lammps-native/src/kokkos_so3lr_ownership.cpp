#include "so3lr/kokkos_so3lr_ownership.hpp"

#include <stdexcept>

namespace so3lr {

void KokkosSo3lrOwnedEvaluator::launch_rank_device(
    const Int64View &atomic_numbers,
    const DoubleView &sr_edge_vectors,
    const IndexView &sr_senders,
    const IndexView &sr_receivers,
    const DoubleView &lr_pair_vectors,
    const IndexView &lr_senders,
    const IndexView &lr_receivers,
    const IndexView &owned_nodes,
    const So3lrEvaluatorWorkspace &workspace) const {
  const std::size_t nodes = atomic_numbers.extent(0);
  if (nodes == 0 || owned_nodes.extent(0) == 0 ||
      workspace.energy_seeds.extent(0) != nodes)
    throw std::runtime_error("SO3LR owned-rank graph is empty or inconsistent");

  const auto seeds = workspace.energy_seeds;
  Kokkos::deep_copy(seeds, 0.0);
  Kokkos::parallel_for(
      "so3lr_owned_energy_seeds",
      Kokkos::RangePolicy<>(0, owned_nodes.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t node = owned_nodes(i);
        if (node < nodes) seeds(node) = 1.0;
      });

  evaluator_.launch_device(
      atomic_numbers, sr_edge_vectors, sr_senders, sr_receivers,
      lr_pair_vectors, lr_senders, lr_receivers, workspace);

  // The dev_26 evaluator exposes learned + physical atomic energies.  Learned
  // atomic energies belong to their owning rank, whereas physical energies
  // belong to the rank that owns the unique LR half-pair.  Preserve every
  // physical contribution but mask the learned contribution with its seed.
  const auto total = workspace.atomic_energies;
  const auto physical = workspace.long_range.atomic_energy;
  Kokkos::parallel_for(
      "so3lr_owned_atomic_energy",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) {
        total(i) = seeds(i) * (total(i) - physical(i)) + physical(i);
      });
}

void launch_pack_reverse_forces_device(
    const Kokkos::View<double *> &atomic_forces,
    const Kokkos::View<std::size_t *> &ghost_indices,
    const Kokkos::View<double *> &packed_forces) {
  const std::size_t ghosts = ghost_indices.extent(0);
  if (atomic_forces.extent(0) % 3 != 0 ||
      packed_forces.extent(0) != ghosts * 3)
    throw std::runtime_error("SO3LR reverse-force pack shape mismatch");
  const std::size_t nodes = atomic_forces.extent(0) / 3;
  Kokkos::parallel_for(
      "so3lr_pack_reverse_forces",
      Kokkos::RangePolicy<>(0, ghosts * 3),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        const std::size_t node = ghost_indices(item);
        if (node < nodes)
          packed_forces(flat) = atomic_forces(node * 3 + component);
      });
}

void launch_unpack_reverse_forces_device(
    const Kokkos::View<double *> &packed_forces,
    const Kokkos::View<std::size_t *> &owner_indices,
    const Kokkos::View<double *> &atomic_forces) {
  const std::size_t owners = owner_indices.extent(0);
  if (atomic_forces.extent(0) % 3 != 0 ||
      packed_forces.extent(0) != owners * 3)
    throw std::runtime_error("SO3LR reverse-force unpack shape mismatch");
  const std::size_t nodes = atomic_forces.extent(0) / 3;
  Kokkos::parallel_for(
      "so3lr_unpack_reverse_forces",
      Kokkos::RangePolicy<>(0, owners * 3),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        const std::size_t node = owner_indices(item);
        if (node < nodes)
          Kokkos::atomic_add(&atomic_forces(node * 3 + component),
                             packed_forces(flat));
      });
}

std::size_t so3lr_reverse_force_buffer_bytes(std::size_t atoms) {
  return atoms * 3 * sizeof(double);
}

}  // namespace so3lr
