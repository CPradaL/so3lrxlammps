#pragma once

#include "so3lr/kokkos_so3lr_evaluator.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace so3lr {

// Thin ownership layer above the self-contained evaluator.  Owned atom
// indices seed learned atomic energies.  LR half-pairs are supplied already
// partitioned by the caller.  The resulting force view contains both owned
// and ghost contributions and is therefore ready for reverse communication.
class KokkosSo3lrOwnedEvaluator {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosSo3lrOwnedEvaluator(const NativeModel &model)
      : evaluator_(model) {}

  void launch_rank_device(
      const Int64View &atomic_numbers,
      const DoubleView &sr_edge_vectors,
      const IndexView &sr_senders,
      const IndexView &sr_receivers,
      const DoubleView &lr_pair_vectors,
      const IndexView &lr_senders,
      const IndexView &lr_receivers,
      const IndexView &owned_nodes,
      const So3lrEvaluatorWorkspace &workspace) const;

  const KokkosSo3lrEvaluator &evaluator() const { return evaluator_; }

 private:
  KokkosSo3lrEvaluator evaluator_;
};

// These two routines mirror the data contract needed by LAMMPS reverse
// communication without depending on LAMMPS headers.  The adapter will map
// its ghost and owner lists to these device views.
void launch_pack_reverse_forces_device(
    const Kokkos::View<double *> &atomic_forces,
    const Kokkos::View<std::size_t *> &ghost_indices,
    const Kokkos::View<double *> &packed_forces);

void launch_unpack_reverse_forces_device(
    const Kokkos::View<double *> &packed_forces,
    const Kokkos::View<std::size_t *> &owner_indices,
    const Kokkos::View<double *> &atomic_forces);

std::size_t so3lr_reverse_force_buffer_bytes(std::size_t atoms);

}  // namespace so3lr
