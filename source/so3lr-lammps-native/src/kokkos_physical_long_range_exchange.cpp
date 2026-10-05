#include "so3lr/kokkos_physical_long_range_exchange.hpp"

#include <stdexcept>

namespace so3lr {

void KokkosPhysicalLongRangeExchange::pack_charge_hirshfeld_device(
    const DoubleView &charges, const DoubleView &hirshfeld,
    const IndexView &selected, const DoubleView &packed) const {
  if (charges.extent(0) != hirshfeld.extent(0) ||
      packed.extent(0) != selected.extent(0) * 2)
    throw std::runtime_error("SO3LR LR output pack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_pack_lr_charge_hirshfeld",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        packed(i * 2) = charges(node);
        packed(i * 2 + 1) = hirshfeld(node);
      });
}

void KokkosPhysicalLongRangeExchange::unpack_charge_hirshfeld_overwrite_device(
    const DoubleView &packed, const IndexView &selected,
    const DoubleView &charges, const DoubleView &hirshfeld) const {
  if (charges.extent(0) != hirshfeld.extent(0) ||
      packed.extent(0) != selected.extent(0) * 2)
    throw std::runtime_error("SO3LR LR output unpack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_unpack_lr_charge_hirshfeld",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        charges(node) = packed(i * 2);
        hirshfeld(node) = packed(i * 2 + 1);
      });
}

void KokkosPhysicalLongRangeExchange::pack_reverse_fields_device(
    const PhysicalLongRangeWorkspace &workspace, const IndexView &selected,
    const DoubleView &packed) const {
  if (packed.extent(0) != selected.extent(0) * 3)
    throw std::runtime_error("SO3LR LR reverse pack shape mismatch");
  const auto energy = workspace.atomic_energy;
  const auto charge = workspace.charge_gradient;
  const auto hirshfeld = workspace.hirshfeld_gradient;
  Kokkos::parallel_for(
      "so3lr_pack_lr_reverse_fields",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        packed(i * 3) = energy(node);
        packed(i * 3 + 1) = charge(node);
        packed(i * 3 + 2) = hirshfeld(node);
      });
}

void KokkosPhysicalLongRangeExchange::zero_reverse_fields_device(
    const PhysicalLongRangeWorkspace &workspace,
    const IndexView &selected) const {
  const auto energy = workspace.atomic_energy;
  const auto charge = workspace.charge_gradient;
  const auto hirshfeld = workspace.hirshfeld_gradient;
  Kokkos::parallel_for(
      "so3lr_zero_lr_reverse_fields",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        energy(node) = 0.0;
        charge(node) = 0.0;
        hirshfeld(node) = 0.0;
      });
}

void KokkosPhysicalLongRangeExchange::unpack_reverse_fields_accumulate_device(
    const DoubleView &packed, const IndexView &selected,
    const DoubleView &atomic_energy, const DoubleView &charge_gradient,
    const DoubleView &hirshfeld_gradient) const {
  if (atomic_energy.extent(0) != charge_gradient.extent(0) ||
      atomic_energy.extent(0) != hirshfeld_gradient.extent(0) ||
      packed.extent(0) != selected.extent(0) * 3)
    throw std::runtime_error("SO3LR LR reverse unpack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_accumulate_lr_reverse_fields",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        // Live periodic-image graphs can request the same physical owner more
        // than once.  Atomic accumulation is therefore a correctness rule,
        // not merely an optimization detail.
        Kokkos::atomic_add(&atomic_energy(node), packed(i * 3));
        Kokkos::atomic_add(&charge_gradient(node), packed(i * 3 + 1));
        Kokkos::atomic_add(&hirshfeld_gradient(node), packed(i * 3 + 2));
      });
}

}  // namespace so3lr
