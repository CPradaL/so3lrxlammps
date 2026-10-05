#include "so3lr/kokkos_node_feature_exchange.hpp"

#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

using Exchange = KokkosNodeFeatureExchange;

void validate(const Exchange::DoubleView &invariant,
              const Exchange::DoubleView &equivariant,
              const Exchange::IndexView &node_indices,
              const Exchange::DoubleView &packed,
              std::size_t invariant_width, std::size_t equivariant_width,
              const char *operation) {
  if (invariant.extent(0) % invariant_width != 0 ||
      equivariant.extent(0) % equivariant_width != 0)
    throw std::runtime_error(std::string(operation) +
                             ": feature shape is not channel aligned");
  const std::size_t inv_nodes = invariant.extent(0) / invariant_width;
  const std::size_t ev_nodes = equivariant.extent(0) / equivariant_width;
  if (inv_nodes != ev_nodes)
    throw std::runtime_error(std::string(operation) +
                             ": invariant/equivariant node mismatch");
  if (packed.extent(0) !=
      node_indices.extent(0) * (invariant_width + equivariant_width))
    throw std::runtime_error(std::string(operation) +
                             ": packed buffer size mismatch");
}

}  // namespace

NodeFeatureExchangeWorkspace::NodeFeatureExchangeWorkspace(
    std::size_t send_nodes, std::size_t receive_nodes, std::size_t row_width)
    : send_buffer("so3lr_feature_exchange_send", send_nodes * row_width),
      receive_buffer("so3lr_feature_exchange_receive",
                     receive_nodes * row_width) {
  if (send_nodes == 0 || receive_nodes == 0)
    throw std::runtime_error("SO3LR feature exchange workspace is empty");
}

void KokkosNodeFeatureExchange::pack_device(
    const DoubleView &invariant, const DoubleView &equivariant,
    const IndexView &node_indices, const DoubleView &packed) const {
  const std::size_t invariant_width = invariant_channels_;
  const std::size_t equivariant_width = equivariant_channels_;
  const std::size_t packed_width = invariant_width + equivariant_width;
  validate(invariant, equivariant, node_indices, packed,
           invariant_width, equivariant_width, "SO3LR feature pack");
  const std::size_t nodes = invariant.extent(0) / invariant_width;
  Kokkos::parallel_for(
      "so3lr_pack_node_features",
      Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t linear) {
        const std::size_t item = linear / packed_width;
        const std::size_t channel = linear % packed_width;
        const std::size_t node = node_indices(item);
        if (node >= nodes) return;
        packed(linear) = channel < invariant_width
                             ? invariant(node * invariant_width + channel)
                             : equivariant(node * equivariant_width +
                                           channel - invariant_width);
      });
}

void KokkosNodeFeatureExchange::unpack_overwrite_device(
    const DoubleView &packed, const IndexView &node_indices,
    const DoubleView &invariant, const DoubleView &equivariant) const {
  const std::size_t invariant_width = invariant_channels_;
  const std::size_t equivariant_width = equivariant_channels_;
  const std::size_t packed_width = invariant_width + equivariant_width;
  validate(invariant, equivariant, node_indices, packed,
           invariant_width, equivariant_width, "SO3LR feature overwrite");
  const std::size_t nodes = invariant.extent(0) / invariant_width;
  Kokkos::parallel_for(
      "so3lr_unpack_node_features",
      Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t linear) {
        const std::size_t item = linear / packed_width;
        const std::size_t channel = linear % packed_width;
        const std::size_t node = node_indices(item);
        if (node >= nodes) return;
        if (channel < invariant_width)
          invariant(node * invariant_width + channel) = packed(linear);
        else
          equivariant(node * equivariant_width + channel - invariant_width) =
              packed(linear);
      });
}

void KokkosNodeFeatureExchange::unpack_accumulate_device(
    const DoubleView &packed, const IndexView &node_indices,
    const DoubleView &invariant, const DoubleView &equivariant) const {
  const std::size_t invariant_width = invariant_channels_;
  const std::size_t equivariant_width = equivariant_channels_;
  const std::size_t packed_width = invariant_width + equivariant_width;
  validate(invariant, equivariant, node_indices, packed,
           invariant_width, equivariant_width, "SO3LR adjoint accumulation");
  const std::size_t nodes = invariant.extent(0) / invariant_width;
  Kokkos::parallel_for(
      "so3lr_accumulate_node_adjoints",
      Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t linear) {
        const std::size_t item = linear / packed_width;
        const std::size_t channel = linear % packed_width;
        const std::size_t node = node_indices(item);
        if (node >= nodes) return;
        if (channel < invariant_width)
          Kokkos::atomic_add(&invariant(node * invariant_width + channel),
                             packed(linear));
        else
          Kokkos::atomic_add(
              &equivariant(node * equivariant_width +
                           channel - invariant_width),
              packed(linear));
      });
}

}  // namespace so3lr
