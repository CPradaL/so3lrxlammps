#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace so3lr {

// Communication payload at an SO3LR transformer-block boundary.  The
// ordering deliberately matches the model: invariant channels first, then
// equivariant channels.  MPI is kept outside this class so the same kernels
// can later be called by either a LAMMPS communicator or a standalone driver.
class KokkosNodeFeatureExchange {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  // SO3LR v1 row layout, used by the default constructor and the v1 tests.
  static constexpr std::size_t invariant_width = 128;
  static constexpr std::size_t equivariant_width = 24;
  static constexpr std::size_t packed_width =
      invariant_width + equivariant_width;

  KokkosNodeFeatureExchange() = default;
  // Any architecture: F invariant and E equivariant channels per node.
  KokkosNodeFeatureExchange(std::size_t invariant_channels,
                            std::size_t equivariant_channels)
      : invariant_channels_(invariant_channels),
        equivariant_channels_(equivariant_channels) {}

  std::size_t row_width() const {
    return invariant_channels_ + equivariant_channels_;
  }

  static std::size_t packed_values(std::size_t nodes) {
    return nodes * packed_width;
  }
  static std::size_t packed_bytes(std::size_t nodes) {
    return packed_values(nodes) * sizeof(double);
  }

  // Pack selected nodes into a contiguous device buffer suitable for direct
  // CUDA-aware MPI transport.
  void pack_device(const DoubleView &invariant,
                   const DoubleView &equivariant,
                   const IndexView &node_indices,
                   const DoubleView &packed) const;

  // Forward owner -> ghost operation: replace ghost values with the values
  // computed by their owning rank.
  void unpack_overwrite_device(const DoubleView &packed,
                               const IndexView &node_indices,
                               const DoubleView &invariant,
                               const DoubleView &equivariant) const;

  // Reverse ghost -> owner operation: add remote adjoint contributions to the
  // owner's local contribution.  Atomic updates make duplicate indices safe.
  void unpack_accumulate_device(const DoubleView &packed,
                                const IndexView &node_indices,
                                const DoubleView &invariant,
                                const DoubleView &equivariant) const;

 private:
  std::size_t invariant_channels_ = invariant_width;
  std::size_t equivariant_channels_ = equivariant_width;
};

struct NodeFeatureExchangeWorkspace {
  using DoubleView = Kokkos::View<double *>;

  NodeFeatureExchangeWorkspace(
      std::size_t send_nodes, std::size_t receive_nodes,
      std::size_t row_width = KokkosNodeFeatureExchange::packed_width);
  DoubleView send_buffer;
  DoubleView receive_buffer;
};

}  // namespace so3lr
