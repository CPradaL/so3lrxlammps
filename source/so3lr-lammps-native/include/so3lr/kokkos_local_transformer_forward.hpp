#pragma once

#include "so3lr/kokkos_feature_transformer_block.hpp"
#include "so3lr/kokkos_transformer_block0.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace so3lr {

// Workspace for a rank-local SR graph.  `nodes` is the rank's local+ghost
// node count and `local_edges` contains only directed edges whose receiver is
// owned by this rank.  Ghost outputs are intentionally incomplete until the
// caller performs owner-to-ghost exchange at a block boundary.
struct LocalTransformerForwardWorkspace {
  LocalTransformerForwardWorkspace(std::size_t nodes,
                                   std::size_t local_edges);
  TransformerBlock0DeviceWorkspace block0;
  FeatureTransformerDeviceWorkspace block1;
  FeatureTransformerDeviceWorkspace block2;
};

class KokkosLocalTransformerForward {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosLocalTransformerForward(const NativeModel &model);

  void launch_block0_device(
      const Int64View &atomic_numbers, const DoubleView &local_distances,
      const DoubleView &local_sh_vectors, const IndexView &local_senders,
      const IndexView &local_receivers,
      const LocalTransformerForwardWorkspace &workspace) const;

  // The caller must publish block-0 owner outputs to its ghost entries before
  // invoking this method.
  void launch_block1_device(
      const DoubleView &local_distances, const DoubleView &local_sh_vectors,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalTransformerForwardWorkspace &workspace) const;

  // The caller must publish block-1 owner outputs to its ghost entries before
  // invoking this method.
  void launch_block2_device(
      const DoubleView &local_distances, const DoubleView &local_sh_vectors,
      const IndexView &local_senders, const IndexView &local_receivers,
      const LocalTransformerForwardWorkspace &workspace) const;

  const DoubleView &block0_final_inv(
      const LocalTransformerForwardWorkspace &workspace) const;
  const DoubleView &block0_final_ev(
      const LocalTransformerForwardWorkspace &workspace) const;
  const DoubleView &block1_final_inv(
      const LocalTransformerForwardWorkspace &workspace) const;
  const DoubleView &block1_final_ev(
      const LocalTransformerForwardWorkspace &workspace) const;
  const DoubleView &block2_final_inv(
      const LocalTransformerForwardWorkspace &workspace) const;
  const DoubleView &block2_final_ev(
      const LocalTransformerForwardWorkspace &workspace) const;

  bool explicit_block_boundaries() const { return true; }
  bool receiver_owned_edge_contract() const { return true; }
  bool device_contract_verified() const { return device_contract_verified_; }

 private:
  KokkosTransformerBlock0 block0_;
  KokkosFeatureTransformerBlock block1_;
  KokkosFeatureTransformerBlock block2_;
  bool device_contract_verified_ = false;
};

}  // namespace so3lr
