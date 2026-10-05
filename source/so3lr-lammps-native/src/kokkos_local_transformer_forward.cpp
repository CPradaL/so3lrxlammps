#include "so3lr/kokkos_local_transformer_forward.hpp"

#include <stdexcept>

namespace so3lr {

LocalTransformerForwardWorkspace::LocalTransformerForwardWorkspace(
    std::size_t nodes, std::size_t local_edges)
    : block0(nodes, local_edges),
      block1(nodes, local_edges),
      block2(nodes, local_edges) {
  if (nodes == 0 || local_edges == 0)
    throw std::runtime_error("SO3LR local transformer workspace is empty");
}

KokkosLocalTransformerForward::KokkosLocalTransformerForward(
    const NativeModel &model)
    : block0_(model), block1_(model, 1), block2_(model, 2) {
  device_contract_verified_ = block0_.device_contract_verified() &&
                              block1_.device_contract_verified() &&
                              block2_.device_contract_verified() &&
                              block1_.block_index() == 1 &&
                              block2_.block_index() == 2;
  if (!device_contract_verified_)
    throw std::runtime_error(
        "SO3LR local transformer device contract failed");
}

void KokkosLocalTransformerForward::launch_block0_device(
    const Int64View &atomic_numbers, const DoubleView &local_distances,
    const DoubleView &local_sh_vectors, const IndexView &local_senders,
    const IndexView &local_receivers,
    const LocalTransformerForwardWorkspace &workspace) const {
  block0_.launch_device(atomic_numbers, local_distances, local_sh_vectors,
                        local_senders, local_receivers, workspace.block0);
}

void KokkosLocalTransformerForward::launch_block1_device(
    const DoubleView &local_distances, const DoubleView &local_sh_vectors,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalTransformerForwardWorkspace &workspace) const {
  block1_.launch_device(workspace.block0.post_attention.final_inv,
                        workspace.block0.post_attention.final_ev,
                        local_distances, local_sh_vectors, local_senders,
                        local_receivers, workspace.block1);
}

void KokkosLocalTransformerForward::launch_block2_device(
    const DoubleView &local_distances, const DoubleView &local_sh_vectors,
    const IndexView &local_senders, const IndexView &local_receivers,
    const LocalTransformerForwardWorkspace &workspace) const {
  block2_.launch_device(workspace.block1.post_attention.final_inv,
                        workspace.block1.post_attention.final_ev,
                        local_distances, local_sh_vectors, local_senders,
                        local_receivers, workspace.block2);
}

const KokkosLocalTransformerForward::DoubleView &
KokkosLocalTransformerForward::block0_final_inv(
    const LocalTransformerForwardWorkspace &workspace) const {
  return workspace.block0.post_attention.final_inv;
}

const KokkosLocalTransformerForward::DoubleView &
KokkosLocalTransformerForward::block0_final_ev(
    const LocalTransformerForwardWorkspace &workspace) const {
  return workspace.block0.post_attention.final_ev;
}

const KokkosLocalTransformerForward::DoubleView &
KokkosLocalTransformerForward::block1_final_inv(
    const LocalTransformerForwardWorkspace &workspace) const {
  return workspace.block1.post_attention.final_inv;
}

const KokkosLocalTransformerForward::DoubleView &
KokkosLocalTransformerForward::block1_final_ev(
    const LocalTransformerForwardWorkspace &workspace) const {
  return workspace.block1.post_attention.final_ev;
}

const KokkosLocalTransformerForward::DoubleView &
KokkosLocalTransformerForward::block2_final_inv(
    const LocalTransformerForwardWorkspace &workspace) const {
  return workspace.block2.post_attention.final_inv;
}

const KokkosLocalTransformerForward::DoubleView &
KokkosLocalTransformerForward::block2_final_ev(
    const LocalTransformerForwardWorkspace &workspace) const {
  return workspace.block2.post_attention.final_ev;
}

}  // namespace so3lr
