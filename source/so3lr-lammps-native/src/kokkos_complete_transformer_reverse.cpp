#include "so3lr/kokkos_complete_transformer_reverse.hpp"

#include <Kokkos_Timer.hpp>

#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

std::size_t complete_workspace_bytes(std::size_t nodes, std::size_t edges,
                                     std::size_t persistent_bytes) {
  // Dev_19 complete attention-input audit (2168 doubles/node and 1052
  // doubles/edge), plus the post-attention forward (1340 doubles/node) and
  // reverse (1312 doubles/node) checkpoints.  External final seeds replace
  // dev_19's attention-boundary seeds, so no additional seed term is needed.
  constexpr std::size_t node_doubles = 2168 + 1340 + 1312;
  constexpr std::size_t edge_doubles = 1052;
  return persistent_bytes + nodes * node_doubles * sizeof(double) +
         edges * edge_doubles * sizeof(double) +
         2 * edges * sizeof(std::size_t);
}

}  // namespace

CompleteTransformerReverseWorkspace::CompleteTransformerReverseWorkspace(
    std::size_t nodes, std::size_t edges, const ArchDims &dims)
    : attention(nodes, edges, dims),
      post_forward(nodes, dims),
      post_reverse(nodes, dims) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR complete reverse workspace is empty");
}


CompleteTransformerReverseWorkspace::CompleteTransformerReverseWorkspace(
    std::size_t nodes, std::size_t edges,
    const CompleteTransformerReverseWorkspace &shared, const ArchDims &dims)
    : attention(nodes, edges, shared.attention, dims),
      post_forward(nodes, dims),
      post_reverse(shared.post_reverse) {}

KokkosCompleteTransformerReverseBlock::
    KokkosCompleteTransformerReverseBlock(const NativeModel &model,
                                           std::size_t block_index)
    : attention_(model, block_index),
      post_(model, block_index),
      block_index_(block_index) {
  persistent_device_bytes_ = attention_.persistent_device_bytes() +
                             post_.persistent_device_bytes();
  shared_kokkos_stream_ = attention_.shared_kokkos_stream() &&
                          post_.shared_kokkos_stream();
  checkpoint_contract_verified_ =
      attention_.block_index() == block_index_ &&
      post_.block_index() == block_index_ &&
      attention_.checkpoint_contract_verified() &&
      post_.checkpoint_contract_verified() && shared_kokkos_stream_;
  if (!checkpoint_contract_verified_)
    throw std::runtime_error("SO3LR complete transformer reverse contract failed");
}

void KokkosCompleteTransformerReverseBlock::launch_forward_device(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const CompleteTransformerReverseWorkspace &workspace) const {
  attention_.launch_forward_device(inv_features, ev_features, distances,
                                   sh_vectors, senders, receivers,
                                   workspace.attention);
  post_.launch_forward_device(
      inv_features, ev_features,
      workspace.attention.forward.invariant_update,
      workspace.attention.forward.equivariant_update,
      workspace.post_forward);
}

void KokkosCompleteTransformerReverseBlock::launch_reverse_device(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const DoubleView &grad_final_inv, const DoubleView &grad_final_ev,
    const CompleteTransformerReverseWorkspace &workspace) const {
  post_.launch_reverse_device(grad_final_inv, grad_final_ev,
                              workspace.post_forward,
                              workspace.post_reverse);
  attention_.launch_reverse_device(
      inv_features, ev_features, distances, sh_vectors, senders, receivers,
      workspace.post_reverse.grad_attention_inv,
      workspace.post_reverse.grad_attention_ev, workspace.attention);
}

CompleteTransformerReverseBenchmark
KokkosCompleteTransformerReverseBlock::benchmark(
    std::size_t nodes, std::size_t edges,
    std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR complete reverse benchmark invalid");
  const ArchDims &dims = attention_.dims();
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t equivariant_width = dims.equivariant_width;
  DoubleView inv("so3lr_dev20_bench_inv", nodes * feature_width);
  DoubleView ev("so3lr_dev20_bench_ev", nodes * equivariant_width);
  DoubleView distances("so3lr_dev20_bench_distances", edges);
  DoubleView sh("so3lr_dev20_bench_sh", edges * equivariant_width);
  IndexView senders("so3lr_dev20_bench_senders", edges);
  IndexView receivers("so3lr_dev20_bench_receivers", edges);
  DoubleView grad_final_inv("so3lr_dev20_bench_grad_final_inv",
                            nodes * feature_width);
  DoubleView grad_final_ev("so3lr_dev20_bench_grad_final_ev",
                           nodes * equivariant_width);
  Kokkos::parallel_for(
      "so3lr_dev20_bench_inv", Kokkos::RangePolicy<>(0, inv.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        inv(i) = (static_cast<double>(i % 251) - 125.0) / 137.0;
        grad_final_inv(i) =
            (static_cast<double>(i % 113) - 56.0) / 127.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev20_bench_ev", Kokkos::RangePolicy<>(0, ev.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        ev(i) = (static_cast<double>(i % 83) - 41.0) / 97.0;
        grad_final_ev(i) =
            (static_cast<double>(i % 47) - 23.0) / 59.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev20_bench_graph", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        receivers(edge) = edge % nodes;
        senders(edge) = (edge * 17 + 11) % nodes;
        distances(edge) =
            0.15 + 4.29 * static_cast<double>(edge % 8192) / 8191.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev20_bench_sh", Kokkos::RangePolicy<>(0, sh.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });

  CompleteTransformerReverseWorkspace workspace(nodes, edges, dims);
  for (int warmup = 0; warmup < 2; ++warmup) {
    launch_forward_device(inv, ev, distances, sh, senders, receivers,
                          workspace);
    launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                          grad_final_inv, grad_final_ev, workspace);
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                          grad_final_inv, grad_final_ev, workspace);
  Kokkos::fence();
  const double reverse_ms = timer.seconds() * 1000.0 /
                            static_cast<double>(repetitions);
  timer.reset();
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    launch_forward_device(inv, ev, distances, sh, senders, receivers,
                          workspace);
    launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                          grad_final_inv, grad_final_ev, workspace);
  }
  Kokkos::fence();
  const double forward_reverse_ms = timer.seconds() * 1000.0 /
                                    static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto output_inv = workspace.attention.grad_inv_features;
  const auto output_ev = workspace.attention.grad_ev_features;
  const auto output_distance = workspace.attention.grad_distances;
  const auto output_sh = workspace.attention.attention_reverse.grad_sh;
  Kokkos::parallel_reduce(
      "so3lr_dev20_bench_checksum", Kokkos::RangePolicy<>(0, 4),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += output_inv(17);
        if (which == 1) update += output_ev(11);
        if (which == 2) update += output_distance(7);
        if (which == 3) update += output_sh(29);
      },
      checksum);
  Kokkos::fence();

  CompleteTransformerReverseBenchmark result;
  result.block_index = block_index_;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes =
      complete_workspace_bytes(nodes, edges, persistent_device_bytes_);
  result.host_boundary_bytes_per_iteration = 0;
  result.reverse_milliseconds = reverse_ms;
  result.forward_reverse_milliseconds = forward_reverse_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (reverse_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
