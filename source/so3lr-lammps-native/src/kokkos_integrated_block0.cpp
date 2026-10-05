#include "so3lr/kokkos_integrated_block0.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

// Every released SO3LR model uses 32-channel attention heads (the capability
// table enforces it). Keeping it a compile-time constant lets the per-head
// channel loops unroll; the head count and every other width come from the
// architecture descriptor at run time.
constexpr std::size_t head_width = 32;

void check_cublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS failure in ") + operation +
                             ": status=" +
                             std::to_string(static_cast<int>(status)));
}

int checked_int(std::size_t value, const char *name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(std::string("cuBLAS dimension exceeds int: ") +
                             name);
  return static_cast<int>(value);
}

template <class View, class Reader>
void fill_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy_host(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

void require_tensor(const TensorRecord &tensor,
                    const std::vector<std::size_t> &shape,
                    const std::string &dtype = "float64") {
  if (tensor.dtype != dtype || tensor.shape != shape)
    throw std::runtime_error("SO3LR integrated tensor contract mismatch: " +
                             tensor.state_key);
}

DoubleView load_double(const NativeModel &model, const std::string &key,
                       const std::vector<std::size_t> &shape) {
  const auto &tensor = model.tensor(key);
  require_tensor(tensor, shape);
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  DoubleView result(key + "_integrated", count);
  fill_view(result, count,
            [&](std::size_t i) { return model.float64(tensor, i); });
  return result;
}

IntegratedFilterParameters load_filter(const NativeModel &model,
                                       const std::string &prefix,
                                       const ArchDims &dims) {
  const std::size_t f = dims.invariant_width;
  const std::size_t r = dims.rbf_width;
  const std::size_t h = dims.ev_filter_hidden;
  const std::size_t degrees = dims.ev_heads;
  IntegratedFilterParameters result;
  result.rbf_weight_0 =
      load_double(model, prefix + ".mlp_rbf.0.weight", {f, r});
  result.rbf_bias_0 =
      load_double(model, prefix + ".mlp_rbf.0.bias", {f});
  result.rbf_weight_1 = load_double(
      model, prefix + ".mlp_rbf.mlp_rbf_layer_1.0.weight", {f, f});
  result.rbf_bias_1 = load_double(
      model, prefix + ".mlp_rbf.mlp_rbf_layer_1.0.bias", {f});
  result.ev_weight_0 =
      load_double(model, prefix + ".mlp_ev.0.weight", {h, degrees});
  result.ev_bias_0 = load_double(model, prefix + ".mlp_ev.0.bias", {h});
  result.ev_weight_1 = load_double(
      model, prefix + ".mlp_ev.mlp_ev_layer_1.0.weight", {f, h});
  result.ev_bias_1 = load_double(
      model, prefix + ".mlp_ev.mlp_ev_layer_1.0.bias", {f});
  return result;
}

std::string transformer_prefix(const NativeModel &model,
                               std::size_t block_index) {
  const auto blocks = static_cast<std::size_t>(
      model.architecture_number("interaction_blocks"));
  if (block_index >= blocks)
    throw std::runtime_error("SO3LR transformer block index out of range");
  return "model.euclidean_transformers." + std::to_string(block_index);
}

double attention_normalization(const NativeModel &model,
                               std::size_t block_index) {
  // Resolved once by the architecture descriptor. Previously this walked the
  // manifest's PyTorch module tree, which a flax-exported model does not have.
  (void)block_index;
  return model.arch().attention_norm_value;
}

void launch_inputs(const Int64View &atomic_numbers,
                   const DoubleView &distances, const DoubleView &embedding,
                   const DoubleView &cutoff, const DoubleView &rbf,
                   const DoubleView &weight, const DoubleView &b,
                   const Int64View &k, const Int64View &k_reverse,
                   double gamma, double rmax, const ArchDims &dims) {
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t rbf_width = dims.rbf_width;
  const std::size_t atoms = atomic_numbers.extent(0);
  const std::size_t edges = distances.extent(0);
  Kokkos::parallel_for(
      "so3lr_integrated_embedding",
      Kokkos::RangePolicy<>(0, atoms * feature_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::int64_t z = atomic_numbers(flat / feature_width);
        const std::size_t channel = flat % feature_width;
        embedding(flat) =
            (z > 0 && z <= 118)
                ? weight(channel * 118 + static_cast<std::size_t>(z - 1))
                : 0.0;
      });
  Kokkos::parallel_for(
      "so3lr_integrated_cutoff_rbf",
      Kokkos::RangePolicy<>(0, edges * rbf_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / rbf_width;
        const std::size_t channel = flat % rbf_width;
        const double distance = distances(edge);
        if (channel == 0) {
          if (distance < rmax) {
            const double x = distance / rmax;
            const double x2 = x * x;
            const double x3 = x2 * x;
            cutoff(edge) = 1.0 - 10.0 * x3 + 15.0 * x3 * x -
                           6.0 * x3 * x2;
          } else {
            cutoff(edge) = 0.0;
          }
        }
        double x = Kokkos::exp(-gamma * distance);
        if (x < 1.0e-6) x = 1.0e-6;
        if (x > 1.0 - 1.0e-6) x = 1.0 - 1.0e-6;
        const double log_poly =
            b(channel) + static_cast<double>(k(channel)) * Kokkos::log(x) +
            static_cast<double>(k_reverse(channel)) * Kokkos::log(1.0 - x);
        rbf(flat) = Kokkos::exp(log_poly);
      });
}

void launch_edge_ev_invariants(
    const DoubleView &ev_features, const IndexView &senders,
    const IndexView &receivers, const DoubleView &cg_rep,
    const IndexView &degree_repeats, const IndexView &degree_offsets,
    std::size_t edges, const DoubleView &edge_ev_invariants,
    const ArchDims &dims) {
  const std::size_t heads = dims.ev_heads;
  const std::size_t equivariant_width = dims.equivariant_width;
  Kokkos::parallel_for(
      "so3lr_edge_l0_contraction",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / heads;
        const std::size_t degree = flat % heads;
        const std::size_t sender = senders(edge);
        const std::size_t receiver = receivers(edge);
        double value = 0.0;
        for (std::size_t local = 0; local < degree_repeats(degree); ++local) {
          const std::size_t channel = degree_offsets(degree) + local;
          const double difference =
              ev_features(sender * equivariant_width + channel) -
              ev_features(receiver * equivariant_width + channel);
          value += difference * difference * cg_rep(channel);
        }
        edge_ev_invariants(flat) = value;
      });
}

void linear(cublasHandle_t handle, const DoubleView &input,
            std::size_t rows, std::size_t input_width,
            std::size_t output_width, const DoubleView &weight,
            const DoubleView &output, double beta) {
  const int m = checked_int(output_width, "output_width");
  const int n = checked_int(rows, "rows");
  const int k = checked_int(input_width, "input_width");
  constexpr double alpha = 1.0;
  check_cublas(
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  weight.data(), k, input.data(), k, &beta, output.data(), m),
      "cublasDgemm(filter)");
}

void bias_silu(const DoubleView &values, const DoubleView &bias,
               std::size_t rows, std::size_t width, const char *label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, rows * width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const double value = values(flat) + bias(flat % width);
        values(flat) = value / (1.0 + Kokkos::exp(-value));
      });
}

void bias_add(const DoubleView &values, const DoubleView &bias,
              std::size_t rows, std::size_t width, const char *label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, rows * width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        values(flat) += bias(flat % width);
      });
}

void launch_filter(cublasHandle_t handle,
                   const IntegratedFilterParameters &parameters,
                   const DoubleView &rbf, const DoubleView &ev,
                   std::size_t edges, const DoubleView &rbf_hidden,
                   const DoubleView &ev_hidden, const DoubleView &output,
                   bool invariant, const ArchDims &dims) {
  const std::size_t f = dims.invariant_width;
  const std::size_t r = dims.rbf_width;
  const std::size_t h = dims.ev_filter_hidden;
  const std::size_t degrees = dims.ev_heads;
  linear(handle, rbf, edges, r, f, parameters.rbf_weight_0, rbf_hidden,
         0.0);
  bias_silu(rbf_hidden, parameters.rbf_bias_0, edges, f,
            invariant ? "so3lr_pipe_inv_rbf_silu"
                      : "so3lr_pipe_ev_rbf_silu");
  linear(handle, rbf_hidden, edges, f, f, parameters.rbf_weight_1,
         output, 0.0);
  bias_add(output, parameters.rbf_bias_1, edges, f,
           invariant ? "so3lr_pipe_inv_rbf_bias"
                     : "so3lr_pipe_ev_rbf_bias");
  linear(handle, ev, edges, degrees, h, parameters.ev_weight_0, ev_hidden,
         0.0);
  bias_silu(ev_hidden, parameters.ev_bias_0, edges, h,
            invariant ? "so3lr_pipe_inv_ev_silu" : "so3lr_pipe_ev_ev_silu");
  linear(handle, ev_hidden, edges, h, f, parameters.ev_weight_1, output,
         1.0);
  bias_add(output, parameters.ev_bias_1, edges, f,
           invariant ? "so3lr_pipe_inv_ev_bias" : "so3lr_pipe_ev_ev_bias");
}

void project(cublasHandle_t handle, const DoubleView &features,
             std::size_t nodes, const DoubleView &weight,
             const DoubleView &output, std::size_t heads,
             std::size_t feature_width) {
  const int m = checked_int(head_width, "head_width");
  const int n = checked_int(nodes, "nodes");
  const int stride = checked_int(feature_width, "feature_width");
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  for (std::size_t head = 0; head < heads; ++head) {
    check_cublas(
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &alpha,
                    weight.data() + head * head_width * head_width, m,
                    features.data() + head * head_width, stride, &beta,
                    output.data() + head * head_width, stride),
        "cublasDgemm(Q/K/V)");
  }
}

void launch_qkv(cublasHandle_t handle, const DoubleView &embedding,
                std::size_t nodes,
                const std::array<DoubleView, 5> &weights,
                const std::array<DoubleView, 5> &outputs,
                const ArchDims &dims) {
  // {q_inv, k_inv, v_inv} split into H1 heads, {q_ev, k_ev} into H2 = one
  // head per degree slot.
  for (std::size_t i = 0; i < weights.size(); ++i)
    project(handle, embedding, nodes, weights[i], outputs[i],
            i < 3 ? dims.inv_heads : dims.ev_heads, dims.invariant_width);
}

// Optional qk_norm: functional_rms_norm over the last axis of q1, k1, q2, k2,
// i.e. per (node, head) over the head channels. The reference applies it after
// gathering to edges; a per-row normalisation commutes with a gather, so doing
// it on the node-level projections is exact and leaves the fused attention
// kernel untouched. Normalises in place and records 1/rms for the adjoint.
// Normalise each (node, head) row of one projection in place and record 1/rms
// in column `slot` of inverse_rms (layout: row * 4 + slot).
void qk_rms_norm_one(const DoubleView &target, std::size_t nodes,
                     std::size_t slot, const DoubleView &inverse_rms,
                     std::size_t heads) {
  Kokkos::parallel_for(
      "so3lr_qk_rms_norm", Kokkos::RangePolicy<>(0, nodes * heads),
      KOKKOS_LAMBDA(const std::size_t row) {
        const std::size_t offset = row * head_width;
        double mean_square = 0.0;
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          const double value = target(offset + channel);
          mean_square += value * value;
        }
        mean_square /= static_cast<double>(head_width);
        const double r = 1.0 / Kokkos::sqrt(mean_square + 1.0e-6);
        for (std::size_t channel = 0; channel < head_width; ++channel)
          target(offset + channel) *= r;
        inverse_rms(row * 4 + slot) = r;
      });
}

// Optional qk_norm: functional_rms_norm over the last axis of q1, k1, q2, k2,
// i.e. per (node, head) over the head channels. The reference applies it after
// gathering to edges; a per-row normalisation commutes with a gather, so doing
// it on the node-level projections is exact and leaves the fused attention
// kernel untouched. qkv = {q_inv, k_inv, v_inv, q_ev, k_ev}; v is not touched.
void launch_qk_rms_norm(const std::array<DoubleView, 5> &qkv, std::size_t nodes,
                        const DoubleView &inverse_rms, const ArchDims &dims) {
  // The attention kernels require H1 == H2, so one row stride serves all four.
  const std::size_t heads = dims.inv_heads;
  qk_rms_norm_one(qkv[0], nodes, 0, inverse_rms, heads);
  qk_rms_norm_one(qkv[1], nodes, 1, inverse_rms, heads);
  qk_rms_norm_one(qkv[3], nodes, 2, inverse_rms, heads);
  qk_rms_norm_one(qkv[4], nodes, 3, inverse_rms, heads);
}

void launch_attention(const std::array<DoubleView, 5> &qkv,
                      const DoubleView &filter_inv,
                      const DoubleView &filter_ev, const DoubleView &sh,
                      const DoubleView &cutoff, const IndexView &senders,
                      const IndexView &receivers,
                      const IndexView &degree_repeats,
                      const IndexView &degree_offsets, std::size_t edges,
                      double norm_inv, double norm_ev,
                      const DoubleView &d_inv, const DoubleView &d_ev,
                      const ArchDims &dims) {
  const std::size_t heads = dims.inv_heads;
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t equivariant_width = dims.equivariant_width;
  Kokkos::deep_copy(d_inv, 0.0);
  Kokkos::deep_copy(d_ev, 0.0);
  const auto q_inv = qkv[0];
  const auto k_inv = qkv[1];
  const auto v_inv = qkv[2];
  const auto q_ev = qkv[3];
  const auto k_ev = qkv[4];
  Kokkos::parallel_for(
      "so3lr_integrated_attention_scatter",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / heads;
        const std::size_t head = flat % heads;
        const std::size_t sender = senders(edge);
        const std::size_t receiver = receivers(edge);
        const std::size_t head_offset = head * head_width;
        const std::size_t edge_offset = edge * feature_width + head_offset;
        const std::size_t q_offset = receiver * feature_width + head_offset;
        const std::size_t kv_offset = sender * feature_width + head_offset;
        double alpha_inv = 0.0;
        double alpha_ev = 0.0;
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          alpha_inv += q_inv(q_offset + channel) *
                       k_inv(kv_offset + channel) *
                       filter_inv(edge_offset + channel);
          alpha_ev += q_ev(q_offset + channel) * k_ev(kv_offset + channel) *
                      filter_ev(edge_offset + channel);
        }
        alpha_inv /= norm_inv;
        alpha_ev /= norm_ev;
        const double weight = cutoff(edge);
        for (std::size_t channel = 0; channel < head_width; ++channel)
          Kokkos::atomic_add(
              &d_inv(receiver * feature_width + head_offset + channel),
              weight * alpha_inv * v_inv(kv_offset + channel));
        for (std::size_t local = 0; local < degree_repeats(head); ++local) {
          const std::size_t channel = degree_offsets(head) + local;
          Kokkos::atomic_add(
              &d_ev(receiver * equivariant_width + channel),
              weight * alpha_ev * sh(edge * equivariant_width + channel));
        }
      });
}

void launch_pipeline(
    cublasHandle_t handle, const Int64View &z, const DoubleView &distances,
    const DoubleView &ev_invariants, const DoubleView &sh,
    const IndexView &senders, const IndexView &receivers,
    const DoubleView &embedding, const DoubleView &cutoff,
    const DoubleView &rbf, const DoubleView &embedding_weight,
    const DoubleView &bernstein_b, const Int64View &bernstein_k,
    const Int64View &bernstein_k_reverse, double gamma, double rmax,
    const IntegratedFilterParameters &filter_inv_params,
    const IntegratedFilterParameters &filter_ev_params,
    const DoubleView &rbf_hidden, const DoubleView &ev_hidden,
    const DoubleView &filter_inv, const DoubleView &filter_ev,
    const std::array<DoubleView, 5> &qkv_weights,
    const std::array<DoubleView, 5> &qkv, const IndexView &degree_repeats,
    const IndexView &degree_offsets, double norm_inv, double norm_ev,
    const DoubleView &d_inv, const DoubleView &d_ev, bool qk_norm,
    const DoubleView &qk_inverse_rms, const ArchDims &dims) {
  const std::size_t nodes = z.extent(0);
  const std::size_t edges = distances.extent(0);
  launch_inputs(z, distances, embedding, cutoff, rbf, embedding_weight,
                bernstein_b, bernstein_k, bernstein_k_reverse, gamma, rmax,
                dims);
  launch_filter(handle, filter_inv_params, rbf, ev_invariants, edges,
                rbf_hidden, ev_hidden, filter_inv, true, dims);
  launch_filter(handle, filter_ev_params, rbf, ev_invariants, edges,
                rbf_hidden, ev_hidden, filter_ev, false, dims);
  launch_qkv(handle, embedding, nodes, qkv_weights, qkv, dims);
  if (qk_norm) launch_qk_rms_norm(qkv, nodes, qk_inverse_rms, dims);
  launch_attention(qkv, filter_inv, filter_ev, sh, cutoff, senders, receivers,
                   degree_repeats, degree_offsets, edges, norm_inv, norm_ev,
                   d_inv, d_ev, dims);
}

std::size_t dynamic_bytes(std::size_t nodes, std::size_t edges,
                          const ArchDims &dims) {
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t equivariant_width = dims.equivariant_width;
  const std::size_t rbf_width = dims.rbf_width;
  const std::size_t heads = dims.ev_heads;
  const std::size_t node_bytes =
      nodes * (sizeof(std::int64_t) +
               (feature_width + 5 * feature_width + feature_width +
                equivariant_width) * sizeof(double));
  const std::size_t edge_doubles =
      edges * (1 + rbf_width + heads + equivariant_width + 1 +
               feature_width + dims.ev_filter_hidden + 2 * feature_width);
  const std::size_t edge_bytes =
      edge_doubles * sizeof(double) + edges * 2 * sizeof(std::size_t);
  return node_bytes + edge_bytes;
}

}  // namespace

IntegratedBlock0DeviceWorkspace::IntegratedBlock0DeviceWorkspace(
    std::size_t nodes, std::size_t edges, const ArchDims &dims)
    : embedding("so3lr_full0_embedding", nodes * dims.invariant_width),
      cutoff("so3lr_full0_cutoff", edges),
      radial_basis("so3lr_full0_rbf", edges * dims.rbf_width),
      edge_ev_invariants("so3lr_feature_edge_ev_invariants",
                         edges * dims.ev_heads),
      rbf_hidden("so3lr_full0_rbf_hidden", edges * dims.invariant_width),
      ev_hidden("so3lr_full0_ev_hidden", edges * dims.ev_filter_hidden),
      filter_inv("so3lr_full0_filter_inv", edges * dims.invariant_width),
      filter_ev("so3lr_full0_filter_ev", edges * dims.invariant_width),
      invariant_update("so3lr_full0_d_inv", nodes * dims.invariant_width),
      equivariant_update("so3lr_full0_d_ev",
                         nodes * dims.equivariant_width),
      qk_inverse_rms("so3lr_full0_qk_inverse_rms",
                     nodes * dims.inv_heads * 4) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR integrated device workspace is empty");
  for (std::size_t i = 0; i < qkv.size(); ++i)
    qkv[i] = DoubleView("so3lr_full0_qkv_" + std::to_string(i),
                        nodes * dims.invariant_width);
}

KokkosIntegratedBlock0::KokkosIntegratedBlock0(
    const NativeModel &model, std::size_t block_index)
    : dims_(model.arch().dims()),
      embedding_weight_(load_double(
          model, "model.inv_feature_embedding.embedding.weight",
          {dims_.invariant_width, 118})),
      bernstein_b_(load_double(
          model, "model.radial_embedding.radial_basis_fn.b",
          {dims_.rbf_width})),
      attention_cg_rep_(load_double(
          model, transformer_prefix(model, block_index) +
                     ".euclidean_attention_block.so3_conv_invariants.cg_rep",
          {dims_.equivariant_width})),
      bernstein_k_("so3lr_integrated_bernstein_k", dims_.rbf_width),
      bernstein_k_reverse_("so3lr_integrated_bernstein_k_reverse",
                           dims_.rbf_width),
      filter_inv_(load_filter(
          model, transformer_prefix(model, block_index) + ".filter_net_inv",
          dims_)),
      filter_ev_(load_filter(
          model, transformer_prefix(model, block_index) + ".filter_net_ev",
          dims_)),
      gamma_(model.float64(
          model.tensor("model.radial_embedding.radial_basis_fn.gamma"), 0)),
      cutoff_radius_(
          model.architecture_number("short_range_cutoff_angstrom")),
      attention_norm_inv_(attention_normalization(model, block_index)),
      attention_norm_ev_(attention_normalization(model, block_index)),
      block_index_(block_index) {
  qk_norm_ = model.arch().qk_norm;
  const std::size_t heads = dims_.inv_heads;
  const std::size_t rbf_width = dims_.rbf_width;
  // The fused attention kernel walks one head index over both the invariant
  // and the per-degree heads, so it needs H1 == H2 and a 32-wide head.
  if (dims_.inv_heads != dims_.ev_heads ||
      dims_.inv_head_width != head_width ||
      dims_.ev_head_width != head_width ||
      model.architecture_number("attention_heads") != heads ||
      model.architecture_number("attention_head_width") != head_width)
    throw std::runtime_error("SO3LR integrated architecture mismatch");
  const auto &k = model.tensor("model.radial_embedding.radial_basis_fn.k");
  const auto &k_reverse =
      model.tensor("model.radial_embedding.radial_basis_fn.k_rev");
  require_tensor(k, {rbf_width}, "int64");
  require_tensor(k_reverse, {rbf_width}, "int64");
  fill_view(bernstein_k_, rbf_width,
            [&](std::size_t i) { return model.int64(k, i); });
  fill_view(bernstein_k_reverse_, rbf_width,
            [&](std::size_t i) { return model.int64(k_reverse, i); });

  const std::string prefix = transformer_prefix(model, block_index_) +
                             ".euclidean_attention_block.";
  constexpr std::array<const char *, 5> names = {
      "W_q_inv", "W_k_inv", "W_v_inv", "W_q_ev", "W_k_ev"};
  for (std::size_t i = 0; i < names.size(); ++i)
    qkv_weights_[i] =
        load_double(model, prefix + names[i], {heads, head_width, head_width});

  // Per-degree-slot multiplicities come from the descriptor, which already
  // validated them; the checkpoint copy must agree slot for slot.
  const auto &arch = model.arch();
  degree_repeats_ = IndexView("so3lr_integrated_degree_repeats", heads);
  degree_offsets_ = IndexView("so3lr_integrated_degree_offsets", heads);
  const auto &repeats = model.tensor(prefix + "degree_repeats");
  require_tensor(repeats, {heads}, "int64");
  for (std::size_t i = 0; i < heads; ++i)
    if (model.int64(repeats, i) !=
        static_cast<std::int64_t>(arch.degree_repeats[i]))
      throw std::runtime_error("SO3LR degree-repeat values changed");
  fill_view(degree_repeats_, heads,
            [&](std::size_t i) { return arch.degree_repeats[i]; });
  fill_view(degree_offsets_, heads,
            [&](std::size_t i) { return arch.degree_offsets[i]; });

  check_cublas(cublasCreate(&handle_), "cublasCreate");
  const auto stream = Kokkos::Cuda().cuda_stream();
  check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
  cudaStream_t configured = nullptr;
  check_cublas(cublasGetStream(handle_, &configured), "cublasGetStream");
  shared_kokkos_stream_ = configured == stream;
  const std::size_t f = dims_.invariant_width;
  const std::size_t h = dims_.ev_filter_hidden;
  const std::size_t filter_doubles = f * rbf_width + f + f * f + f +
                                     h * heads + h + f * h + f;
  persistent_device_bytes_ =
      (f * 118 + rbf_width + dims_.equivariant_width + 2 * filter_doubles +
       5 * heads * head_width * head_width) * sizeof(double) +
      2 * rbf_width * sizeof(std::int64_t) + 2 * heads * sizeof(std::size_t);
  Kokkos::fence();
}

KokkosIntegratedBlock0::~KokkosIntegratedBlock0() {
  if (handle_ != nullptr) {
    Kokkos::fence();
    static_cast<void>(cublasDestroy(handle_));
  }
}

void KokkosIntegratedBlock0::launch_device(
    const Kokkos::View<std::int64_t *> &atomic_numbers,
    const Kokkos::View<double *> &distances,
    const Kokkos::View<double *> &ev_invariants,
    const Kokkos::View<double *> &sh_vectors,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const IntegratedBlock0DeviceWorkspace &workspace) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  const std::size_t rbf_width = dims_.rbf_width;
  const std::size_t heads = dims_.inv_heads;
  const std::size_t nodes = atomic_numbers.extent(0);
  const std::size_t edges = distances.extent(0);
  if (nodes == 0 || edges == 0 || ev_invariants.extent(0) != edges * heads ||
      sh_vectors.extent(0) != edges * equivariant_width ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      workspace.embedding.extent(0) != nodes * feature_width ||
      workspace.radial_basis.extent(0) != edges * rbf_width ||
      workspace.invariant_update.extent(0) != nodes * feature_width ||
      workspace.equivariant_update.extent(0) != nodes * equivariant_width)
    throw std::runtime_error("SO3LR integrated device-view contract mismatch");
  launch_pipeline(
      handle_, atomic_numbers, distances, ev_invariants, sh_vectors, senders,
      receivers, workspace.embedding, workspace.cutoff,
      workspace.radial_basis, embedding_weight_, bernstein_b_, bernstein_k_,
      bernstein_k_reverse_, gamma_, cutoff_radius_, filter_inv_, filter_ev_,
      workspace.rbf_hidden, workspace.ev_hidden, workspace.filter_inv,
      workspace.filter_ev, qkv_weights_, workspace.qkv, degree_repeats_,
      degree_offsets_, attention_norm_inv_, attention_norm_ev_,
      workspace.invariant_update, workspace.equivariant_update, qk_norm_,
      workspace.qk_inverse_rms, dims_);
}

void KokkosIntegratedBlock0::launch_features_device(
    const Kokkos::View<double *> &inv_features,
    const Kokkos::View<double *> &ev_features,
    const Kokkos::View<double *> &distances,
    const Kokkos::View<double *> &sh_vectors,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const IntegratedBlock0DeviceWorkspace &workspace) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  const std::size_t rbf_width = dims_.rbf_width;
  const std::size_t heads = dims_.inv_heads;
  const std::size_t nodes = inv_features.extent(0) / feature_width;
  const std::size_t edges = distances.extent(0);
  if (nodes == 0 || edges == 0 ||
      inv_features.extent(0) != nodes * feature_width ||
      ev_features.extent(0) != nodes * equivariant_width ||
      sh_vectors.extent(0) != edges * equivariant_width ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      workspace.edge_ev_invariants.extent(0) != edges * heads ||
      workspace.radial_basis.extent(0) != edges * rbf_width ||
      workspace.invariant_update.extent(0) != nodes * feature_width ||
      workspace.equivariant_update.extent(0) != nodes * equivariant_width)
    throw std::runtime_error("SO3LR feature attention contract mismatch");
  Int64View no_atomic_numbers;
  DoubleView no_embedding;
  launch_inputs(no_atomic_numbers, distances, no_embedding, workspace.cutoff,
                workspace.radial_basis, embedding_weight_, bernstein_b_,
                bernstein_k_, bernstein_k_reverse_, gamma_, cutoff_radius_,
                dims_);
  launch_edge_ev_invariants(
      ev_features, senders, receivers, attention_cg_rep_, degree_repeats_,
      degree_offsets_, edges, workspace.edge_ev_invariants, dims_);
  launch_filter(handle_, filter_inv_, workspace.radial_basis,
                workspace.edge_ev_invariants, edges, workspace.rbf_hidden,
                workspace.ev_hidden, workspace.filter_inv, true, dims_);
  launch_filter(handle_, filter_ev_, workspace.radial_basis,
                workspace.edge_ev_invariants, edges, workspace.rbf_hidden,
                workspace.ev_hidden, workspace.filter_ev, false, dims_);
  launch_qkv(handle_, inv_features, nodes, qkv_weights_, workspace.qkv, dims_);
  if (qk_norm_)
    launch_qk_rms_norm(workspace.qkv, nodes, workspace.qk_inverse_rms, dims_);
  launch_attention(
      workspace.qkv, workspace.filter_inv, workspace.filter_ev, sh_vectors,
      workspace.cutoff, senders, receivers, degree_repeats_, degree_offsets_,
      edges, attention_norm_inv_, attention_norm_ev_,
      workspace.invariant_update, workspace.equivariant_update, dims_);
}

IntegratedBlock0Results KokkosIntegratedBlock0::evaluate(
    const std::vector<std::int64_t> &atomic_numbers,
    const std::vector<double> &distances,
    const std::vector<double> &ev_invariants,
    const std::vector<double> &sh_vectors,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  const std::size_t rbf_width = dims_.rbf_width;
  const std::size_t heads = dims_.inv_heads;
  const std::size_t nodes = atomic_numbers.size();
  const std::size_t edges = distances.size();
  if (nodes == 0 || edges == 0 || senders.size() != edges ||
      receivers.size() != edges || ev_invariants.size() != edges * heads ||
      sh_vectors.size() != edges * equivariant_width)
    throw std::runtime_error("SO3LR integrated input contract mismatch");
  if (std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR integrated edge index out of range");

  Int64View z("so3lr_pipe_z", nodes);
  DoubleView distance("so3lr_pipe_distance", edges);
  DoubleView ev("so3lr_pipe_ev_invariants", edges * heads);
  DoubleView sh("so3lr_pipe_sh", edges * equivariant_width);
  IndexView sender("so3lr_pipe_sender", edges);
  IndexView receiver("so3lr_pipe_receiver", edges);
  fill_view(z, nodes, [&](std::size_t i) { return atomic_numbers[i]; });
  fill_view(distance, edges, [&](std::size_t i) { return distances[i]; });
  fill_view(ev, edges * heads,
            [&](std::size_t i) { return ev_invariants[i]; });
  fill_view(sh, edges * equivariant_width,
            [&](std::size_t i) { return sh_vectors[i]; });
  fill_view(sender, edges, [&](std::size_t i) { return senders[i]; });
  fill_view(receiver, edges, [&](std::size_t i) { return receivers[i]; });

  DoubleView embedding("so3lr_pipe_embedding", nodes * feature_width);
  DoubleView cutoff("so3lr_pipe_cutoff", edges);
  DoubleView rbf("so3lr_pipe_rbf", edges * rbf_width);
  DoubleView rbf_hidden("so3lr_pipe_rbf_hidden", edges * feature_width);
  DoubleView ev_hidden("so3lr_pipe_ev_hidden", edges * dims_.ev_filter_hidden);
  DoubleView filter_inv("so3lr_pipe_filter_inv", edges * feature_width);
  DoubleView filter_ev("so3lr_pipe_filter_ev", edges * feature_width);
  std::array<DoubleView, 5> qkv;
  for (std::size_t i = 0; i < qkv.size(); ++i)
    qkv[i] = DoubleView("so3lr_pipe_qkv_" + std::to_string(i),
                        nodes * feature_width);
  DoubleView d_inv("so3lr_pipe_d_inv", nodes * feature_width);
  DoubleView d_ev("so3lr_pipe_d_ev", nodes * equivariant_width);
  DoubleView qk_rms("so3lr_qk_inverse_rms_local", z.extent(0) * heads * 4);
  launch_pipeline(
      handle_, z, distance, ev, sh, sender, receiver, embedding, cutoff, rbf,
      embedding_weight_, bernstein_b_, bernstein_k_, bernstein_k_reverse_,
      gamma_, cutoff_radius_, filter_inv_, filter_ev_, rbf_hidden, ev_hidden,
      filter_inv, filter_ev, qkv_weights_, qkv, degree_repeats_,
      degree_offsets_, attention_norm_inv_, attention_norm_ev_, d_inv, d_ev,
        qk_norm_, qk_rms, dims_);
  Kokkos::fence();
  IntegratedBlock0Results result;
  result.embedding = copy_host(embedding);
  result.radial_basis = copy_host(rbf);
  result.cutoff = copy_host(cutoff);
  result.invariant_update = copy_host(d_inv);
  result.equivariant_update = copy_host(d_ev);
  return result;
}

IntegratedBlock0Benchmark KokkosIntegratedBlock0::benchmark(
    std::size_t nodes, std::size_t edges, std::size_t repetitions) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  const std::size_t rbf_width = dims_.rbf_width;
  const std::size_t heads = dims_.inv_heads;
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR integrated benchmark sizes invalid");
  Int64View z("so3lr_pipe_bench_z", nodes);
  DoubleView distance("so3lr_pipe_bench_distance", edges);
  DoubleView ev("so3lr_pipe_bench_ev", edges * heads);
  DoubleView sh("so3lr_pipe_bench_sh", edges * equivariant_width);
  IndexView sender("so3lr_pipe_bench_sender", edges);
  IndexView receiver("so3lr_pipe_bench_receiver", edges);
  Kokkos::parallel_for(
      "so3lr_pipe_bench_nodes", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  Kokkos::parallel_for(
      "so3lr_pipe_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        distance(edge) =
            0.05 + 4.449 * static_cast<double>(edge % 8192) / 8191.0;
        sender(edge) = (edge * 17 + 11) % nodes;
        receiver(edge) = edge % nodes;
      });
  Kokkos::deep_copy(ev, 0.0);
  Kokkos::parallel_for(
      "so3lr_pipe_bench_sh",
      Kokkos::RangePolicy<>(0, edges * equivariant_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 61) - 30.0) / 37.0;
      });

  DoubleView embedding("so3lr_pipe_bench_embedding", nodes * feature_width);
  DoubleView cutoff("so3lr_pipe_bench_cutoff", edges);
  DoubleView rbf("so3lr_pipe_bench_rbf", edges * rbf_width);
  DoubleView rbf_hidden("so3lr_pipe_bench_rbf_hidden", edges * feature_width);
  DoubleView ev_hidden("so3lr_pipe_bench_ev_hidden", edges * dims_.ev_filter_hidden);
  DoubleView filter_inv("so3lr_pipe_bench_filter_inv", edges * feature_width);
  DoubleView filter_ev("so3lr_pipe_bench_filter_ev", edges * feature_width);
  std::array<DoubleView, 5> qkv;
  for (std::size_t i = 0; i < qkv.size(); ++i)
    qkv[i] = DoubleView("so3lr_pipe_bench_qkv_" + std::to_string(i),
                        nodes * feature_width);
  DoubleView d_inv("so3lr_pipe_bench_d_inv", nodes * feature_width);
  DoubleView d_ev("so3lr_pipe_bench_d_ev", nodes * equivariant_width);

  auto run_input = [&]() {
    launch_inputs(z, distance, embedding, cutoff, rbf, embedding_weight_,
                  bernstein_b_, bernstein_k_, bernstein_k_reverse_, gamma_,
                  cutoff_radius_, dims_);
  };
  auto run_filters = [&]() {
    launch_filter(handle_, filter_inv_, rbf, ev, edges, rbf_hidden, ev_hidden,
                  filter_inv, true, dims_);
    launch_filter(handle_, filter_ev_, rbf, ev, edges, rbf_hidden, ev_hidden,
                  filter_ev, false, dims_);
  };
  auto run_qkv = [&]() {
    launch_qkv(handle_, embedding, nodes, qkv_weights_, qkv, dims_);
  };
  auto run_attention = [&]() {
    launch_attention(qkv, filter_inv, filter_ev, sh, cutoff, sender, receiver,
                     degree_repeats_, degree_offsets_, edges,
                     attention_norm_inv_, attention_norm_ev_, d_inv, d_ev,
                     dims_);
  };
  auto time_phase = [&](auto launch) {
    launch();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i) launch();
    Kokkos::fence();
    return timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  };

  // Populate dependencies, then measure each phase with an explicit fence.
  run_input();
  run_filters();
  run_qkv();
  run_attention();
  Kokkos::fence();
  const double input_ms = time_phase(run_input);
  const double filter_ms = time_phase(run_filters);
  const double qkv_ms = time_phase(run_qkv);
  const double attention_ms = time_phase(run_attention);

  DoubleView qk_rms("so3lr_qk_inverse_rms_local", z.extent(0) * heads * 4);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_pipeline(
        handle_, z, distance, ev, sh, sender, receiver, embedding, cutoff, rbf,
        embedding_weight_, bernstein_b_, bernstein_k_, bernstein_k_reverse_,
        gamma_, cutoff_radius_, filter_inv_, filter_ev_, rbf_hidden, ev_hidden,
        filter_inv, filter_ev, qkv_weights_, qkv, degree_repeats_,
        degree_offsets_, attention_norm_inv_, attention_norm_ev_, d_inv, d_ev,
        qk_norm_, qk_rms, dims_);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_pipeline(
        handle_, z, distance, ev, sh, sender, receiver, embedding, cutoff, rbf,
        embedding_weight_, bernstein_b_, bernstein_k_, bernstein_k_reverse_,
        gamma_, cutoff_radius_, filter_inv_, filter_ev_, rbf_hidden, ev_hidden,
        filter_inv, filter_ev, qkv_weights_, qkv, degree_repeats_,
        degree_offsets_, attention_norm_inv_, attention_norm_ev_, d_inv, d_ev,
        qk_norm_, qk_rms, dims_);
  Kokkos::fence();
  const double pipeline_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);

  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_pipe_checksum", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0)
          update += d_inv((nodes / 3) * feature_width + 17);
        else
          update += d_ev((nodes / 2) * equivariant_width + 9);
      },
      checksum);
  Kokkos::fence();
  IntegratedBlock0Benchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes = persistent_device_bytes_ + dynamic_bytes(nodes, edges, dims_);
  result.host_boundary_bytes_per_iteration = 0;
  result.input_milliseconds = input_ms;
  result.filter_milliseconds = filter_ms;
  result.qkv_milliseconds = qkv_ms;
  result.attention_milliseconds = attention_ms;
  result.phase_sum_milliseconds =
      input_ms + filter_ms + qkv_ms + attention_ms;
  result.pipeline_milliseconds = pipeline_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (pipeline_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
