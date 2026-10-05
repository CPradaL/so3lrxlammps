#include "so3lr/kokkos_block2_attention_input_reverse.hpp"

#include <Kokkos_Timer.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;

// See kokkos_integrated_block0.cpp: head width is fixed, the rest is taken
// from the architecture descriptor.
constexpr std::size_t head_width = 32;
// Upper bound on interaction depth for the per-block profile counters.
constexpr std::size_t kMaxProfiledBlocks = 8;

// SO3LR_STAGE3_OPTIM_DEV10_REVERSE_KERNEL_PROFILE
using ProfileClock = std::chrono::steady_clock;

bool reverse_kernel_profile_requested() {
  static const bool requested = [] {
    const char *value = std::getenv("SO3_NATIVE_REVERSE_KERNEL_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return requested;
}

std::size_t reverse_kernel_profile_interval() {
  static const std::size_t interval = [] {
    const char *value =
        std::getenv("SO3_NATIVE_REVERSE_KERNEL_PROFILE_EVERY");
    if (value == nullptr || value[0] == '\0') return std::size_t{20};
    char *end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0
               ? static_cast<std::size_t>(parsed)
               : std::size_t{20};
  }();
  return interval;
}

int reverse_kernel_profile_rank() {
  static const int rank = [] {
    const char *value = std::getenv("SLURM_PROCID");
    if (value == nullptr || value[0] == '\0')
      value = std::getenv("OMPI_COMM_WORLD_RANK");
    if (value == nullptr || value[0] == '\0') return 0;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value ? static_cast<int>(parsed) : 0;
  }();
  return rank;
}

std::array<std::size_t, kMaxProfiledBlocks> &reverse_kernel_profile_calls() {
  static std::array<std::size_t, kMaxProfiledBlocks> calls = {};
  return calls;
}

double profile_milliseconds(ProfileClock::time_point begin,
                            ProfileClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

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

void require_tensor(const TensorRecord &tensor,
                    const std::vector<std::size_t> &shape,
                    const std::string &dtype = "float64") {
  if (tensor.dtype != dtype || tensor.shape != shape)
    throw std::runtime_error("SO3LR block2 reverse tensor mismatch: " +
                             tensor.state_key);
}

DoubleView load_double(const NativeModel &model, const std::string &key,
                       const std::vector<std::size_t> &shape) {
  const auto &tensor = model.tensor(key);
  require_tensor(tensor, shape);
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  DoubleView result(key + "_block2_reverse", count);
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
  result.ev_bias_0 =
      load_double(model, prefix + ".mlp_ev.0.bias", {h});
  result.ev_weight_1 = load_double(
      model, prefix + ".mlp_ev.mlp_ev_layer_1.0.weight", {f, h});
  result.ev_bias_1 = load_double(
      model, prefix + ".mlp_ev.mlp_ev_layer_1.0.bias", {f});
  return result;
}

void linear(cublasHandle_t handle, const DoubleView &input,
            std::size_t rows, std::size_t input_width,
            std::size_t output_width, const DoubleView &weight,
            const DoubleView &output) {
  const int m = checked_int(output_width, "linear_output_width");
  const int n = checked_int(rows, "linear_rows");
  const int k = checked_int(input_width, "linear_input_width");
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  check_cublas(
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  weight.data(), k, input.data(), k, &beta, output.data(), m),
      "cublasDgemm(block2-filter-recompute)");
}

void linear_input_gradient(cublasHandle_t handle,
                           const DoubleView &output_gradient,
                           std::size_t rows, std::size_t output_width,
                           std::size_t input_width,
                           const DoubleView &weight,
                           const DoubleView &input_gradient,
                           double beta) {
  const int m = checked_int(input_width, "reverse_input_width");
  const int n = checked_int(rows, "reverse_rows");
  const int k = checked_int(output_width, "reverse_output_width");
  constexpr double alpha = 1.0;
  check_cublas(
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  weight.data(), m, output_gradient.data(), k, &beta,
                  input_gradient.data(), m),
      "cublasDgemm(block2-filter-reverse)");
}

void add_bias(const DoubleView &values, const DoubleView &bias,
              std::size_t rows, std::size_t width, const char *label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, rows * width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        values(flat) += bias(flat % width);
      });
}

void apply_silu_derivative(const DoubleView &preactivation,
                           const DoubleView &gradient, std::size_t count,
                           const char *label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, count),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double x = preactivation(i);
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-x));
        gradient(i) *= sigmoid + x * sigmoid * (1.0 - sigmoid);
      });
}

void reverse_filter(cublasHandle_t handle,
                    const IntegratedFilterParameters &parameters,
                    const DoubleView &rbf,
                    const DoubleView &edge_ev_invariants,
                    const DoubleView &filter_gradient, std::size_t edges,
                    const DoubleView &preactivation,
                    const DoubleView &hidden_gradient,
                    const DoubleView &grad_rbf,
                    const DoubleView &grad_edge_ev_invariants,
                    bool accumulate, const char *prefix,
                    const ArchDims &dims) {
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t rbf_width = dims.rbf_width;
  const std::size_t ev_hidden = dims.ev_filter_hidden;
  const std::size_t heads = dims.ev_heads;
  linear(handle, rbf, edges, rbf_width, feature_width,
         parameters.rbf_weight_0, preactivation);
  add_bias(preactivation, parameters.rbf_bias_0, edges, feature_width,
           prefix);
  linear_input_gradient(handle, filter_gradient, edges, feature_width,
                        feature_width, parameters.rbf_weight_1,
                        hidden_gradient, 0.0);
  apply_silu_derivative(preactivation, hidden_gradient,
                        edges * feature_width, prefix);
  linear_input_gradient(handle, hidden_gradient, edges, feature_width,
                        rbf_width, parameters.rbf_weight_0, grad_rbf,
                        accumulate ? 1.0 : 0.0);

  // The spherical-filter hidden width is F/4. In v1 that happens to equal the
  // radial-basis width (32), which the original code used here.
  linear(handle, edge_ev_invariants, edges, heads, ev_hidden,
         parameters.ev_weight_0, preactivation);
  add_bias(preactivation, parameters.ev_bias_0, edges, ev_hidden, prefix);
  linear_input_gradient(handle, filter_gradient, edges, feature_width,
                        ev_hidden, parameters.ev_weight_1,
                        hidden_gradient, 0.0);
  apply_silu_derivative(preactivation, hidden_gradient, edges * ev_hidden,
                        prefix);
  linear_input_gradient(handle, hidden_gradient, edges, ev_hidden, heads,
                        parameters.ev_weight_0,
                        grad_edge_ev_invariants,
                        accumulate ? 1.0 : 0.0);
}

void reverse_qkv(cublasHandle_t handle,
                 const std::array<DoubleView, 5> &qkv_gradient,
                 const std::array<DoubleView, 5> &weights,
                 std::size_t nodes, const DoubleView &grad_inv_features,
                 const ArchDims &dims) {
  const std::size_t feature_width = dims.invariant_width;
  const int m = checked_int(head_width, "qkv_reverse_head_width");
  const int n = checked_int(nodes, "qkv_reverse_nodes");
  const int stride = checked_int(feature_width, "qkv_reverse_stride");
  constexpr double alpha = 1.0;
  constexpr double beta = 1.0;
  for (std::size_t which = 0; which < weights.size(); ++which) {
    const std::size_t heads = which < 3 ? dims.inv_heads : dims.ev_heads;
    for (std::size_t head = 0; head < heads; ++head) {
      check_cublas(
          cublasDgemm(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha,
              weights[which].data() + head * head_width * head_width, m,
              qkv_gradient[which].data() + head * head_width, stride,
              &beta, grad_inv_features.data() + head * head_width, stride),
          "cublasDgemm(block2-QKV-reverse)");
    }
  }
}

void reverse_edge_invariants(
    const DoubleView &ev_features, const IndexView &senders,
    const IndexView &receivers, const DoubleView &cg_rep,
    const IndexView &degree_repeats, const IndexView &degree_offsets,
    const DoubleView &grad_edge_ev_invariants,
    const DoubleView &grad_ev_features, const ArchDims &dims) {
  const std::size_t heads = dims.ev_heads;
  const std::size_t equivariant_width = dims.equivariant_width;
  const std::size_t edges = senders.extent(0);
  Kokkos::parallel_for(
      "so3lr_block2_edge_invariant_reverse",
      Kokkos::RangePolicy<>(0, edges * heads),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / heads;
        const std::size_t degree = flat % heads;
        const std::size_t sender = senders(edge);
        const std::size_t receiver = receivers(edge);
        const double upstream = grad_edge_ev_invariants(flat);
        for (std::size_t local = 0; local < degree_repeats(degree); ++local) {
          const std::size_t channel = degree_offsets(degree) + local;
          const std::size_t sender_index =
              sender * equivariant_width + channel;
          const std::size_t receiver_index =
              receiver * equivariant_width + channel;
          const double difference =
              ev_features(sender_index) - ev_features(receiver_index);
          const double gradient =
              upstream * 2.0 * difference * cg_rep(channel);
          Kokkos::atomic_add(&grad_ev_features(sender_index), gradient);
          Kokkos::atomic_add(&grad_ev_features(receiver_index), -gradient);
        }
      });
}

void reverse_radial_and_cutoff(
    const DoubleView &distances, const DoubleView &radial_basis,
    const DoubleView &grad_radial_basis,
    const DoubleView &grad_cutoff, const Int64View &k,
    const Int64View &k_reverse, double gamma, double rmax,
    const DoubleView &grad_distances, const ArchDims &dims) {
  const std::size_t rbf_width = dims.rbf_width;
  const std::size_t edges = distances.extent(0);
  Kokkos::parallel_for(
      "so3lr_block2_radial_cutoff_reverse",
      Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        const double distance = distances(edge);
        double gradient = 0.0;
        if (distance < rmax) {
          const double x = distance / rmax;
          const double one_minus_x = 1.0 - x;
          const double derivative =
              -30.0 * x * x * one_minus_x * one_minus_x / rmax;
          gradient += grad_cutoff(edge) * derivative;
        }
        const double raw_x = Kokkos::exp(-gamma * distance);
        if (raw_x > 1.0e-6 && raw_x < 1.0 - 1.0e-6) {
          for (std::size_t channel = 0; channel < rbf_width; ++channel) {
            const double log_derivative =
                -gamma * static_cast<double>(k(channel)) +
                gamma * static_cast<double>(k_reverse(channel)) * raw_x /
                    (1.0 - raw_x);
            const std::size_t index = edge * rbf_width + channel;
            gradient += grad_radial_basis(index) * radial_basis(index) *
                        log_derivative;
          }
        }
        grad_distances(edge) = gradient;
      });
}

std::size_t complete_workspace_bytes(std::size_t nodes, std::size_t edges,
                                     std::size_t persistent_bytes,
                                     const ArchDims &dims) {
  const std::size_t heads = dims.inv_heads;
  const std::size_t feature_width = dims.invariant_width;
  const std::size_t equivariant_width = dims.equivariant_width;
  const std::size_t rbf_width = dims.rbf_width;
  // Standalone audit footprint: four node inputs/seeds/results plus both
  // forward and dev_18 workspaces, edge inputs, filter reverse scratch, and
  // geometry-boundary gradients.
  const std::size_t node_doubles =
      2 * (feature_width + equivariant_width) +
      (feature_width + 5 * feature_width + feature_width +
       equivariant_width) +
      (feature_width + equivariant_width + 5 * feature_width) +
      feature_width + equivariant_width;
  const std::size_t edge_doubles =
      (1 + equivariant_width) +
      (1 + rbf_width + heads + feature_width + rbf_width +
       2 * feature_width) +
      (2 * feature_width + equivariant_width + 1) +
      (1 + rbf_width + heads + 2 * feature_width);
  return persistent_bytes + nodes * node_doubles * sizeof(double) +
         edges * edge_doubles * sizeof(double) +
         2 * edges * sizeof(std::size_t);
}

}  // namespace

// Adjoint of the in-place QK RMS normalisation done in the forward pass
// (kokkos_integrated_block0.cpp, launch_qk_rms_norm). With y = x r stored in
// place of x and r = 1/rms kept per (node, head) row:
//   dx = r (dy - y mean(y dy))
// Rewrites the gradient in place, turning d/d(normalised) into d/d(raw).
static void qk_rms_norm_backward_one(const DoubleView &gradient, const DoubleView &normalised,
                              std::size_t nodes, std::size_t slot,
                              const DoubleView &inverse_rms,
                              std::size_t heads) {
  Kokkos::parallel_for(
      "so3lr_qk_rms_norm_reverse", Kokkos::RangePolicy<>(0, nodes * heads),
      KOKKOS_LAMBDA(const std::size_t row) {
        const std::size_t offset = row * head_width;
        double mean_y_dy = 0.0;
        for (std::size_t channel = 0; channel < head_width; ++channel)
          mean_y_dy += normalised(offset + channel) * gradient(offset + channel);
        mean_y_dy /= static_cast<double>(head_width);
        const double r = inverse_rms(row * 4 + slot);
        for (std::size_t channel = 0; channel < head_width; ++channel)
          gradient(offset + channel) =
              r * (gradient(offset + channel) -
                   normalised(offset + channel) * mean_y_dy);
      });
}

Block2AttentionInputReverseWorkspace::Block2AttentionInputReverseWorkspace(
    std::size_t nodes, std::size_t edges, const ArchDims &dims)
    : forward(nodes, edges, dims),
      attention_reverse(nodes, edges, dims),
      grad_inv_features("so3lr_block2_reverse_grad_inv",
                        nodes * dims.invariant_width),
      grad_ev_features("so3lr_block2_reverse_grad_ev",
                       nodes * dims.equivariant_width),
      grad_distances("so3lr_block2_reverse_grad_distances", edges),
      grad_radial_basis("so3lr_block2_reverse_grad_rbf",
                        edges * dims.rbf_width),
      grad_edge_ev_invariants("so3lr_block2_reverse_grad_edge_ev",
                              edges * dims.ev_heads),
      // Shared by the radial (width F) and spherical (width F/4) hidden layers.
      filter_preactivation("so3lr_block2_reverse_filter_preactivation",
                           edges * dims.invariant_width),
      filter_hidden_gradient("so3lr_block2_reverse_filter_hidden_gradient",
                             edges * dims.invariant_width) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR block2 reverse workspace is empty");
}

Block2AttentionInputReverseWorkspace::Block2AttentionInputReverseWorkspace(
    std::size_t nodes, std::size_t edges,
    const Block2AttentionInputReverseWorkspace &shared, const ArchDims &dims)
    : forward(nodes, edges, dims),
      attention_reverse(nodes, edges, shared.attention_reverse),
      grad_inv_features(shared.grad_inv_features),
      grad_ev_features(shared.grad_ev_features),
      // These two outputs survive until the final three-block geometry sum.
      grad_distances("so3lr_block_reverse_geometry_grad_distances", edges),
      grad_radial_basis(shared.grad_radial_basis),
      grad_edge_ev_invariants(shared.grad_edge_ev_invariants),
      filter_preactivation(shared.filter_preactivation),
      filter_hidden_gradient(shared.filter_hidden_gradient) {
  if (nodes == 0 || edges == 0 ||
      shared.grad_inv_features.extent(0) != nodes * dims.invariant_width ||
      shared.grad_radial_basis.extent(0) != edges * dims.rbf_width)
    throw std::runtime_error("SO3LR shared block reverse scratch mismatch");
}

KokkosBlock2AttentionInputReverse::KokkosBlock2AttentionInputReverse(
    const NativeModel &model, std::size_t block_index)
    : dims_(model.arch().dims()),
      forward_(model, block_index),
      attention_reverse_(model, block_index),
      filter_inv_(load_filter(
          model, "model.euclidean_transformers." +
                     std::to_string(block_index) + ".filter_net_inv",
          dims_)),
      filter_ev_(load_filter(
          model, "model.euclidean_transformers." +
                     std::to_string(block_index) + ".filter_net_ev",
          dims_)),
      attention_cg_rep_(load_double(
          model,
          "model.euclidean_transformers." + std::to_string(block_index) +
              ".euclidean_attention_block.so3_conv_invariants.cg_rep",
          {dims_.equivariant_width})),
      bernstein_b_(load_double(
          model, "model.radial_embedding.radial_basis_fn.b",
          {dims_.rbf_width})),
      bernstein_k_("so3lr_block2_reverse_bernstein_k", dims_.rbf_width),
      bernstein_k_reverse_("so3lr_block2_reverse_bernstein_k_reverse",
                           dims_.rbf_width),
      degree_repeats_("so3lr_block2_reverse_degree_repeats", dims_.ev_heads),
      degree_offsets_("so3lr_block2_reverse_degree_offsets", dims_.ev_heads),
      gamma_(model.float64(
          model.tensor("model.radial_embedding.radial_basis_fn.gamma"), 0)),
      cutoff_radius_(
          model.architecture_number("short_range_cutoff_angstrom")),
      block_index_(block_index) {
  qk_norm_ = model.arch().qk_norm;
  const std::size_t heads = dims_.inv_heads;
  const std::size_t rbf_width = dims_.rbf_width;
  const auto blocks = static_cast<std::size_t>(
      model.architecture_number("interaction_blocks"));
  if (block_index_ >= blocks || block_index_ >= kMaxProfiledBlocks)
    throw std::runtime_error("SO3LR generalized reverse index out of range");
  if (dims_.ev_heads != heads || dims_.inv_head_width != head_width ||
      dims_.ev_head_width != head_width)
    throw std::runtime_error("SO3LR block reverse head layout unsupported");
  const std::string attention =
      "model.euclidean_transformers." + std::to_string(block_index_) +
      ".euclidean_attention_block.";
  constexpr std::array<const char *, 5> names = {
      "W_q_inv", "W_k_inv", "W_v_inv", "W_q_ev", "W_k_ev"};
  for (std::size_t i = 0; i < names.size(); ++i)
    qkv_weights_[i] =
        load_double(model, attention + names[i], {heads, head_width,
                                                  head_width});
  const auto &k = model.tensor("model.radial_embedding.radial_basis_fn.k");
  const auto &k_reverse =
      model.tensor("model.radial_embedding.radial_basis_fn.k_rev");
  require_tensor(k, {rbf_width}, "int64");
  require_tensor(k_reverse, {rbf_width}, "int64");
  fill_view(bernstein_k_, rbf_width,
            [&](std::size_t i) { return model.int64(k, i); });
  fill_view(bernstein_k_reverse_, rbf_width,
            [&](std::size_t i) { return model.int64(k_reverse, i); });
  const auto &arch = model.arch();
  const auto &repeats = model.tensor(attention + "degree_repeats");
  require_tensor(repeats, {heads}, "int64");
  for (std::size_t i = 0; i < heads; ++i)
    if (model.int64(repeats, i) !=
        static_cast<std::int64_t>(arch.degree_repeats[i]))
      throw std::runtime_error("SO3LR block2 degree repeats changed");
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
  const std::size_t duplicated_filter_doubles =
      2 * (f * rbf_width + f + f * f + f + h * heads + h + f * h + f);
  const std::size_t qkv_doubles = 5 * heads * head_width * head_width;
  persistent_device_bytes_ =
      forward_.persistent_device_bytes() +
      attention_reverse_.persistent_device_bytes() +
      (duplicated_filter_doubles + qkv_doubles + dims_.equivariant_width +
       rbf_width) * sizeof(double) +
      2 * rbf_width * sizeof(std::int64_t) +
      2 * heads * sizeof(std::size_t);
  checkpoint_contract_verified_ =
      forward_.block_index() == block_index_ &&
      forward_.shared_kokkos_stream() &&
      attention_reverse_.block_index() == block_index_ &&
      attention_reverse_.checkpoint_contract_verified() &&
      shared_kokkos_stream_;
  if (!checkpoint_contract_verified_)
    throw std::runtime_error("SO3LR block2 integrated reverse contract failed");
  Kokkos::fence();
}

KokkosBlock2AttentionInputReverse::~KokkosBlock2AttentionInputReverse() {
  if (handle_ != nullptr) {
    Kokkos::fence();
    static_cast<void>(cublasDestroy(handle_));
  }
}

void KokkosBlock2AttentionInputReverse::launch_forward_device(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const Block2AttentionInputReverseWorkspace &workspace) const {
  forward_.launch_features_device(inv_features, ev_features, distances,
                                  sh_vectors, senders, receivers,
                                  workspace.forward);
}

void KokkosBlock2AttentionInputReverse::launch_reverse_device(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &distances, const DoubleView &sh_vectors,
    const IndexView &senders, const IndexView &receivers,
    const DoubleView &grad_attention_inv,
    const DoubleView &grad_attention_ev,
    const Block2AttentionInputReverseWorkspace &workspace) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  const std::size_t heads = dims_.inv_heads;
  const std::size_t nodes = inv_features.extent(0) / feature_width;
  const std::size_t edges = distances.extent(0);
  if (nodes == 0 || edges == 0 ||
      inv_features.extent(0) != nodes * feature_width ||
      ev_features.extent(0) != nodes * equivariant_width ||
      sh_vectors.extent(0) != edges * equivariant_width ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      grad_attention_inv.extent(0) != nodes * feature_width ||
      grad_attention_ev.extent(0) != nodes * equivariant_width ||
      workspace.grad_inv_features.extent(0) != nodes * feature_width ||
      workspace.grad_ev_features.extent(0) != nodes * equivariant_width ||
      workspace.grad_distances.extent(0) != edges)
    throw std::runtime_error("SO3LR block2 integrated reverse view mismatch");

  const bool profile_requested = reverse_kernel_profile_requested();
  std::size_t profile_call = 0;
  bool profile_sample = false;
  if (profile_requested) {
    profile_call = ++reverse_kernel_profile_calls().at(block_index_);
    const auto every = reverse_kernel_profile_interval();
    profile_sample = (profile_call - 1) % every == 0;
  }
  const auto mark = [&]() {
    if (!profile_sample) return ProfileClock::time_point{};
    Kokkos::fence();
    return ProfileClock::now();
  };

  const auto profile_start = mark();
  attention_reverse_.launch_reverse_device(
      workspace.forward.qkv, workspace.forward.filter_inv,
      workspace.forward.filter_ev, sh_vectors, workspace.forward.cutoff,
      senders, receivers, grad_attention_inv, grad_attention_ev,
      workspace.attention_reverse);
  const auto attention_done = mark();
  // qkv = {q_inv, k_inv, v_inv, q_ev, k_ev}; slots follow the forward pass.
  if (qk_norm_) {
    const auto &grad = workspace.attention_reverse.grad_qkv;
    const auto &qkv = workspace.forward.qkv;
    const auto &inverse_rms = workspace.forward.qk_inverse_rms;
    qk_rms_norm_backward_one(grad[0], qkv[0], nodes, 0, inverse_rms, heads);
    qk_rms_norm_backward_one(grad[1], qkv[1], nodes, 1, inverse_rms, heads);
    qk_rms_norm_backward_one(grad[3], qkv[3], nodes, 2, inverse_rms, heads);
    qk_rms_norm_backward_one(grad[4], qkv[4], nodes, 3, inverse_rms, heads);
  }

  // The post-attention boundary is inv+d_inv and ev+d_ev, so these copies are
  // the residual branches. Projection and edge-invariant derivatives are then
  // accumulated into the same block-input gradients.
  Kokkos::deep_copy(workspace.grad_inv_features, grad_attention_inv);
  Kokkos::deep_copy(workspace.grad_ev_features, grad_attention_ev);
  const auto residual_done = mark();
  reverse_qkv(handle_, workspace.attention_reverse.grad_qkv, qkv_weights_,
              nodes, workspace.grad_inv_features, dims_);
  const auto qkv_done = mark();
  reverse_filter(handle_, filter_inv_, workspace.forward.radial_basis,
                 workspace.forward.edge_ev_invariants,
                 workspace.attention_reverse.grad_filter_inv, edges,
                 workspace.filter_preactivation,
                 workspace.filter_hidden_gradient,
                 workspace.grad_radial_basis,
                 workspace.grad_edge_ev_invariants, false,
                 "so3lr_block2_filter_inv_reverse", dims_);
  const auto filter_inv_done = mark();
  reverse_filter(handle_, filter_ev_, workspace.forward.radial_basis,
                 workspace.forward.edge_ev_invariants,
                 workspace.attention_reverse.grad_filter_ev, edges,
                 workspace.filter_preactivation,
                 workspace.filter_hidden_gradient,
                 workspace.grad_radial_basis,
                 workspace.grad_edge_ev_invariants, true,
                 "so3lr_block2_filter_ev_reverse", dims_);
  const auto filter_ev_done = mark();
  reverse_edge_invariants(
      ev_features, senders, receivers, attention_cg_rep_, degree_repeats_,
      degree_offsets_, workspace.grad_edge_ev_invariants,
      workspace.grad_ev_features, dims_);
  const auto edge_invariants_done = mark();
  reverse_radial_and_cutoff(
      distances, workspace.forward.radial_basis,
      workspace.grad_radial_basis,
      workspace.attention_reverse.grad_cutoff, bernstein_k_,
      bernstein_k_reverse_, gamma_, cutoff_radius_,
      workspace.grad_distances, dims_);
  const auto radial_cutoff_done = mark();

  if (profile_sample) {
    std::printf(
        "SO3LR_SR_BLOCK_REVERSE_KERNEL_PROFILE rank=%d call=%zu block=%zu "
        "nodes=%zu edges=%zu attention_ms=%.6f residual_ms=%.6f "
        "qkv_ms=%.6f filter_inv_ms=%.6f filter_ev_ms=%.6f "
        "edge_invariants_ms=%.6f radial_cutoff_ms=%.6f total_ms=%.6f\n",
        reverse_kernel_profile_rank(), profile_call, block_index_, nodes,
        edges, profile_milliseconds(profile_start, attention_done),
        profile_milliseconds(attention_done, residual_done),
        profile_milliseconds(residual_done, qkv_done),
        profile_milliseconds(qkv_done, filter_inv_done),
        profile_milliseconds(filter_inv_done, filter_ev_done),
        profile_milliseconds(filter_ev_done, edge_invariants_done),
        profile_milliseconds(edge_invariants_done, radial_cutoff_done),
        profile_milliseconds(profile_start, radial_cutoff_done));
    std::fflush(stdout);
  }
}

Block2AttentionInputReverseBenchmark
KokkosBlock2AttentionInputReverse::benchmark(
    std::size_t nodes, std::size_t edges,
    std::size_t repetitions) const {
  const std::size_t feature_width = dims_.invariant_width;
  const std::size_t equivariant_width = dims_.equivariant_width;
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR block2 reverse benchmark invalid");
  DoubleView inv("so3lr_dev19_bench_inv", nodes * feature_width);
  DoubleView ev("so3lr_dev19_bench_ev", nodes * equivariant_width);
  DoubleView distances("so3lr_dev19_bench_distances", edges);
  DoubleView sh("so3lr_dev19_bench_sh", edges * equivariant_width);
  IndexView senders("so3lr_dev19_bench_senders", edges);
  IndexView receivers("so3lr_dev19_bench_receivers", edges);
  DoubleView grad_inv("so3lr_dev19_bench_grad_inv", nodes * feature_width);
  DoubleView grad_ev("so3lr_dev19_bench_grad_ev", nodes * equivariant_width);
  Kokkos::parallel_for(
      "so3lr_dev19_bench_inv", Kokkos::RangePolicy<>(0, inv.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        inv(i) = (static_cast<double>(i % 251) - 125.0) / 137.0;
        grad_inv(i) = (static_cast<double>(i % 113) - 56.0) / 127.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev19_bench_ev", Kokkos::RangePolicy<>(0, ev.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        ev(i) = (static_cast<double>(i % 83) - 41.0) / 97.0;
        grad_ev(i) = (static_cast<double>(i % 47) - 23.0) / 59.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev19_bench_graph", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        receivers(edge) = edge % nodes;
        senders(edge) = (edge * 17 + 11) % nodes;
        distances(edge) =
            0.15 + 4.29 * static_cast<double>(edge % 8192) / 8191.0;
      });
  Kokkos::parallel_for(
      "so3lr_dev19_bench_sh",
      Kokkos::RangePolicy<>(0, sh.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        sh(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });
  Block2AttentionInputReverseWorkspace workspace(nodes, edges, dims_);
  for (int warmup = 0; warmup < 2; ++warmup) {
    launch_forward_device(inv, ev, distances, sh, senders, receivers,
                          workspace);
    launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                          grad_inv, grad_ev, workspace);
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                          grad_inv, grad_ev, workspace);
  Kokkos::fence();
  const double reverse_ms = timer.seconds() * 1000.0 /
                            static_cast<double>(repetitions);
  timer.reset();
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    launch_forward_device(inv, ev, distances, sh, senders, receivers,
                          workspace);
    launch_reverse_device(inv, ev, distances, sh, senders, receivers,
                          grad_inv, grad_ev, workspace);
  }
  Kokkos::fence();
  const double forward_reverse_ms = timer.seconds() * 1000.0 /
                                    static_cast<double>(repetitions);
  const auto output_inv = workspace.grad_inv_features;
  const auto output_ev = workspace.grad_ev_features;
  const auto output_distance = workspace.grad_distances;
  const auto output_sh = workspace.attention_reverse.grad_sh;
  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev19_bench_checksum", Kokkos::RangePolicy<>(0, 4),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += output_inv(17);
        if (which == 1) update += output_ev(11);
        if (which == 2) update += output_distance(7);
        if (which == 3) update += output_sh(29);
      },
      checksum);
  Kokkos::fence();
  Block2AttentionInputReverseBenchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes =
      complete_workspace_bytes(nodes, edges, persistent_device_bytes_, dims_);
  result.host_boundary_bytes_per_iteration = 0;
  result.reverse_milliseconds = reverse_ms;
  result.forward_reverse_milliseconds = forward_reverse_ms;
  result.edges_per_second =
      static_cast<double>(edges) / (reverse_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
