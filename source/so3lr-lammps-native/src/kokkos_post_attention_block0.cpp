#include "so3lr/kokkos_post_attention_block0.hpp"

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
using IndexView = Kokkos::View<std::size_t *>;

// Widths come from the architecture descriptor. Each launcher binds the
// names below as locals so the kernel bodies read exactly as they did when
// these were v1 compile-time constants:
//   inv_width = F, ev_width = E, degree_count = H2, interaction_width = F + H2.
#define SO3LR_POST_ATTENTION_WIDTHS(dims)                                   \
  const std::size_t inv_width = (dims).invariant_width;                    \
  const std::size_t ev_width = (dims).equivariant_width;                   \
  const std::size_t degree_count = (dims).ev_heads;                        \
  const std::size_t interaction_width = (dims).interaction_width;          \
  static_cast<void>(inv_width);                                            \
  static_cast<void>(ev_width);                                             \
  static_cast<void>(degree_count);                                         \
  static_cast<void>(interaction_width)

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
    throw std::runtime_error("SO3LR post-attention tensor mismatch: " +
                             tensor.state_key);
}

DoubleView load_double(const NativeModel &model, const std::string &key,
                       const std::vector<std::size_t> &shape) {
  const auto &tensor = model.tensor(key);
  require_tensor(tensor, shape);
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  DoubleView result(key + "_post_attention", count);
  fill_view(result, count,
            [&](std::size_t i) { return model.float64(tensor, i); });
  return result;
}


void linear(cublasHandle_t handle, const DoubleView &input,
            std::size_t rows, std::size_t input_width,
            std::size_t output_width, const DoubleView &weight,
            const DoubleView &output) {
  const int m = checked_int(output_width, "output_width");
  const int n = checked_int(rows, "rows");
  const int k = checked_int(input_width, "input_width");
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  check_cublas(
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  weight.data(), k, input.data(), k, &beta, output.data(), m),
      "cublasDgemm(post-attention)");
}

void layer_norm(const DoubleView &input, const DoubleView &weight,
                const DoubleView &bias, std::size_t nodes, double epsilon,
                const DoubleView &output, const char *label,
                const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::size_t offset = node * inv_width;
        double mean = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel)
          mean += input(offset + channel);
        mean /= static_cast<double>(inv_width);
        double variance = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel) {
          const double difference = input(offset + channel) - mean;
          variance += difference * difference;
        }
        variance /= static_cast<double>(inv_width);
        const double inverse_sigma = 1.0 / Kokkos::sqrt(variance + epsilon);
        for (std::size_t channel = 0; channel < inv_width; ++channel)
          output(offset + channel) =
              (input(offset + channel) - mean) * inverse_sigma *
                  weight(channel) +
              bias(channel);
      });
}

// nn.RMSNorm(use_scale=False): y = x * rsqrt(mean(x^2) + eps). No parameters.
void rms_norm(const DoubleView &input, std::size_t nodes, double epsilon,
              const DoubleView &output, const char *label,
              const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::size_t offset = node * inv_width;
        double mean_square = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel) {
          const double value = input(offset + channel);
          mean_square += value * value;
        }
        mean_square /= static_cast<double>(inv_width);
        const double inverse_rms = 1.0 / Kokkos::sqrt(mean_square + epsilon);
        for (std::size_t channel = 0; channel < inv_width; ++channel)
          output(offset + channel) = input(offset + channel) * inverse_rms;
      });
}

// The LayerAffine branch is the unchanged v1 kernel, so a v1 model runs the
// exact same instruction stream it always has.
void apply_norm(NormMode mode, const DoubleView &input, const DoubleView &weight,
                const DoubleView &bias, std::size_t nodes, double epsilon,
                const DoubleView &output, const char *label,
                const ArchDims &dims) {
  switch (mode) {
    case NormMode::LayerAffine:
      layer_norm(input, weight, bias, nodes, epsilon, output, label, dims);
      return;
    case NormMode::RmsNoScale:
      rms_norm(input, nodes, epsilon, output, label, dims);
      return;
    case NormMode::Identity:
      Kokkos::deep_copy(output, input);
      return;
  }
}

void launch_attention_residual_norm1(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &d_att_inv, const DoubleView &d_att_ev,
    const DoubleView &ln_weight, const DoubleView &ln_bias, NormMode mode,
    std::size_t nodes, double epsilon, const DoubleView &att_inv,
    const DoubleView &att_ev, const DoubleView &norm1, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      "so3lr_post_attention_residual_inv",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        att_inv(i) = inv_features(i) + d_att_inv(i);
      });
  Kokkos::parallel_for(
      "so3lr_post_attention_residual_ev",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        att_ev(i) = ev_features(i) + d_att_ev(i);
      });
  apply_norm(mode, att_inv, ln_weight, ln_bias, nodes, epsilon, norm1,
             "so3lr_post_attention_layer_norm_1", dims);
}

void launch_residual_mlp1(cublasHandle_t handle, const DoubleView &norm1,
                          const DoubleView &weight1,
                          const DoubleView &bias1,
                          const DoubleView &weight2,
                          const DoubleView &bias2, std::size_t nodes,
                          const DoubleView &activation,
                          const DoubleView &hidden,
                          const DoubleView &mlp_output,
                          const DoubleView &post_mlp, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      "so3lr_post_attention_mlp_input_silu",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = norm1(i);
        activation(i) = value / (1.0 + Kokkos::exp(-value));
      });
  linear(handle, activation, nodes, inv_width, inv_width, weight1, hidden);
  Kokkos::parallel_for(
      "so3lr_post_attention_mlp_hidden_silu",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = hidden(i) + bias1(i % inv_width);
        hidden(i) = value / (1.0 + Kokkos::exp(-value));
      });
  linear(handle, hidden, nodes, inv_width, inv_width, weight2, mlp_output);
  Kokkos::parallel_for(
      "so3lr_post_attention_mlp_output_residual",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        mlp_output(i) += bias2(i % inv_width);
        post_mlp(i) = norm1(i) + mlp_output(i);
      });
}

void launch_interaction(cublasHandle_t handle, const DoubleView &att_ev,
                        const DoubleView &post_mlp,
                        const DoubleView &cg_rep,
                        const IndexView &degree_repeats,
                        const IndexView &degree_offsets,
                        const DoubleView &weight, const DoubleView &bias,
                        std::size_t nodes, const DoubleView &ev_invariants,
                        const DoubleView &concatenated,
                        const DoubleView &transformed, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      "so3lr_post_attention_l0_contraction",
      Kokkos::RangePolicy<>(0, nodes * degree_count),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / degree_count;
        const std::size_t degree = flat % degree_count;
        double value = 0.0;
        for (std::size_t local = 0; local < degree_repeats(degree); ++local) {
          const std::size_t channel = degree_offsets(degree) + local;
          const double ev = att_ev(node * ev_width + channel);
          value += ev * ev * cg_rep(channel);
        }
        ev_invariants(flat) = value;
      });
  Kokkos::parallel_for(
      "so3lr_post_attention_concat",
      Kokkos::RangePolicy<>(0, nodes * interaction_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / interaction_width;
        const std::size_t channel = flat % interaction_width;
        concatenated(flat) =
            channel < inv_width
                ? post_mlp(node * inv_width + channel)
                : ev_invariants(node * degree_count + channel - inv_width);
      });
  linear(handle, concatenated, nodes, interaction_width, interaction_width,
         weight, transformed);
  Kokkos::parallel_for(
      "so3lr_post_attention_interaction_bias",
      Kokkos::RangePolicy<>(0, nodes * interaction_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        transformed(i) += bias(i % interaction_width);
      });
}

void launch_final_residual_norm2(
    const DoubleView &att_ev, const DoubleView &post_mlp,
    const DoubleView &transformed, const IndexView &degree_offsets,
    const DoubleView &ln_weight, const DoubleView &ln_bias, NormMode mode,
    std::size_t nodes, double epsilon, const DoubleView &pre_norm2,
    const DoubleView &final_inv, const DoubleView &final_ev,
    const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      "so3lr_post_attention_interaction_inv_residual",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / inv_width;
        const std::size_t channel = flat % inv_width;
        pre_norm2(flat) =
            post_mlp(flat) + transformed(node * interaction_width + channel);
      });
  Kokkos::parallel_for(
      "so3lr_post_attention_interaction_ev_residual",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / ev_width;
        const std::size_t channel = flat % ev_width;
        std::size_t degree = 0;
        for (std::size_t slot = 1; slot < degree_count; ++slot)
          if (channel >= degree_offsets(slot)) degree = slot;
        const double gate =
            transformed(node * interaction_width + inv_width + degree);
        final_ev(flat) = att_ev(flat) * (1.0 + gate);
      });
  apply_norm(mode, pre_norm2, ln_weight, ln_bias, nodes, epsilon, final_inv,
             "so3lr_post_attention_layer_norm_2", dims);
}

void launch_pipeline(
    cublasHandle_t handle, const DoubleView &inv_features,
    const DoubleView &ev_features, const DoubleView &d_att_inv,
    const DoubleView &d_att_ev, const DoubleView &ln1_weight,
    const DoubleView &ln1_bias, const DoubleView &mlp_weight1,
    const DoubleView &mlp_bias1, const DoubleView &mlp_weight2,
    const DoubleView &mlp_bias2, const DoubleView &interaction_weight,
    const DoubleView &interaction_bias, const DoubleView &cg_rep,
    const IndexView &degree_repeats, const IndexView &degree_offsets,
    const DoubleView &ln2_weight, const DoubleView &ln2_bias,
    NormMode norm1_mode, NormMode norm2_mode,
    std::size_t nodes, double epsilon, const DoubleView &att_inv,
    const DoubleView &att_ev, const DoubleView &norm1,
    const DoubleView &activation, const DoubleView &hidden,
    const DoubleView &mlp_output, const DoubleView &post_mlp,
    const DoubleView &ev_invariants, const DoubleView &concatenated,
    const DoubleView &transformed, const DoubleView &pre_norm2,
    const DoubleView &final_inv, const DoubleView &final_ev,
    const ArchDims &dims) {
  launch_attention_residual_norm1(
      inv_features, ev_features, d_att_inv, d_att_ev, ln1_weight, ln1_bias,
      norm1_mode, nodes, epsilon, att_inv, att_ev, norm1, dims);
  launch_residual_mlp1(handle, norm1, mlp_weight1, mlp_bias1, mlp_weight2,
                       mlp_bias2, nodes, activation, hidden, mlp_output,
                       post_mlp, dims);
  launch_interaction(handle, att_ev, post_mlp, cg_rep, degree_repeats,
                     degree_offsets, interaction_weight, interaction_bias,
                     nodes, ev_invariants, concatenated, transformed, dims);
  launch_final_residual_norm2(att_ev, post_mlp, transformed, degree_offsets,
                              ln2_weight, ln2_bias, norm2_mode, nodes, epsilon,
                              pre_norm2, final_inv, final_ev, dims);
}

std::size_t dynamic_bytes(std::size_t nodes, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  const std::size_t doubles_per_node =
      2 * inv_width + 2 * ev_width +  // four inputs
      8 * inv_width + 2 * ev_width +  // residual/norm/MLP/final buffers
      degree_count + 2 * interaction_width;
  return nodes * doubles_per_node * sizeof(double);
}

}  // namespace

PostAttentionBlock0DeviceWorkspace::PostAttentionBlock0DeviceWorkspace(
    std::size_t nodes, const ArchDims &dims)
    : PostAttentionBlock0DeviceWorkspace(nodes, dims.invariant_width,
                                         dims.equivariant_width,
                                         dims.ev_heads) {}

PostAttentionBlock0DeviceWorkspace::PostAttentionBlock0DeviceWorkspace(
    std::size_t nodes, std::size_t inv_width, std::size_t ev_width,
    std::size_t degree_count)
    : attention_residual_inv("so3lr_full0_att_inv", nodes * inv_width),
      attention_residual_ev("so3lr_full0_att_ev", nodes * ev_width),
      layer_norm_1("so3lr_full0_norm1", nodes * inv_width),
      mlp_activation("so3lr_full0_mlp_activation", nodes * inv_width),
      mlp_hidden("so3lr_full0_mlp_hidden", nodes * inv_width),
      mlp_output("so3lr_full0_mlp_output", nodes * inv_width),
      post_mlp_inv("so3lr_full0_post_mlp", nodes * inv_width),
      interaction_ev_invariants("so3lr_full0_ev_invariants",
                                nodes * degree_count),
      interaction_concatenated("so3lr_full0_concat",
                               nodes * (inv_width + degree_count)),
      interaction_transformed("so3lr_full0_transformed",
                              nodes * (inv_width + degree_count)),
      pre_layer_norm_2_inv("so3lr_full0_pre_norm2", nodes * inv_width),
      final_inv("so3lr_full0_final_inv", nodes * inv_width),
      final_ev("so3lr_full0_final_ev", nodes * ev_width) {
  if (nodes == 0)
    throw std::runtime_error("SO3LR post-attention device workspace is empty");
}

KokkosPostAttentionBlock0::KokkosPostAttentionBlock0(
    const NativeModel &model, std::size_t block_index)
    : dims_(model.arch().dims()), block_index_(block_index) {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  const auto blocks = static_cast<std::size_t>(
      model.architecture_number("interaction_blocks"));
  if (block_index_ >= blocks)
    throw std::runtime_error("SO3LR post-attention block index out of range");
  const std::string block =
      "model.euclidean_transformers." + std::to_string(block_index_);
  const So3lrArchitecture &arch = model.arch();
  // Both SO3LR generations have the first norm and the first residual MLP and
  // no second residual MLP. The second norm is optional (present in SO3LR v1).
  if (arch.norm_1 == NormMode::Identity || !arch.residual_mlp_1 ||
      arch.residual_mlp_2)
    throw std::runtime_error("SO3LR transformer residual contract changed");
  norm_1_mode_ = arch.norm_1;
  norm_2_mode_ = arch.norm_2;
  layer_norm_epsilon_ = arch.layer_norm_epsilon;
  if (std::abs(layer_norm_epsilon_ - 1.0e-6) > 1.0e-15)
    throw std::runtime_error("SO3LR transformer LayerNorm epsilon changed");

  // Only an affine LayerNorm has parameters; RMSNorm(use_scale=False) has none.
  if (norm_1_mode_ == NormMode::LayerAffine) {
    layer_norm_1_weight_ =
        load_double(model, block + ".layer_norm_inv_1.weight", {inv_width});
    layer_norm_1_bias_ =
        load_double(model, block + ".layer_norm_inv_1.bias", {inv_width});
  }
  mlp_1_weight_1_ =
      load_double(model, block + ".mlp_1.1.weight", {inv_width, inv_width});
  mlp_1_bias_1_ = load_double(model, block + ".mlp_1.1.bias", {inv_width});
  mlp_1_weight_2_ =
      load_double(model, block + ".mlp_1.3.weight", {inv_width, inv_width});
  mlp_1_bias_2_ = load_double(model, block + ".mlp_1.3.bias", {inv_width});
  const std::string interaction = block + ".interaction_block";
  interaction_weight_ =
      load_double(model, interaction + ".linear_layer.weight",
                  {interaction_width, interaction_width});
  interaction_bias_ = load_double(model, interaction + ".linear_layer.bias",
                                  {interaction_width});
  interaction_cg_rep_ = load_double(
      model, interaction + ".so3_conv_invariants.cg_rep", {ev_width});
  if (norm_2_mode_ == NormMode::LayerAffine) {
    layer_norm_2_weight_ =
        load_double(model, block + ".layer_norm_inv_2.weight", {inv_width});
    layer_norm_2_bias_ =
        load_double(model, block + ".layer_norm_inv_2.bias", {inv_width});
  }

  const auto &repeats = model.tensor(interaction + ".degree_repeats");
  require_tensor(repeats, {degree_count}, "int64");
  for (std::size_t degree = 0; degree < degree_count; ++degree)
    if (model.int64(repeats, degree) !=
        static_cast<std::int64_t>(arch.degree_repeats[degree]))
      throw std::runtime_error("SO3LR interaction degree repeats changed");
  if (interaction_width != inv_width + degree_count)
    throw std::runtime_error("SO3LR interaction width mismatch");
  degree_repeats_ = IndexView("so3lr_post_degree_repeats", degree_count);
  degree_offsets_ = IndexView("so3lr_post_degree_offsets", degree_count);
  fill_view(degree_repeats_, degree_count,
            [&](std::size_t i) { return arch.degree_repeats[i]; });
  fill_view(degree_offsets_, degree_count,
            [&](std::size_t i) { return arch.degree_offsets[i]; });

  check_cublas(cublasCreate(&handle_), "cublasCreate");
  const auto stream = Kokkos::Cuda().cuda_stream();
  check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
  cudaStream_t configured = nullptr;
  check_cublas(cublasGetStream(handle_, &configured), "cublasGetStream");
  shared_kokkos_stream_ = configured == stream;
  const std::size_t persistent_doubles =
      2 * inv_width + 2 * (inv_width * inv_width + inv_width) +
      interaction_width * interaction_width + interaction_width + ev_width +
      2 * inv_width;
  persistent_device_bytes_ = persistent_doubles * sizeof(double) +
                             2 * degree_count * sizeof(std::size_t);
  checkpoint_contract_verified_ = true;
  Kokkos::fence();
}

KokkosPostAttentionBlock0::~KokkosPostAttentionBlock0() {
  if (handle_ != nullptr) {
    Kokkos::fence();
    static_cast<void>(cublasDestroy(handle_));
  }
}

void KokkosPostAttentionBlock0::launch_device(
    const Kokkos::View<double *> &inv_features,
    const Kokkos::View<double *> &ev_features,
    const Kokkos::View<double *> &d_attention_inv,
    const Kokkos::View<double *> &d_attention_ev,
    const PostAttentionBlock0DeviceWorkspace &workspace) const {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  const std::size_t nodes = inv_features.extent(0) / inv_width;
  if (nodes == 0 || inv_features.extent(0) != nodes * inv_width ||
      ev_features.extent(0) != nodes * ev_width ||
      d_attention_inv.extent(0) != nodes * inv_width ||
      d_attention_ev.extent(0) != nodes * ev_width ||
      workspace.final_inv.extent(0) != nodes * inv_width ||
      workspace.final_ev.extent(0) != nodes * ev_width)
    throw std::runtime_error("SO3LR post-attention device-view contract mismatch");
  launch_pipeline(
      handle_, inv_features, ev_features, d_attention_inv, d_attention_ev,
      layer_norm_1_weight_, layer_norm_1_bias_, mlp_1_weight_1_,
      mlp_1_bias_1_, mlp_1_weight_2_, mlp_1_bias_2_, interaction_weight_,
      interaction_bias_, interaction_cg_rep_, degree_repeats_,
      degree_offsets_, layer_norm_2_weight_, layer_norm_2_bias_,
      norm_1_mode_, norm_2_mode_, nodes,
      layer_norm_epsilon_, workspace.attention_residual_inv,
      workspace.attention_residual_ev, workspace.layer_norm_1,
      workspace.mlp_activation, workspace.mlp_hidden, workspace.mlp_output,
      workspace.post_mlp_inv, workspace.interaction_ev_invariants,
      workspace.interaction_concatenated, workspace.interaction_transformed,
      workspace.pre_layer_norm_2_inv, workspace.final_inv,
      workspace.final_ev, dims_);
}

PostAttentionBlock0Results KokkosPostAttentionBlock0::evaluate(
    const std::vector<double> &inv_features,
    const std::vector<double> &ev_features,
    const std::vector<double> &d_attention_inv,
    const std::vector<double> &d_attention_ev, std::size_t nodes) const {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  if (nodes == 0 || inv_features.size() != nodes * inv_width ||
      ev_features.size() != nodes * ev_width ||
      d_attention_inv.size() != nodes * inv_width ||
      d_attention_ev.size() != nodes * ev_width)
    throw std::runtime_error("SO3LR post-attention input contract mismatch");

  DoubleView inv("so3lr_post_inv", nodes * inv_width);
  DoubleView ev("so3lr_post_ev", nodes * ev_width);
  DoubleView d_inv("so3lr_post_d_att_inv", nodes * inv_width);
  DoubleView d_ev("so3lr_post_d_att_ev", nodes * ev_width);
  fill_view(inv, inv_features.size(),
            [&](std::size_t i) { return inv_features[i]; });
  fill_view(ev, ev_features.size(),
            [&](std::size_t i) { return ev_features[i]; });
  fill_view(d_inv, d_attention_inv.size(),
            [&](std::size_t i) { return d_attention_inv[i]; });
  fill_view(d_ev, d_attention_ev.size(),
            [&](std::size_t i) { return d_attention_ev[i]; });

  DoubleView att_inv("so3lr_post_att_inv", nodes * inv_width);
  DoubleView att_ev("so3lr_post_att_ev", nodes * ev_width);
  DoubleView norm1("so3lr_post_norm1", nodes * inv_width);
  DoubleView activation("so3lr_post_mlp_activation", nodes * inv_width);
  DoubleView hidden("so3lr_post_mlp_hidden", nodes * inv_width);
  DoubleView mlp_output("so3lr_post_mlp_output", nodes * inv_width);
  DoubleView post_mlp("so3lr_post_mlp_residual", nodes * inv_width);
  DoubleView invariants("so3lr_post_ev_invariants", nodes * degree_count);
  DoubleView concatenated("so3lr_post_concat", nodes * interaction_width);
  DoubleView transformed("so3lr_post_transformed", nodes * interaction_width);
  DoubleView pre_norm2("so3lr_post_pre_norm2", nodes * inv_width);
  DoubleView final_inv("so3lr_post_final_inv", nodes * inv_width);
  DoubleView final_ev("so3lr_post_final_ev", nodes * ev_width);
  launch_pipeline(
      handle_, inv, ev, d_inv, d_ev, layer_norm_1_weight_,
      layer_norm_1_bias_, mlp_1_weight_1_, mlp_1_bias_1_, mlp_1_weight_2_,
      mlp_1_bias_2_, interaction_weight_, interaction_bias_,
      interaction_cg_rep_, degree_repeats_, degree_offsets_,
      layer_norm_2_weight_, layer_norm_2_bias_, norm_1_mode_,
        norm_2_mode_, nodes, layer_norm_epsilon_,
      att_inv, att_ev, norm1, activation, hidden, mlp_output, post_mlp,
      invariants, concatenated, transformed, pre_norm2, final_inv, final_ev,
        dims_);
  Kokkos::fence();
  PostAttentionBlock0Results result;
  result.attention_residual_inv = copy_host(att_inv);
  result.attention_residual_ev = copy_host(att_ev);
  result.layer_norm_1 = copy_host(norm1);
  result.post_mlp_inv = copy_host(post_mlp);
  result.interaction_ev_invariants = copy_host(invariants);
  result.interaction_transformed = copy_host(transformed);
  result.pre_layer_norm_2_inv = copy_host(pre_norm2);
  result.final_inv = copy_host(final_inv);
  result.final_ev = copy_host(final_ev);
  return result;
}

PostAttentionBlock0Benchmark KokkosPostAttentionBlock0::benchmark(
    std::size_t nodes, std::size_t repetitions) const {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  if (nodes == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR post-attention benchmark sizes invalid");
  DoubleView inv("so3lr_post_bench_inv", nodes * inv_width);
  DoubleView ev("so3lr_post_bench_ev", nodes * ev_width);
  DoubleView d_inv("so3lr_post_bench_d_inv", nodes * inv_width);
  DoubleView d_ev("so3lr_post_bench_d_ev", nodes * ev_width);
  Kokkos::parallel_for(
      "so3lr_post_bench_inv_init",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = (static_cast<double>(i % 257) - 128.0) / 181.0;
        inv(i) = value;
        d_inv(i) = 0.15 * value + 0.02;
      });
  Kokkos::parallel_for(
      "so3lr_post_bench_ev_init",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = (static_cast<double>(i % 61) - 30.0) / 43.0;
        ev(i) = value;
        d_ev(i) = -0.12 * value + 0.01;
      });
  DoubleView att_inv("so3lr_post_bench_att_inv", nodes * inv_width);
  DoubleView att_ev("so3lr_post_bench_att_ev", nodes * ev_width);
  DoubleView norm1("so3lr_post_bench_norm1", nodes * inv_width);
  DoubleView activation("so3lr_post_bench_activation", nodes * inv_width);
  DoubleView hidden("so3lr_post_bench_hidden", nodes * inv_width);
  DoubleView mlp_output("so3lr_post_bench_mlp_output", nodes * inv_width);
  DoubleView post_mlp("so3lr_post_bench_post_mlp", nodes * inv_width);
  DoubleView invariants("so3lr_post_bench_invariants", nodes * degree_count);
  DoubleView concatenated("so3lr_post_bench_concat", nodes * interaction_width);
  DoubleView transformed("so3lr_post_bench_transformed", nodes * interaction_width);
  DoubleView pre_norm2("so3lr_post_bench_pre_norm2", nodes * inv_width);
  DoubleView final_inv("so3lr_post_bench_final_inv", nodes * inv_width);
  DoubleView final_ev("so3lr_post_bench_final_ev", nodes * ev_width);

  auto run_residual_norm1 = [&]() {
    launch_attention_residual_norm1(
        inv, ev, d_inv, d_ev, layer_norm_1_weight_, layer_norm_1_bias_,
        norm_1_mode_, nodes, layer_norm_epsilon_, att_inv, att_ev, norm1,
        dims_);
  };
  auto run_mlp = [&]() {
    launch_residual_mlp1(handle_, norm1, mlp_1_weight_1_, mlp_1_bias_1_,
                         mlp_1_weight_2_, mlp_1_bias_2_, nodes, activation,
                         hidden, mlp_output, post_mlp, dims_);
  };
  auto run_interaction = [&]() {
    launch_interaction(handle_, att_ev, post_mlp, interaction_cg_rep_,
                       degree_repeats_, degree_offsets_, interaction_weight_,
                       interaction_bias_, nodes, invariants, concatenated,
                       transformed, dims_);
  };
  auto run_residual_norm2 = [&]() {
    launch_final_residual_norm2(
        att_ev, post_mlp, transformed, degree_offsets_, layer_norm_2_weight_,
        layer_norm_2_bias_, norm_2_mode_, nodes, layer_norm_epsilon_,
        pre_norm2, final_inv, final_ev, dims_);
  };
  auto time_phase = [&](auto launch) {
    launch();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (std::size_t i = 0; i < repetitions; ++i) launch();
    Kokkos::fence();
    return timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  };

  run_residual_norm1();
  run_mlp();
  run_interaction();
  run_residual_norm2();
  Kokkos::fence();
  const double residual_norm1_ms = time_phase(run_residual_norm1);
  const double mlp_ms = time_phase(run_mlp);
  const double interaction_ms = time_phase(run_interaction);
  const double residual_norm2_ms = time_phase(run_residual_norm2);

  for (int warmup = 0; warmup < 2; ++warmup)
    launch_pipeline(
        handle_, inv, ev, d_inv, d_ev, layer_norm_1_weight_,
        layer_norm_1_bias_, mlp_1_weight_1_, mlp_1_bias_1_, mlp_1_weight_2_,
        mlp_1_bias_2_, interaction_weight_, interaction_bias_,
        interaction_cg_rep_, degree_repeats_, degree_offsets_,
        layer_norm_2_weight_, layer_norm_2_bias_, norm_1_mode_,
        norm_2_mode_, nodes, layer_norm_epsilon_,
        att_inv, att_ev, norm1, activation, hidden, mlp_output, post_mlp,
        invariants, concatenated, transformed, pre_norm2, final_inv, final_ev,
        dims_);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_pipeline(
        handle_, inv, ev, d_inv, d_ev, layer_norm_1_weight_,
        layer_norm_1_bias_, mlp_1_weight_1_, mlp_1_bias_1_, mlp_1_weight_2_,
        mlp_1_bias_2_, interaction_weight_, interaction_bias_,
        interaction_cg_rep_, degree_repeats_, degree_offsets_,
        layer_norm_2_weight_, layer_norm_2_bias_, norm_1_mode_,
        norm_2_mode_, nodes, layer_norm_epsilon_,
        att_inv, att_ev, norm1, activation, hidden, mlp_output, post_mlp,
        invariants, concatenated, transformed, pre_norm2, final_inv, final_ev,
        dims_);
  Kokkos::fence();
  const double pipeline_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_post_bench_checksum", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += final_inv((nodes / 3) * inv_width + 17);
        if (which == 1) update += final_ev((nodes / 2) * ev_width + 9);
      },
      checksum);
  Kokkos::fence();
  PostAttentionBlock0Benchmark result;
  result.nodes = nodes;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes = persistent_device_bytes_ + dynamic_bytes(nodes, dims_);
  result.host_boundary_bytes_per_iteration = 0;
  result.residual_norm1_milliseconds = residual_norm1_ms;
  result.residual_mlp1_milliseconds = mlp_ms;
  result.interaction_milliseconds = interaction_ms;
  result.residual_norm2_milliseconds = residual_norm2_ms;
  result.phase_sum_milliseconds = residual_norm1_ms + mlp_ms + interaction_ms +
                                  residual_norm2_ms;
  result.pipeline_milliseconds = pipeline_ms;
  result.nodes_per_second =
      static_cast<double>(nodes) / (pipeline_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
