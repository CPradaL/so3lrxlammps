#pragma once

// Runtime architecture descriptor for the native SO3LR runtime.
//
// The `.so3lr` manifest already describes the trained model's shape; until
// 0.3.0 the runtime only ever compared that description against compile-time
// constants. This header turns the description into the single source of truth
// from which kernels size themselves.
//
// Host-only: no Kokkos, no CUDA. `ArchDims` is the trivially copyable subset
// that may be captured by value inside a device lambda.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace so3lr {

class NativeModel;

// Output-head kinds the runtime knows how to name. Parsing a head name that is
// not in this list is a capability failure, not a silent skip.
enum class HeadKind {
  AtomicEnergy,
  PartialCharges,
  HirshfeldRatios,
  C6Ratios,
};

struct HeadSpec {
  HeadKind kind = HeadKind::AtomicEnergy;
  std::string manifest_name;  // e.g. "atomic_energy_output_block"
  std::string state_prefix;   // e.g. "model.atomic_energy_output_block"
};

// Short-range repulsion family.
enum class RepulsionKind {
  None,
  LearnedZbl,  // v1: ten softplus-decoded parameters
  NlhTable,    // parameter-free (92,92,3) lookup tables
};

// How attention logits are normalised (already runtime in 0.2.0-rc1, mirrored
// here so every consumer reads one structure).
enum class AttentionNormKind {
  Identity,
  SqrtNumFeatures,
  AvgNumNeighbors,
};

// Which normalisation a transformer block applies at a given position.
// LayerAffine: nn.LayerNorm with learnable scale and bias (v1).
// RmsNoScale:  nn.RMSNorm(use_scale=False), parameter-free.
// Identity:    the norm is disabled (layer_normalization_2 = false).
enum class NormMode {
  Identity,
  LayerAffine,
  RmsNoScale,
};

// Dimensions a device kernel may need. Trivially copyable; 32-bit because no
// SO3LR dimension approaches 2^32 and a narrower closure is cheaper.
struct ArchDims {
  std::uint32_t invariant_width = 0;      // F
  std::uint32_t inv_heads = 0;            // H1
  std::uint32_t inv_head_width = 0;       // F / H1
  std::uint32_t ev_heads = 0;             // H2 == len(degrees)
  std::uint32_t ev_head_width = 0;        // F / H2
  std::uint32_t rbf_width = 0;            // R
  std::uint32_t equivariant_width = 0;    // E == sum(degree_repeats)
  std::uint32_t interaction_width = 0;    // F + H2
  std::uint32_t sh_width = 0;             // distinct (l,m) count
  std::uint32_t max_degree = 0;           // Lmax
  std::uint32_t ev_filter_hidden = 0;     // F / 4, spherical filter hidden
  std::uint32_t head_hidden_width = 0;    // energy/charge head hidden layer
};

// The SO3LR v1 widths (F=128, 4 heads of 32, degrees {1,2,3,4}, R=32). Used as
// the default by workspaces and kernels that predate the descriptor, so the
// standalone v1 kernel tests keep compiling and running unchanged.
inline ArchDims v1_arch_dims() {
  ArchDims d;
  d.invariant_width = 128;
  d.inv_heads = 4;
  d.inv_head_width = 32;
  d.ev_heads = 4;
  d.ev_head_width = 32;
  d.rbf_width = 32;
  d.equivariant_width = 24;
  d.interaction_width = 132;
  d.sh_width = 24;
  d.max_degree = 4;
  d.ev_filter_hidden = 32;
  d.head_hidden_width = 128;
  return d;
}

// Everything that follows from the per-degree multiplicities alone.
//
// Split out of the descriptor because it is the part of the derivation that a
// duplicated degree list (e.g. {1,1,2,2,3,3,4,4}) makes non-obvious,
// and because it can then be tested without a model file.
struct DegreeLayout {
  std::vector<int> degrees;                  // l per feature block
  std::vector<std::size_t> degree_repeats;   // 2l+1 per feature block
  std::vector<std::size_t> degree_offsets;   // exclusive prefix sum
  int max_degree = 0;
  std::size_t equivariant_width = 0;         // sum(degree_repeats)
  std::size_t sh_width = 0;                  // distinct (l,m) count
  std::vector<std::size_t> feature_to_sh_channel;
  bool duplicated = false;
};

// Throws std::runtime_error when `repeats` is not a non-decreasing sequence of
// odd numbers of the form 2l+1 with l >= 1.
DegreeLayout compute_degree_layout(const std::vector<std::size_t> &repeats);

struct So3lrArchitecture {
  // --- interaction stack ------------------------------------------------
  std::size_t layers = 0;              // interaction_blocks
  std::size_t invariant_width = 0;     // F
  std::size_t inv_heads = 0;           // H1
  std::size_t inv_head_width = 0;      // F / H1
  std::size_t ev_heads = 0;            // H2 (== len(degrees))
  std::size_t ev_head_width = 0;       // F / H2
  std::size_t rbf_width = 0;           // R
  std::size_t equivariant_width = 0;   // E
  std::size_t interaction_width = 0;   // F + H2

  // --- degrees ----------------------------------------------------------
  // `degrees` may repeat a value (e.g. {1,1,2,2,3,3,4,4}). When it
  // does, `equivariant_width` exceeds `sh_width` and `feature_to_sh_channel`
  // is the non-trivial expansion map from a feature channel to the unique
  // spherical-harmonic channel it replicates.
  std::vector<int> degrees;
  std::vector<std::size_t> degree_repeats;   // 2l+1 per degree slot
  std::vector<std::size_t> degree_offsets;   // exclusive prefix sum
  int max_degree = 0;
  std::size_t sh_width = 0;
  std::vector<std::size_t> feature_to_sh_channel;
  bool degrees_duplicated = false;

  // --- element tables ---------------------------------------------------
  std::size_t atomic_number_capacity = 0;  // energy-head embedding extent
  std::size_t lr_element_capacity = 0;     // charge/Hirshfeld embedding extent
  // Hidden width of the energy and charge heads (rows of their first layer).
  // Usually equal to F; it can be smaller (e.g. 256 features to 128 hidden units).
  std::size_t head_hidden_width = 0;

  // --- geometry and physics --------------------------------------------
  double short_range_cutoff = 0.0;
  double long_range_cutoff = 0.0;
  double embedding_scale = 1.0;
  std::size_t num_embeddings = 1;
  AttentionNormKind attention_norm = AttentionNormKind::Identity;
  double avg_num_neighbors = 1.0;
  // The divisor the attention kernels actually apply: 1, sqrt(head width) or
  // the average neighbour count. Resolved once here so no kernel has to walk
  // the manifest's module tree to find it.
  double attention_norm_value = 1.0;
  RepulsionKind repulsion = RepulsionKind::None;

  // --- optional operator flags (all false for a v1 checkpoint) ----------------
  bool use_rms_norm = false;
  bool qk_norm = false;
  bool use_residual_scalars = false;
  bool layer_normalization_1 = true;
  bool layer_normalization_2 = true;
  NormMode norm_1 = NormMode::LayerAffine;  // after the attention residual
  NormMode norm_2 = NormMode::LayerAffine;  // after the interaction residual
  bool residual_mlp_1 = true;
  bool residual_mlp_2 = false;
  double layer_norm_epsilon = 1.0e-6;
  bool qk_nonlinearity_identity = true;

  // --- output heads, in manifest order ----------------------------------
  std::vector<HeadSpec> heads;

  // --- convenience ------------------------------------------------------
  ArchDims dims() const;
  bool has_head(HeadKind kind) const;
  const HeadSpec &head(HeadKind kind) const;   // throws when absent
  std::string summary() const;                 // one line, for logs and errors
};

// Derive the descriptor from a fully validated model. Throws
// std::runtime_error with a message naming the offending field when the
// manifest is internally inconsistent.
So3lrArchitecture describe_architecture(const NativeModel &model);

const char *to_string(HeadKind kind);
const char *to_string(RepulsionKind kind);
const char *to_string(AttentionNormKind kind);
const char *to_string(NormMode mode);

}  // namespace so3lr
