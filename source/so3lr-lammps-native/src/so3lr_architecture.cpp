#include "so3lr/so3lr_architecture.hpp"

#include "so3lr/native_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace so3lr {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("SO3LR architecture: " + message);
}

const Json *find(const Json &object, const std::string &key) {
  const auto &entries = object.object();
  const auto found = entries.find(key);
  return found == entries.end() ? nullptr : &found->second;
}

std::size_t required_size(const Json &architecture, const std::string &key) {
  const Json *entry = find(architecture, key);
  if (entry == nullptr) fail("manifest is missing architecture key '" + key + "'");
  return static_cast<std::size_t>(entry->unsigned_integer());
}

double optional_number(const Json &architecture, const std::string &key,
                       double fallback) {
  const Json *entry = find(architecture, key);
  return entry == nullptr ? fallback : entry->number();
}

bool optional_flag(const Json &architecture, const std::string &key,
                   bool fallback) {
  const Json *entry = find(architecture, key);
  if (entry == nullptr) return fallback;
  // The exporter writes booleans as 0/1 numbers in some places and as JSON
  // booleans in others; accept both rather than depending on which.
  try {
    return entry->boolean();
  } catch (const std::exception &) {
    return entry->number() != 0.0;
  }
}

// The per-degree multiplicities are carried as an int64 tensor rather than a
// manifest field. Block 0's attention block always has one.
std::vector<std::size_t> read_degree_repeats(const NativeModel &model,
                                             std::size_t expected_count) {
  const std::string key =
      "model.euclidean_transformers.0.euclidean_attention_block.degree_repeats";
  const TensorRecord &record = model.tensor(key);
  if (record.dtype != "int64" || record.shape.size() != 1)
    fail("tensor '" + key + "' must be a one-dimensional int64 tensor");
  if (record.shape[0] != expected_count)
    fail("degree_repeats has " + std::to_string(record.shape[0]) +
         " entries but the manifest declares euclidean_degree_channels = " +
         std::to_string(expected_count));

  std::vector<std::size_t> repeats;
  repeats.reserve(expected_count);
  for (std::size_t i = 0; i < expected_count; ++i) {
    const std::int64_t value = model.int64(record, i);
    // The 2l+1 rule and the ordering rule belong to compute_degree_layout;
    // here we only guarantee the cast below is meaningful.
    if (value < 1)
      fail("degree_repeats[" + std::to_string(i) + "] = " +
           std::to_string(value) + " is not positive");
    repeats.push_back(static_cast<std::size_t>(value));
  }
  return repeats;
}

// A first extent of a named tensor, when the tensor is optional.
std::size_t leading_extent_or(const NativeModel &model, const std::string &key,
                              std::size_t fallback) {
  try {
    const TensorRecord &record = model.tensor(key);
    return record.shape.empty() ? fallback : record.shape[0];
  } catch (const std::exception &) {
    return fallback;
  }
}

const std::map<std::string, HeadKind> &head_registry() {
  static const std::map<std::string, HeadKind> registry = {
      {"atomic_energy_output_block", HeadKind::AtomicEnergy},
      {"partial_charges_output_block", HeadKind::PartialCharges},
      {"hirshfeld_output_block", HeadKind::HirshfeldRatios},
      {"c6_ratios_output_block", HeadKind::C6Ratios},
  };
  return registry;
}

AttentionNormKind parse_attention_norm(const std::string &mode) {
  if (mode == "identity") return AttentionNormKind::Identity;
  if (mode == "sqrt_num_features") return AttentionNormKind::SqrtNumFeatures;
  if (mode == "avg_num_neighbors") return AttentionNormKind::AvgNumNeighbors;
  fail("unsupported message_normalization '" + mode + "'");
}


// Attributes of a module whose path ends with `suffix`, or nullptr. Only
// PyTorch-exported manifests carry a module tree; a flax export states the
// same facts as architecture fields instead.
const Json *module_attributes(const NativeModel &model, const std::string &suffix) {
  const Json *modules = find(model.manifest(), "modules");
  if (modules == nullptr) return nullptr;
  for (const auto &module : modules->array()) {
    const auto &path = module.at("path").string();
    if (path.size() >= suffix.size() &&
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0)
      return &module.at("attributes");
  }
  return nullptr;
}

struct AttentionNorm {
  AttentionNormKind kind = AttentionNormKind::Identity;
  double avg_num_neighbors = 1.0;
};

AttentionNorm read_attention_norm(const NativeModel &model) {
  // Preferred: explicit manifest fields. A PyTorch-exported manifest has
  // neither, so fall back to the module tree -- which is the only place the
  // 0.2.0-rc1 exporter recorded them.
  const Json &architecture = model.manifest().at("architecture");
  if (const Json *mode = find(architecture, "message_normalization")) {
    AttentionNorm norm;
    norm.kind = parse_attention_norm(mode->string());
    if (const Json *average = find(architecture, "avg_num_neighbors"))
      norm.avg_num_neighbors = average->number();
    else if (norm.kind == AttentionNormKind::AvgNumNeighbors)
      fail("message_normalization is 'avg_num_neighbors' but the manifest "
           "declares no avg_num_neighbors");
    return norm;
  }

  const std::string suffix = "model.euclidean_transformers.0.euclidean_attention_block";
  for (const auto &module : model.manifest().at("modules").array()) {
    const auto &path = module.at("path").string();
    if (path.size() < suffix.size() ||
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    const Json &attributes = module.at("attributes");
    AttentionNorm norm;
    norm.kind = parse_attention_norm(attributes.at("message_normalization").string());
    if (const Json *average = find(attributes, "avg_num_neighbors"))
      norm.avg_num_neighbors = average->number();
    return norm;
  }
  fail("manifest has no euclidean_attention_block module for block 0");
}

}  // namespace

const char *to_string(HeadKind kind) {
  switch (kind) {
    case HeadKind::AtomicEnergy: return "atomic_energy";
    case HeadKind::PartialCharges: return "partial_charges";
    case HeadKind::HirshfeldRatios: return "hirshfeld_ratios";
    case HeadKind::C6Ratios: return "c6_ratios";
  }
  return "unknown";
}

const char *to_string(RepulsionKind kind) {
  switch (kind) {
    case RepulsionKind::None: return "none";
    case RepulsionKind::LearnedZbl: return "zbl_learned";
    case RepulsionKind::NlhTable: return "nlh_table";
  }
  return "unknown";
}

const char *to_string(AttentionNormKind kind) {
  switch (kind) {
    case AttentionNormKind::Identity: return "identity";
    case AttentionNormKind::SqrtNumFeatures: return "sqrt_num_features";
    case AttentionNormKind::AvgNumNeighbors: return "avg_num_neighbors";
  }
  return "unknown";
}

const char *to_string(NormMode mode) {
  switch (mode) {
    case NormMode::Identity: return "identity";
    case NormMode::LayerAffine: return "layer_norm";
    case NormMode::RmsNoScale: return "rms_norm";
  }
  return "unknown";
}

ArchDims So3lrArchitecture::dims() const {
  ArchDims d;
  d.invariant_width = static_cast<std::uint32_t>(invariant_width);
  d.inv_heads = static_cast<std::uint32_t>(inv_heads);
  d.inv_head_width = static_cast<std::uint32_t>(inv_head_width);
  d.ev_heads = static_cast<std::uint32_t>(ev_heads);
  d.ev_head_width = static_cast<std::uint32_t>(ev_head_width);
  d.rbf_width = static_cast<std::uint32_t>(rbf_width);
  d.equivariant_width = static_cast<std::uint32_t>(equivariant_width);
  d.interaction_width = static_cast<std::uint32_t>(interaction_width);
  d.sh_width = static_cast<std::uint32_t>(sh_width);
  d.max_degree = static_cast<std::uint32_t>(max_degree);
  d.ev_filter_hidden = static_cast<std::uint32_t>(invariant_width / 4);
  d.head_hidden_width = static_cast<std::uint32_t>(head_hidden_width);
  return d;
}

bool So3lrArchitecture::has_head(HeadKind kind) const {
  return std::any_of(heads.begin(), heads.end(),
                     [kind](const HeadSpec &h) { return h.kind == kind; });
}

const HeadSpec &So3lrArchitecture::head(HeadKind kind) const {
  for (const HeadSpec &spec : heads)
    if (spec.kind == kind) return spec;
  fail(std::string("model has no '") + to_string(kind) + "' output head");
}

std::string So3lrArchitecture::summary() const {
  std::ostringstream out;
  out << "layers=" << layers << " features=" << invariant_width
      << " heads=" << inv_heads << "x" << inv_head_width
      << " degrees=[";
  for (std::size_t i = 0; i < degrees.size(); ++i)
    out << (i ? "," : "") << degrees[i];
  out << "] ev_width=" << equivariant_width << " sh_width=" << sh_width
      << " rbf=" << rbf_width << " r_cut=" << short_range_cutoff
      << " r_cut_lr=" << long_range_cutoff
      << " repulsion=" << to_string(repulsion) << " heads={";
  for (std::size_t i = 0; i < heads.size(); ++i)
    out << (i ? "," : "") << to_string(heads[i].kind);
  out << "}";
  return out.str();
}

DegreeLayout compute_degree_layout(const std::vector<std::size_t> &repeats) {
  DegreeLayout layout;
  if (repeats.empty()) fail("the model declares no equivariant degrees");

  layout.degree_repeats = repeats;
  layout.degree_offsets.reserve(repeats.size());
  layout.degrees.reserve(repeats.size());

  std::size_t running = 0;
  for (std::size_t i = 0; i < repeats.size(); ++i) {
    const std::size_t repeat = repeats[i];
    if (repeat < 3 || repeat % 2 == 0)
      fail("degree_repeats[" + std::to_string(i) + "] = " +
           std::to_string(repeat) +
           " is not of the form 2l+1 with l >= 1 (degree 0 is carried by the "
           "invariant stream, not by the equivariant one)");
    layout.degree_offsets.push_back(running);
    running += repeat;
    layout.degrees.push_back(static_cast<int>((repeat - 1) / 2));
  }
  layout.equivariant_width = running;

  if (!std::is_sorted(layout.degrees.begin(), layout.degrees.end()))
    fail("degrees must be sorted non-decreasing; the Clebsch-Gordan reduction "
         "groups by sorted unique degree while the segment identifiers follow "
         "the given order, and the two disagree otherwise");
  layout.max_degree = layout.degrees.back();

  // Distinct spherical-harmonic channels, l = 1..Lmax ascending. A duplicated
  // degree list reuses the same harmonics across several feature blocks, so
  // sh_width can be smaller than equivariant_width.
  std::vector<std::size_t> unique_offset(
      static_cast<std::size_t>(layout.max_degree) + 1, 0);
  std::size_t unique_running = 0;
  for (int l = 1; l <= layout.max_degree; ++l) {
    unique_offset[static_cast<std::size_t>(l)] = unique_running;
    unique_running += static_cast<std::size_t>(2 * l + 1);
  }
  layout.sh_width = unique_running;
  layout.duplicated = layout.equivariant_width != layout.sh_width;

  // The expansion map used by the geometry kernel: feature channel -> the
  // unique harmonic it replicates. The identity when nothing is duplicated,
  // which is what keeps the v1 path free of any extra indirection.
  layout.feature_to_sh_channel.resize(layout.equivariant_width);
  for (std::size_t d = 0; d < repeats.size(); ++d) {
    const std::size_t base =
        unique_offset[static_cast<std::size_t>(layout.degrees[d])];
    for (std::size_t k = 0; k < layout.degree_repeats[d]; ++k)
      layout.feature_to_sh_channel[layout.degree_offsets[d] + k] = base + k;
  }
  return layout;
}

So3lrArchitecture describe_architecture(const NativeModel &model) {
  const Json &architecture = model.manifest().at("architecture");
  So3lrArchitecture arch;

  arch.layers = required_size(architecture, "interaction_blocks");
  arch.invariant_width = required_size(architecture, "invariant_features");
  arch.inv_heads = required_size(architecture, "attention_heads");
  arch.inv_head_width = required_size(architecture, "attention_head_width");
  arch.ev_heads = required_size(architecture, "euclidean_degree_channels");
  arch.rbf_width = required_size(architecture, "radial_basis_features");
  arch.atomic_number_capacity = required_size(architecture, "atomic_number_capacity");

  if (arch.layers == 0) fail("interaction_blocks must be at least 1");
  if (arch.invariant_width == 0) fail("invariant_features must be at least 1");
  if (arch.inv_heads == 0 || arch.ev_heads == 0)
    fail("attention_heads and euclidean_degree_channels must be at least 1");
  if (arch.inv_heads * arch.inv_head_width != arch.invariant_width)
    fail("attention_heads * attention_head_width (" +
         std::to_string(arch.inv_heads * arch.inv_head_width) +
         ") does not equal invariant_features (" +
         std::to_string(arch.invariant_width) + ")");
  if (arch.invariant_width % arch.ev_heads != 0)
    fail("invariant_features is not divisible by euclidean_degree_channels");
  arch.ev_head_width = arch.invariant_width / arch.ev_heads;
  arch.interaction_width = arch.invariant_width + arch.ev_heads;

  // --- degrees ----------------------------------------------------------
  const DegreeLayout layout =
      compute_degree_layout(read_degree_repeats(model, arch.ev_heads));
  arch.degrees = layout.degrees;
  arch.degree_repeats = layout.degree_repeats;
  arch.degree_offsets = layout.degree_offsets;
  arch.max_degree = layout.max_degree;
  arch.equivariant_width = layout.equivariant_width;
  arch.sh_width = layout.sh_width;
  arch.degrees_duplicated = layout.duplicated;
  arch.feature_to_sh_channel = layout.feature_to_sh_channel;

  // --- geometry and physics --------------------------------------------
  arch.short_range_cutoff = optional_number(architecture, "short_range_cutoff_angstrom", 0.0);
  arch.long_range_cutoff = optional_number(architecture, "long_range_cutoff_angstrom", 0.0);
  arch.embedding_scale = optional_number(architecture, "embedding_scale", 1.0);
  arch.num_embeddings = static_cast<std::size_t>(
      optional_number(architecture, "num_embeddings", 1.0));
  const AttentionNorm norm = read_attention_norm(model);
  arch.attention_norm = norm.kind;
  arch.avg_num_neighbors = norm.avg_num_neighbors;
  switch (arch.attention_norm) {
    case AttentionNormKind::Identity:
      arch.attention_norm_value = 1.0;
      break;
    case AttentionNormKind::SqrtNumFeatures:
      // Historically computed from the invariant head width; keep it exactly.
      arch.attention_norm_value =
          std::sqrt(static_cast<double>(arch.inv_head_width));
      break;
    case AttentionNormKind::AvgNumNeighbors:
      arch.attention_norm_value = arch.avg_num_neighbors;
      break;
  }
  // `repulsion_type` is explicit; `zbl_enabled` is the v1 manifest's implicit
  // spelling of the same thing and is the fallback.
  if (const Json *kind = find(architecture, "repulsion_type")) {
    const std::string &name = kind->string();
    if (name == "zbl_learned") arch.repulsion = RepulsionKind::LearnedZbl;
    else if (name == "nlh") arch.repulsion = RepulsionKind::NlhTable;
    else if (name == "none") arch.repulsion = RepulsionKind::None;
    else fail("unknown repulsion_type '" + name + "'");
  } else {
    arch.repulsion = optional_flag(architecture, "zbl_enabled", false)
                         ? RepulsionKind::LearnedZbl
                         : RepulsionKind::None;
  }

  // --- optional operator flags (absent from a v1 manifest) --------------------
  arch.use_rms_norm = optional_flag(architecture, "use_rms_norm", false);
  arch.qk_norm = optional_flag(architecture, "qk_norm", false);
  arch.use_residual_scalars = optional_flag(architecture, "use_residual_scalars", false);
  // Block contract: explicit manifest fields when present, otherwise the
  // PyTorch module tree, otherwise the v1 defaults.
  const Json *block0 = module_attributes(model, "model.euclidean_transformers.0");
  auto block_flag = [&](const std::string &key, bool fallback) {
    if (const Json *entry = find(architecture, key)) return entry->boolean();
    if (block0 != nullptr)
      if (const Json *entry = find(*block0, key)) return entry->boolean();
    return fallback;
  };
  arch.layer_normalization_1 = block_flag("layer_normalization_1", true);
  arch.layer_normalization_2 = block_flag("layer_normalization_2", true);
  // The same module choice (use_rms_norm) applies to both positions; each
  // position can independently be switched off.
  const NormMode enabled_norm =
      arch.use_rms_norm ? NormMode::RmsNoScale : NormMode::LayerAffine;
  arch.norm_1 = arch.layer_normalization_1 ? enabled_norm : NormMode::Identity;
  arch.norm_2 = arch.layer_normalization_2 ? enabled_norm : NormMode::Identity;
  arch.residual_mlp_1 = block_flag("residual_mlp_1", true);
  arch.residual_mlp_2 = block_flag("residual_mlp_2", false);

  if (const Json *eps = find(architecture, "layer_norm_epsilon")) {
    arch.layer_norm_epsilon = eps->number();
  } else if (const Json *ln =
                 module_attributes(model, "model.euclidean_transformers.0.layer_norm_inv_1")) {
    if (const Json *entry = find(*ln, "eps")) arch.layer_norm_epsilon = entry->number();
  }

  if (const Json *qk = find(architecture, "qk_nonlinearity")) {
    arch.qk_nonlinearity_identity = qk->string() == "identity";
  } else if (const Json *modules = find(model.manifest(), "modules")) {
    const std::string suffix =
        "model.euclidean_transformers.0.euclidean_attention_block.qk_non_linearity";
    for (const auto &module : modules->array()) {
      const auto &path = module.at("path").string();
      if (path.size() >= suffix.size() &&
          path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        arch.qk_nonlinearity_identity =
            module.at("type").string() == "torch.nn.modules.linear.Identity";
        break;
      }
    }
  }

  // --- output heads -----------------------------------------------------
  const Json *head_names = find(architecture, "output_heads");
  if (head_names == nullptr) fail("manifest is missing architecture key 'output_heads'");
  for (const Json &entry : head_names->array()) {
    const std::string &name = entry.string();
    const auto found = head_registry().find(name);
    if (found == head_registry().end())
      fail("unknown output head '" + name + "'; this build implements " +
           "atomic_energy_output_block, partial_charges_output_block, "
           "hirshfeld_output_block and c6_ratios_output_block");
    arch.heads.push_back(HeadSpec{found->second, name, "model." + name});
  }
  if (!arch.has_head(HeadKind::AtomicEnergy))
    fail("model declares no atomic energy output head");

  // The long-range element capacity is not a manifest field; it is the extent
  // of the per-element embeddings in the charge and Hirshfeld heads.
  arch.lr_element_capacity = 0;
  if (arch.has_head(HeadKind::PartialCharges))
    arch.lr_element_capacity = leading_extent_or(
        model, arch.head(HeadKind::PartialCharges).state_prefix +
                   ".atomic_embedding.weight", 0);
  if (arch.lr_element_capacity == 0 && arch.has_head(HeadKind::HirshfeldRatios))
    arch.lr_element_capacity = leading_extent_or(
        model, arch.head(HeadKind::HirshfeldRatios).state_prefix +
                   ".v_shift_embedding.weight", 0);

  arch.head_hidden_width = leading_extent_or(
      model, arch.head(HeadKind::AtomicEnergy).state_prefix + ".layers.0.weight",
      arch.invariant_width);

  return arch;
}

}  // namespace so3lr
