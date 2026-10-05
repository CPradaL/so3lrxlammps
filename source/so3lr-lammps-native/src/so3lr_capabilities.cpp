#include "so3lr/so3lr_capabilities.hpp"

#include "so3lr/native_model.hpp"
#include "so3lr/so3lr_architecture.hpp"

#include <sstream>
#include <stdexcept>

namespace so3lr {
namespace {

std::string join(const std::set<std::size_t> &values) {
  std::ostringstream out;
  bool first = true;
  for (std::size_t value : values) {
    out << (first ? "" : ", ") << value;
    first = false;
  }
  return out.str();
}

// Report a dimension this build has no kernels for.
void check_dimension(std::vector<std::string> &misses,
                     const std::set<std::size_t> &supported, std::size_t actual,
                     const std::string &path, const std::string &noun) {
  if (supported.empty() || supported.count(actual) != 0) return;
  std::ostringstream out;
  out << "unsupported " << noun << "\n    " << path << " = " << actual
      << "\n    this build implements " << noun << " in {" << join(supported)
      << "}";
  misses.push_back(out.str());
}

// `evidence` is the manifest line that proves the model wants the operator,
// quoted verbatim so a user can find it in their own file.
void check_flag(std::vector<std::string> &misses, bool requested,
                bool implemented, const std::string &evidence,
                const std::string &explanation) {
  if (!requested || implemented) return;
  misses.push_back("unsupported operator\n    " + evidence + "\n    " +
                   explanation);
}

}  // namespace

const RuntimeCapabilities &runtime_capabilities() {
  static const RuntimeCapabilities caps = [] {
    RuntimeCapabilities c;

    // ---- dimensions --------------------------------------------------
    // The MPI path loops over blocks; the fused geometry-adjoint sum captures
    // per-block views in a fixed-size array (kMaxInteractionBlocks = 8 in
    // kokkos_local_cartesian_forces.cpp).
    c.interaction_blocks = {1, 2, 3, 4, 5, 6, 7, 8};
    // Every width is descriptor-driven in the MPI path. These are the values
    // of the released models (v1/s/m: 128 and 32; l: 256 and 64); others are
    // refused until a checkpoint exists to validate them against.
    c.invariant_features = {128, 256};
    c.radial_basis_features = {32, 64};
    // The per-thread accumulator in kokkos_attention_scatter_reverse.cpp must
    // keep a compile-time extent; only instantiated widths are loadable.
    c.attention_head_width = {32};
    // kokkos_learned_energy_forces.cpp hand-expands l = 1..4.
    c.max_supported_degree = 4;

    // ---- structure ---------------------------------------------------
    // kokkos_attention_scatter_block0.cpp:172 computes both attention streams
    // in one thread, which requires the two head counts to coincide.
    c.require_heads_equal_degrees = true;
    // The geometry kernel evaluates the distinct harmonics once and expands
    // them to the feature layout; the force path contracts the gradient back
    // (kokkos_learned_energy_forces.cpp, kokkos_local_cartesian_forces.cpp).
    // The per-node gate-gradient array bounds the degree slots at 8.
    c.allow_duplicated_degrees = true;

    // ---- optional operators ---------------------------------------------------
    // RMSNorm and QK normalisation live in the shared per-block kernels.
    // Residual scalars, the C6 head and NLH repulsion are wired into the MPI
    // path (pair_style so3lr/native/mpi); the single-rank chain refuses them
    // explicitly in its constructor.
    c.rms_norm = true;
    c.qk_norm = true;
    c.residual_scalars = true;
    c.nlh_repulsion = true;
    c.c6_ratios_head = true;

    // ---- manifest surface ---------------------------------------------
    c.supported_schemas = {"so3lr-native-model-v2"};

    // Every architecture key this build knows how to interpret -- or knows it
    // may safely ignore. A key outside this set is a hard failure.
    c.known_architecture_keys = {
        "atomic_number_capacity",
        "attention_head_width",
        "attention_heads",
        "avg_num_neighbors",
        "embedding_scale",
        "energy_theory_level_selected",
        "euclidean_degree_channels",
        "interaction_blocks",
        "invariant_features",
        "layer_norm_epsilon",
        "long_range_cutoff_angstrom",
        "lr_bohr_angstrom",
        "lr_dispersion_cuton_angstrom",
        "lr_dispersion_scale",
        "lr_electrostatic_cuton_angstrom",
        "lr_electrostatic_ke",
        "lr_electrostatic_sigma",
        "lr_fine_structure",
        "lr_hartree_ev",
        "lr_pair_scale",
        "layer_normalization_1",
        "layer_normalization_2",
        "message_normalization",
        "module_type_registry",
        "native_long_range_observables",
        "native_long_range_terms",
        "native_short_range_physical_terms",
        "num_embeddings",
        "output_heads",
        "parameter_elements",
        "physical_reference_table_state_keys",
        "physical_reference_table_tensor_count",
        "qk_norm",
        "qk_nonlinearity",
        "radial_basis_features",
        "repulsion_type",
        "residual_mlp_1",
        "residual_mlp_2",
        "runtime_element_mapping",
        "short_range_cutoff_angstrom",
        "state_tensor_count",
        "use_residual_scalars",
        "use_rms_norm",
        "zbl_a1", "zbl_a2", "zbl_a3", "zbl_a4",
        "zbl_c1", "zbl_c2", "zbl_c3", "zbl_c4",
        "zbl_cutoff_function",
        "zbl_d",
        "zbl_enabled",
        "zbl_ke",
        "zbl_p",
        "zbl_switch_off_angstrom",
    };

    c.supported_module_types = {
        "so3krates_torch.blocks.embedding.ChargeSpinEmbedding",
        "so3krates_torch.blocks.embedding.EuclideanEmbedding",
        "so3krates_torch.blocks.embedding.InvariantEmbedding",
        "so3krates_torch.blocks.euclidean_transformer.EuclideanAttentionBlock",
        "so3krates_torch.blocks.euclidean_transformer.EuclideanTransformer",
        "so3krates_torch.blocks.euclidean_transformer.FilterNet",
        "so3krates_torch.blocks.euclidean_transformer.InteractionBlock",
        "so3krates_torch.blocks.output_block.AtomicEnergyOutputHead",
        "so3krates_torch.blocks.output_block.DipoleVecOutputHead",
        "so3krates_torch.blocks.output_block.HirshfeldOutputHead",
        "so3krates_torch.blocks.output_block.PartialChargesOutputHead",
        "so3krates_torch.blocks.physical_potentials.DispersionInteraction",
        "so3krates_torch.blocks.physical_potentials.ElectrostaticInteraction",
        "so3krates_torch.blocks.physical_potentials.ZBLRepulsion",
        "so3krates_torch.blocks.radial_basis.BernsteinBasis",
        "so3krates_torch.blocks.radial_basis.ComputeRBF",
        "so3krates_torch.blocks.so3_conv_invariants.L0Contraction",
        "so3krates_torch.calculator.lammps_mliap_so3.So3EdgeForcesWrapper",
        "so3krates_torch.modules.cutoff.PhysNetCutoff",
        "so3krates_torch.modules.models.SO3LR",
        "so3krates_torch.modules.spherical_harmonics.RealSphericalHarmonics",
        "torch.nn.modules.activation.SiLU",
        "torch.nn.modules.container.ModuleList",
        "torch.nn.modules.container.Sequential",
        "torch.nn.modules.linear.Identity",
        "torch.nn.modules.linear.Linear",
        "torch.nn.modules.normalization.LayerNorm",
        "torch.nn.modules.sparse.Embedding",
    };
    return c;
  }();
  return caps;
}

std::vector<std::string> capability_violations(const NativeModel &model,
                                               const So3lrArchitecture &arch) {
  const RuntimeCapabilities &caps = runtime_capabilities();
  std::vector<std::string> misses;

  // ---- schema ----------------------------------------------------------
  const std::string &schema = model.manifest().at("schema").string();
  if (caps.supported_schemas.count(schema) == 0) {
    std::ostringstream out;
    out << "unsupported manifest schema\n    schema = \"" << schema
        << "\"\n    this build reads";
    for (const std::string &supported : caps.supported_schemas)
      out << " \"" << supported << "\"";
    misses.push_back(out.str());
  }

  // ---- dimensions ------------------------------------------------------
  check_dimension(misses, caps.interaction_blocks, arch.layers,
                  "architecture.interaction_blocks", "interaction block counts");
  check_dimension(misses, caps.invariant_features, arch.invariant_width,
                  "architecture.invariant_features", "invariant feature widths");
  check_dimension(misses, caps.radial_basis_features, arch.rbf_width,
                  "architecture.radial_basis_features", "radial basis sizes");
  check_dimension(misses, caps.attention_head_width, arch.inv_head_width,
                  "architecture.attention_head_width", "attention head widths");

  if (caps.max_supported_degree != 0 &&
      static_cast<std::size_t>(arch.max_degree) > caps.max_supported_degree) {
    std::ostringstream out;
    out << "unsupported maximum spherical degree\n    max(degrees) = "
        << arch.max_degree
        << "\n    this build hand-expands the real spherical harmonics up to l = "
        << caps.max_supported_degree;
    misses.push_back(out.str());
  }

  // ---- structure -------------------------------------------------------
  if (caps.require_heads_equal_degrees && arch.inv_heads != arch.ev_heads) {
    std::ostringstream out;
    out << "unsupported attention topology\n    architecture.attention_heads = "
        << arch.inv_heads << ", architecture.euclidean_degree_channels = "
        << arch.ev_heads
        << "\n    this build evaluates both attention streams in one kernel, "
           "which requires them to be equal";
    misses.push_back(out.str());
  }
  if (!caps.allow_duplicated_degrees && arch.degrees_duplicated) {
    std::ostringstream out;
    out << "unsupported degree list\n    degrees repeat a value "
           "(equivariant_width = "
        << arch.equivariant_width << ", distinct harmonics = " << arch.sh_width
        << ")\n    this build requires one feature block per spherical degree";
    misses.push_back(out.str());
  }

  // ---- operator flags ---------------------------------------------------
  check_flag(misses, arch.use_rms_norm, caps.rms_norm,
             "architecture.use_rms_norm = true",
             "this build implements LayerNorm only");
  check_flag(misses, arch.qk_norm, caps.qk_norm, "architecture.qk_norm = true",
             "query/key normalization is not implemented");
  check_flag(misses, arch.use_residual_scalars, caps.residual_scalars,
             "architecture.use_residual_scalars = true",
             "per-layer residual scalars are not implemented");
  check_flag(misses, arch.repulsion == RepulsionKind::NlhTable,
             caps.nlh_repulsion, "architecture.repulsion = \"nlh\"",
             "this build implements learnable ZBL repulsion only");
  check_flag(misses, arch.has_head(HeadKind::C6Ratios), caps.c6_ratios_head,
             "architecture.output_heads contains \"c6_ratios_output_block\"",
             "the C6 ratio head is not implemented");

  // ---- unknown architecture keys ---------------------------------------
  // Deliberately strict: see the header for why.
  for (const auto &entry : model.manifest().at("architecture").object()) {
    if (caps.known_architecture_keys.count(entry.first) != 0) continue;
    misses.push_back(
        "unknown architecture key\n    architecture." + entry.first +
        "\n    this build cannot tell whether ignoring it changes the result; "
        "upgrade the runtime or re-export with a known exporter");
  }

  // ---- unknown module types --------------------------------------------
  for (const auto &entry :
       model.manifest().at("architecture").at("module_type_registry").object()) {
    if (caps.supported_module_types.count(entry.first) != 0) continue;
    std::ostringstream out;
    out << "unsupported module type\n    module_type_registry[\"" << entry.first
        << "\"] = " << static_cast<std::size_t>(entry.second.number())
        << "\n    this build has no kernel for it";
    misses.push_back(out.str());
  }

  return misses;
}

void negotiate_capabilities(const NativeModel &model,
                            const So3lrArchitecture &arch,
                            const std::string &origin) {
  const std::vector<std::string> misses = capability_violations(model, arch);
  if (misses.empty()) return;
  std::ostringstream out;
  out << "so3lr: this build cannot execute the model";
  if (!origin.empty()) out << " at " << origin;
  out << "\n  " << arch.summary();
  for (const std::string &miss : misses) out << "\n  " << miss;
  throw std::runtime_error(out.str());
}

}  // namespace so3lr
