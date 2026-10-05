// Acceptance test for the runtime architecture descriptor (stage A1).
//
// The descriptor replaces ~20 translation units' worth of file-local
// `constexpr` constants. Its correctness criterion is therefore not "is it
// self-consistent" but "does it reproduce, exactly, the literals it replaces"
// for the v1 foundation checkpoint. Every assertion below is one of those
// literals, quoted from the file it came from.

#include "so3lr/native_model.hpp"
#include "so3lr/so3lr_architecture.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

template <class T>
void expect_eq(const char *label, const T &actual, const T &expected) {
  if (actual == expected) return;
  std::fprintf(stderr, "  FAIL %s\n", label);
  ++failures;
}

void expect_close(const char *label, double actual, double expected) {
  const double tolerance = 1e-12 * (1.0 + (expected < 0 ? -expected : expected));
  const double error = actual - expected;
  if (error <= tolerance && -error <= tolerance) return;
  std::fprintf(stderr, "  FAIL %s: %.17g != %.17g\n", label, actual, expected);
  ++failures;
}

void expect_true(const char *label, bool condition) {
  if (condition) return;
  std::fprintf(stderr, "  FAIL %s\n", label);
  ++failures;
}

// The v1 foundation contract, as it appears in the sources being generalized.
void check_v1_foundation(const so3lr::So3lrArchitecture &arch) {
  // src/native_model.cpp:394 (the gate this descriptor is replacing)
  expect_eq<std::size_t>("interaction_blocks == 3", arch.layers, 3);
  expect_eq<std::size_t>("invariant_features == 128", arch.invariant_width, 128);
  expect_eq<std::size_t>("radial_basis_features == 32", arch.rbf_width, 32);

  // src/kokkos_integrated_block0.cpp:19-23 and ~19 duplicate copies
  expect_eq<std::size_t>("heads == 4", arch.inv_heads, 4);
  expect_eq<std::size_t>("head_width == 32", arch.inv_head_width, 32);
  expect_eq<std::size_t>("equivariant_width == 24", arch.equivariant_width, 24);

  // src/kokkos_post_attention_block0.cpp:20-21
  expect_eq<std::size_t>("degree_count == 4", arch.ev_heads, 4);
  expect_eq<std::size_t>("interaction_width == 132", arch.interaction_width, 132);

  // The {3,5,7,9} literal asserted in six separate translation units.
  const std::vector<std::size_t> expected_repeats = {3, 5, 7, 9};
  expect_eq("degree_repeats == {3,5,7,9}", arch.degree_repeats, expected_repeats);
  const std::vector<std::size_t> expected_offsets = {0, 3, 8, 15};
  expect_eq("degree_offsets == {0,3,8,15}", arch.degree_offsets, expected_offsets);
  const std::vector<int> expected_degrees = {1, 2, 3, 4};
  expect_eq("degrees == {1,2,3,4}", arch.degrees, expected_degrees);
  expect_eq("max_degree == 4", arch.max_degree, 4);

  // src/kokkos_learned_energy_forces.cpp:14-63 evaluates 24 harmonics, l=1..4.
  expect_eq<std::size_t>("sh_width == 24", arch.sh_width, 24);
  expect_true("degrees are not duplicated", !arch.degrees_duplicated);

  // With no duplication the expansion map must be the identity, so that the
  // generalized geometry kernel compiles to the copy loop that exists today.
  bool identity = arch.feature_to_sh_channel.size() == arch.equivariant_width;
  for (std::size_t i = 0; identity && i < arch.feature_to_sh_channel.size(); ++i)
    identity = arch.feature_to_sh_channel[i] == i;
  expect_true("feature_to_sh_channel is the identity", identity);

  // src/kokkos_output_heads.cpp:17-19
  expect_eq<std::size_t>("atomic_number_capacity == 118", arch.atomic_number_capacity, 118);
  expect_eq<std::size_t>("lr_element_capacity == 100", arch.lr_element_capacity, 100);

  // Cutoffs and the embedding scale, already runtime in 0.2.0-rc1.
  expect_close("short_range_cutoff == 4.5", arch.short_range_cutoff, 4.5);
  expect_close("long_range_cutoff == 12.0", arch.long_range_cutoff, 12.0);
  expect_close("embedding_scale == sqrt(3)", arch.embedding_scale,
               1.7320508075688772);
  expect_true("attention norm is avg_num_neighbors",
              arch.attention_norm == so3lr::AttentionNormKind::AvgNumNeighbors);
  expect_true("repulsion is learned ZBL",
              arch.repulsion == so3lr::RepulsionKind::LearnedZbl);

  // src/kokkos_output_heads.cpp:137-143 required exactly this triple, in order.
  expect_eq<std::size_t>("three output heads", arch.heads.size(), 3);
  if (arch.heads.size() == 3) {
    expect_true("head 0 is atomic energy",
                arch.heads[0].kind == so3lr::HeadKind::AtomicEnergy);
    expect_true("head 1 is partial charges",
                arch.heads[1].kind == so3lr::HeadKind::PartialCharges);
    expect_true("head 2 is Hirshfeld ratios",
                arch.heads[2].kind == so3lr::HeadKind::HirshfeldRatios);
  }
  expect_true("no C6 head in v1", !arch.has_head(so3lr::HeadKind::C6Ratios));

  // A v1 manifest carries none of the optional operator flags; they must default off
  // rather than be absent-and-undefined.
  expect_true("use_rms_norm is off", !arch.use_rms_norm);
  expect_true("qk_norm is off", !arch.qk_norm);
  expect_true("use_residual_scalars is off", !arch.use_residual_scalars);
}

// Invariants that must hold for *any* architecture, not just the v1 one.
void check_general_invariants(const so3lr::So3lrArchitecture &arch) {
  expect_true("layers >= 1", arch.layers >= 1);
  expect_eq("heads * head_width == features",
            arch.inv_heads * arch.inv_head_width, arch.invariant_width);
  expect_eq("ev_heads * ev_head_width == features",
            arch.ev_heads * arch.ev_head_width, arch.invariant_width);
  expect_eq("ev_heads == len(degrees)", arch.ev_heads, arch.degrees.size());
  expect_eq("ev_heads == len(degree_repeats)", arch.ev_heads,
            arch.degree_repeats.size());
  expect_eq("interaction_width == features + ev_heads",
            arch.interaction_width, arch.invariant_width + arch.ev_heads);

  std::size_t running = 0;
  bool offsets_ok = true;
  for (std::size_t d = 0; d < arch.ev_heads; ++d) {
    offsets_ok = offsets_ok && arch.degree_offsets[d] == running;
    offsets_ok = offsets_ok &&
                 arch.degree_repeats[d] ==
                     static_cast<std::size_t>(2 * arch.degrees[d] + 1);
    running += arch.degree_repeats[d];
  }
  expect_true("degree_offsets is the exclusive prefix sum", offsets_ok);
  expect_eq("equivariant_width == sum(degree_repeats)",
            arch.equivariant_width, running);

  std::size_t unique = 0;
  for (int l = 1; l <= arch.max_degree; ++l)
    unique += static_cast<std::size_t>(2 * l + 1);
  expect_eq("sh_width == sum_{l=1..Lmax}(2l+1)", arch.sh_width, unique);
  expect_eq("feature_to_sh_channel spans the ev width",
            arch.feature_to_sh_channel.size(), arch.equivariant_width);

  bool map_in_range = true;
  for (std::size_t channel : arch.feature_to_sh_channel)
    map_in_range = map_in_range && channel < arch.sh_width;
  expect_true("feature_to_sh_channel stays inside sh_width", map_in_range);
  expect_eq("degrees_duplicated iff E != sh_width", arch.degrees_duplicated,
            arch.equivariant_width != arch.sh_width);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.so3lr>\n", argv[0]);
    return 2;
  }
  try {
    const so3lr::NativeModel model = so3lr::NativeModel::load(argv[1]);
    const so3lr::So3lrArchitecture &arch = model.arch();
    std::printf("architecture: %s\n", arch.summary().c_str());
    check_general_invariants(arch);
    check_v1_foundation(arch);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "architecture_descriptor=FAIL (%s)\n", error.what());
    return 1;
  }
  if (failures != 0) {
    std::fprintf(stderr, "architecture_descriptor=FAIL (%d checks)\n", failures);
    return 1;
  }
  std::printf("architecture_descriptor=PASS\n");
  return 0;
}
