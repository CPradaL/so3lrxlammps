#pragma once

// Capability negotiation between a `.so3lr` file and this build.
//
// FORMAT.md already states the contract this implements:
//
//   "Loading a file proves data integrity, not that a runtime implements every
//    module type. libso3lr must compare the manifest module registry and model
//    family against its compiled capability table before executing the model."
//
// Until 0.3.0 that table did not exist; `native_model.cpp` instead rejected
// anything that was not exactly the v1 foundation contract with the single
// message "unsupported architecture contract". This header replaces that with
// a declared table, checked field by field, reporting every miss at once and
// naming the descriptor path responsible for each.
//
// Two properties are deliberate:
//
//   * Limits live in ONE place (`runtime_capabilities()`), so relaxing one as a
//     stage lands is a one-line edit rather than a search.
//   * An architecture key this build has never heard of is a HARD FAILURE.
//     That is counter-intuitive -- a newer exporter breaks an older runtime
//     loudly -- and it is the only mechanism here that prevents a future
//     upstream flag from silently producing a wrong-but-loadable file.

#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace so3lr {

class NativeModel;
struct So3lrArchitecture;

struct RuntimeCapabilities {
  // Dimensions this build has kernels for. An empty set means "any value".
  std::set<std::size_t> interaction_blocks;
  std::set<std::size_t> invariant_features;
  std::set<std::size_t> radial_basis_features;
  std::set<std::size_t> attention_head_width;
  std::size_t max_supported_degree = 0;

  // Structural restrictions.
  bool require_heads_equal_degrees = true;
  bool allow_duplicated_degrees = false;

  // Operator flags this build implements. A model that sets a flag absent
  // from this list is refused by name.
  bool rms_norm = false;
  bool qk_norm = false;
  bool residual_scalars = false;
  bool nlh_repulsion = false;
  bool c6_ratios_head = false;

  // Manifest surface this build understands.
  std::set<std::string> known_architecture_keys;
  std::set<std::string> supported_module_types;
  std::set<std::string> supported_schemas;
};

// The capability table compiled into this build.
const RuntimeCapabilities &runtime_capabilities();

// Check a model against it. Returns every violation, most structural first;
// an empty vector means the model is executable by this build.
std::vector<std::string> capability_violations(const NativeModel &model,
                                               const So3lrArchitecture &arch);

// Convenience wrapper used by the loader: throws std::runtime_error carrying
// every violation when the vector above is non-empty.
void negotiate_capabilities(const NativeModel &model,
                            const So3lrArchitecture &arch,
                            const std::string &origin);

}  // namespace so3lr
