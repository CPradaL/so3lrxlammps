#pragma once

#include "so3lr/native_model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace so3lr {

// SO3LR's total-charge and spin embeddings (SpookyNet-style, mlff
// ChargeSpinEmbedSparse). For one system with total charge Q, or with
// psi = multiplicity - 1 unpaired electrons, every atom i gets
//
//   a_i = psi * softplus(q_Z(i) . k / sqrt(F)) / sum_j softplus(q_Z(j) . k / sqrt(F))
//   e_i = x + W3 silu(W1 silu(x)),   x = a_i v,
//
// with k, v chosen by the sign of psi. The result is added to the element
// embedding before the embedding scale. It depends only on the composition
// and on (Q, psi), never on positions, so it is one table per system and adds
// no force term. For Q = 0 and psi = 0 it is exactly zero, which is why
// neutral closed-shell results never needed it.
struct ChargeSpinEmbedding {
  struct Module {
    std::vector<double> wq;      // (F, 118): column Z-1 is the query of element Z
    std::vector<double> wk, wv;  // (2, F): row 0 for psi >= 0, row 1 for psi < 0
    std::vector<double> w1, w3;  // (F, F), torch (out, in) layout
  };
  std::size_t width = 0;  // F
  Module charge, spin;
  bool has_charge = false, has_spin = false;
};

ChargeSpinEmbedding load_charge_spin_embedding(const NativeModel &model);

// Element counts of one system, indexed by Z (entry 0 unused).
using ElementCounts = std::array<std::int64_t, 119>;

// Per-element offset table (118 x F, row Z-1) for one system, to be added to
// the element embedding *before* the embedding scale. Rows of elements absent
// from `counts` are zero. Throws if the model lacks an embedding that a
// nonzero charge or spin would need.
std::vector<double> charge_spin_offset_table(const ChargeSpinEmbedding &embedding,
                                             const ElementCounts &counts,
                                             double total_charge,
                                             double unpaired_electrons);

// Empty if (charge, multiplicity) is consistent with the electron count of
// `counts`; otherwise a message saying why not.
std::string charge_multiplicity_problem(const ElementCounts &counts,
                                        double total_charge,
                                        double multiplicity);

}  // namespace so3lr
