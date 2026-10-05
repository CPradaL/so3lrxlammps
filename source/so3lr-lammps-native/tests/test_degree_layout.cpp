// Unit test for the degree-layout derivation (stage A1).
//
// This is the part of the descriptor that a duplicated degree list makes
// non-obvious, and it is the prerequisite for models whose
// degrees are {1,1,2,2,3,3,4,4}: eight feature blocks spanning 48 channels,
// but still only 24 distinct spherical harmonics. Testing it needs no model
// file and no GPU, so the hardest piece of the width generalization is
// verifiable long before the kernels that will consume it exist.

#include "so3lr/so3lr_architecture.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

template <class T>
void expect_eq(const std::string &label, const T &actual, const T &expected) {
  if (actual == expected) return;
  std::fprintf(stderr, "  FAIL %s\n", label.c_str());
  ++failures;
}

void expect_true(const std::string &label, bool condition) {
  if (condition) return;
  std::fprintf(stderr, "  FAIL %s\n", label.c_str());
  ++failures;
}

void expect_throws(const std::string &label,
                   const std::vector<std::size_t> &repeats) {
  try {
    so3lr::compute_degree_layout(repeats);
  } catch (const std::exception &) {
    return;
  }
  std::fprintf(stderr, "  FAIL %s: accepted but should have been refused\n",
               label.c_str());
  ++failures;
}

// SO3LR v1: degrees {1,2,3,4}, no duplication.
void check_plain_degrees() {
  const so3lr::DegreeLayout layout = so3lr::compute_degree_layout({3, 5, 7, 9});
  expect_eq<std::vector<int>>("degrees", layout.degrees, {1, 2, 3, 4});
  expect_eq<std::vector<std::size_t>>("offsets", layout.degree_offsets,
                                      {0, 3, 8, 15});
  expect_eq<std::size_t>("equivariant_width", layout.equivariant_width, 24);
  expect_eq<std::size_t>("sh_width", layout.sh_width, 24);
  expect_eq("max_degree", layout.max_degree, 4);
  expect_true("not duplicated", !layout.duplicated);

  // The v1 path must see the identity map, so the generalized geometry kernel
  // reduces to the copy loop that exists today and costs nothing extra.
  bool identity = layout.feature_to_sh_channel.size() == 24;
  for (std::size_t i = 0; identity && i < 24; ++i)
    identity = layout.feature_to_sh_channel[i] == i;
  expect_true("feature_to_sh_channel is the identity", identity);
}

// Duplicated degrees {1,1,2,2,3,3,4,4}.
void check_duplicated_degrees() {
  const so3lr::DegreeLayout layout =
      so3lr::compute_degree_layout({3, 3, 5, 5, 7, 7, 9, 9});
  expect_eq<std::vector<int>>("degrees", layout.degrees,
                              {1, 1, 2, 2, 3, 3, 4, 4});
  expect_eq<std::vector<std::size_t>>("offsets", layout.degree_offsets,
                                      {0, 3, 6, 11, 16, 23, 30, 39});
  expect_eq<std::size_t>("equivariant_width", layout.equivariant_width, 48);
  expect_eq<std::size_t>("sh_width", layout.sh_width, 24);
  expect_eq("max_degree", layout.max_degree, 4);
  expect_true("duplicated", layout.duplicated);

  // Each pair of blocks sharing a degree must map onto the *same* unique
  // harmonic channels: that is exactly what makes one evaluation of 24
  // harmonics sufficient for 48 feature channels.
  const std::vector<std::size_t> expected = {
      0,  1,  2,            // l=1, first copy   (unique 0..2)
      0,  1,  2,            // l=1, second copy
      3,  4,  5,  6,  7,    // l=2, first copy   (unique 3..7)
      3,  4,  5,  6,  7,    // l=2, second copy
      8,  9, 10, 11, 12, 13, 14,          // l=3, first copy  (unique 8..14)
      8,  9, 10, 11, 12, 13, 14,          // l=3, second copy
      15, 16, 17, 18, 19, 20, 21, 22, 23, // l=4, first copy  (unique 15..23)
      15, 16, 17, 18, 19, 20, 21, 22, 23, // l=4, second copy
  };
  expect_eq("feature_to_sh_channel", layout.feature_to_sh_channel, expected);
}

// A short list, to confirm nothing assumes Lmax == 4.
void check_truncated_degrees() {
  const so3lr::DegreeLayout layout = so3lr::compute_degree_layout({3, 5});
  expect_eq<std::size_t>("equivariant_width", layout.equivariant_width, 8);
  expect_eq<std::size_t>("sh_width", layout.sh_width, 8);
  expect_eq("max_degree", layout.max_degree, 2);
  expect_true("not duplicated", !layout.duplicated);
}

void check_rejections() {
  expect_throws("empty degree list", {});
  expect_throws("even multiplicity", {3, 4});
  // 2l+1 with l=0 is 1: degree 0 lives in the invariant stream.
  expect_throws("degree zero", {1, 3});
  // cg_rep groups by sorted unique degree while segment_ids follows the given
  // order; an unsorted list silently mis-pairs them in the reference too.
  expect_throws("unsorted degrees", {5, 3});
}

}  // namespace

int main() {
  check_plain_degrees();
  check_duplicated_degrees();
  check_truncated_degrees();
  check_rejections();
  if (failures != 0) {
    std::fprintf(stderr, "degree_layout=FAIL (%d checks)\n", failures);
    return 1;
  }
  std::printf("degree_layout=PASS\n");
  return 0;
}
