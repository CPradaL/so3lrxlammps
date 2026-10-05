// `so3lr_capability_probe <model.so3lr>`
//
// Answers one question: can this build execute this model? On success it
// prints the architecture summary and exits 0. On failure it prints every
// capability violation, each naming the descriptor path responsible, and exits
// 1. It is both the driver for the negotiation tests and a useful diagnostic
// to hand a user whose checkpoint is refused.

#include "so3lr/native_model.hpp"
#include "so3lr/so3lr_architecture.hpp"

#include <cstdio>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.so3lr>\n", argv[0]);
    return 2;
  }
  try {
    const so3lr::NativeModel model = so3lr::NativeModel::load(argv[1]);
    std::printf("capability_probe=PASS\n%s\n", model.arch().summary().c_str());
    return 0;
  } catch (const std::exception &error) {
    std::printf("capability_probe=FAIL\n%s\n", error.what());
    return 1;
  }
}
