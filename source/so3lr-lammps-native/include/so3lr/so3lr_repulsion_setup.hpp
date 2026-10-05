#pragma once

#include "so3lr/kokkos_physical_zbl.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

namespace so3lr {

// The short-range repulsion of a native model: learnable ZBL (SO3LR v1) or
// the parameter-free NLH pair tables. Shared by the so3lr/native/mpi
// and so3lr/turbo pair styles so both evaluate the same repulsion.
struct So3lrRepulsion {
  PhysicalZblParameters parameters;
  Kokkos::View<double *> nlh_a, nlh_b;  // empty unless parameters.kind == 1
};

PhysicalZblParameters load_zbl_parameters(const NativeModel &model);

// Parameters plus, for NLH, the (Z, Z, 3) A and B tables copied to the device.
So3lrRepulsion load_so3lr_repulsion(const NativeModel &model);

}  // namespace so3lr
