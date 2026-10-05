#ifdef PAIR_CLASS
// clang-format off
PairStyle(so3lr/native/mpi,PairSO3LRNativeMPI);
// clang-format on
#else

#ifndef LMP_PAIR_SO3LR_NATIVE_MPI_H
#define LMP_PAIR_SO3LR_NATIVE_MPI_H

#include "pair.h"

#include "so3lr/kokkos_local_cartesian_forces.hpp"
#include "so3lr/kokkos_physical_long_range_model.hpp"
#include "so3lr/kokkos_physical_zbl.hpp"
#include "so3lr/native_model.hpp"
#include "so3lr/so3lr_charge_spin.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace so3lr_lammps_native_detail {
struct PersistentCache;
}

namespace LAMMPS_NS {

// Production ownership-aware MPI implementation of the native SO3LR model.
class PairSO3LRNativeMPI : public Pair {
 public:
  explicit PairSO3LRNativeMPI(class LAMMPS *);
  ~PairSO3LRNativeMPI() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;

 private:
  std::string model_path_;
  double short_range_cutoff_ = 0.0;
  double long_range_cutoff_ = 0.0;
  std::vector<std::int64_t> type_to_atomic_number_;
  class NeighList *short_range_list_ = nullptr;
  class NeighList *long_range_list_ = nullptr;
  std::unique_ptr<so3lr::NativeModel> model_;
  std::unique_ptr<so3lr::KokkosLocalCartesianForces> force_model_;
  std::unique_ptr<so3lr::KokkosPhysicalLongRangeModel> long_range_model_;
  so3lr::PhysicalZblParameters zbl_parameters_;
  Kokkos::View<double *> nlh_a_, nlh_b_;  // NLH repulsion tables, if the model uses them
  // System total charge and spin multiplicity (pair_style arguments 2 and 3).
  double total_charge_ = 0.0;
  double multiplicity_ = 1.0;
  so3lr::ChargeSpinEmbedding charge_spin_;
  Kokkos::View<double *> charge_spin_table_;  // 118 x F; empty when neutral singlet
  bool initialized_kokkos_here_ = false;
  bool reported_ = false;
  std::size_t evaluation_count_ = 0;
  std::unique_ptr<so3lr_lammps_native_detail::PersistentCache> persistent_;
  std::size_t plan_cache_hits_ = 0;
  std::size_t plan_cache_rebuilds_ = 0;
  std::size_t topology_cache_hits_ = 0;
  std::size_t topology_cache_rebuilds_ = 0;
  std::size_t workspace_cache_hits_ = 0;
  std::size_t workspace_cache_rebuilds_ = 0;
  std::size_t payload_cache_hits_ = 0;
  std::size_t payload_cache_growths_ = 0;

  void allocate();
  void setup_charge_spin();
};

}  // namespace LAMMPS_NS

#endif
#endif
