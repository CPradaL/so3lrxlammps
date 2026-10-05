#pragma once

#include "so3lr/kokkos_learned_energy_forces.hpp"
#include "so3lr/kokkos_physical_long_range.hpp"
#include "so3lr/kokkos_physical_zbl.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace so3lr {

struct So3lrEvaluatorWorkspace {
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;

  So3lrEvaluatorWorkspace(std::size_t nodes, std::size_t sr_edges,
                          std::size_t lr_pairs);
  So3lrEvaluatorWorkspace(std::size_t sr_nodes, std::size_t owned_nodes,
                          std::size_t sr_edges, std::size_t lr_pairs);
  GeometryVJPWorkspace sr_geometry;
  LearnedEnergyReverseWorkspace chain;
  PhysicalZblWorkspace zbl;
  PhysicalLongRangeWorkspace long_range;
  DoubleView energy_seeds;
  DoubleView partial_charge_seeds;
  DoubleView hirshfeld_seeds;
  Int64View owned_atomic_numbers;
  DoubleView owned_partial_charges;
  DoubleView owned_hirshfeld_ratios;
  DoubleView atomic_energies;
  DoubleView atomic_forces;
};

class KokkosSo3lrEvaluator {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosSo3lrEvaluator(const NativeModel &model);

  void launch_device(
      const Int64View &atomic_numbers,
      const DoubleView &sr_edge_vectors,
      const IndexView &sr_senders,
      const IndexView &sr_receivers,
      const DoubleView &lr_pair_vectors,
      const IndexView &lr_senders,
      const IndexView &lr_receivers,
      const So3lrEvaluatorWorkspace &workspace) const;

  void launch_owned_device(
      const Int64View &atomic_numbers, std::size_t owned_nodes,
      const IndexView &sr_node_owners,
      const DoubleView &sr_edge_vectors,
      const IndexView &sr_senders, const IndexView &sr_receivers,
      const DoubleView &lr_pair_vectors,
      const IndexView &lr_senders, const IndexView &lr_receivers,
      const So3lrEvaluatorWorkspace &workspace) const;

  const DoubleView &atomic_energies(
      const So3lrEvaluatorWorkspace &workspace) const {
    return workspace.atomic_energies;
  }
  const DoubleView &atomic_forces(
      const So3lrEvaluatorWorkspace &workspace) const {
    return workspace.atomic_forces;
  }
  const DoubleView &partial_charges(
      const So3lrEvaluatorWorkspace &workspace) const {
    return chain_.partial_charges(workspace.chain);
  }
  const DoubleView &hirshfeld_ratios(
      const So3lrEvaluatorWorkspace &workspace) const {
    return chain_.hirshfeld_ratios(workspace.chain);
  }
  const PhysicalZblWorkspace &zbl(
      const So3lrEvaluatorWorkspace &workspace) const {
    return workspace.zbl;
  }
  const PhysicalLongRangeWorkspace &long_range(
      const So3lrEvaluatorWorkspace &workspace) const {
    return workspace.long_range;
  }
  const PhysicalLongRangeParameters &long_range_parameters() const {
    return parameters_;
  }
  bool self_contained_model_contract() const {
    return self_contained_model_contract_;
  }
  bool shared_kokkos_stream() const { return chain_.shared_kokkos_stream(); }
  std::size_t persistent_device_bytes() const {
    return persistent_device_bytes_;
  }

 private:
  KokkosLearnedEnergyReverseChain chain_;
  DoubleView reference_alphas_;
  DoubleView reference_c6_;
  PhysicalZblParameters zbl_parameters_;
  PhysicalLongRangeParameters parameters_;
  std::size_t persistent_device_bytes_ = 0;
  bool self_contained_model_contract_ = false;
};

std::size_t so3lr_evaluator_workspace_bytes(
    std::size_t nodes, std::size_t sr_edges, std::size_t lr_pairs,
    std::size_t persistent_bytes);
std::size_t so3lr_evaluator_workspace_bytes(
    std::size_t sr_nodes, std::size_t owned_nodes,
    std::size_t sr_edges, std::size_t lr_pairs,
    std::size_t persistent_bytes);

}  // namespace so3lr
