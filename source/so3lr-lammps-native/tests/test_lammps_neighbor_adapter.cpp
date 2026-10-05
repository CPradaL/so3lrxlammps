#include "so3lr/lammps_neighbor_adapter.hpp"
#include "so3lr/native_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_text(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open fixture " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<std::size_t> sizes(const so3lr::Json &value) {
  std::vector<std::size_t> result;
  for (const auto &item : value.array())
    result.push_back(static_cast<std::size_t>(item.unsigned_integer()));
  return result;
}

std::vector<std::int64_t> int64s(const so3lr::Json &value) {
  std::vector<std::int64_t> result;
  for (const auto &item : value.array())
    result.push_back(static_cast<std::int64_t>(item.unsigned_integer()));
  return result;
}

std::vector<int> ints(const so3lr::Json &value) {
  std::vector<int> result;
  for (const auto &item : value.array())
    result.push_back(static_cast<int>(item.unsigned_integer()));
  return result;
}

std::vector<double> doubles(const so3lr::Json &value) {
  std::vector<double> result;
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

std::vector<double> positions(const so3lr::Json &value) {
  std::vector<double> result;
  for (const auto &row : value.array())
    for (const auto &item : row.array()) result.push_back(item.number());
  return result;
}

so3lr::LammpsAtomSnapshot atoms(const so3lr::Json &value) {
  return {static_cast<std::size_t>(value.at("nlocal").unsigned_integer()),
          int64s(value.at("tags")), ints(value.at("types")),
          positions(value.at("positions"))};
}

so3lr::LammpsNeighborSnapshot neighbors(const so3lr::Json &value) {
  return {sizes(value.at("ilist")), sizes(value.at("offsets")),
          sizes(value.at("neighbors"))};
}

double compare_doubles(const std::vector<double> &actual,
                       const std::vector<double> &expected,
                       const std::string &label) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    if (std::abs(actual[i] - expected[i]) > 2.0e-14)
      throw std::runtime_error(label + " mismatch at " + std::to_string(i));
  }
  return maximum;
}

template <class T>
void compare_exact(const std::vector<T> &actual, const std::vector<T> &expected,
                   const std::string &label) {
  if (actual != expected) throw std::runtime_error(label + " mismatch");
}

double compare_graph(const so3lr::LammpsCompactGraph &actual,
                     const so3lr::Json &expected, const std::string &label) {
  compare_exact(actual.source_atom_rows, sizes(expected.at("source_atom_rows")),
                label + " source rows");
  compare_exact(actual.tags, int64s(expected.at("tags")), label + " tags");
  compare_exact(actual.atomic_numbers, int64s(expected.at("atomic_numbers")),
                label + " atomic numbers");
  compare_exact(actual.owned_local, sizes(expected.at("owned_local")),
                label + " owners");
  compare_exact(actual.ghost_local, sizes(expected.at("ghost_local")),
                label + " ghosts");
  compare_exact(actual.senders, sizes(expected.at("senders")),
                label + " senders");
  compare_exact(actual.receivers, sizes(expected.at("receivers")),
                label + " receivers");
  compare_exact(actual.original_neighbor_entries,
                sizes(expected.at("original_neighbor_entries")),
                label + " original entries");
  return compare_doubles(actual.vectors, doubles(expected.at("vectors")),
                         label + " vectors");
}

bool throws_newton(const so3lr::LammpsAtomSnapshot &atom_data,
                   const so3lr::LammpsNeighborSnapshot &sr,
                   const so3lr::LammpsNeighborSnapshot &lr,
                   const std::vector<std::int64_t> &zmap,
                   double sr_cutoff, double lr_cutoff) {
  try {
    (void)so3lr::build_lammps_native_graphs(
        atom_data, sr, lr, zmap, sr_cutoff, lr_cutoff, false);
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: TEST FIXTURE.json");
    const auto fixture = so3lr::Json::parse(read_text(argv[1]));
    if (fixture.at("schema").string() !=
        "so3lr-lammps-neighbor-adapter-fixture-v1")
      throw std::runtime_error("unexpected LAMMPS adapter fixture");
    const auto zmap = int64s(fixture.at("type_to_atomic_number"));
    const double sr_cutoff = fixture.at("short_range_cutoff").number();
    const double lr_cutoff = fixture.at("long_range_cutoff").number();
    std::size_t sr_edges = 0;
    std::size_t lr_pairs = 0;
    double maximum_vector_error = 0.0;
    const auto &ranks = fixture.at("ranks").array();
    if (ranks.size() != 2) throw std::runtime_error("expected two ranks");
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
      const auto atom_data = atoms(ranks[rank].at("atoms"));
      const auto sr = neighbors(ranks[rank].at("full_sr"));
      const auto lr = neighbors(ranks[rank].at("half_lr"));
      const auto graphs = so3lr::build_lammps_native_graphs(
          atom_data, sr, lr, zmap, sr_cutoff, lr_cutoff, true);
      maximum_vector_error = std::max(
          maximum_vector_error,
          compare_graph(graphs.short_range, ranks[rank].at("expected_sr"),
                        "rank SR"));
      maximum_vector_error = std::max(
          maximum_vector_error,
          compare_graph(graphs.long_range, ranks[rank].at("expected_lr"),
                        "rank LR"));
      sr_edges += graphs.short_range.interactions();
      lr_pairs += graphs.long_range.interactions();
      if (!throws_newton(atom_data, sr, lr, zmap, sr_cutoff, lr_cutoff))
        throw std::runtime_error("newton pair off was accepted");
    }
    if (sr_edges != 34 || lr_pairs != 39)
      throw std::runtime_error("global adapter interaction count mismatch");

    const auto &periodic = fixture.at("periodic_image_case");
    const auto periodic_atoms = atoms(periodic.at("atoms"));
    const auto periodic_graphs = so3lr::build_lammps_native_graphs(
        periodic_atoms, neighbors(periodic.at("full_sr")),
        neighbors(periodic.at("half_lr")), zmap, sr_cutoff, lr_cutoff, true);
    maximum_vector_error = std::max(
        maximum_vector_error,
        compare_graph(periodic_graphs.short_range,
                      periodic.at("expected_sr"), "periodic SR"));
    maximum_vector_error = std::max(
        maximum_vector_error,
        compare_graph(periodic_graphs.long_range,
                      periodic.at("expected_lr"), "periodic LR"));
    if (periodic_graphs.long_range.tags.size() != 2 ||
        periodic_graphs.long_range.tags[0] != 7 ||
        periodic_graphs.long_range.tags[1] != 7 ||
        std::abs(periodic_graphs.long_range.vectors[0] + 10.0) > 1.0e-14)
      throw std::runtime_error("periodic duplicate-tag image was collapsed");

    std::cout << "global_sr_edges=" << sr_edges << '\n'
              << "global_lr_pairs=" << lr_pairs << '\n'
              << "maximum_vector_abs_error=" << maximum_vector_error << '\n'
              << "periodic_same_tag_rows_preserved=2\n"
              << "type_to_atomic_number_mapping=PASS\n"
              << "owned_ghost_compaction=PASS\n"
              << "receiver_owned_full_sr_list=PASS\n"
              << "unique_half_lr_list=PASS\n"
              << "separate_sr_lr_compaction=PASS\n"
              << "periodic_image_vectors=PASS\n"
              << "duplicate_periodic_tags_preserved=PASS\n"
              << "newton_pair_contract=PASS\n"
              << "LAMMPS_NEIGHBOR_ADAPTER_DEV36=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "LAMMPS_NEIGHBOR_ADAPTER_DEV36=FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
