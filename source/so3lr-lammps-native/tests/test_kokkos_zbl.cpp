#include "so3lr/kokkos_physical_zbl.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

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

std::vector<double> numbers(const so3lr::Json &value) {
  std::vector<double> result;
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

template <class View, class Values>
void upload(const View &view, const Values &values) {
  if (view.extent(0) != values.size()) throw std::runtime_error("fixture size mismatch");
  auto host = Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(view, host);
}

double max_error(const Kokkos::View<double *> &actual,
                 const std::vector<double> &expected) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
  double result = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i)
    result = std::max(result, std::abs(host(i) - expected[i]));
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: test_kokkos_zbl MODEL FIXTURE\n";
    return 2;
  }
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    const auto &root = fixture.object();
    const auto z_values = numbers(root.at("atomic_numbers"));
    const auto distance_values = numbers(root.at("distances"));
    const auto sender_values = numbers(root.at("senders"));
    const auto receiver_values = numbers(root.at("receivers"));
    const auto expected_atomic = numbers(root.at("atomic_energy"));
    const auto expected_edge = numbers(root.at("edge_energy"));
    const auto expected_gradient = numbers(root.at("radial_gradient"));
    const std::size_t nodes = z_values.size();
    const std::size_t edges = distance_values.size();

    Kokkos::View<std::int64_t *> z("zbl_test_z", nodes);
    Kokkos::View<double *> distance("zbl_test_distance", edges);
    Kokkos::View<std::size_t *> sender("zbl_test_sender", edges);
    Kokkos::View<std::size_t *> receiver("zbl_test_receiver", edges);
    upload(z, z_values);
    upload(distance, distance_values);
    upload(sender, sender_values);
    upload(receiver, receiver_values);

    so3lr::PhysicalZblParameters parameters;
    parameters.ke = model.architecture_number("zbl_ke");
    parameters.cutoff = model.architecture_number("short_range_cutoff_angstrom");
    parameters.switch_off = model.architecture_number("zbl_switch_off_angstrom");
    parameters.p = model.architecture_number("zbl_p");
    parameters.d = model.architecture_number("zbl_d");
    for (int k = 0; k < 4; ++k) {
      parameters.a[k] = model.architecture_number("zbl_a" + std::to_string(k + 1));
      parameters.c[k] = model.architecture_number("zbl_c" + std::to_string(k + 1));
    }
    so3lr::PhysicalZblWorkspace workspace(nodes, edges);
    so3lr::launch_so3lr_zbl_device(z, nodes, distance, sender, receiver,
                                   parameters, workspace);
    Kokkos::fence();
    const double atomic_error = max_error(workspace.atomic_energy, expected_atomic);
    const double edge_error = max_error(workspace.edge_energy, expected_edge);
    const double gradient_error = max_error(workspace.edge_radial_gradient,
                                            expected_gradient);
    std::cout.precision(17);
    std::cout << "zbl_atomic_energy_max_abs_error=" << atomic_error << '\n'
              << "zbl_edge_energy_max_abs_error=" << edge_error << '\n'
              << "zbl_radial_gradient_max_abs_error=" << gradient_error << '\n';
    if (atomic_error > 2.0e-11 || edge_error > 2.0e-11 ||
        gradient_error > 2.0e-10)
      throw std::runtime_error("native ZBL differs from PyTorch oracle");
    std::cout << "SO3LR_NATIVE_DEV56_ZBL_TEST=PASS\n";
  } catch (const std::exception &exc) {
    std::cerr << "SO3LR_NATIVE_DEV56_ZBL_TEST=FAIL: " << exc.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
  return status;
}
