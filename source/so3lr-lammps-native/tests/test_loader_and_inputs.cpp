#include "so3lr/native_model.hpp"

#include <algorithm>
#include <cmath>
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
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<double> numbers(const so3lr::Json &value) {
  std::vector<double> result;
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

std::vector<std::vector<double>> matrix(const so3lr::Json &value) {
  std::vector<std::vector<double>> result;
  for (const auto &row : value.array()) result.push_back(numbers(row));
  return result;
}

void close(double actual, double expected, double atol, double rtol, const std::string &label) {
  const double error = std::abs(actual-expected);
  if (error > atol + rtol*std::abs(expected))
    throw std::runtime_error(label + " mismatch actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

}  // namespace

int main(int argc, char **argv) try {
  if (argc != 3) throw std::runtime_error("usage: test MODEL FIXTURE.json");
  const auto model = so3lr::NativeModel::load(argv[1]);
  const auto fixture = so3lr::Json::parse(read_text(argv[2]));

  if (model.tensor_count() != 211 || model.tensor_bytes() != 5452492) throw std::runtime_error("unexpected model tensor totals");
  close(model.architecture_number("short_range_cutoff_angstrom"), 4.5, 0, 0, "SR cutoff");
  close(model.architecture_number("long_range_cutoff_angstrom"), 12.0, 0, 0, "LR cutoff");

  const auto &atomic_map = model.tensor("atomic_numbers_map");
  if (atomic_map.shape != std::vector<std::size_t>{2} || model.int64(atomic_map,0) != 1 || model.int64(atomic_map,1) != 8)
    throw std::runtime_error("water element map mismatch");

  const auto distances = numbers(fixture.at("distances"));
  const auto expected_cutoff = numbers(fixture.at("physnet_cutoff"));
  const auto expected_rbf = matrix(fixture.at("bernstein_rbf"));
  const auto atomic_numbers_json = fixture.at("atomic_numbers").array();
  const auto expected_embedding = matrix(fixture.at("invariant_embedding"));
  if (distances.size() != expected_cutoff.size() || distances.size() != expected_rbf.size()) throw std::runtime_error("fixture distance dimensions mismatch");

  const double rmax = model.architecture_number("short_range_cutoff_angstrom");
  double cutoff_max_error = 0;
  for (std::size_t i=0; i<distances.size(); ++i) {
    const double x = distances[i]/rmax;
    const double actual = distances[i] < rmax ? 1.0-6.0*std::pow(x,5)+15.0*std::pow(x,4)-10.0*std::pow(x,3) : 0.0;
    cutoff_max_error = std::max(cutoff_max_error, std::abs(actual-expected_cutoff[i]));
    close(actual, expected_cutoff[i], 2e-13, 2e-13, "PhysNet cutoff");
  }

  const auto &b = model.tensor("model.radial_embedding.radial_basis_fn.b");
  const auto &k = model.tensor("model.radial_embedding.radial_basis_fn.k");
  const auto &krev = model.tensor("model.radial_embedding.radial_basis_fn.k_rev");
  const auto &gamma_tensor = model.tensor("model.radial_embedding.radial_basis_fn.gamma");
  const double gamma = model.float64(gamma_tensor,0);
  double rbf_max_error = 0;
  for (std::size_t i=0; i<distances.size(); ++i) {
    const double x = std::clamp(std::exp(-gamma*distances[i]), 1e-6, 1.0-1e-6);
    for (std::size_t channel=0; channel<32; ++channel) {
      const double log_poly = model.float64(b,channel) + static_cast<double>(model.int64(k,channel))*std::log(x) + static_cast<double>(model.int64(krev,channel))*std::log(1.0-x);
      const double actual = std::exp(log_poly);
      rbf_max_error = std::max(rbf_max_error, std::abs(actual-expected_rbf.at(i).at(channel)));
      close(actual, expected_rbf.at(i).at(channel), 3e-13, 3e-13, "Bernstein RBF");
    }
  }

  const auto &weight = model.tensor("model.inv_feature_embedding.embedding.weight");
  if (weight.shape != std::vector<std::size_t>({128,118})) throw std::runtime_error("embedding shape mismatch");
  if (atomic_numbers_json.size() != expected_embedding.size()) throw std::runtime_error("embedding fixture dimensions mismatch");
  double embedding_max_error = 0;
  for (std::size_t atom=0; atom<atomic_numbers_json.size(); ++atom) {
    const auto z = static_cast<std::size_t>(atomic_numbers_json[atom].unsigned_integer());
    for (std::size_t feature=0; feature<128; ++feature) {
      const double actual = model.float64(weight, feature*118+z);
      embedding_max_error = std::max(embedding_max_error, std::abs(actual-expected_embedding.at(atom).at(feature)));
      close(actual, expected_embedding.at(atom).at(feature), 0, 0, "invariant embedding");
    }
  }

  auto corrupt = model.bytes();
  corrupt.back() ^= 1;
  bool rejected = false;
  try { static_cast<void>(so3lr::NativeModel::load_bytes(std::move(corrupt))); }
  catch (const std::exception &) { rejected = true; }
  if (!rejected) throw std::runtime_error("C++ loader accepted corrupted payload");

  std::cout << "tensor_count=" << model.tensor_count() << '\n'
            << "tensor_bytes=" << model.tensor_bytes() << '\n'
            << "short_range_cutoff=4.5\nlong_range_cutoff=12\n"
            << "input_distances=" << distances.size() << '\n'
            << "embedding_atoms=" << atomic_numbers_json.size() << '\n'
            << "cutoff_max_abs_error=" << cutoff_max_error << '\n'
            << "rbf_max_abs_error=" << rbf_max_error << '\n'
            << "embedding_max_abs_error=" << embedding_max_error << '\n'
            << "cpp_corruption_detected=1\n"
            << "native_loader=PASS\ninput_operator_equivalence=PASS\nSO3LR_NATIVE_DEV2_TEST=PASS\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "SO3LR_NATIVE_DEV2_TEST=FAIL reason=" << error.what() << '\n';
  return 1;
}

