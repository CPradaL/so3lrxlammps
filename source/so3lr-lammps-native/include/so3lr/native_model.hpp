#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "so3lr/so3lr_architecture.hpp"

namespace so3lr {

class Json {
 public:
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;
  using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

  Json() : value_(nullptr) {}
  explicit Json(Storage value) : value_(std::move(value)) {}

  static Json parse(const std::string &text);
  const Json &at(const std::string &key) const;
  const Array &array() const;
  const Object &object() const;
  const std::string &string() const;
  double number() const;
  std::uint64_t unsigned_integer() const;
  bool boolean() const;

 private:
  Storage value_;
};

struct TensorRecord {
  std::string name;
  std::string state_key;
  std::string dtype;
  std::string role;
  std::vector<std::size_t> shape;
  std::size_t offset = 0;
  std::size_t nbytes = 0;
  std::string sha256;
};

class NativeModel {
 public:
  static NativeModel load(const std::string &path);
  // `origin` is used only to name the file in capability-failure messages.
  static NativeModel load_bytes(std::vector<std::uint8_t> bytes,
                                const std::string &origin = std::string());

  const Json &manifest() const { return manifest_; }
  const TensorRecord &tensor(const std::string &state_key) const;
  std::size_t tensor_count() const { return tensors_.size(); }
  std::size_t tensor_bytes() const;
  const So3lrArchitecture &arch() const { return arch_; }
  double architecture_number(const std::string &key) const;
  double model_number(const std::string &key) const;
  std::int64_t int64(const TensorRecord &tensor, std::size_t index) const;
  double float64(const TensorRecord &tensor, std::size_t index) const;
  std::vector<std::uint8_t> bytes() const { return bytes_; }

 private:
  std::vector<std::uint8_t> bytes_;
  Json manifest_;
  std::vector<TensorRecord> tensors_;
  std::map<std::string, std::size_t> by_state_key_;
  So3lrArchitecture arch_;
  std::string origin_;
  std::size_t payload_begin_ = 0;
};

std::string sha256_hex(const std::uint8_t *data, std::size_t size);

}  // namespace so3lr

