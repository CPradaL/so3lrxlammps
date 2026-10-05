#include "so3lr/native_model.hpp"

#include "so3lr/so3lr_capabilities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace so3lr {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("SO3LR native model: " + message);
}

class JsonParser {
 public:
  explicit JsonParser(const std::string &text) : text_(text) {}

  Json parse() {
    Json result = value();
    whitespace();
    if (position_ != text_.size()) fail("trailing JSON data");
    return result;
  }

 private:
  void whitespace() {
    while (position_ < text_.size() &&
           (text_[position_] == ' ' || text_[position_] == '\n' ||
            text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
  }

  char take() {
    if (position_ == text_.size()) fail("unexpected end of JSON");
    return text_[position_++];
  }

  bool consume(char expected) {
    whitespace();
    if (position_ < text_.size() && text_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void literal(const char *word) {
    const std::size_t count = std::strlen(word);
    if (text_.compare(position_, count, word) != 0) fail("invalid JSON literal");
    position_ += count;
  }

  static void append_utf8(std::string &result, std::uint32_t code) {
    if (code <= 0x7f) {
      result.push_back(static_cast<char>(code));
    } else if (code <= 0x7ff) {
      result.push_back(static_cast<char>(0xc0 | (code >> 6)));
      result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else if (code <= 0xffff) {
      result.push_back(static_cast<char>(0xe0 | (code >> 12)));
      result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
      result.push_back(static_cast<char>(0xf0 | (code >> 18)));
      result.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
  }

  std::uint32_t hex4() {
    std::uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = take();
      result <<= 4;
      if (c >= '0' && c <= '9') result |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') result |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') result |= static_cast<unsigned>(c - 'A' + 10);
      else fail("invalid JSON unicode escape");
    }
    return result;
  }

  std::string string_value() {
    whitespace();
    if (take() != '"') fail("expected JSON string");
    std::string result;
    while (true) {
      const char c = take();
      if (c == '"') break;
      if (static_cast<unsigned char>(c) < 0x20) fail("control byte in JSON string");
      if (c != '\\') {
        result.push_back(c);
        continue;
      }
      const char escaped = take();
      switch (escaped) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          std::uint32_t code = hex4();
          if (code >= 0xd800 && code <= 0xdbff) {
            if (take() != '\\' || take() != 'u') fail("invalid surrogate pair");
            const std::uint32_t low = hex4();
            if (low < 0xdc00 || low > 0xdfff) fail("invalid surrogate pair");
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
          }
          append_utf8(result, code);
          break;
        }
        default: fail("invalid JSON escape");
      }
    }
    return result;
  }

  Json number_value() {
    whitespace();
    const std::size_t begin = position_;
    if (position_ < text_.size() && text_[position_] == '-') ++position_;
    if (position_ == text_.size()) fail("invalid JSON number");
    if (text_[position_] == '0') ++position_;
    else {
      if (text_[position_] < '1' || text_[position_] > '9') fail("invalid JSON number");
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
    }
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      if (position_ == text_.size() || text_[position_] < '0' || text_[position_] > '9') fail("invalid JSON fraction");
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
      if (position_ == text_.size() || text_[position_] < '0' || text_[position_] > '9') fail("invalid JSON exponent");
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
    }
    const std::string token = text_.substr(begin, position_ - begin);
    char *end = nullptr;
    const double parsed = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size() || !std::isfinite(parsed)) fail("invalid JSON number");
    return Json(Json::Storage(parsed));
  }

  Json value() {
    whitespace();
    if (position_ == text_.size()) fail("unexpected end of JSON");
    const char c = text_[position_];
    if (c == '"') return Json(Json::Storage(string_value()));
    if (c == '{') return object_value();
    if (c == '[') return array_value();
    if (c == 't') { literal("true"); return Json(Json::Storage(true)); }
    if (c == 'f') { literal("false"); return Json(Json::Storage(false)); }
    if (c == 'n') { literal("null"); return Json(Json::Storage(nullptr)); }
    return number_value();
  }

  Json array_value() {
    if (!consume('[')) fail("expected JSON array");
    Json::Array result;
    if (consume(']')) return Json(Json::Storage(std::move(result)));
    while (true) {
      result.push_back(value());
      if (consume(']')) break;
      if (!consume(',')) fail("expected comma in JSON array");
    }
    return Json(Json::Storage(std::move(result)));
  }

  Json object_value() {
    if (!consume('{')) fail("expected JSON object");
    Json::Object result;
    if (consume('}')) return Json(Json::Storage(std::move(result)));
    while (true) {
      const std::string key = string_value();
      if (!consume(':')) fail("expected colon in JSON object");
      if (!result.emplace(key, value()).second) fail("duplicate JSON object key");
      if (consume('}')) break;
      if (!consume(',')) fail("expected comma in JSON object");
    }
    return Json(Json::Storage(std::move(result)));
  }

  const std::string &text_;
  std::size_t position_ = 0;
};

constexpr std::array<std::uint32_t, 64> kSha = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

std::uint32_t rotate_right(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

std::array<std::uint8_t, 32> sha256(const std::uint8_t *data, std::size_t size) {
  std::vector<std::uint8_t> padded(data, data + size);
  padded.push_back(0x80);
  while (padded.size() % 64 != 56) padded.push_back(0);
  const std::uint64_t bits = static_cast<std::uint64_t>(size) * 8;
  for (int shift = 56; shift >= 0; shift -= 8) padded.push_back(static_cast<std::uint8_t>(bits >> shift));
  std::array<std::uint32_t, 8> h = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  for (std::size_t base = 0; base < padded.size(); base += 64) {
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      const std::size_t p = base + static_cast<std::size_t>(4 * i);
      w[i] = (static_cast<std::uint32_t>(padded[p]) << 24) |
             (static_cast<std::uint32_t>(padded[p + 1]) << 16) |
             (static_cast<std::uint32_t>(padded[p + 2]) << 8) | padded[p + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const auto s0 = rotate_right(w[i-15],7) ^ rotate_right(w[i-15],18) ^ (w[i-15] >> 3);
      const auto s1 = rotate_right(w[i-2],17) ^ rotate_right(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      const auto s1=rotate_right(e,6)^rotate_right(e,11)^rotate_right(e,25);
      const auto ch=(e&f)^((~e)&g);
      const auto t1=hh+s1+ch+kSha[i]+w[i];
      const auto s0=rotate_right(a,2)^rotate_right(a,13)^rotate_right(a,22);
      const auto maj=(a&b)^(a&c)^(b&c);
      const auto t2=s0+maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
  }
  std::array<std::uint8_t, 32> result{};
  for (std::size_t i = 0; i < h.size(); ++i)
    for (int j = 0; j < 4; ++j) result[4*i+j]=static_cast<std::uint8_t>(h[i] >> (24-8*j));
  return result;
}

std::uint32_t little_u32(const std::uint8_t *p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint64_t little_u64(const std::uint8_t *p) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
  return value;
}

std::size_t dtype_bytes(const std::string &dtype) {
  if (dtype == "float64" || dtype == "int64") return 8;
  if (dtype == "float32" || dtype == "int32") return 4;
  if (dtype == "uint8" || dtype == "bool") return 1;
  fail("unsupported tensor dtype " + dtype);
}

std::size_t checked_size(const Json &value, const std::string &what) {
  const std::uint64_t number = value.unsigned_integer();
  if (number > std::numeric_limits<std::size_t>::max()) fail(what + " exceeds size_t");
  return static_cast<std::size_t>(number);
}

}  // namespace

Json Json::parse(const std::string &text) { return JsonParser(text).parse(); }
const Json &Json::at(const std::string &key) const {
  const auto &items = object();
  const auto found = items.find(key);
  if (found == items.end()) fail("missing JSON key " + key);
  return found->second;
}
const Json::Array &Json::array() const {
  const auto *result = std::get_if<Array>(&value_);
  if (!result) fail("JSON value is not an array");
  return *result;
}
const Json::Object &Json::object() const {
  const auto *result = std::get_if<Object>(&value_);
  if (!result) fail("JSON value is not an object");
  return *result;
}
const std::string &Json::string() const {
  const auto *result = std::get_if<std::string>(&value_);
  if (!result) fail("JSON value is not a string");
  return *result;
}
double Json::number() const {
  const auto *result = std::get_if<double>(&value_);
  if (!result) fail("JSON value is not a number");
  return *result;
}
std::uint64_t Json::unsigned_integer() const {
  const double value = number();
  if (value < 0 || std::floor(value) != value || value > 9007199254740991.0) fail("JSON value is not an exact unsigned integer");
  return static_cast<std::uint64_t>(value);
}
bool Json::boolean() const {
  const auto *result = std::get_if<bool>(&value_);
  if (!result) fail("JSON value is not a boolean");
  return *result;
}

std::string sha256_hex(const std::uint8_t *data, std::size_t size) {
  const auto digest = sha256(data, size);
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const auto byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
  return out.str();
}

NativeModel NativeModel::load(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open " + path);
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0) fail("cannot determine model size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!bytes.empty()) input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) fail("cannot read complete model");
  return load_bytes(std::move(bytes), path);
}

NativeModel NativeModel::load_bytes(std::vector<std::uint8_t> bytes,
                                    const std::string &origin) {

  constexpr std::size_t header = 128;
  constexpr std::array<std::uint8_t, 8> magic = {'S','O','3','L','R','N','1',0};
  if (bytes.size() < header) fail("file is shorter than fixed header");
  if (!std::equal(magic.begin(), magic.end(), bytes.begin())) fail("wrong magic");
  if (little_u32(bytes.data()+8) != 1 || little_u32(bytes.data()+12) != 0) fail("unsupported version or flags");
  const std::size_t manifest_size = static_cast<std::size_t>(little_u64(bytes.data()+16));
  const std::size_t payload_size = static_cast<std::size_t>(little_u64(bytes.data()+24));
  if (manifest_size > bytes.size()-header || payload_size != bytes.size()-header-manifest_size) fail("header length mismatch");
  for (std::size_t i = 96; i < 128; ++i) if (bytes[i] != 0) fail("nonzero reserved header bytes");
  const std::uint8_t *manifest_data = bytes.data()+header;
  const std::uint8_t *payload_data = manifest_data+manifest_size;
  const auto manifest_digest = sha256(manifest_data, manifest_size);
  const auto payload_digest = sha256(payload_data, payload_size);
  if (!std::equal(manifest_digest.begin(), manifest_digest.end(), bytes.begin()+32)) fail("manifest SHA-256 mismatch");
  if (!std::equal(payload_digest.begin(), payload_digest.end(), bytes.begin()+64)) fail("payload SHA-256 mismatch");

  NativeModel result;
  result.origin_ = origin;
  result.bytes_ = std::move(bytes);
  result.payload_begin_ = header + manifest_size;
  const std::string manifest_text(reinterpret_cast<const char *>(result.bytes_.data()+header), manifest_size);
  result.manifest_ = Json::parse(manifest_text);
  if (result.manifest_.at("format").string() != "so3lr-native" || result.manifest_.at("format_version").unsigned_integer() != 1) fail("manifest format mismatch");
  if (result.manifest_.at("endianness").string() != "little" || result.manifest_.at("tensor_alignment").unsigned_integer() != 64) fail("manifest endianness/alignment mismatch");
  if (result.manifest_.at("model_family").string() != "SO3LR") fail("wrong model family");

  std::set<std::string> names, state_keys;
  std::size_t previous_end = 0, total = 0;
  const auto &records = result.manifest_.at("tensors").array();
  for (const auto &item : records) {
    TensorRecord tensor;
    tensor.name = item.at("name").string();
    tensor.state_key = item.at("state_key").string();
    tensor.dtype = item.at("dtype").string();
    tensor.role = item.at("role").string();
    tensor.offset = checked_size(item.at("offset"), "tensor offset");
    tensor.nbytes = checked_size(item.at("nbytes"), "tensor byte count");
    tensor.sha256 = item.at("sha256").string();
    std::size_t elements = 1;
    for (const auto &extent_json : item.at("shape").array()) {
      const std::size_t extent = checked_size(extent_json, "tensor extent");
      if (extent != 0 && elements > std::numeric_limits<std::size_t>::max()/extent) fail("tensor shape overflow");
      elements *= extent;
      tensor.shape.push_back(extent);
    }
    if (!names.insert(tensor.name).second || !state_keys.insert(tensor.state_key).second) fail("duplicate tensor name/state key");
    if (tensor.offset % 64 != 0 || tensor.nbytes != elements*dtype_bytes(tensor.dtype)) fail("tensor alignment/size mismatch");
    if (tensor.offset < previous_end || tensor.offset > payload_size || tensor.nbytes > payload_size-tensor.offset) fail("tensor overlap/range error");
    for (std::size_t p = previous_end; p < tensor.offset; ++p) if (payload_data[p] != 0) fail("nonzero payload padding");
    if (sha256_hex(payload_data+tensor.offset, tensor.nbytes) != tensor.sha256) fail("per-tensor SHA-256 mismatch");
    previous_end = tensor.offset + tensor.nbytes;
    total += tensor.nbytes;
    result.by_state_key_.emplace(tensor.state_key, result.tensors_.size());
    result.tensors_.push_back(std::move(tensor));
  }
  if (previous_end != payload_size) fail("unreferenced trailing payload bytes");
  if (result.manifest_.at("tensor_count").unsigned_integer() != result.tensors_.size() || result.manifest_.at("tensor_bytes").unsigned_integer() != total) fail("tensor totals mismatch");
  // The descriptor is derived once, from the validated manifest, and is
  // from here on the single source of truth for every kernel dimension.
  result.arch_ = describe_architecture(result);
  // Integrity is now established; whether this build can *execute* the
  // model is a separate question, answered against a declared capability
  // table rather than against three inline literals.
  negotiate_capabilities(result, result.arch_, result.origin_);
  return result;
}

const TensorRecord &NativeModel::tensor(const std::string &state_key) const {
  const auto found = by_state_key_.find(state_key);
  if (found == by_state_key_.end()) fail("missing tensor " + state_key);
  return tensors_.at(found->second);
}
std::size_t NativeModel::tensor_bytes() const {
  std::size_t total = 0;
  for (const auto &tensor : tensors_) total += tensor.nbytes;
  return total;
}
double NativeModel::architecture_number(const std::string &key) const { return manifest_.at("architecture").at(key).number(); }
double NativeModel::model_number(const std::string &key) const {
  const auto &architecture = manifest_.at("architecture").object();
  const auto direct = architecture.find(key);
  if (direct != architecture.end()) return direct->second.number();
  for (const auto &module : manifest_.at("modules").array()) {
    if (module.at("type").string() ==
        "so3krates_torch.modules.models.SO3LR")
      return module.at("attributes").at(key).number();
  }
  fail("missing SO3LR model number " + key);
}
std::int64_t NativeModel::int64(const TensorRecord &tensor, std::size_t index) const {
  if (tensor.dtype != "int64" || index >= tensor.nbytes/8) fail("invalid int64 tensor access");
  std::uint64_t bits = little_u64(bytes_.data()+payload_begin_+tensor.offset+8*index);
  std::int64_t value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
double NativeModel::float64(const TensorRecord &tensor, std::size_t index) const {
  if (tensor.dtype != "float64" || index >= tensor.nbytes/8) fail("invalid float64 tensor access");
  const std::uint64_t bits = little_u64(bytes_.data()+payload_begin_+tensor.offset+8*index);
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace so3lr
