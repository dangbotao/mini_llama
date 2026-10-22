#include "mini_llama/gguf.h"

#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <unordered_set>
#include <limits>
#include <iostream>
#include <utility>

namespace mini_llama {

namespace {

constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;

// Q4_0、Q4_1、Q8_0 三种 GGML 量化格式中，kQuantBlockSize 都固定为 32 个元素
// 每块大小计算方式为：
    // | 类型 | 每块元素数 | 量化数据 | 辅助参数 | 每块大小 |
    // |---|---:|---:|---:|---:|
    // | Q4_0 | 32 | 32 × 4 bit = 16 B | FP16 缩放值 `d`：2 B | 18 B |
    // | Q4_1 | 32 | 32 × 4 bit = 16 B | FP16 `d` + FP16 `m`：4 B | 20 B |
    // | Q8_0 | 32 | 32 × 8 bit = 32 B | FP16 缩放值 `d`：2 B | 34 B |
constexpr uint64_t kQuantBlockSize = 32;

// 从文件中读取一个按照小端字节序存储的数值
    // Read：从文件流中读取数据。
    // LE：Little Endian，即小端字节序
template <typename T>
bool ReadLe(std::ifstream& f, T& out) {
  char buf[sizeof(T)]; // 使用buffer预存，保证完整读取后再修改out
  if (!f.read(buf, sizeof(T))) {
    return false;
  }
  std::memcpy(&out, buf, sizeof(T));
  return true;
}

// 读入临时对象，与char buf + memcpy类似
// template <typename T>
// bool ReadLe(std::ifstream& f, T& out) {
//   T temp{};
//   if (!f.read(reinterpret_cast<char*>(&temp), sizeof(T))) {
//     return false;
//   }
//   out = temp;
//   return true;
// }

// 直接读入out，返回false时，read()可能已经把部分字节写入out
// template <typename T>
// bool ReadLe(std::ifstream& f, T& out) {
//   return static_cast<bool>(
//       f.read(reinterpret_cast<char*>(&out), sizeof(T)));
// }

bool ReadU32(std::ifstream& f, uint32_t& out) { return ReadLe(f, out); }

bool ReadU64(std::ifstream& f, uint64_t& out) { return ReadLe(f, out); }

// GGUF string
// [长度前缀，占8字节，值为key占用的字节数][key内容]
// 例如 general.name 的存储是：[uint64_t: 12][12字节: general.name] - 共20字节
// qwen2.attention.head_count_kv：[uint64_t: 29][29字节: qwen2.attention.head_count_kv] - 共37字节
bool ReadString(std::ifstream& f, std::string& out) {
  uint64_t len = 0;

  // 先读取string的length，然后读取对应字节数的值
  if (!ReadU64(f, len)) {
    return false;
  }
  if (len > kMaxStringBytes ||
      len >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }

  out.resize(len);
  if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
    return false;
  }
  return true;
}

// 函数返回字符串字面量，在程序生命周期中静态存储，不允许修改
// 因此返回const char*，指向只读字符数组首元素的指针
// 可直接当作c风格字符串输出
const char* GgufValueTypeName(uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
      return "uint8";
    case kGgufTypeInt8:
      return "int8";
    case kGgufTypeUint16:
      return "uint16";
    case kGgufTypeInt16:
      return "int16";
    case kGgufTypeUint32:
      return "uint32";
    case kGgufTypeInt32:
      return "int32";
    case kGgufTypeFloat32:
      return "float32";
    case kGgufTypeBool:
      return "bool";
    case kGgufTypeString:
      return "string";
    case kGgufTypeArray:
      return "array";
    case kGgufTypeUint64:
      return "uint64";
    case kGgufTypeInt64:
      return "int64";
    case kGgufTypeFloat64:
      return "float64";
    default:
      return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Type name helpers
// ---------------------------------------------------------------------------
const char* GgmlTypeName(uint32_t type) {
  switch (type) {
    case kGgmlTypeF32:
      return "F32";
    case kGgmlTypeF16:
      return "F16";
    case kGgmlTypeQ40:
      return "Q4_0";
    case kGgmlTypeQ41:
      return "Q4_1";
    case kGgmlTypeQ80:
      return "Q8_0";
    default:
      return "UNKNOWN";
  }
}

// 跳过metadata中不需要保存的数据，保证文件指针正确停在下一个字段的起始位置
bool SkipValue(std::ifstream& f, uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
    case kGgufTypeInt8: {
      char dummy;
      return f.read(&dummy, 1).good();
    }
    case kGgufTypeUint16:
    case kGgufTypeInt16: {
      char dummy[2];
      return f.read(dummy, 2).good();
    }
    case kGgufTypeUint32:
    case kGgufTypeInt32:
    case kGgufTypeFloat32: {
      char dummy[4];
      return f.read(dummy, 4).good();
    }
    case kGgufTypeBool: {
      char dummy;
      return f.read(&dummy, 1).good();
    }
    case kGgufTypeString: {
      std::string dummy;
      return ReadString(f, dummy);
    }
    case kGgufTypeUint64:
    case kGgufTypeInt64:
    case kGgufTypeFloat64: {
      char dummy[8];
      return f.read(dummy, 8).good();
    }
    case kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// Read a scalar value and convert to string representation.
bool ReadValueAsString(std::ifstream& f, uint32_t value_type, std::string& value_str) {
    std::ostringstream oss;
    switch (value_type) {
        case kGgufTypeUint8: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            // uint8_t 和 int8_t 通常分别是unsigned char和signed char的别名
            // 输出流可能会当成字符处理，转换成更宽的整数类型，确保输出数值文本
            // 例如，uint8_t v = 65，期望输出65，而不是A
            oss << static_cast<unsigned int>(v);
            break;
        }
        case kGgufTypeInt8: {
            int8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << static_cast<int>(v);
            break;
        }
        case kGgufTypeUint16: {
            uint16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeInt16: {
            int16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeUint32: {
            uint32_t v = 0;
            if (!ReadU32(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeInt32: {
            int32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeFloat32: {
            float v = 0.0f;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeBool: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << (v ? "true" : "false");
            break;
        }
        case kGgufTypeString: {
            if (!ReadString(f, value_str)) {
                return false;
            }
            return true;
        }
        case kGgufTypeUint64: {
            uint64_t v = 0;
            if (!ReadU64(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeInt64: {
            int64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeFloat64: {
            double v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        // Array type in Metadata
            // [uint32_t elem_type]
            // [uint64_t element_count]
            // [element 0]
            // [element 1]
        // 例如：
        // tokenizer.ggml.tokens 是字符串数组
            // ARRAY
            // ├─ elem_type = STRING
            // ├─ len = 151936
            // ├─ [字符串长度]["!"]
            // ├─ [字符串长度]["\""]
            // ├─ [字符串长度]["#"]
            // └─ ...
        // tokenizer.ggml.token_type 是整数数组
            // ARRAY
            // ├─ elem_type = INT32
            // ├─ len = 151936
            // ├─ 4字节整数
            // ├─ 4字节整数
            // └─ ...

        // 对于数组类型，这里跳过数组中真实的数据，只保存一个概述：
        // [len x element_type]
        case kGgufTypeArray: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }

            // for example: tokenizer.ggml.tokens [151936 x string]
            oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
            for (uint64_t i = 0; i < len; ++i) {
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            break;
        }
        default:
            return false;
    }
    value_str = oss.str();
    return true;
}

// 从metadata中读取alignment，保证tensor 数据从特定字节边界开始，
// 例如 32 或 64 字节对齐。这样 CPU SIMD、内存映射和 GPU 数据传输能更高效
uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
  uint64_t alignment = kDefaultAlignment;
  for (const auto& kv : metadata) {
    if (kv.key != "general.alignment") {
      continue;
    }
    try {
      size_t parsed = 0;
      // stoull 是“string to unsigned long long”的缩写，用于将字符串转换为 unsigned long long
      // parsed 用于接收成功解析的字符数量
      // 例如："32abc"只能成功解析两个字符
      alignment = std::stoull(kv.value_str, &parsed);
      if (parsed != kv.value_str.size()) { // 字符没能完全解析，存在非法字符
        return 0;
      }
    } catch (const std::exception&) {
      return 0;
    }
  }
  if (alignment == 0 || alignment > kMaxAlignment) {
    return 0;
  }

  // 检查aligment是否为2的幂
    //   2的幂在二进制中只有一个 1。例如：
    //     32 = 100000
    //     31 = 011111
    // 按位与结果为0
    // 而不是2的幂，例如24：
    // 24 = 11000
    // 23 = 10111
    // 结果不为0
  if ((alignment & (alignment - 1)) != 0) {
    return 0;
  }
  return alignment;
}

// 安全乘法，防止溢出   out = a*b
bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

// 安全加法， out = a + b
bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
  // 如果使用a + b判断，溢出时会重新从零开始，因此必须写成减法形式
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    return false;
  }
  out = a + b;
  return true;
}

// 安全计算tensor的元素总数
bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
  if (tensor.shape.empty()) {
    return false;
  }
  out = 1;
  for (uint64_t dim : tensor.shape) {
    if (dim == 0) {
      return false;
    }
    if (!MultiplyChecked(out, dim, out)) {
      return false;
    }
  }
  return true;
}

// 安全计算 GGUF 张量实际占用的字节数
// F32 和 F16 每个元素分别占 4 字节和 2 字节，
// Q4_0、Q4_1、Q8_0 量化类型则要求元素数能被量化块大小32整除，
// 并按每块 18、20、34 字节计算
bool ComputeTensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
  uint64_t num_elements = 0;
  if (!TensorNumel(tensor, num_elements)) {
    return false;
  }

  switch (tensor.type) {
    case kGgmlTypeF32:
      return MultiplyChecked(num_elements, 4, out);
    case kGgmlTypeF16:
      return MultiplyChecked(num_elements, 2, out);
    case kGgmlTypeQ40: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize; // 每块kQuantBlockSize个元素
      return MultiplyChecked(n_blocks, 18, out); // 总字节数 = n_blocks x 18 (每块占18个字节)
    }
    case kGgmlTypeQ41: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 20, out);
    }
    case kGgmlTypeQ80: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 34, out);
    }
    default:
      out = 0;
      return false;
  }
}
}

bool GgufReader::TensorDataSize(const GgufTensorInfo& info,
                                uint64_t& out_size) {
  return ComputeTensorDataSize(info, out_size);
}

// Header：4 + 4 + 8 + 8 = 24 bytes
bool GgufReader::ReadHeader(std::ifstream& f) {
    // Magic: GGUF
    char magic[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        load_error = "invalid GGUF magic";
        return false;
    }

    // Version
    if (!ReadU32(f, version)) {
        load_error = "failed to read version";
        return false;
    }
    if (version != 3) {
        load_error = "unsupported GGUF version: " + std::to_string(version);
        return false;
    }

    // n_tensors, n_metadata
    if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
        load_error = "failed to read header counts";
        return false;
    }
    if (n_tensors > kMaxTensorEntries) {
        load_error = "too many tensors: " + std::to_string(n_tensors);
        return false;
    }
    if (n_metadata > kMaxMetadataEntries) {
        load_error = "too many metadata entries: " +
                     std::to_string(n_metadata);
        return false;
    }
    return true;
}

// Metadata KV array - key, type, value
bool GgufReader::ReadMetadata(std::ifstream& f) {
    metadata.reserve(n_metadata);
    metadata_index_.reserve(n_metadata);
    std::unordered_set<std::string> metadata_keys;
    for (uint64_t i = 0; i < n_metadata; ++i) {
        GgufMetadataKv kv;
        // read key
        if (!ReadString(f, kv.key)) {
            load_error =
                "failed to read metadata key at FlatIndex " + std::to_string(i);
            return false;
        }
        // 检查key是否重复
        auto [iter, inserted] = metadata_keys.insert(kv.key); // [kv.key的迭代器，插入是否成功]
        if (!inserted) {
            load_error =
                "duplicate metadata key: " + kv.key;
            return false;
        }
        // read type
        if (!ReadU32(f, kv.type)) {
            load_error = "failed to read metadata type for key: " + kv.key;
            return false;
        }

        // The value starts immediately after its type field.  Keep this
        // absolute offset so later typed reads can seek directly to it.
        std::streampos value_pos = f.tellg();
        if (value_pos < 0) {
            load_error = "failed to determine metadata value offset for key: " +
                         kv.key;
            return false;
        }
        kv.value_offset = static_cast<uint64_t>(value_pos);

        // read value according to type, represent as string
        if (!ReadValueAsString(f, kv.type, kv.value_str)) {
            load_error = "failed to read metadata value for key: " + kv.key;
            return false;
        }
        metadata.push_back(std::move(kv));
        metadata_index_.emplace(metadata.back().key, metadata.size() - 1);
    }
    return true;
}

// Tensor info
bool GgufReader::ReadTensorInfos(std::ifstream& f) {
    tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo info;

        // name : GGUF string
        if (!ReadString(f, info.name)) {
            load_error =
                "failed to read tensor name at FlatIndex " + std::to_string(i);
            return false;
        }

        // n_dim: uint32
        info.n_dims = 0;
        if (!ReadU32(f, info.n_dims)) {
            load_error = "failed to read tensor n_dims for: " + info.name;
            return false;
        }

        if (info.n_dims == 0 || info.n_dims > kMaxTensorDims) {
            load_error = "invalid tensor n_dims for: " + info.name;
            return false;
        }

        // shape: uint64[n_dim]
        info.shape.resize(info.n_dims);
        for (uint32_t d = 0; d < info.n_dims; ++d) {
            if (!ReadU64(f, info.shape[d])) {
                load_error = "failed to read tensor shape for: " + info.name;
                return false;
            }
            if (info.shape[d] == 0) {
                load_error = "invalid zero tensor dimension for: " + info.name;
                return false;
            }
        }

        // type: uint32
        if (!ReadU32(f, info.type)) {
            load_error = "failed to read tensor type for: " + info.name;
            return false;
        }

        // offset: uint64
        // info.offset：当前 Tensor 相对于数据区起点（data_offset)的偏移
        if (!ReadU64(f, info.offset)) {
            load_error = "failed to read tensor offset for: " + info.name;
            return false;
        }
        tensors.push_back(std::move(info));
    }

    // 检查是否存在重名tensor
    std::unordered_set<std::string> names;
    for (const auto& tensor : tensors) {
        // names.insert返回 std::pair<iterator, bool>
        if (!names.insert(tensor.name).second) {
        load_error = "duplicate tensor name: " + tensor.name;
        return false;
        }
    }
    return true;
}

// Compute data offset (aligned to general.alignment, default 32)
bool GgufReader::ComputeDataOffset(std::ifstream& f) {
    std::streampos current_pos = f.tellg();
    if (current_pos < 0) {
        load_error = "failed to determine GGUF data offset";
        return false;
    }
    data_offset = static_cast<uint64_t>(current_pos);

    uint64_t alignment = ParseAlignment(metadata);
    if (alignment == 0) {
        load_error = "invalid general.alignment";
        return false;
    }

    // 向上对齐
    // 末尾的 % alignment，处理已经对齐的情况
    uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
    data_offset += padding;
    return true;
}

// 验证file_size与tensor.offset + tensor_size匹配
bool GgufReader::ValidateTensorRanges(std::ifstream& f) {

    // 获取文件大小
    // 将指针移动到文件末尾
    f.seekg(0, std::ios::end); // std::ios::end → 以文件末尾作为参照位置，偏移0个字节
    std::streampos end = f.tellg(); // 获取当前指针位置（文件末尾）

    // 检查文件末尾是否小于offset
    if (end < 0) {
        load_error = "failed to determine GGUF file size";
        return false;
    }
    uint64_t file_size = static_cast<uint64_t>(end);
    if (file_size < data_offset) {
        load_error = "GGUF file ended before tensor data section";
        return false;
    }

    // 检查每个Tensor的权重数据存储范围是否合法（是否超过GGUF文件大小）
    for (const auto& tensor : tensors) {
        uint64_t tensor_size = 0;
        if (!TensorDataSize(tensor, tensor_size)) {
        load_error = "invalid tensor data size for: " + tensor.name;
        return false;
        }
        if (tensor_size == 0) {
        continue;
        }

        // tensor_start = data_offset + tensor.offset
        uint64_t tensor_start = 0;
        if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
        load_error = "tensor offset overflow for: " + tensor.name;
        return false;
        }

        // tensor_end = tensor_start + tensor_size
        uint64_t tensor_end = 0;
        if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
        load_error = "tensor data range overflow for: " + tensor.name;
        return false;
        }

        // tensor_end > file_size，非法
        if (tensor_end > file_size) {
        load_error = "tensor data outside file bounds: " + tensor.name;
        return false;
        }
    }
    return true;
}

// ---- gguf 格式
// 文件头
// ├─ magic
// ├─ version
// ├─ n_tensors
// └─ n_metadata
// Metadata[n_metadata]
// ├─ key
// ├─ value type
// └─ value
// TensorInfo[n_tensors]
// ├─ name
// ├─ n_dims
// ├─ shape
// ├─ tensor type
// └─ offset
// padding
// Tensor真实数据区
bool GgufReader::Load(const std::string& path) {
    version = 0;
    n_tensors = 0;
    n_metadata = 0;
    metadata.clear();
    metadata_index_.clear();
    tensors.clear();
    data_offset = 0;
    load_error.clear();
    path_.clear();

    std::ifstream f(path, std::ios::binary); // 以二进制打开文件
    if (!f.is_open()) {
        load_error = "failed to open: " + path;
        return false;
    }

    bool loaded = ReadHeader(f) &&
                  ReadMetadata(f) &&
                  ReadTensorInfos(f) &&
                  ComputeDataOffset(f) &&
                  ValidateTensorRanges(f);
    if (loaded) {
        path_ = path;
    }
    return loaded;
}

// 找到key对应的GgufMetadataKv
const GgufMetadataKv* GgufReader::FindMetadata(
    const std::string& key) const {
    auto it = metadata_index_.find(key);
    if (it == metadata_index_.end() || it->second >= metadata.size()) {
        return nullptr;
    }
    return &metadata[it->second];
}

// 根据 metadata 的 key 快速定位其在 GGUF 文件中的真实值：
    // 它先通过索引找到对应的 metadata 条目
    // 然后以二进制方式重新打开 path_ 指向的 GGUF 文件，
    // 将文件指针移动到 value_offset 指定的数据起点，
    // 并通过 out_type 返回该值的类型
bool GgufReader::OpenMetadataValue(
    const std::string& key, std::ifstream& file,
    uint32_t& out_type) const {
    const GgufMetadataKv* metadata_entry = FindMetadata(key);
    if (metadata_entry == nullptr || path_.empty()) {
        return false;
    }
    if (metadata_entry->value_offset >
        static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }

    file.open(path_, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    // 将文件指针移动到 value_offset 指定的数据起点
    file.seekg(static_cast<std::streamoff>(metadata_entry->value_offset),
               std::ios::beg);
    if (!file.good()) {
        return false;
    }
    out_type = metadata_entry->type;
    return true;
}

// 获取int类型的metadata数据
bool GgufReader::GetMetadataInt(const std::string& key, int64_t& out) const {
    std::ifstream file;
    uint32_t type = 0;
    // 根据key找到数据类型type和文件指针的位置file
    if (!OpenMetadataValue(key, file, type)) {
        return false;
    }

    switch (type) {
        case kGgufTypeUint8: {
            uint8_t value = 0;
            if (!ReadLe(file, value)) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeInt8: {
            int8_t value = 0;
            if (!ReadLe(file, value)) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeUint16: {
            uint16_t value = 0;
            if (!ReadLe(file, value)) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeInt16: {
            int16_t value = 0;
            if (!ReadLe(file, value)) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeUint32: {
            uint32_t value = 0;
            if (!ReadU32(file, value)) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeInt32: {
            int32_t value = 0;
            if (!ReadLe(file, value)) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeUint64: {
            uint64_t value = 0;
            if (!ReadU64(file, value) ||
                value > static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max())) {
                return false;
            }
            out = static_cast<int64_t>(value);
            return true;
        }
        case kGgufTypeInt64: {
            int64_t value = 0;
            if (!ReadLe(file, value)) {
                return false;
            }
            out = value;
            return true;
        }
        default:
            return false;
    }
}

bool GgufReader::GetMetadataFloat(const std::string& key, double& out) const {
    std::ifstream file;
    uint32_t type = 0;
    if (!OpenMetadataValue(key, file, type)) {
        return false;
    }

    if (type == kGgufTypeFloat32) {
        float value = 0.0f;
        if (!ReadLe(file, value)) {
            return false;
        }
        out = static_cast<double>(value);
        return true;
    }
    if (type == kGgufTypeFloat64) {
        double value = 0.0;
        if (!ReadLe(file, value)) {
            return false;
        }
        out = value;
        return true;
    }
    return false;
}

bool GgufReader::GetMetadataString(const std::string& key,
                                   std::string& out) const {
    std::ifstream file;
    uint32_t type = 0;
    if (!OpenMetadataValue(key, file, type) || type != kGgufTypeString) {
        return false;
    }

    std::string value;
    if (!ReadString(file, value)) {
        return false;
    }
    out = std::move(value);
    return true;
}

bool GgufReader::GetMetadataStringArray(
    const std::string& key, std::vector<std::string>& out) const {
    std::ifstream file;
    uint32_t type = 0;
    if (!OpenMetadataValue(key, file, type) || type != kGgufTypeArray) {
        return false;
    }

    uint32_t element_type = 0;
    uint64_t length = 0;
    if (!ReadU32(file, element_type) || !ReadU64(file, length) ||
        element_type != kGgufTypeString || length > kMaxArrayElements) {
        return false;
    }

    std::vector<std::string> values;
    values.reserve(static_cast<std::size_t>(length));
    for (uint64_t i = 0; i < length; ++i) {
        std::string value;
        if (!ReadString(file, value)) {
            return false;
        }
        values.push_back(std::move(value));
    }
    out = std::move(values);
    return true;
}

bool GgufReader::GetMetadataIntArray(
    const std::string& key, std::vector<int32_t>& out) const {
    std::ifstream file;
    uint32_t type = 0;
    if (!OpenMetadataValue(key, file, type) || type != kGgufTypeArray) {
        return false;
    }

    uint32_t element_type = 0;
    uint64_t length = 0;
    if (!ReadU32(file, element_type) || !ReadU64(file, length) ||
        element_type != kGgufTypeInt32 || length > kMaxArrayElements) {
        return false;
    }

    std::vector<int32_t> values;
    values.reserve(static_cast<std::size_t>(length));
    for (uint64_t i = 0; i < length; ++i) {
        int32_t value = 0;
        if (!ReadLe(file, value)) {
            return false;
        }
        values.push_back(value);
    }
    out = std::move(values);
    return true;
}

// 从已加载的 GGUF 文件中读取指定 tensor 的原始二进制数据
bool GgufReader::ReadTensorRaw(const GgufTensorInfo& info,
                               std::vector<uint8_t>& out) const {
    if (path_.empty()) {
        return false;
    }

    // 计算实际占用字节数
    uint64_t tensor_size = 0;
    if (!TensorDataSize(info, tensor_size) || tensor_size == 0 ||
        tensor_size >
            static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        tensor_size > static_cast<uint64_t>(
                          std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    uint64_t tensor_start = 0;
    if (!AddChecked(data_offset, info.offset, tensor_start) ||
        tensor_start > static_cast<uint64_t>(
                           std::numeric_limits<std::streamoff>::max())) {
        return false;
    }

    // 文件指针移动到tensor_start
    std::ifstream file(path_, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
    if (!file.good()) {
        return false;
    }

    // 读取tensor_size长度的字节
    std::vector<uint8_t> bytes(static_cast<std::size_t>(tensor_size));
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(tensor_size));
    if (static_cast<uint64_t>(file.gcount()) != tensor_size) {
        return false; // 实际读取字节数
    }

    out = std::move(bytes);
    return true;
}

// ---------------------------------------------------------------------------
// InspectGguf
// ---------------------------------------------------------------------------
void InspectGguf(const GgufReader& reader) {
  std::cout << "=== GGUF Info ===" << std::endl;
  std::cout << "  version:    " << reader.version << std::endl;
  std::cout << "  n_tensors:  " << reader.n_tensors << std::endl;
  std::cout << "  n_metadata: " << reader.n_metadata << std::endl;
  std::cout << "  data_offset: " << reader.data_offset << std::endl;

  if (!reader.metadata.empty()) {
    std::cout << std::endl;
    std::cout << "=== Metadata ===" << std::endl;
    for (const auto& kv : reader.metadata) {
      std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
                << "): " << kv.value_str << std::endl;
    }
  }

  if (!reader.tensors.empty()) {
    std::cout << std::endl;
    std::cout << "=== Tensors ===" << std::endl;
    for (const auto& t : reader.tensors) {
      std::cout << "  " << t.name << std::endl;
      std::cout << "    shape:  [";
      for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << t.shape[i];
      }
      std::cout << "]" << std::endl;
      std::cout << "    dtype:  " << GgmlTypeName(t.type) << " (" << t.type
                << ")" << std::endl;
      std::cout << "    offset: " << t.offset << std::endl;
    }
  }
}
}
