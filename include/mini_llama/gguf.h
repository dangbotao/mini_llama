#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// GGUF value types
// ---------------------------------------------------------------------------
// metadata的value类型，枚举编号由 GGUF 文件格式规范定义
enum GgufValueType : uint32_t {
  kGgufTypeUint8 = 0,
  kGgufTypeInt8 = 1,
  kGgufTypeUint16 = 2,
  kGgufTypeInt16 = 3,
  kGgufTypeUint32 = 4,
  kGgufTypeInt32 = 5,
  kGgufTypeFloat32 = 6,
  kGgufTypeBool = 7,
  kGgufTypeString = 8,
  kGgufTypeArray = 9,
  kGgufTypeUint64 = 10,
  kGgufTypeInt64 = 11,
  kGgufTypeFloat64 = 12,
};

// ---------------------------------------------------------------------------
// GGML types
// ---------------------------------------------------------------------------
enum GgmlType : uint32_t {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ40 = 2,
  kGgmlTypeQ41 = 3,
  kGgmlTypeQ80 = 8,
};

// ---------------------------------------------------------------------------
// GGUF tensor info
// ---------------------------------------------------------------------------
struct GgufTensorInfo {
  std::string name;
  uint32_t n_dims;
  std::vector<uint64_t> shape;
  uint32_t type = 0;    // GGML type
  uint64_t offset = 0;  // offset from start of data section
};

// ---------------------------------------------------------------------------
// GGUF metadata KV
// ---------------------------------------------------------------------------
struct GgufMetadataKv {
  std::string key;
  uint32_t type = 0;      // GGUF value type
  // value 在 GGUF 文件中的绝对字节偏移，指向 type 字段之后。
  uint64_t value_offset = 0;

  // 转换为string，统一保存和展示 metadata，而不是用于后续模型计算。
  // value_str 只是可读摘要，并不保存数组中的所有元素。
  std::string value_str;  // human-readable representation
};

// ---------------------------------------------------------------------------
// Minimal GGUF reader (v3)
// ---------------------------------------------------------------------------
struct GgufReader {

public:
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;

  // 真实数据起始位置
  uint64_t data_offset = 0;  // absolute file offset where tensor data begins

  // Load and parse a GGUF file. Returns false on error (message in load_error).
  bool Load(const std::string& path);

  // 通过名称 key 快速查找 metadata；不存在时返回 nullptr。
  const GgufMetadataKv* FindMetadata(const std::string& key) const;

  // 返回解析出的 metadata 描述信息，只允许上层只读访问。
  const std::vector<GgufMetadataKv>& GetMetadataEntries() const {
    return metadata_;
  }

  // 返回解析出的 tensor 描述信息，只允许上层只读访问。
  const std::vector<GgufTensorInfo>& GetTensors() const { return tensors_; }

  // 根据key，直接读取 typed metadata
  bool GetMetadataInt(const std::string& key, int64_t& out) const;
  bool GetMetadataFloat(const std::string& key, double& out) const;
  bool GetMetadataString(const std::string& key, std::string& out) const;
  bool GetMetadataStringArray(const std::string& key,
                              std::vector<std::string>& out) const;
  bool GetMetadataIntArray(const std::string& key,
                           std::vector<int32_t>& out) const;

  // 读取 tensor 数据区中的原始字节（uint8_t）
  bool ReadTensorRaw(const GgufTensorInfo& info,
                     std::vector<uint8_t>& out) const;

  // 根据 tensor 的 GGML 类型和 shape 安全计算原始数据字节数。
  // 该计算不依赖具体 reader 实例，因此声明为 static。
  static bool TensorDataSize(const GgufTensorInfo& info,
                             uint64_t& out_size);

  std::string load_error;
private:
  // metadata 名称 -> metadata vector 下标。
  std::unordered_map<std::string, std::size_t> metadata_index_;
  std::vector<GgufMetadataKv> metadata_;
  std::vector<GgufTensorInfo> tensors_;
  std::string path_;

  bool OpenMetadataValue(const std::string& key, std::ifstream& file,
                         uint32_t& out_type) const;

  bool ReadHeader(std::ifstream& file);
  bool ReadMetadata(std::ifstream& file);
  bool ReadTensorInfos(std::ifstream& file);
  bool ComputeDataOffset(std::ifstream& file);
  bool ValidateTensorRanges(std::ifstream& file);

};

// Print GGUF contents to stdout (for inspect-gguf).
void InspectGguf(const GgufReader& reader);

}  // namespace mini_llama
