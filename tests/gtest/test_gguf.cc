#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"

namespace mini_llama {

namespace {

void AppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

void AppendU64Le(std::vector<char>& data, uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

void AppendString(std::vector<char>& data, const std::string& value) {
  AppendU64Le(data, value.size());
  data.insert(data.end(), value.begin(), value.end());
}

void AppendMetadataKey(std::vector<char>& data, const std::string& key,
                       uint32_t type) {
  AppendString(data, key);
  AppendU32Le(data, type);
}

void AppendFloat32Le(std::vector<char>& data, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU32Le(data, bits);
}

std::filesystem::path WriteTypedMetadataFixture() {
  std::vector<char> data{'G', 'G', 'U', 'F'};
  AppendU32Le(data, 3);  // version
  AppendU64Le(data, 0);  // n_tensors
  AppendU64Le(data, 6);  // n_metadata

  AppendMetadataKey(data, "general.alignment", kGgufTypeUint32);
  AppendU32Le(data, 32);

  AppendMetadataKey(data, "test.int", kGgufTypeInt32);
  AppendU32Le(data, static_cast<uint32_t>(-7));

  AppendMetadataKey(data, "test.float", kGgufTypeFloat32);
  AppendFloat32Le(data, 1.5f);

  AppendMetadataKey(data, "test.string", kGgufTypeString);
  AppendString(data, "hello");

  AppendMetadataKey(data, "test.strings", kGgufTypeArray);
  AppendU32Le(data, kGgufTypeString);
  AppendU64Le(data, 2);
  AppendString(data, "one");
  AppendString(data, "two");

  AppendMetadataKey(data, "test.ints", kGgufTypeArray);
  AppendU32Le(data, kGgufTypeInt32);
  AppendU64Le(data, 3);
  AppendU32Le(data, static_cast<uint32_t>(-1));
  AppendU32Le(data, 0);
  AppendU32Le(data, 7);

  while (data.size() % 32 != 0) {
    data.push_back(0);
  }

  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      "mini_llama_typed_metadata.gguf";
  std::ofstream file(path, std::ios::binary);
  file.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

}  // namespace

// ---------------------------------------------------------------------------
// GGUF reader tests
// ---------------------------------------------------------------------------
TEST(GgufTest, TestGgufLoadTestFile) {
  GgufReader reader;
  bool ok = reader.Load("models/tiny/test.gguf");
  if (!ok) {
    FAIL() << "failed to load test.gguf: " << reader.load_error;
  }
  EXPECT_EQ(reader.version, 3);
  EXPECT_EQ(reader.n_tensors, 1);
  EXPECT_EQ(reader.n_metadata, 3);
}

TEST(GgufTest, TestGgufMetadata) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  EXPECT_EQ(reader.metadata.size(), 3);
  EXPECT_EQ(reader.metadata[0].key, "general.architecture");
  EXPECT_EQ(reader.metadata[0].value_str, "llama");
  EXPECT_EQ(reader.metadata[1].key, "general.name");
  EXPECT_EQ(reader.metadata[1].value_str, "tiny-test");
  EXPECT_EQ(reader.metadata[2].key, "general.alignment");
  EXPECT_EQ(reader.metadata[2].value_str, "64");
}

// 验证 tensor 描述信息能否正确解析tensor
TEST(GgufTest, TestGgufTensorInfo) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  EXPECT_EQ(reader.tensors.size(), 1);
  EXPECT_EQ(reader.tensors[0].name, "token_embd.weight");
  EXPECT_EQ(reader.tensors[0].shape.size(), 2);
  EXPECT_EQ(reader.tensors[0].shape[0], 128);
  EXPECT_EQ(reader.tensors[0].shape[1], 32);
  EXPECT_EQ(reader.tensors[0].type, kGgmlTypeF32);
  EXPECT_EQ(reader.tensors[0].offset, 0);
}

// 验证 value_offset 是否真的指向 metadata value 的起始位置
TEST(GgufTest, TestGgufMetadataIndex) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  // 验证metadata通过key快速索引GgufMetadataKv
  for (const auto& kv : reader.metadata) {
    const GgufMetadataKv* found = reader.FindMetadata(kv.key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found, &kv);
    EXPECT_LT(found->value_offset, reader.data_offset); // 小于
  }
  EXPECT_EQ(reader.FindMetadata("missing.key"), nullptr);

  // 验证 architecture_metadata->value_offset 是否真的指向 metadata value 的起始位置
  // "general.architecture" : "llama"
  const GgufMetadataKv* architecture_metadata =
      reader.FindMetadata("general.architecture");
  ASSERT_NE(architecture_metadata, nullptr); // not equal

  std::ifstream file("models/tiny/test.gguf", std::ios::binary);
  ASSERT_TRUE(file.is_open());
  // 文件指针跳转到value的开头
  file.seekg(
      static_cast<std::streamoff>(architecture_metadata->value_offset));

  // GGUF string格式，前8个字节为前缀，储存长度信息，期望为5
  uint64_t length = 0;
  ASSERT_TRUE(static_cast<bool>(file.read(reinterpret_cast<char*>(&length),
                                          sizeof(length))));
  ASSERT_EQ(length, 5);
  // 紧随其后5个字节为实际字符，"llama"
  std::string architecture(length, '\0');
  ASSERT_TRUE(static_cast<bool>(file.read(architecture.data(),
                                          static_cast<std::streamsize>(length))));
  EXPECT_EQ(architecture, "llama");
}

// 验证各个 typed metadata getter
// general.alignment = 32
// test.int          = -7
// test.float        = 1.5
// test.string       = "hello"
// test.strings      = {"one", "two"}
// test.ints         = {-1, 0, 7}
// 还测试两种失败情况：
// GetMetadataInt("test.string", ...)
// 类型不匹配，应返回 false；
// GetMetadataString("missing.key", ...)
// key 不存在，也应返回 false。
TEST(GgufTest, TestTypedMetadataAccess) {
  std::filesystem::path path = WriteTypedMetadataFixture();
  GgufReader reader;
  ASSERT_TRUE(reader.Load(path.string())) << reader.load_error;

  int64_t integer = 0;
  EXPECT_TRUE(reader.GetMetadataInt("general.alignment", integer));
  EXPECT_EQ(integer, 32);
  EXPECT_TRUE(reader.GetMetadataInt("test.int", integer));
  EXPECT_EQ(integer, -7);

  double floating = 0.0;
  EXPECT_TRUE(reader.GetMetadataFloat("test.float", floating));
  EXPECT_DOUBLE_EQ(floating, 1.5);

  std::string string_value;
  EXPECT_TRUE(reader.GetMetadataString("test.string", string_value));
  EXPECT_EQ(string_value, "hello");

  std::vector<std::string> strings;
  EXPECT_TRUE(reader.GetMetadataStringArray("test.strings", strings));
  EXPECT_EQ(strings, (std::vector<std::string>{"one", "two"}));

  std::vector<int32_t> integers;
  EXPECT_TRUE(reader.GetMetadataIntArray("test.ints", integers));
  EXPECT_EQ(integers, (std::vector<int32_t>{-1, 0, 7}));

  EXPECT_FALSE(reader.GetMetadataInt("test.string", integer));
  EXPECT_FALSE(reader.GetMetadataString("missing.key", string_value));
  std::filesystem::remove(path);
}

// 验证 GgufReader::TensorDataSize() 对不同数据格式的字节数计算。
// 对于 32 个元素：
// F32  ：32 × 4 = 128 字节
// F16  ：32 × 2 = 64 字节
// Q4_0 ：1 个块 = 18 字节
// Q4_1 ：1 个块 = 20 字节
// Q8_0 ：1 个块 = 34 字节
TEST(GgufTest, TestTensorDataSizeSupportsQuantizedTypes) {
  GgufTensorInfo info;
  info.shape = {32};
  uint64_t size = 0;

  info.type = kGgmlTypeF32;
  EXPECT_TRUE(GgufReader::TensorDataSize(info, size));
  EXPECT_EQ(size, 128);

  info.type = kGgmlTypeF16;
  EXPECT_TRUE(GgufReader::TensorDataSize(info, size));
  EXPECT_EQ(size, 64);

  info.type = kGgmlTypeQ40;
  EXPECT_TRUE(GgufReader::TensorDataSize(info, size));
  EXPECT_EQ(size, 18);

  info.type = kGgmlTypeQ41;
  EXPECT_TRUE(GgufReader::TensorDataSize(info, size));
  EXPECT_EQ(size, 20);

  info.type = kGgmlTypeQ80;
  EXPECT_TRUE(GgufReader::TensorDataSize(info, size));
  EXPECT_EQ(size, 34);

  info.shape = {33};
  EXPECT_FALSE(GgufReader::TensorDataSize(info, size));
  info.shape = {32};
  info.type = 999;
  EXPECT_FALSE(GgufReader::TensorDataSize(info, size));
}

// 验证能否从 GGUF tensor 数据区读取完整的原始字节
TEST(GgufTest, TestReadTensorRaw) {
  GgufReader reader;
  ASSERT_TRUE(reader.Load("models/tiny/test.gguf")) << reader.load_error;
  ASSERT_EQ(reader.tensors.size(), 1);
  uint64_t size = 0;
  ASSERT_TRUE(GgufReader::TensorDataSize(reader.tensors[0], size));
  EXPECT_EQ(size, 128 * 32 * sizeof(float));

  std::vector<uint8_t> bytes;
  EXPECT_TRUE(reader.ReadTensorRaw(reader.tensors[0], bytes));
  EXPECT_EQ(bytes.size(), 128U * 32U * sizeof(float));
}

// 验证当前 GGUF reader 能处理真实的量化模型
TEST(GgufTest, TestBundledQ80ModelParsesWhenAvailable) {
  const std::filesystem::path path =
      "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::exists(path) ||
      std::filesystem::file_size(path) < 1024 * 1024) {
    GTEST_SKIP() << "bundled Q8_0 model is not available";
  }

  GgufReader reader;
  ASSERT_TRUE(reader.Load(path.string())) << reader.load_error;
  EXPECT_EQ(reader.n_tensors, 290);

  std::size_t q80_count = 0;
  for (const auto& tensor : reader.tensors) {
    if (tensor.type == kGgmlTypeQ80) {
      ++q80_count;
    }
  }
  EXPECT_GT(q80_count, 0); // 大于
}

}  // namespace mini_llama
