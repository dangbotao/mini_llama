// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mini_llama/model.h"
#include "mini_llama/loader/loader.h"

namespace mini_llama {
namespace {

// 一次性读取整个文本文件并返回字符串
// model.json → std::string
std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// 生成临时测试文件路径
std::filesystem::path TestTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

// 把字符串写入文件，通常用于生成修改过的临时 model.json
void WriteTextFile(const std::filesystem::path& path,
                   const std::string& content) {
  std::ofstream out(path);
  out << content;
}

// 将字符串中第一次出现的 from 替换成 to
bool ReplaceOnce(std::string& text, const std::string& from,
                 const std::string& to) {
  size_t pos = text.find(from);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

// 先找到 marker，再从该位置之后寻找并替换 from
bool ReplaceAfter(std::string& text, const std::string& marker,
                  const std::string& from, const std::string& to) {
  size_t marker_pos = text.find(marker);
  if (marker_pos == std::string::npos) {
    return false;
  }
  size_t pos = text.find(from, marker_pos);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

bool ParseManifestFails(const std::string& json, const std::string& filename) {
  std::filesystem::path path = TestTempPath(filename);
  WriteTextFile(path, json);
  try {
    ParseManifest(path.string());
    std::filesystem::remove(path);
    return false;
  } catch (const std::runtime_error&) {
    std::filesystem::remove(path);
    return true;
  }
}

}  // namespace

TEST(LoaderTest, TestLoaderManifestParsesMetadata) {
  ModelManifest manifest = ParseManifest("models/tiny/model.json");
  EXPECT_EQ(manifest.config.vocab_size, 128);
  EXPECT_EQ(manifest.config.n_layers, 2);
  EXPECT_EQ(manifest.tensors.size(), 21);
  EXPECT_EQ(manifest.tensors[0].name, "token_embedding");
  EXPECT_EQ(manifest.tensors[0].dtype, "float32");
  EXPECT_EQ(manifest.tensors[0].offset, 0);
  EXPECT_EQ(manifest.tensors[0].byte_size, 16384);
}

// 故意制造两个同名张量，检查 ParseManifest() 能否拒绝它
TEST(LoaderTest, TestLoaderRejectsDuplicateTensorName) {
  std::string json = ReadTextFile("models/tiny/model.json");
  // 把 JSON 中第一次出现的："name": "final_norm" 替换成："name": "token_embedding"
  // 替换成功，出现同名张量
  EXPECT_TRUE(ReplaceOnce(json, "\"name\": \"final_norm\"",
                          "\"name\": \"token_embedding\""));
  // 触发ValidateManifest中的报错，ParseManifestFails返回true
  EXPECT_TRUE(ParseManifestFails(json, "duplicate_tensor.json"));
}

// 故意把 lm_head 的 shape 改错，检查 ValidateManifest() 能否发现
// 实际上对比的是json中config推导的值与tensors提供的元数据是否一致
TEST(LoaderTest, TestLoaderRejectsRequiredShapeMismatch) {
  // 读取正确 JSON
  // → 只修改 lm_head 的 shape
  // → ParseManifest 解析
  // → ValidateManifest 发现 shape 不匹配
  // → 抛异常
  // → 测试通过
  std::string json = ReadTextFile("models/tiny/model.json");
  // 把lm_head 的 shape 从 [128, 32] 改为 [64, 64]
  EXPECT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "\"shape\": [\n        128,\n        32\n      ]",
                           "\"shape\": [\n        64,\n        64\n      ]"));
  EXPECT_TRUE(ParseManifestFails(json, "shape_mismatch.json"));
}

TEST(LoaderTest, TestLoaderRejectsOverlappingTensorRanges) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"", "\"offset\": 115840",
                           "\"offset\": 115712"));
  EXPECT_TRUE(ParseManifestFails(json, "overlap.json"));
}

TEST(LoaderTest, TestLoaderRejectsMissingDtype) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "      \"dtype\": \"float32\",\n", ""));
  EXPECT_TRUE(ParseManifestFails(json, "missing_dtype.json"));
}

TEST(LoaderTest, TestLoaderRejectsFractionalIntegerConfig) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceOnce(json, "\"n_heads\": 4", "\"n_heads\": 4.5"));
  EXPECT_TRUE(ParseManifestFails(json, "fractional_integer_config.json"));
}

TEST(LoaderTest, TestLoaderRejectsZeroHeads) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceOnce(json, "\"n_heads\": 4", "\"n_heads\": 0"));
  EXPECT_TRUE(ParseManifestFails(json, "zero_heads_config.json"));
}

TEST(LoaderTest, TestLoaderRejectsNonfiniteFloatConfig) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(
      ReplaceOnce(json, "\"rope_theta\": 10000.0", "\"rope_theta\": NaN"));
  EXPECT_TRUE(ParseManifestFails(json, "nonfinite_float_config.json"));
}

// 权重文件末尾如果多出未知字节，LoadModel() 必须加载失败
// 它主要防止损坏的权重文件、错误版本文件或未知尾部数据被误认为合法模型。
TEST(LoaderTest, TestLoaderRejectsTrailingWeightBytes) {
  std::filesystem::path weights_path = TestTempPath("trailing_model.bin");
  {
    std::ifstream in("models/tiny/model.bin", std::ios::binary);
    std::ofstream out(weights_path, std::ios::binary);
    out << in.rdbuf();
    char extra = 0;
    out.write(&extra, 1);
  }

  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", weights_path.string());
  std::filesystem::remove(weights_path);
  EXPECT_FALSE(model.loaded);
}

TEST(LoaderTest, LoadsTinyModelCorrectly) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");

  ASSERT_TRUE(model.loaded);

  EXPECT_EQ(model.config.vocab_size, 128);
  EXPECT_EQ(model.config.n_layers, 2);
  ASSERT_EQ(model.layers.size(), 2);

  EXPECT_EQ(model.token_embedding.shape,
            std::vector<int>({128, 32}));
  EXPECT_EQ(model.layers[0].wq.shape,
            std::vector<int>({32, 32}));
  EXPECT_EQ(model.final_norm.shape,
            std::vector<int>({32}));
  EXPECT_EQ(model.lm_head.shape,
            std::vector<int>({128, 32}));

  ASSERT_FALSE(model.token_embedding.data.empty());
  EXPECT_NEAR(model.token_embedding.data[0],
              0.005577072f, 1e-6f);
}

}  // namespace mini_llama
