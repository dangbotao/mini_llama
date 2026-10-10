#pragma once

#include "mini_llama/model.h"

#include <vector>
#include <string>

namespace mini_llama {

struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t offset = 0;
    size_t byte_size = 0;
};

// model.json 中的 tokenizer 描述。
// 这里只保存创建 tokenizer 所需的信息，不包含实际词表对象。
struct TokenizerInfo {
  std::string type = "ascii";
  std::string path;
  int bos_id = 1;
  int eos_id = 2;
  int unk_id = 0;
};

// 存储解析json获得的模型结构信息
struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
};

// 解析json文件，得到模型清单 ModelManifest
ModelManifest ParseManifest(const std::string& json_path);

// 校验模型配置与张量元数据的一致性。
// LoadModel(ModelManifest, ...) 也会调用该函数，以便安全地接受手工构造的 manifest。
void ValidateManifest(const ModelManifest& manifest);

}
