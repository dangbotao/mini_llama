#pragma once

#include <string>
#include <unordered_map>

#include "mini_llama/gguf.h"
#include "mini_llama/model.h"

namespace mini_llama {

class GgufModelLoader {
 public:
  // reader must remain alive for the lifetime of this loader.
  explicit GgufModelLoader(const GgufReader& reader) : reader_(reader) {}

  MiniLlamaModel Load();

 private:
  ModelConfig BuildConfig() const;
  void BuildTensorMap();
  void LoadWeights(MiniLlamaModel& model) const;

  QuantizedTensor LoadWeight(const std::string& name) const;
  Tensor LoadTensorF32(const std::string& name) const;
  Tensor LoadOptionalTensorF32(const std::string& name) const;

  const GgufReader& reader_;

  // 指针指向 reader_.tensors 中的元素。
  // reader_ 是只读引用，因此这些指针在 loader 生命周期内保持有效。
  std::unordered_map<std::string, const GgufTensorInfo*> tensor_map_;
};

}  // namespace mini_llama
