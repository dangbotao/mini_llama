#include "mini_llama/kv_cache.h"
#include <cstring>
#include <stdexcept>

namespace mini_llama {

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

// write specific k and v to kv_cache
// k:[n_kv_heads, head_dim]
// v:[n_kv_heads, head_dim]
void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
    if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
        throw std::invalid_argument(
            "KvCache::Write expected matching [n_kv_heads, head_dim] tensors");
    }

    if (layer < 0 || layer >= keys.shape[0]) {
        throw std::out_of_range("KvCache::Write layer out of range");
    }

    if (pos < 0 || pos >= keys.shape[1]) {
        throw std::out_of_range("KvCache::Write position out of range");
    }

    if (k.shape[0] != keys.shape[2] ||
        k.shape[1] != keys.shape[3]) {
        throw std::invalid_argument(
            "KvCache::Write tensor shape does not match cache shape");
    }

    const size_t layer_stride =
        static_cast<size_t>(keys.shape[1]) * keys.shape[2] * keys.shape[3];
    const size_t pos_stride =
        static_cast<size_t>(keys.shape[2]) * keys.shape[3];

    const size_t offset =
        static_cast<size_t>(layer) * layer_stride +
        static_cast<size_t>(pos) * pos_stride;

    // keys shape: [n_layers, max_seq_len, n_kv_heads, head_dim]
    // We need to write at [layer, pos, :, :]
    const size_t count = pos_stride;
    std::memcpy(keys.data.data() + offset,
            k.data.data(),
            count * sizeof(float));
    std::memcpy(values.data.data() + offset,
            v.data.data(),
            count * sizeof(float));
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
    if (keys.shape.size() < 4) {
        throw std::logic_error("KvCache::KeyPtr expects a 4D tensor");
    }

    if (layer < 0 || layer >= keys.shape[0] || pos < 0 || pos >= keys.shape[1] ||
        kv_head < 0 || kv_head >= keys.shape[2]) {
    throw std::out_of_range("KvCache::KeyPtr FlatIndex out of range");
    }

    const size_t pos_count = static_cast<size_t>(keys.shape[1]);
    const size_t head_count = static_cast<size_t>(keys.shape[2]);
    const size_t head_dim = static_cast<size_t>(keys.shape[3]);

    const size_t offset =
        static_cast<size_t>(layer) * pos_count * head_count * head_dim +
        static_cast<size_t>(pos) * head_count * head_dim +
        static_cast<size_t>(kv_head) * head_dim;

    return keys.data.data() + offset;
}

const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
    if (keys.shape.size() < 4) {
        throw std::logic_error("KvCache::KeyPtr expects a 4D tensor");
    }

    if (layer < 0 || layer >= values.shape[0] || pos < 0 ||
        pos >= values.shape[1] || kv_head < 0 || kv_head >= values.shape[2]) {
        throw std::out_of_range("KvCache::ValuePtr FlatIndex out of range");
    }

    const size_t pos_count = static_cast<size_t>(values.shape[1]);
    const size_t head_count = static_cast<size_t>(values.shape[2]);
    const size_t head_dim = static_cast<size_t>(values.shape[3]);

    const size_t offset =
        static_cast<size_t>(layer) * pos_count * head_count * head_dim +
        static_cast<size_t>(pos) * head_count * head_dim +
        static_cast<size_t>(kv_head) * head_dim;

    return values.data.data() + offset;
}

}