# tensor 实现顺序
1. 写 data、shape 和构造函数声明。
2. 写 CheckedNumel。
3. 完成构造函数、size()、num_elements()、num_dims()。
4. 写 CheckRank 和 CheckAxisIndex。
5. 独立实现并测试 FlatIndex。
6. 基于 FlatIndex 实现通用 At。
7. 实现 At1～At4。
8. 实现 RowPtr。
9. 实现 AssertShape 和 ReshapeChecked。
10. 最后补打印函数和 MakeTensor*。

# 普通编译及测试
cmake -S . -B build \
  -DMINI_LLAMA_BUILD_GTEST=ON

cmake --build build -j

./build/mini-llama-gtests \
  --gtest_filter=OpsTest.TestRope

#  SIMD编译及测试
在项目根目录执行：

```bash
cmake -S . -B build-avx2 \
  -DMINI_LLAMA_BUILD_GTEST=ON \
  -DCMAKE_CXX_FLAGS="-mavx2 -mfma"

cmake --build build-avx2 -j
```

运行全部测试：

```bash
ctest --test-dir build-avx2 --output-on-failure
```

只运行刚添加的 SIMD 测试：

```bash
./build-avx2/mini-llama-gtests \
  --gtest_filter=LinearTest.TestLinearSimdAndThreadedSimdMatchNaive
```

# 越界问题
复杂越界问题可以用 AddressSanitizer，它会直接给出发生越界的源码行：
cmake -S . -B build-asan \
  -DMINI_LLAMA_BUILD_GTEST=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"

cmake --build build-asan -j
./build-asan/mini-llama-gtests \
  --gtest_filter=TokenizerTest.TestEncodeChineseCharacter

# 运行 GGUF 模型

模型文件不纳入 Git，请放置在 `models/chat/` 目录下：

```bash
MINI_LLAMA_PERF=1 ./build/mini-llama \
  models/chat/Qwen2.5-3B-Instruct-Q4_0.gguf
```
