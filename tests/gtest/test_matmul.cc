#include <gtest/gtest.h>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {

TEST(MatmulTest, TestMatmulNaiveResult) {
    Tensor a = MakeTensor2D(2, 3);
    Tensor b = MakeTensor2D(3, 2);

    // a = [[1, 2, 3],
    //      [4, 5, 6]]
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;
    a[4] = 5.0f;
    a[5] = 6.0f;

    // b = [[ 7,  8],
    //      [ 9, 10],
    //      [11, 12]]
    b[0] = 7.0f;
    b[1] = 8.0f;
    b[2] = 9.0f;
    b[3] = 10.0f;
    b[4] = 11.0f;
    b[5] = 12.0f;

    Tensor c = MatmulDispatch(a, b, MatmulMode::kNaive);

    ASSERT_EQ(c.shape, (std::vector<int>{2, 2}));
    EXPECT_FLOAT_EQ(c.At2(0, 0), 58.0f);
    EXPECT_FLOAT_EQ(c.At2(0, 1), 64.0f);
    EXPECT_FLOAT_EQ(c.At2(1, 0), 139.0f);
    EXPECT_FLOAT_EQ(c.At2(1, 1), 154.0f);
}

TEST(MatmulTest, TestMatmulNaiveRejectsIncompatibleShapes) {
    Tensor a = MakeTensor2D(2, 3);
    Tensor b = MakeTensor2D(4, 2);

    EXPECT_THROW(
        MatmulDispatch(a, b, MatmulMode::kNaive),
        std::runtime_error);
}

TEST(MatmulTest, TestMatmulNaiveRejectsNonMatrixInputs) {
    Tensor a = MakeTensor1D(3);
    Tensor b = MakeTensor2D(3, 2);

    EXPECT_THROW(
        MatmulDispatch(a, b, MatmulMode::kNaive),
        std::runtime_error);
}

TEST(MatmulTest, TestMatmulThreadedMatchesNaive) {
    // Five output rows split across three threads also exercises remainder
    // distribution in ParallelFor.
    SetThreadCount(3);

    Tensor a = MakeTensor2D(5, 7);
    Tensor b = MakeTensor2D(7, 4);

    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 9) - 2) * 0.25f;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.5f;
    }

    const Tensor expected =
        MatmulDispatch(a, b, MatmulMode::kNaive);
    const Tensor actual =
        MatmulDispatch(a, b, MatmulMode::kThreaded);

    ASSERT_EQ(actual.shape, expected.shape);
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-6f)
            << "flat index: " << i;
    }
}

TEST(LinearTest, TestLinearNaiveWithVectorInput) {
    Tensor x = MakeTensor1D(3);
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;

    // weight = [[1, 2, 3],
    //           [4, 5, 6]]
    Tensor weight = MakeTensor2D(2, 3);
    weight[0] = 1.0f;
    weight[1] = 2.0f;
    weight[2] = 3.0f;
    weight[3] = 4.0f;
    weight[4] = 5.0f;
    weight[5] = 6.0f;

    Tensor y = LinearDispatch(x, weight, MatmulMode::kNaive);

    ASSERT_EQ(y.shape, (std::vector<int>{2}));
    EXPECT_FLOAT_EQ(y[0], 14.0f);
    EXPECT_FLOAT_EQ(y[1], 32.0f);
}

TEST(LinearTest, TestLinearNaiveWithSingletonBatchInput) {
    // Linear accepts a 2D input only when its first dimension is 1.
    Tensor x = MakeTensor2D(1, 3);
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;

    Tensor weight = MakeTensor2D(2, 3);
    weight[0] = 1.0f;
    weight[1] = 2.0f;
    weight[2] = 3.0f;
    weight[3] = 4.0f;
    weight[4] = 5.0f;
    weight[5] = 6.0f;

    Tensor y = LinearDispatch(x, weight, MatmulMode::kNaive);

    ASSERT_EQ(y.shape, (std::vector<int>{1, 2}));
    EXPECT_FLOAT_EQ(y.At2(0, 0), 14.0f);
    EXPECT_FLOAT_EQ(y.At2(0, 1), 32.0f);
}

TEST(LinearTest, TestLinearNaiveRejectsMultipleRows) {
    Tensor x = MakeTensor2D(2, 3);
    Tensor weight = MakeTensor2D(2, 3);

    EXPECT_THROW(
        LinearDispatch(x, weight, MatmulMode::kNaive),
        std::runtime_error);
}

TEST(LinearTest, TestLinearThreadedMatchesNaive) {
    SetThreadCount(3);

    Tensor x = MakeTensor2D(1, 100);
    Tensor weight = MakeTensor2D(500, 100);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = i;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = i;
    }

    const Tensor expected =
        LinearDispatch(x, weight, MatmulMode::kNaive);
    const Tensor actual =
        LinearDispatch(x, weight, MatmulMode::kThreaded);

    ASSERT_EQ(actual.shape, expected.shape);
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-6f)
            << "flat index: " << i;
    }
}

TEST(LinearTest, TestLinearSimdAndThreadedSimdMatchNaive) {
    // 27 exercises the 16-element SIMD loop, the 8-element SIMD loop, and
    // the scalar tail. Seven output rows across three threads also exercises
    // uneven task distribution in LinearThreadedSimd.
    constexpr int in_features = 27;
    constexpr int out_features = 7;
    SetThreadCount(3);

    Tensor x = MakeTensor1D(in_features);
    Tensor weight = MakeTensor2D(out_features, in_features);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] =
            static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0625f;
    }

    const Tensor expected =
        LinearDispatch(x, weight, MatmulMode::kNaive);
    const Tensor simd =
        LinearDispatch(x, weight, MatmulMode::kSimd);
    const Tensor threaded_simd =
        LinearDispatch(x, weight, MatmulMode::kThreadedSimd);

    ASSERT_EQ(simd.shape, expected.shape);
    ASSERT_EQ(threaded_simd.shape, expected.shape);
    ASSERT_EQ(simd.size(), expected.size());
    ASSERT_EQ(threaded_simd.size(), expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(simd[i], expected[i], 1e-5f)
            << "LinearSimd flat index: " << i;
        EXPECT_NEAR(threaded_simd[i], expected[i], 1e-5f)
            << "LinearThreadedSimd flat index: " << i;
    }
}

} // namespace mini_llama
