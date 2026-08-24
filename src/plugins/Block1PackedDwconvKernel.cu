#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 4;
constexpr int32_t kOutputRowsPerTile = 2;
constexpr int32_t kFixedSpatialExtent = 88;
constexpr int32_t kFixedChannelPairs = 128; // 256channels / 2
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluScaledCubic = 0.035677406936883926F; // scale * 0.044715 常量折叠

__device__ __forceinline__ void multiplyAddHalf2(
    __half2& accumulator,
    const __half2& input,
    const __half2& weight
) {
    accumulator = __hfma2(input, weight, accumulator);
}

// 卷积在 half2 中用 __hfma2 进行 FP16 累加；GELU 前只转换一次为
// float2，保留原有 FP32 多项式。两个 tanh 输入仍打包为 half2，使用
// SM75+ 的 tanh.approx.f16x2 一次处理两个 channel。
__device__ __forceinline__ __half2 convertFloat2Output(
    const float2& accumulator
) {
    const float xSquared0 = accumulator.x * accumulator.x;
    const float xSquared1 = accumulator.y * accumulator.y;
    const float tanhInput0 = accumulator.x *
        fmaf(kGeluScaledCubic, xSquared0, kGeluScale);
    const float tanhInput1 = accumulator.y *
        fmaf(kGeluScaledCubic, xSquared1, kGeluScale);
    const __half2 packedTanh = h2tanh_approx(
        __floats2half2_rn(tanhInput0, tanhInput1)
    );
    const float2 tanhValue = __half22float2(packedTanh);
    const float halfValue0 = 0.5F * accumulator.x;
    const float halfValue1 = 0.5F * accumulator.y;
    return __floats2half2_rn(
        fmaf(halfValue0, tanhValue.x, halfValue0),
        fmaf(halfValue1, tanhValue.y, halfValue1)
    );
}

__device__ __forceinline__ __half2 convertOutput(
    const __half2& packedAccumulator
) {
    return convertFloat2Output(__half22float2(packedAccumulator));
}

// 同一行的 4 个相邻输出总共只需要 6 个不同的输入位置。每个 3-tap
// 权重加载、转换一次后，分别复用到 4 个输出上。
__device__ __forceinline__ void accumulateRowTile(
    __half2 (&accumulator)[kOutputsPerTile],
    const __half2 (&inputTile)[6],
    const __half2 (&weight)[3]
) {
#pragma unroll
    for (int32_t outputColumn = 0;
         outputColumn < kOutputsPerTile;
         ++outputColumn) {
#pragma unroll
        for (int32_t kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {
            multiplyAddHalf2(
                accumulator[outputColumn],
                inputTile[outputColumn + kernelColumn],
                weight[kernelColumn]
            );
        }
    }
}

__device__ __forceinline__ void loadWeightRow(
    const __half2* packedWeight,
    int32_t kernelRow,
    int32_t channelPair,
    int32_t channelPairs,
    __half2 (&weight)[3]
) {
    const int32_t rowOffset = kernelRow * 3 * channelPairs + channelPair;
#pragma unroll
    for (int32_t kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {
        weight[kernelColumn] =
            packedWeight[rowOffset + kernelColumn * channelPairs];
    }
}

__device__ __forceinline__ void loadInteriorInputRow(
    const __half2* inputRowStart,
    __half2 (&inputTile)[6]
) {
#pragma unroll
    for (int32_t inputColumn = 0; inputColumn < 6; ++inputColumn) {
        inputTile[inputColumn] =
            inputRowStart[inputColumn * kFixedChannelPairs];
    }
}

__device__ __forceinline__ void storeOutputRow(
    __half2* outputRowStart,
    const __half2 (&accumulator)[kOutputsPerTile]
) {
#pragma unroll
    for (int32_t outputColumn = 0;
         outputColumn < kOutputsPerTile;
         ++outputColumn) {
        outputRowStart[outputColumn * kFixedChannelPairs] =
            convertOutput(accumulator[outputColumn]);
    }
}

__device__ __forceinline__ __half2 loadBoundaryInput(
    const __half2* inputRowStart,
    int32_t column,
    int32_t channelPair
) {
    if (column < 0 || column >= kFixedSpatialExtent) {
        return __float2half2_rn(0.0F);
    }
    return inputRowStart[
        static_cast<int64_t>(column) * kFixedChannelPairs + channelPair
    ];
}

// 固定 88x88 的边界 tile 数量很少，使用完整 padding 逻辑保证首尾行列
// 正确；内部 tile 则由下面的 4x2 专用 kernel 走无分支快速路径。
__device__ __forceinline__ void accumulateBoundaryRow(
    const __half2* input,
    const __half2* packedWeight,
    int32_t outputRow,
    int32_t tileColumn,
    int32_t channelPair,
    int32_t channelPairs,
    __half2 (&accumulator)[kOutputsPerTile]
) {
#pragma unroll
    for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
        const int32_t inputRow = outputRow + kernelRow - 1;
        if (inputRow < 0 || inputRow >= kFixedSpatialExtent) {
            continue;
        }

        const int64_t rowOffset =
            static_cast<int64_t>(inputRow) * kFixedSpatialExtent *
            channelPairs;
        const __half2* inputRowStart = input + rowOffset;

        __half2 weight[3];
        loadWeightRow(
            packedWeight,
            kernelRow,
            channelPair,
            channelPairs,
            weight
        );

        __half2 inputTile[6];
        const int32_t firstInputColumn = tileColumn - 1;
#pragma unroll
        for (int32_t inputColumn = 0; inputColumn < 6; ++inputColumn) {
            inputTile[inputColumn] = loadBoundaryInput(
                inputRowStart,
                firstInputColumn + inputColumn,
                channelPair
            );
        }
        accumulateRowTile(accumulator, inputTile, weight);
    }
}

// 固定 88x88 热路径采用 4x2 二维 tile。相邻两个输出行共同需要 4 行
// 输入，中间两行只加载一次。计算按输入行展开：任一时刻最多保留两行
// weight 和一行 input，upper 完成后立即转换并写回，缩短寄存器生命周期。
__global__ void block1PackedDwconvHalf2Tile2DKernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ packedWeight,
    const __half2* __restrict__ packedBias,
    __half2* __restrict__ output
) {
    constexpr int32_t channelPairs = kFixedChannelPairs;
    const int32_t channelPair = static_cast<int32_t>(threadIdx.x);
    const int32_t tileColumn =
        static_cast<int32_t>(blockIdx.x) * kOutputsPerTile;
    const int32_t tileRow =
        static_cast<int32_t>(blockIdx.y) * kOutputRowsPerTile;
    const __half2 bias = packedBias[channelPair];
    __half2 upper[kOutputsPerTile] = {bias, bias, bias, bias};
    __half2 lower[kOutputsPerTile] = {bias, bias, bias, bias};

    const int64_t outputPairOffset =
        (static_cast<int64_t>(tileRow) * kFixedSpatialExtent +
         tileColumn) *
            channelPairs +
        channelPair;
    constexpr int64_t outputRowStride =
        static_cast<int64_t>(kFixedSpatialExtent) * channelPairs;

    const bool isInterior =
        tileRow > 0 && tileRow + kOutputRowsPerTile < kFixedSpatialExtent &&
        tileColumn > 0 &&
        tileColumn + kOutputsPerTile < kFixedSpatialExtent;
    if (isInterior) {
        constexpr int64_t rowPairStride =
            static_cast<int64_t>(kFixedSpatialExtent) * channelPairs;
        const int64_t firstInputPairOffset =
            static_cast<int64_t>(tileRow - 1) * rowPairStride +
            static_cast<int64_t>(tileColumn - 1) * channelPairs +
            channelPair;

        __half2 weightA[3];
        __half2 weightB[3];
        __half2 inputTile[6];

        // 输入第 0 行只贡献 upper，weightA 保存卷积核第 0 行。
        loadWeightRow(
            packedWeight, 0, channelPair, channelPairs, weightA
        );
        loadInteriorInputRow(input + firstInputPairOffset, inputTile);
        accumulateRowTile(upper, inputTile, weightA);

        // 输入第 1 行同时贡献 upper 和 lower；此时只保留 weightA/B。
        loadWeightRow(
            packedWeight, 1, channelPair, channelPairs, weightB
        );
        loadInteriorInputRow(
            input + firstInputPairOffset + rowPairStride,
            inputTile
        );
        accumulateRowTile(upper, inputTile, weightB);
        accumulateRowTile(lower, inputTile, weightA);

        // weightA 的第 0 行已经用完，直接复用为第 2 行。
        loadWeightRow(
            packedWeight, 2, channelPair, channelPairs, weightA
        );
        loadInteriorInputRow(
            input + firstInputPairOffset + 2 * rowPairStride,
            inputTile
        );
        accumulateRowTile(upper, inputTile, weightA);
        accumulateRowTile(lower, inputTile, weightB);

        storeOutputRow(output + outputPairOffset, upper);

        // upper 已经写回，最后一行 input 只继续完成 lower。
        loadInteriorInputRow(
            input + firstInputPairOffset + 3 * rowPairStride,
            inputTile
        );
        accumulateRowTile(lower, inputTile, weightA);
    } else {
        accumulateBoundaryRow(
            input,
            packedWeight,
            tileRow,
            tileColumn,
            channelPair,
            channelPairs,
            upper
        );
        storeOutputRow(output + outputPairOffset, upper);
        accumulateBoundaryRow(
            input,
            packedWeight,
            tileRow + 1,
            tileColumn,
            channelPair,
            channelPairs,
            lower
        );
    }

    storeOutputRow(output + outputPairOffset + outputRowStride, lower);
}

void launchFixedTile2DKernel(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    const dim3& grid,
    const dim3& block,
    cudaStream_t stream
) noexcept {
    block1PackedDwconvHalf2Tile2DKernel
        <<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(input),
            static_cast<const __half2*>(packedWeight),
            static_cast<const __half2*>(packedBias),
            static_cast<__half2*>(output)
        );
}

} // 匿名命名空间

int32_t launchBlock1PackedDwconv(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    cudaStream_t stream
) noexcept {
    if (input == nullptr || packedWeight == nullptr || packedBias == nullptr ||
        output == nullptr) {
        return -1;
    }

    const dim3 block(kFixedChannelPairs);
    const dim3 grid(
        kFixedSpatialExtent / kOutputsPerTile,
        kFixedSpatialExtent / kOutputRowsPerTile
    );
    launchFixedTile2DKernel(
        input,
        packedWeight,
        packedBias,
        output,
        grid,
        block,
        stream
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // 命名空间 egcinet::plugins
