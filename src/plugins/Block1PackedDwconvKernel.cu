#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 8;
constexpr int32_t kInputsPerTile = kOutputsPerTile + 2;
constexpr int32_t kBoundaryOutputsPerChunk = 4;
constexpr int32_t kOutputRowsPerTile = 2;
constexpr int32_t kFixedSpatialExtent = 88;
constexpr int32_t kFixedChannelPairs = 128; // 256channels / 2
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluScaledCubic = 0.035677406936883926F; // scale * 0.044715 常量折叠

static_assert(kOutputsPerTile % kBoundaryOutputsPerChunk == 0);

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

// 同一行的相邻输出总共只需要 OutputCount+2 个不同输入位置。每个
// 3-tap 权重加载、转换一次后，分别复用到整个横向 tile。
template <int32_t OutputCount>
__device__ __forceinline__ void accumulateRowTile(
    __half2 (&accumulator)[OutputCount],
    const __half2 (&inputTile)[OutputCount + 2],
    const __half2 (&weight)[3]
) {
    static_assert(OutputCount > 0 && OutputCount <= kOutputsPerTile);
#pragma unroll
    for (int32_t outputColumn = 0;
         outputColumn < OutputCount;
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
    __half2 (&inputTile)[kInputsPerTile]
) {
#pragma unroll
    for (int32_t inputColumn = 0;
         inputColumn < kInputsPerTile;
         ++inputColumn) {
        inputTile[inputColumn] = inputRowStart[
            inputColumn * kFixedChannelPairs
        ];
    }
}

template <int32_t OutputCount>
__device__ __forceinline__ void storeOutputRow(
    __half2* outputRowStart,
    const __half2 (&accumulator)[OutputCount]
) {
    static_assert(OutputCount > 0 && OutputCount <= kOutputsPerTile);
#pragma unroll
    for (int32_t outputColumn = 0;
         outputColumn < OutputCount;
         ++outputColumn) {
        outputRowStart[outputColumn * kFixedChannelPairs] =
            convertOutput(accumulator[outputColumn]);
    }
}

// 边界 CTA 仍按 2 行滑动 tile 处理；只有输入行加载器负责上下左右
// padding。这样 upper/lower 能共享中间两行输入，三行权重也只加载一次。
template <int32_t OutputCount>
__device__ __forceinline__ void loadBoundaryInputRow(
    const __half2* input,
    int32_t inputRow,
    int32_t tileColumn,
    int32_t channelPair,
    __half2 (&inputTile)[OutputCount + 2]
) {
    static_assert(OutputCount > 0 && OutputCount <= kOutputsPerTile);
    if (inputRow < 0 || inputRow >= kFixedSpatialExtent) {
#pragma unroll
        for (int32_t inputColumn = 0;
             inputColumn < OutputCount + 2;
             ++inputColumn) {
            inputTile[inputColumn] = __float2half2_rn(0.0F);
        }
        return;
    }

    const int64_t rowOffset =
        static_cast<int64_t>(inputRow) * kFixedSpatialExtent *
        kFixedChannelPairs;
    const __half2* inputRowStart = input + rowOffset + channelPair;
    if (tileColumn == 0) {
        inputTile[0] = __float2half2_rn(0.0F);
#pragma unroll
        for (int32_t inputColumn = 1;
             inputColumn < OutputCount + 2;
             ++inputColumn) {
            inputTile[inputColumn] = inputRowStart[
                (inputColumn - 1) * kFixedChannelPairs
            ];
        }
        return;
    }

    const __half2* firstInput = inputRowStart +
        static_cast<int64_t>(tileColumn - 1) * kFixedChannelPairs;
    if (tileColumn + OutputCount == kFixedSpatialExtent) {
#pragma unroll
        for (int32_t inputColumn = 0;
             inputColumn < OutputCount + 1;
             ++inputColumn) {
            inputTile[inputColumn] =
                firstInput[inputColumn * kFixedChannelPairs];
        }
        inputTile[OutputCount + 1] = __float2half2_rn(0.0F);
        return;
    }

#pragma unroll
    for (int32_t inputColumn = 0;
         inputColumn < OutputCount + 2;
         ++inputColumn) {
        inputTile[inputColumn] =
            firstInput[inputColumn * kFixedChannelPairs];
    }
}

// 8 列边界 tile 分成两个 4 列小段，避免 16 个 upper/lower 累加器
// 同时存活。每个小段仍复用四行 input，保持 40-register 内部路径的
// occupancy；首尾 padding 由模板化行加载器处理。
template <int32_t OutputCount>
__device__ __forceinline__ void processBoundaryChunk(
    const __half2* input,
    const __half2* packedWeight,
    __half2* outputRowStart,
    int32_t tileRow,
    int32_t tileColumn,
    int32_t channelPair,
    const __half2& bias
) {
    constexpr int32_t channelPairs = kFixedChannelPairs;
    __half2 upper[OutputCount];
    __half2 lower[OutputCount];
#pragma unroll
    for (int32_t outputColumn = 0;
         outputColumn < OutputCount;
         ++outputColumn) {
        upper[outputColumn] = bias;
        lower[outputColumn] = bias;
    }

    __half2 weightA[3];
    __half2 weightB[3];
    __half2 inputTile[OutputCount + 2];

    loadWeightRow(packedWeight, 0, channelPair, channelPairs, weightA);
    loadBoundaryInputRow<OutputCount>(
        input, tileRow - 1, tileColumn, channelPair, inputTile
    );
    accumulateRowTile<OutputCount>(upper, inputTile, weightA);

    loadWeightRow(packedWeight, 1, channelPair, channelPairs, weightB);
    loadBoundaryInputRow<OutputCount>(
        input, tileRow, tileColumn, channelPair, inputTile
    );
    accumulateRowTile<OutputCount>(upper, inputTile, weightB);
    accumulateRowTile<OutputCount>(lower, inputTile, weightA);

    loadWeightRow(packedWeight, 2, channelPair, channelPairs, weightA);
    loadBoundaryInputRow<OutputCount>(
        input, tileRow + 1, tileColumn, channelPair, inputTile
    );
    accumulateRowTile<OutputCount>(upper, inputTile, weightA);
    accumulateRowTile<OutputCount>(lower, inputTile, weightB);

    storeOutputRow<OutputCount>(outputRowStart, upper);

    loadBoundaryInputRow<OutputCount>(
        input, tileRow + 2, tileColumn, channelPair, inputTile
    );
    accumulateRowTile<OutputCount>(lower, inputTile, weightA);
    constexpr int64_t outputRowStride =
        static_cast<int64_t>(kFixedSpatialExtent) * kFixedChannelPairs;
    storeOutputRow<OutputCount>(outputRowStart + outputRowStride, lower);
}

// 热路径固定采用 8x2 tile。相邻输出行共同需要 4 行输入，中间两行
// 只加载一次；upper 完成后立即转换并写回，缩短寄存器生命周期。
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
        __half2 upper[kOutputsPerTile];
        __half2 lower[kOutputsPerTile];
#pragma unroll
        for (int32_t outputColumn = 0;
             outputColumn < kOutputsPerTile;
             ++outputColumn) {
            upper[outputColumn] = bias;
            lower[outputColumn] = bias;
        }
        constexpr int64_t rowPairStride =
            static_cast<int64_t>(kFixedSpatialExtent) * channelPairs;
        const int64_t firstInputPairOffset =
            static_cast<int64_t>(tileRow - 1) * rowPairStride +
            static_cast<int64_t>(tileColumn - 1) * channelPairs +
            channelPair;

        __half2 weightA[3];
        __half2 weightB[3];
        __half2 inputTile[kInputsPerTile];

        // 输入第 0 行只贡献 upper，weightA 保存卷积核第 0 行。
        loadWeightRow(
            packedWeight, 0, channelPair, channelPairs, weightA
        );
        loadInteriorInputRow(input + firstInputPairOffset, inputTile);
        accumulateRowTile<kOutputsPerTile>(upper, inputTile, weightA);

        // 输入第 1 行同时贡献 upper 和 lower；此时只保留 weightA/B。
        loadWeightRow(
            packedWeight, 1, channelPair, channelPairs, weightB
        );
        loadInteriorInputRow(
            input + firstInputPairOffset + rowPairStride,
            inputTile
        );
        accumulateRowTile<kOutputsPerTile>(upper, inputTile, weightB);
        accumulateRowTile<kOutputsPerTile>(lower, inputTile, weightA);

        // weightA 的第 0 行已经用完，直接复用为第 2 行。
        loadWeightRow(
            packedWeight, 2, channelPair, channelPairs, weightA
        );
        loadInteriorInputRow(
            input + firstInputPairOffset + 2 * rowPairStride,
            inputTile
        );
        accumulateRowTile<kOutputsPerTile>(upper, inputTile, weightA);
        accumulateRowTile<kOutputsPerTile>(lower, inputTile, weightB);

        storeOutputRow<kOutputsPerTile>(output + outputPairOffset, upper);

        // upper 已经写回，最后一行 input 只继续完成 lower。
        loadInteriorInputRow(
            input + firstInputPairOffset + 3 * rowPairStride,
            inputTile
        );
        accumulateRowTile<kOutputsPerTile>(lower, inputTile, weightA);
        storeOutputRow<kOutputsPerTile>(
            output + outputPairOffset + outputRowStride,
            lower
        );
        return;
    }

    processBoundaryChunk<kBoundaryOutputsPerChunk>(
        input,
        packedWeight,
        output + outputPairOffset,
        tileRow,
        tileColumn,
        channelPair,
        bias
    );
    if constexpr (kOutputsPerTile > kBoundaryOutputsPerChunk) {
        processBoundaryChunk<kBoundaryOutputsPerChunk>(
            input,
            packedWeight,
            output + outputPairOffset +
                kBoundaryOutputsPerChunk * kFixedChannelPairs,
            tileRow,
            tileColumn + kBoundaryOutputsPerChunk,
            channelPair,
            bias
        );
    }
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
