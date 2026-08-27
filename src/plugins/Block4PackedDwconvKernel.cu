#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 4;
constexpr int32_t kInputsPerTile = kOutputsPerTile + 2;
constexpr int32_t kOutputRowsPerTile = 2;
constexpr int32_t kSpatialExtent = 11;
constexpr int32_t kTailOutputs = kSpatialExtent % kOutputsPerTile;
constexpr int32_t kChannelPairs = 1024; // 2048 channels / 2
constexpr int32_t kChannelPairsPerBlock = 128;
constexpr int32_t kChannelTiles =
    kChannelPairs / kChannelPairsPerBlock;
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluScaledCubic =
    0.035677406936883926F; // scale * 0.044715 常量折叠

static_assert(kChannelPairs % kChannelPairsPerBlock == 0);
static_assert(kTailOutputs > 0);

__device__ __forceinline__ void multiplyAddHalf2(
    __half2& accumulator,
    const __half2& input,
    const __half2& weight
) {
    accumulator = __hfma2(input, weight, accumulator);
}

// 卷积使用 half2 FP16 累加；GELU 前只转换一次为 float2，后续多项式
// 和最终 FMA 仍保持 FP32。
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

__device__ __forceinline__ void loadWeightRow(
    const __half2* packedWeight,
    int32_t kernelRow,
    int32_t channelPair,
    __half2 (&weight)[3]
) {
    const int32_t rowOffset =
        kernelRow * 3 * kChannelPairs + channelPair;
#pragma unroll
    for (int32_t kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {
        weight[kernelColumn] =
            packedWeight[rowOffset + kernelColumn * kChannelPairs];
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
        inputTile[inputColumn] =
            inputRowStart[inputColumn * kChannelPairs];
    }
}

template <int32_t OutputCount>
__device__ __forceinline__ void storeOutputRow(
    __half2* outputRowStart,
    const __half2 (&accumulator)[kOutputsPerTile]
) {
    static_assert(OutputCount > 0 && OutputCount <= kOutputsPerTile);
#pragma unroll
    for (int32_t outputColumn = 0; outputColumn < OutputCount; ++outputColumn) {
        outputRowStart[outputColumn * kChannelPairs] =
            convertOutput(accumulator[outputColumn]);
    }
}

// 4 个相邻输出复用六个输入与同一组 3-tap 权重。循环长度全部
// 固定，nvcc 展开后与手写累加序列等价，同时保持边界和内部路径一致。
template <int32_t OutputCount>
__device__ __forceinline__ void accumulateRowTile(
    __half2 (&accumulator)[kOutputsPerTile],
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

template <int32_t OutputCount>
__device__ __forceinline__ void loadBoundaryInputRow(
    const __half2* input,
    int32_t inputRow,
    int32_t tileColumn,
    int32_t channelPair,
    __half2 (&inputTile)[OutputCount + 2]
) {
    static_assert(OutputCount > 0 && OutputCount <= kOutputsPerTile);
    const int64_t rowOffset =
        static_cast<int64_t>(inputRow) * kSpatialExtent * kChannelPairs;
    const __half2* inputRowStart = input + rowOffset + channelPair;
    if constexpr (OutputCount < kOutputsPerTile) {
        const __half2* firstInput = inputRowStart +
            static_cast<int64_t>(tileColumn - 1) * kChannelPairs;
#pragma unroll
        for (int32_t inputColumn = 0;
             inputColumn < OutputCount + 1;
             ++inputColumn) {
            inputTile[inputColumn] =
                firstInput[inputColumn * kChannelPairs];
        }
        inputTile[OutputCount + 1] = __float2half2_rn(0.0F);
        return;
    }

    if (tileColumn == 0) {
        inputTile[0] = __float2half2_rn(0.0F);
#pragma unroll
        for (int32_t inputColumn = 1;
             inputColumn < OutputCount + 2;
             ++inputColumn) {
            inputTile[inputColumn] =
                inputRowStart[(inputColumn - 1) * kChannelPairs];
        }
        return;
    }

    const __half2* firstInput = inputRowStart +
        static_cast<int64_t>(tileColumn - 1) * kChannelPairs;
#pragma unroll
    for (int32_t inputColumn = 0;
         inputColumn < OutputCount + 2;
         ++inputColumn) {
        inputTile[inputColumn] =
            firstInput[inputColumn * kChannelPairs];
    }
}

// 11x11 的边界 CTA 占多数，因此边界路径也使用四输入行滑动调度。
// hasLowerOutput 只在最后一排 CTA 为 false，判断对整个 CTA 一致。
template <int32_t OutputCount>
__device__ __forceinline__ void processBoundaryTile(
    const __half2* input,
    const __half2* packedWeight,
    __half2* outputRowStart,
    int32_t tileRow,
    int32_t tileColumn,
    int32_t channelPair,
    bool hasLowerOutput,
    __half2 (&upper)[kOutputsPerTile],
    __half2 (&lower)[kOutputsPerTile]
) {
    __half2 weightA[3];
    __half2 weightB[3];
    __half2 inputTile[OutputCount + 2];

    loadWeightRow(packedWeight, 0, channelPair, weightA);
    if (tileRow > 0) {
        loadBoundaryInputRow<OutputCount>(
            input, tileRow - 1, tileColumn, channelPair, inputTile
        );
        accumulateRowTile<OutputCount>(upper, inputTile, weightA);
    }

    loadWeightRow(packedWeight, 1, channelPair, weightB);
    loadBoundaryInputRow<OutputCount>(
        input, tileRow, tileColumn, channelPair, inputTile
    );
    accumulateRowTile<OutputCount>(upper, inputTile, weightB);
    if (hasLowerOutput) {
        accumulateRowTile<OutputCount>(lower, inputTile, weightA);
    }

    if (hasLowerOutput) {
        loadWeightRow(packedWeight, 2, channelPair, weightA);
        loadBoundaryInputRow<OutputCount>(
            input, tileRow + 1, tileColumn, channelPair, inputTile
        );
        accumulateRowTile<OutputCount>(upper, inputTile, weightA);
        accumulateRowTile<OutputCount>(lower, inputTile, weightB);
    }

    storeOutputRow<OutputCount>(outputRowStart, upper);

    if (hasLowerOutput) {
        loadBoundaryInputRow<OutputCount>(
            input, tileRow + 2, tileColumn, channelPair, inputTile
        );
        accumulateRowTile<OutputCount>(lower, inputTile, weightA);
        constexpr int64_t outputRowStride =
            static_cast<int64_t>(kSpatialExtent) * kChannelPairs;
        storeOutputRow<OutputCount>(outputRowStart + outputRowStride, lower);
    }
}

// 固定 11x11x2048 的 4x2 token-major tile。channel 方向拆成八个
// 128-thread CTA；最后一个水平 tile 只有 3 个有效输出，最后一个垂直
// tile 只有上行有效。计算按输入行展开，最多保留两行 weight 和一行
// input；upper 完成后立即写回。
__global__ void block4PackedDwconvHalf2Tile2DKernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ packedWeight,
    const __half2* __restrict__ packedBias,
    __half2* __restrict__ output
) {
    const int32_t channelPair =
        static_cast<int32_t>(blockIdx.z) * kChannelPairsPerBlock +
        static_cast<int32_t>(threadIdx.x);
    const int32_t tileColumn =
        static_cast<int32_t>(blockIdx.x) * kOutputsPerTile;
    const int32_t tileRow =
        static_cast<int32_t>(blockIdx.y) * kOutputRowsPerTile;
    const bool isTailTile =
        tileColumn + kOutputsPerTile > kSpatialExtent;
    const bool hasLowerOutput = tileRow + 1 < kSpatialExtent;

    const __half2 bias = packedBias[channelPair];
    __half2 upper[kOutputsPerTile] = {bias, bias, bias, bias};
    __half2 lower[kOutputsPerTile] = {bias, bias, bias, bias};

    const int64_t outputPairOffset =
        (static_cast<int64_t>(tileRow) * kSpatialExtent + tileColumn) *
            kChannelPairs +
        channelPair;
    constexpr int64_t outputRowStride =
        static_cast<int64_t>(kSpatialExtent) * kChannelPairs;

    const bool isInterior =
        tileRow > 0 && tileRow + kOutputRowsPerTile < kSpatialExtent &&
        tileColumn > 0 && tileColumn + kOutputsPerTile < kSpatialExtent;
    if (isInterior) {
        constexpr int64_t rowPairStride =
            static_cast<int64_t>(kSpatialExtent) * kChannelPairs;
        const int64_t firstInputPairOffset =
            static_cast<int64_t>(tileRow - 1) * rowPairStride +
            static_cast<int64_t>(tileColumn - 1) * kChannelPairs +
            channelPair;

        __half2 weightA[3];
        __half2 weightB[3];
        __half2 inputTile[kInputsPerTile];

        loadWeightRow(packedWeight, 0, channelPair, weightA);
        loadInteriorInputRow(input + firstInputPairOffset, inputTile);
        accumulateRowTile<kOutputsPerTile>(upper, inputTile, weightA);

        loadWeightRow(packedWeight, 1, channelPair, weightB);
        loadInteriorInputRow(
            input + firstInputPairOffset + rowPairStride,
            inputTile
        );
        accumulateRowTile<kOutputsPerTile>(upper, inputTile, weightB);
        accumulateRowTile<kOutputsPerTile>(lower, inputTile, weightA);

        loadWeightRow(packedWeight, 2, channelPair, weightA);
        loadInteriorInputRow(
            input + firstInputPairOffset + 2 * rowPairStride,
            inputTile
        );
        accumulateRowTile<kOutputsPerTile>(upper, inputTile, weightA);
        accumulateRowTile<kOutputsPerTile>(lower, inputTile, weightB);

        storeOutputRow<kOutputsPerTile>(output + outputPairOffset, upper);

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

    if (isTailTile) {
        processBoundaryTile<kTailOutputs>(
            input,
            packedWeight,
            output + outputPairOffset,
            tileRow,
            tileColumn,
            channelPair,
            hasLowerOutput,
            upper,
            lower
        );
        return;
    }

    processBoundaryTile<kOutputsPerTile>(
        input,
        packedWeight,
        output + outputPairOffset,
        tileRow,
        tileColumn,
        channelPair,
        hasLowerOutput,
        upper,
        lower
    );
}

void launchKernel(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    cudaStream_t stream
) noexcept {
    const dim3 block(kChannelPairsPerBlock);
    const dim3 grid(
        (kSpatialExtent + kOutputsPerTile - 1) / kOutputsPerTile,
        (kSpatialExtent + kOutputRowsPerTile - 1) / kOutputRowsPerTile,
        kChannelTiles
    );
    block4PackedDwconvHalf2Tile2DKernel
        <<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(input),
            static_cast<const __half2*>(packedWeight),
            static_cast<const __half2*>(packedBias),
            static_cast<__half2*>(output)
        );
}

} // namespace

int32_t launchBlock4PackedDwconv(
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

    launchKernel(input, packedWeight, packedBias, output, stream);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // namespace egcinet::plugins
