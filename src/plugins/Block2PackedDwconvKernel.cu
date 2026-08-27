#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 8;
constexpr int32_t kInputsPerTile = kOutputsPerTile + 2;
constexpr int32_t kBoundaryOutputsPerChunk = 4;
constexpr int32_t kOutputRowsPerTile = 2;
constexpr int32_t kSpatialExtent = 44;
constexpr int32_t kTailOutputs = kSpatialExtent % kOutputsPerTile;
constexpr int32_t kChannelPairs = 256; // 512 channels / 2
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

// 与 Block1 保持相同的 PyTorch tanh GELU 近似和常量折叠。
// 卷积使用 half2 FP16 累加，GELU 前只转换一次为 float2；保留 FP32
// 多项式与最终 FMA，并把两路 tanh 合并为一条 f16x2 指令。
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
    const __half2 (&accumulator)[OutputCount]
) {
    static_assert(OutputCount > 0 && OutputCount <= kOutputsPerTile);
#pragma unroll
    for (int32_t outputColumn = 0; outputColumn < OutputCount; ++outputColumn) {
        outputRowStart[outputColumn * kChannelPairs] =
            convertOutput(accumulator[outputColumn]);
    }
}

// 同一输出行的 8 个相邻位置只加载 10 个输入 half2，并复用同一组 3-tap
// 权重。循环边界均为编译期常量，nvcc 会将其完全展开。
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
    if (tileColumn + OutputCount == kSpatialExtent) {
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

#pragma unroll
    for (int32_t inputColumn = 0;
         inputColumn < OutputCount + 2;
         ++inputColumn) {
        inputTile[inputColumn] =
            firstInput[inputColumn * kChannelPairs];
    }
}

// 边界 CTA 与内部 CTA 使用相同的四输入行滑动调度。padding 只发生在
// loadBoundaryInputRow；中间两行 input 及三行 weight 不再为 lower 重载。
template <int32_t OutputCount>
__device__ __forceinline__ void processBoundaryTile(
    const __half2* input,
    const __half2* packedWeight,
    __half2* outputRowStart,
    int32_t tileRow,
    int32_t tileColumn,
    int32_t channelPair,
    const __half2& bias
) {
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
    accumulateRowTile<OutputCount>(lower, inputTile, weightA);

    loadWeightRow(packedWeight, 2, channelPair, weightA);
    loadBoundaryInputRow<OutputCount>(
        input, tileRow + 1, tileColumn, channelPair, inputTile
    );
    accumulateRowTile<OutputCount>(upper, inputTile, weightA);
    accumulateRowTile<OutputCount>(lower, inputTile, weightB);

    storeOutputRow<OutputCount>(outputRowStart, upper);

    if (tileRow + 2 < kSpatialExtent) {
        loadBoundaryInputRow<OutputCount>(
            input, tileRow + 2, tileColumn, channelPair, inputTile
        );
        accumulateRowTile<OutputCount>(lower, inputTile, weightA);
    }
    constexpr int64_t outputRowStride =
        static_cast<int64_t>(kSpatialExtent) * kChannelPairs;
    storeOutputRow<OutputCount>(outputRowStart + outputRowStride, lower);
}

// 固定 44x44x512 的 8x2 tile。相邻输出行共享中间两行输入；channel
// 方向拆成 2 个 128-thread CTA，覆盖全部 256 个 half2 channel pair，
// 计算按输入行展开，任一时刻最多保留两行 weight 和一行 input；upper
// 完成后立即转换并写回。尾 tile 使用编译期固定的四列写回，避免越界。
__global__ void block2PackedDwconvHalf2Tile2DKernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ packedWeight,
    const __half2* __restrict__ packedBias,
    __half2* __restrict__ output
) {
    const int32_t channelTile = static_cast<int32_t>(blockIdx.z);
    const int32_t channelPair =
        channelTile * kChannelPairsPerBlock +
        static_cast<int32_t>(threadIdx.x);
    const int32_t tileColumn =
        static_cast<int32_t>(blockIdx.x) * kOutputsPerTile;
    const int32_t tileRow =
        static_cast<int32_t>(blockIdx.y) * kOutputRowsPerTile;
    const bool isTailTile =
        tileColumn + kOutputsPerTile > kSpatialExtent;

    const __half2 bias = packedBias[channelPair];

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

        storeOutputRow<kOutputsPerTile>(
            output + outputPairOffset,
            upper
        );

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
            bias
        );
        return;
    }

    processBoundaryTile<kBoundaryOutputsPerChunk>(
        input,
        packedWeight,
        output + outputPairOffset,
        tileRow,
        tileColumn,
        channelPair,
        bias
    );
    processBoundaryTile<kBoundaryOutputsPerChunk>(
        input,
        packedWeight,
        output + outputPairOffset +
            kBoundaryOutputsPerChunk * kChannelPairs,
        tileRow,
        tileColumn + kBoundaryOutputsPerChunk,
        channelPair,
        bias
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
        kSpatialExtent / kOutputRowsPerTile,
        kChannelTiles
    );
    block2PackedDwconvHalf2Tile2DKernel
        <<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(input),
            static_cast<const __half2*>(packedWeight),
            static_cast<const __half2*>(packedBias),
            static_cast<__half2*>(output)
        );
}

} // namespace

int32_t launchBlock2PackedDwconv(
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
