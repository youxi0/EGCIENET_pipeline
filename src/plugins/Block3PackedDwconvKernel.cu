#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 4;
constexpr int32_t kOutputRowsPerTile = 2;
constexpr int32_t kSpatialExtent = 22;
constexpr int32_t kChannelPairs = 640; // 1280channels / 2
constexpr int32_t kChannelPairsPerBlock = 128;
constexpr int32_t kChannelTiles =
    kChannelPairs / kChannelPairsPerBlock;
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluScaledCubic =
    0.035677406936883926F; // scale * 0.044715 常量折叠

static_assert(kChannelPairs % kChannelPairsPerBlock == 0);

__device__ __forceinline__ float2 loadHalf2AsFloat2(const __half2* address) {
    return __half22float2(*address);
}

__device__ __forceinline__ void multiplyAddHalf2(
    float2& accumulator,
    const float2& input,
    const float2& weight
) {
    accumulator.x = fmaf(input.x, weight.x, accumulator.x);
    accumulator.y = fmaf(input.y, weight.y, accumulator.y);
}

// 与 Block1 保持相同的 PyTorch tanh GELU 近似和常量折叠。
// 保留 FP32 多项式与最终 FMA，只把两路 tanh 合并为一条 f16x2 指令。
__device__ __forceinline__ __half2 convertOutput(const float2& accumulator) {
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

// 同一输出行的 4 个相邻位置只加载 6 个输入 half2，并复用同一组 3-tap
// 权重。Block3 的 channel 维由 blockIdx.z 分成五个 128-pair CTA。
__device__ __forceinline__ void accumulateRowTile(
    float2& accumulator0,
    float2& accumulator1,
    float2& accumulator2,
    float2& accumulator3,
    const float2& input0,
    const float2& input1,
    const float2& input2,
    const float2& input3,
    const float2& input4,
    const float2& input5,
    const float2& weight0,
    const float2& weight1,
    const float2& weight2
) {
    multiplyAddHalf2(accumulator0, input0, weight0);
    multiplyAddHalf2(accumulator0, input1, weight1);
    multiplyAddHalf2(accumulator0, input2, weight2);

    multiplyAddHalf2(accumulator1, input1, weight0);
    multiplyAddHalf2(accumulator1, input2, weight1);
    multiplyAddHalf2(accumulator1, input3, weight2);

    multiplyAddHalf2(accumulator2, input2, weight0);
    multiplyAddHalf2(accumulator2, input3, weight1);
    multiplyAddHalf2(accumulator2, input4, weight2);

    multiplyAddHalf2(accumulator3, input3, weight0);
    multiplyAddHalf2(accumulator3, input4, weight1);
    multiplyAddHalf2(accumulator3, input5, weight2);
}

__device__ __forceinline__ void loadWeightRow(
    const __half2* packedWeight,
    int32_t kernelRow,
    int32_t channelPair,
    float2& weight0,
    float2& weight1,
    float2& weight2
) {
    const int32_t rowOffset =
        kernelRow * 3 * kChannelPairs + channelPair;
    weight0 = loadHalf2AsFloat2(packedWeight + rowOffset);
    weight1 = loadHalf2AsFloat2(packedWeight + rowOffset + kChannelPairs);
    weight2 =
        loadHalf2AsFloat2(packedWeight + rowOffset + 2 * kChannelPairs);
}

__device__ __forceinline__ void loadInteriorInputRowTile(
    const __half2* inputRowStart,
    float2& input0,
    float2& input1,
    float2& input2,
    float2& input3,
    float2& input4,
    float2& input5
) {
    input0 = loadHalf2AsFloat2(inputRowStart);
    input1 = loadHalf2AsFloat2(inputRowStart + kChannelPairs);
    input2 = loadHalf2AsFloat2(inputRowStart + 2 * kChannelPairs);
    input3 = loadHalf2AsFloat2(inputRowStart + 3 * kChannelPairs);
    input4 = loadHalf2AsFloat2(inputRowStart + 4 * kChannelPairs);
    input5 = loadHalf2AsFloat2(inputRowStart + 5 * kChannelPairs);
}

__device__ __forceinline__ float2 loadBoundaryInput(
    const __half2* inputRowStart,
    int32_t column,
    int32_t channelPair
) {
    if (column < 0 || column >= kSpatialExtent) {
        return make_float2(0.0F, 0.0F);
    }
    return loadHalf2AsFloat2(
        inputRowStart +
        static_cast<int64_t>(column) * kChannelPairs + channelPair
    );
}

// 22 不能被横向 tile=4 整除，所以最后一个 tile 只有两个有效输出。
// 边界路径对六个输入列做完整检查；所有判断在 CTA 内一致，不产生 warp
// divergence。输出写回处再跳过最后两个无效位置。
__device__ __forceinline__ void accumulateBoundaryRow(
    const __half2* input,
    const __half2* packedWeight,
    int32_t outputRow,
    int32_t tileColumn,
    int32_t channelPair,
    float2& accumulator0,
    float2& accumulator1,
    float2& accumulator2,
    float2& accumulator3
) {
#pragma unroll
    for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
        const int32_t inputRow = outputRow + kernelRow - 1;
        if (inputRow < 0 || inputRow >= kSpatialExtent) {
            continue;
        }

        const int64_t rowOffset =
            static_cast<int64_t>(inputRow) * kSpatialExtent * kChannelPairs;
        const __half2* inputRowStart = input + rowOffset;

        float2 weight0;
        float2 weight1;
        float2 weight2;
        loadWeightRow(
            packedWeight,
            kernelRow,
            channelPair,
            weight0,
            weight1,
            weight2
        );

        const int32_t firstInputColumn = tileColumn - 1;
        const float2 input0 = loadBoundaryInput(
            inputRowStart, firstInputColumn, channelPair
        );
        const float2 input1 = loadBoundaryInput(
            inputRowStart, firstInputColumn + 1, channelPair
        );
        const float2 input2 = loadBoundaryInput(
            inputRowStart, firstInputColumn + 2, channelPair
        );
        const float2 input3 = loadBoundaryInput(
            inputRowStart, firstInputColumn + 3, channelPair
        );
        const float2 input4 = loadBoundaryInput(
            inputRowStart, firstInputColumn + 4, channelPair
        );
        const float2 input5 = loadBoundaryInput(
            inputRowStart, firstInputColumn + 5, channelPair
        );

        accumulateRowTile(
            accumulator0,
            accumulator1,
            accumulator2,
            accumulator3,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            weight0,
            weight1,
            weight2
        );
    }
}

// 固定 22x22x1280 的 4x2 tile。相邻输出行共享中间两行输入；channel
// 方向拆成 5 个 128-thread CTA，避免旧通用路径的 640-thread 大 block，
// 同时保持与 Block1 相同的 half2/FP32 累加方式。
// __launch_bounds__(一个 block 最大线程数,一个 SM 最少希望同时运行几个 block)
__global__ __launch_bounds__(128, 10) void block3PackedDwconvHalf2Tile2DKernel(
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

    const float2 bias = loadHalf2AsFloat2(packedBias + channelPair);
    float2 upper0 = bias;
    float2 upper1 = bias;
    float2 upper2 = bias;
    float2 upper3 = bias;
    float2 lower0 = bias;
    float2 lower1 = bias;
    float2 lower2 = bias;
    float2 lower3 = bias;

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

        float2 weight00;
        float2 weight01;
        float2 weight02;
        loadWeightRow(
            packedWeight, 0, channelPair, weight00, weight01, weight02
        );
        {
            float2 input0;
            float2 input1;
            float2 input2;
            float2 input3;
            float2 input4;
            float2 input5;
            loadInteriorInputRowTile(
                input + firstInputPairOffset,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5
            );
            accumulateRowTile(
                upper0,
                upper1,
                upper2,
                upper3,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5,
                weight00,
                weight01,
                weight02
            );
        }

        float2 weight10;
        float2 weight11;
        float2 weight12;
        loadWeightRow(
            packedWeight, 1, channelPair, weight10, weight11, weight12
        );
        {
            float2 input0;
            float2 input1;
            float2 input2;
            float2 input3;
            float2 input4;
            float2 input5;
            loadInteriorInputRowTile(
                input + firstInputPairOffset + rowPairStride,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5
            );
            accumulateRowTile(
                upper0,
                upper1,
                upper2,
                upper3,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5,
                weight10,
                weight11,
                weight12
            );
            accumulateRowTile(
                lower0,
                lower1,
                lower2,
                lower3,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5,
                weight00,
                weight01,
                weight02
            );
        }

        float2 weight20;
        float2 weight21;
        float2 weight22;
        loadWeightRow(
            packedWeight, 2, channelPair, weight20, weight21, weight22
        );
        {
            float2 input0;
            float2 input1;
            float2 input2;
            float2 input3;
            float2 input4;
            float2 input5;
            loadInteriorInputRowTile(
                input + firstInputPairOffset + 2 * rowPairStride,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5
            );
            accumulateRowTile(
                upper0,
                upper1,
                upper2,
                upper3,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5,
                weight20,
                weight21,
                weight22
            );
            accumulateRowTile(
                lower0,
                lower1,
                lower2,
                lower3,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5,
                weight10,
                weight11,
                weight12
            );
        }

        {
            float2 input0;
            float2 input1;
            float2 input2;
            float2 input3;
            float2 input4;
            float2 input5;
            loadInteriorInputRowTile(
                input + firstInputPairOffset + 3 * rowPairStride,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5
            );
            accumulateRowTile(
                lower0,
                lower1,
                lower2,
                lower3,
                input0,
                input1,
                input2,
                input3,
                input4,
                input5,
                weight20,
                weight21,
                weight22
            );
        }
    } else {
        accumulateBoundaryRow(
            input,
            packedWeight,
            tileRow,
            tileColumn,
            channelPair,
            upper0,
            upper1,
            upper2,
            upper3
        );
        accumulateBoundaryRow(
            input,
            packedWeight,
            tileRow + 1,
            tileColumn,
            channelPair,
            lower0,
            lower1,
            lower2,
            lower3
        );
    }

    const int64_t outputPairOffset =
        (static_cast<int64_t>(tileRow) * kSpatialExtent +
         tileColumn) *
            kChannelPairs +
        channelPair;
    constexpr int64_t outputRowStride =
        static_cast<int64_t>(kSpatialExtent) * kChannelPairs;

    output[outputPairOffset] = convertOutput(upper0);
    output[outputPairOffset + outputRowStride] =
        convertOutput(lower0);
    if (tileColumn + 1 < kSpatialExtent) {
        output[outputPairOffset + kChannelPairs] =
            convertOutput(upper1);
        output[outputPairOffset + outputRowStride + kChannelPairs] =
            convertOutput(lower1);
    }
    if (tileColumn + 2 < kSpatialExtent) {
        output[outputPairOffset + 2 * kChannelPairs] =
            convertOutput(upper2);
        output[outputPairOffset + outputRowStride + 2 * kChannelPairs] =
            convertOutput(lower2);
    }
    if (tileColumn + 3 < kSpatialExtent) {
        output[outputPairOffset + 3 * kChannelPairs] =
            convertOutput(upper3);
        output[outputPairOffset + outputRowStride + 3 * kChannelPairs] =
            convertOutput(lower3);
    }
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
    block3PackedDwconvHalf2Tile2DKernel
        <<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(input),
            static_cast<const __half2*>(packedWeight),
            static_cast<const __half2*>(packedBias),
            static_cast<__half2*>(output)
        );
}

} // namespace

int32_t launchBlock3PackedDwconv(
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
