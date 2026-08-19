#include "plugins/Block1PackedDwconvKernel.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 4;
constexpr int32_t kOutputRowsPerTile = 2;
constexpr int32_t kFixedSpatialExtent = 88;
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluScaledCubic = 0.035677406936883926F; // scale * 0.044715 常量折叠

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

// PyTorch GELU 的标准 tanh 近似。__tanhf 直接使用 GPU 快速近似指令；与
// QuickGELU 相比误差更小，与精确 erff 相比则避免昂贵的 device function。
__device__ __forceinline__ float fastGelu(float value) {
    const float xSquared = value * value;
    const float tanhInput =
        value * fmaf(kGeluScaledCubic, xSquared, kGeluScale);
    const float tanhValue = __tanhf(tanhInput);

    const float halfValue = 0.5F * value;
    return fmaf(halfValue, tanhValue, halfValue);
}

template <bool kFuseGelu>
__device__ __forceinline__ __half2 convertOutput(const float2& accumulator) {
    if constexpr (kFuseGelu) {
        return __floats2half2_rn(
            fastGelu(accumulator.x),
            fastGelu(accumulator.y)
        );
    }
    return __floats2half2_rn(accumulator.x, accumulator.y);
}

// 同一行的 4 个相邻输出总共只需要 6 个不同的输入位置。每个 3-tap
// 权重加载、转换一次后，分别复用到 4 个输出上。
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
    int32_t channelPairs,
    float2& weight0,
    float2& weight1,
    float2& weight2
) {
    const int32_t rowOffset = kernelRow * 3 * channelPairs + channelPair;
    weight0 = loadHalf2AsFloat2(packedWeight + rowOffset);
    weight1 = loadHalf2AsFloat2(packedWeight + rowOffset + channelPairs);
    weight2 = loadHalf2AsFloat2(packedWeight + rowOffset + 2 * channelPairs);
}

__device__ __forceinline__ void loadInteriorInputRowTile(
    const __half2* inputRowStart,
    int32_t channelPairs,
    float2& input0,
    float2& input1,
    float2& input2,
    float2& input3,
    float2& input4,
    float2& input5
) {
    input0 = loadHalf2AsFloat2(inputRowStart);
    input1 = loadHalf2AsFloat2(inputRowStart + channelPairs);
    input2 = loadHalf2AsFloat2(inputRowStart + 2 * channelPairs);
    input3 = loadHalf2AsFloat2(inputRowStart + 3 * channelPairs);
    input4 = loadHalf2AsFloat2(inputRowStart + 4 * channelPairs);
    input5 = loadHalf2AsFloat2(inputRowStart + 5 * channelPairs);
}

// 固定 88x88 的边界 tile 数量很少，使用完整 padding 逻辑保证首尾行列
// 正确；内部 tile 则由下面的 4x2 专用 kernel 走无分支快速路径。
__device__ __forceinline__ void accumulateFixedBoundaryRow(
    const __half2* input,
    const __half2* packedWeight,
    int32_t batchIndex,
    int32_t outputRow,
    int32_t tileColumn,
    int32_t channelPair,
    int32_t channelPairs,
    float2& accumulator0,
    float2& accumulator1,
    float2& accumulator2,
    float2& accumulator3
) {
    const float2 zero = make_float2(0.0F, 0.0F);
#pragma unroll
    for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
        const int32_t inputRow = outputRow + kernelRow - 1;
        if (inputRow < 0 || inputRow >= kFixedSpatialExtent) {
            continue;
        }

        const int64_t inputRowPairOffset =
            (static_cast<int64_t>(batchIndex) * kFixedSpatialExtent + inputRow) *
                kFixedSpatialExtent * channelPairs +
            channelPair;
        const __half2* inputRowStart = input + inputRowPairOffset;

        float2 weight0;
        float2 weight1;
        float2 weight2;
        loadWeightRow(
            packedWeight,
            kernelRow,
            channelPair,
            channelPairs,
            weight0,
            weight1,
            weight2
        );

        float2 input0 = zero;
        if (tileColumn > 0) {
            input0 = loadHalf2AsFloat2(
                inputRowStart + (tileColumn - 1) * channelPairs
            );
        }
        const float2 input1 = loadHalf2AsFloat2(
            inputRowStart + tileColumn * channelPairs
        );
        const float2 input2 = loadHalf2AsFloat2(
            inputRowStart + (tileColumn + 1) * channelPairs
        );
        const float2 input3 = loadHalf2AsFloat2(
            inputRowStart + (tileColumn + 2) * channelPairs
        );
        const float2 input4 = loadHalf2AsFloat2(
            inputRowStart + (tileColumn + 3) * channelPairs
        );
        float2 input5 = zero;
        if (tileColumn + kOutputsPerTile < kFixedSpatialExtent) {
            input5 = loadHalf2AsFloat2(
                inputRowStart + (tileColumn + kOutputsPerTile) * channelPairs
            );
        }

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

// 固定 88x88 热路径采用 4x2 二维 tile。相邻两个输出行共同需要 4 行
// 输入，其中间两行只加载一次；每个权重行只跨相邻两个输入行存活，控制
// 寄存器生命周期，避免同时常驻全部 9 个 half2 权重。
template <bool kFuseGelu>
__global__ __launch_bounds__(128, 10) void block1PackedDwconvHalf2Tile2DKernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ packedWeight,
    const __half2* __restrict__ packedBias,
    __half2* __restrict__ output,
    int32_t channelPairs
) {
    const int32_t channelPair = static_cast<int32_t>(threadIdx.x);
    const int32_t tileColumn =
        static_cast<int32_t>(blockIdx.x) * kOutputsPerTile;
    const int32_t tileRow =
        static_cast<int32_t>(blockIdx.y) * kOutputRowsPerTile;
    const int32_t batchIndex = static_cast<int32_t>(blockIdx.z);

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
        tileRow > 0 && tileRow + kOutputRowsPerTile < kFixedSpatialExtent &&
        tileColumn > 0 &&
        tileColumn + kOutputsPerTile < kFixedSpatialExtent;
    if (isInterior) {
        const int64_t rowPairStride =
            static_cast<int64_t>(kFixedSpatialExtent) * channelPairs;
        const int64_t firstInputPairOffset =
            (static_cast<int64_t>(batchIndex) * kFixedSpatialExtent +
             tileRow - 1) *
                rowPairStride +
            static_cast<int64_t>(tileColumn - 1) * channelPairs +
            channelPair;

        float2 weight00;
        float2 weight01;
        float2 weight02;
        loadWeightRow(
            packedWeight,
            0,
            channelPair,
            channelPairs,
            weight00,
            weight01,
            weight02
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
                channelPairs,
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
            packedWeight,
            1,
            channelPair,
            channelPairs,
            weight10,
            weight11,
            weight12
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
                channelPairs,
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
            packedWeight,
            2,
            channelPair,
            channelPairs,
            weight20,
            weight21,
            weight22
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
                channelPairs,
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
                channelPairs,
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
        accumulateFixedBoundaryRow(
            input,
            packedWeight,
            batchIndex,
            tileRow,
            tileColumn,
            channelPair,
            channelPairs,
            upper0,
            upper1,
            upper2,
            upper3
        );
        accumulateFixedBoundaryRow(
            input,
            packedWeight,
            batchIndex,
            tileRow + 1,
            tileColumn,
            channelPair,
            channelPairs,
            lower0,
            lower1,
            lower2,
            lower3
        );
    }

    const int64_t outputPairOffset =
        ((static_cast<int64_t>(batchIndex) * kFixedSpatialExtent + tileRow) *
             kFixedSpatialExtent +
         tileColumn) *
            channelPairs +
        channelPair;
    const int64_t outputRowStride =
        static_cast<int64_t>(kFixedSpatialExtent) * channelPairs;

    output[outputPairOffset] = convertOutput<kFuseGelu>(upper0);
    output[outputPairOffset + channelPairs] = convertOutput<kFuseGelu>(upper1);
    output[outputPairOffset + 2 * channelPairs] =
        convertOutput<kFuseGelu>(upper2);
    output[outputPairOffset + 3 * channelPairs] =
        convertOutput<kFuseGelu>(upper3);
    output[outputPairOffset + outputRowStride] =
        convertOutput<kFuseGelu>(lower0);
    output[outputPairOffset + outputRowStride + channelPairs] =
        convertOutput<kFuseGelu>(lower1);
    output[outputPairOffset + outputRowStride + 2 * channelPairs] =
        convertOutput<kFuseGelu>(lower2);
    output[outputPairOffset + outputRowStride + 3 * channelPairs] =
        convertOutput<kFuseGelu>(lower3);
}

// 一个 CTA 计算同一 batch、同一行的连续 4 个空间位置；一个线程负责
// 一个 channel-pair。blockIdx 直接表达 column-tile/row/batch，因此热路径
// 不再把线性线程编号除法、取模成 token、row 和 column。
template <bool kUseFixed88Shape, bool kFuseGelu>
__global__ void block1PackedDwconvHalf2Kernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ packedWeight,
    const __half2* __restrict__ packedBias,
    __half2* __restrict__ output,
    int32_t height,
    int32_t width,
    int32_t channelPairs
) {
    const int32_t kernelHeight =
        kUseFixed88Shape ? kFixedSpatialExtent : height;
    const int32_t kernelWidth =
        kUseFixed88Shape ? kFixedSpatialExtent : width;
    const int32_t channelPair = static_cast<int32_t>(threadIdx.x);
    const int32_t tileColumn =
        static_cast<int32_t>(blockIdx.x) * kOutputsPerTile;
    const int32_t row = static_cast<int32_t>(blockIdx.y);
    const int32_t batchIndex = static_cast<int32_t>(blockIdx.z);

    const float2 bias = loadHalf2AsFloat2(packedBias + channelPair);
    float2 accumulator0 = bias;
    float2 accumulator1 = bias;
    float2 accumulator2 = bias;
    float2 accumulator3 = bias;

    const int64_t batchRowPairOffset =
        (static_cast<int64_t>(batchIndex) * kernelHeight + row) * kernelWidth *
        channelPairs;
    const int64_t tileOutputPairOffset =
        batchRowPairOffset +
        static_cast<int64_t>(tileColumn) * channelPairs + channelPair;

    // 绝大多数 CTA 都在图像内部，走无边界判断的快速路径。每一行只加载
    // 6 个不同输入和 3 个权重，完成 4 个输出的 12 次 half2 卷积累加。
    const bool isInterior =
        row > 0 && row + 1 < kernelHeight && tileColumn > 0 &&
        tileColumn + kOutputsPerTile < kernelWidth;
    if (isInterior) {
#pragma unroll
        for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
            const int32_t inputRow = row + kernelRow - 1;
            const int64_t inputRowPairOffset =
                (static_cast<int64_t>(batchIndex) * kernelHeight + inputRow) *
                    kernelWidth * channelPairs +
                static_cast<int64_t>(tileColumn - 1) * channelPairs +
                channelPair;
            const __half2* inputRowStart = input + inputRowPairOffset;

            const int32_t weightRowPairOffset =
                kernelRow * 3 * channelPairs + channelPair;
            const float2 weight0 =
                loadHalf2AsFloat2(packedWeight + weightRowPairOffset);
            const float2 weight1 = loadHalf2AsFloat2(
                packedWeight + weightRowPairOffset + channelPairs
            );
            const float2 weight2 = loadHalf2AsFloat2(
                packedWeight + weightRowPairOffset + 2 * channelPairs
            );

            const float2 input0 = loadHalf2AsFloat2(inputRowStart);
            const float2 input1 =
                loadHalf2AsFloat2(inputRowStart + channelPairs);
            const float2 input2 =
                loadHalf2AsFloat2(inputRowStart + 2 * channelPairs);
            const float2 input3 =
                loadHalf2AsFloat2(inputRowStart + 3 * channelPairs);
            const float2 input4 =
                loadHalf2AsFloat2(inputRowStart + 4 * channelPairs);
            const float2 input5 =
                loadHalf2AsFloat2(inputRowStart + 5 * channelPairs);

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
    } else {
        // 边界 CTA 仍使用相同的横向复用方式，只把 padding 区域显式置零。
        const float2 zero = make_float2(0.0F, 0.0F);
#pragma unroll
        for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
            const int32_t inputRow = row + kernelRow - 1;
            if (inputRow < 0 || inputRow >= kernelHeight) {
                continue;
            }

            const int64_t inputRowPairOffset =
                (static_cast<int64_t>(batchIndex) * kernelHeight + inputRow) *
                    kernelWidth * channelPairs +
                channelPair;
            const __half2* inputRowStart = input + inputRowPairOffset;

            const int32_t weightRowPairOffset =
                kernelRow * 3 * channelPairs + channelPair;
            const float2 weight0 =
                loadHalf2AsFloat2(packedWeight + weightRowPairOffset);
            const float2 weight1 = loadHalf2AsFloat2(
                packedWeight + weightRowPairOffset + channelPairs
            );
            const float2 weight2 = loadHalf2AsFloat2(
                packedWeight + weightRowPairOffset + 2 * channelPairs
            );

            float2 input0;
            float2 input1;
            float2 input2;
            float2 input3;
            float2 input4;
            float2 input5;
            if constexpr (kUseFixed88Shape) {
                // 88 可以被每个 CTA 的 4 个输出整除，因此中心 4 个输入必定
                // 有效，只需分别处理左右两侧的 halo。
                input0 = zero;
                if (tileColumn > 0) {
                    input0 = loadHalf2AsFloat2(
                        inputRowStart +
                        (tileColumn - 1) * channelPairs
                    );
                }
                input1 = loadHalf2AsFloat2(
                    inputRowStart + tileColumn * channelPairs
                );
                input2 = loadHalf2AsFloat2(
                    inputRowStart + (tileColumn + 1) * channelPairs
                );
                input3 = loadHalf2AsFloat2(
                    inputRowStart + (tileColumn + 2) * channelPairs
                );
                input4 = loadHalf2AsFloat2(
                    inputRowStart + (tileColumn + 3) * channelPairs
                );
                input5 = zero;
                if (tileColumn + kOutputsPerTile < kFixedSpatialExtent) {
                    input5 = loadHalf2AsFloat2(
                        inputRowStart +
                        (tileColumn + kOutputsPerTile) * channelPairs
                    );
                }
            } else {
                // 通用尺寸的最后一个 tile 可能不足 4 个输出，因此保留
                // 所有逐列边界检查。
                const int32_t firstInputColumn = tileColumn - 1;
                input0 = zero;
                input1 = zero;
                input2 = zero;
                input3 = zero;
                input4 = zero;
                input5 = zero;
                if (firstInputColumn >= 0 && firstInputColumn < kernelWidth) {
                    input0 = loadHalf2AsFloat2(
                        inputRowStart + firstInputColumn * channelPairs
                    );
                }
                if (firstInputColumn + 1 >= 0 &&
                    firstInputColumn + 1 < kernelWidth) {
                    input1 = loadHalf2AsFloat2(
                        inputRowStart +
                        (firstInputColumn + 1) * channelPairs
                    );
                }
                if (firstInputColumn + 2 >= 0 &&
                    firstInputColumn + 2 < kernelWidth) {
                    input2 = loadHalf2AsFloat2(
                        inputRowStart +
                        (firstInputColumn + 2) * channelPairs
                    );
                }
                if (firstInputColumn + 3 >= 0 &&
                    firstInputColumn + 3 < kernelWidth) {
                    input3 = loadHalf2AsFloat2(
                        inputRowStart +
                        (firstInputColumn + 3) * channelPairs
                    );
                }
                if (firstInputColumn + 4 >= 0 &&
                    firstInputColumn + 4 < kernelWidth) {
                    input4 = loadHalf2AsFloat2(
                        inputRowStart +
                        (firstInputColumn + 4) * channelPairs
                    );
                }
                if (firstInputColumn + 5 >= 0 &&
                    firstInputColumn + 5 < kernelWidth) {
                    input5 = loadHalf2AsFloat2(
                        inputRowStart +
                        (firstInputColumn + 5) * channelPairs
                    );
                }
            }

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

    // V11 仅写回 DWConv；V12 在同一 kernel 中执行 FastGELU，避免额外的
    // Myelin kernel launch 和中间 FP16 张量往返。
    output[tileOutputPairOffset] = convertOutput<kFuseGelu>(accumulator0);
    if constexpr (kUseFixed88Shape) {
        output[tileOutputPairOffset + channelPairs] =
            convertOutput<kFuseGelu>(accumulator1);
        output[tileOutputPairOffset + 2 * channelPairs] =
            convertOutput<kFuseGelu>(accumulator2);
        output[tileOutputPairOffset + 3 * channelPairs] =
            convertOutput<kFuseGelu>(accumulator3);
    } else {
        if (tileColumn + 1 < kernelWidth) {
            output[tileOutputPairOffset + channelPairs] =
                convertOutput<kFuseGelu>(accumulator1);
        }
        if (tileColumn + 2 < kernelWidth) {
            output[tileOutputPairOffset + 2 * channelPairs] =
                convertOutput<kFuseGelu>(accumulator2);
        }
        if (tileColumn + 3 < kernelWidth) {
            output[tileOutputPairOffset + 3 * channelPairs] =
                convertOutput<kFuseGelu>(accumulator3);
        }
    }
}

template <bool kUseFixed88Shape, bool kFuseGelu>
void launchKernel(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    const dim3& grid,
    const dim3& block,
    int32_t height,
    int32_t width,
    int32_t channelPairs,
    cudaStream_t stream
) noexcept {
    block1PackedDwconvHalf2Kernel<kUseFixed88Shape, kFuseGelu>
        <<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(input),
            static_cast<const __half2*>(packedWeight),
            static_cast<const __half2*>(packedBias),
            static_cast<__half2*>(output),
            height,
            width,
            channelPairs
        );
}

template <bool kFuseGelu>
void launchFixedTile2DKernel(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    const dim3& grid,
    const dim3& block,
    int32_t channelPairs,
    cudaStream_t stream
) noexcept {
    block1PackedDwconvHalf2Tile2DKernel<kFuseGelu>
        <<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(input),
            static_cast<const __half2*>(packedWeight),
            static_cast<const __half2*>(packedBias),
            static_cast<__half2*>(output),
            channelPairs
        );
}

} // 匿名命名空间

int32_t launchBlock1PackedDwconv(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channels,
    bool fuseGelu,
    cudaStream_t stream
) noexcept {
    if (input == nullptr || packedWeight == nullptr || packedBias == nullptr ||
        output == nullptr || batch <= 0 || height <= 0 || width <= 0 ||
        channels <= 0 || (channels & 1) != 0) {
        return -1;
    }

    const int32_t channelPairs = channels / 2;
    if (channelPairs > 1024 || height > 65535 || batch > 65535) {
        return -1;
    }

    const dim3 block(static_cast<uint32_t>(channelPairs));

    const bool useFixed88Shape =
        height == kFixedSpatialExtent && width == kFixedSpatialExtent;
    const bool useFixedTile2D = useFixed88Shape && channelPairs <= 128;
    if (useFixedTile2D) {
        const dim3 fixedGrid(
            static_cast<uint32_t>(
                kFixedSpatialExtent / kOutputsPerTile),
            static_cast<uint32_t>(
                kFixedSpatialExtent / kOutputRowsPerTile),
            static_cast<uint32_t>(batch)
        );
        if (fuseGelu) {
            launchFixedTile2DKernel<true>(
                input,
                packedWeight,
                packedBias,
                output,
                fixedGrid,
                block,
                channelPairs,
                stream
            );
        } else {
            launchFixedTile2DKernel<false>(
                input,
                packedWeight,
                packedBias,
                output,
                fixedGrid,
                block,
                channelPairs,
                stream
            );
        }
    } else {
        const dim3 grid(
            static_cast<uint32_t>(
                (width + kOutputsPerTile - 1) / kOutputsPerTile),
            static_cast<uint32_t>(height),
            static_cast<uint32_t>(batch)
        );
        if (useFixed88Shape && fuseGelu) {
            launchKernel<true, true>(
                input,
                packedWeight,
                packedBias,
                output,
                grid,
                block,
                height,
                width,
                channelPairs,
                stream
            );
        } else if (useFixed88Shape) {
            launchKernel<true, false>(
                input,
                packedWeight,
                packedBias,
                output,
                grid,
                block,
                height,
                width,
                channelPairs,
                stream
            );
        } else if (fuseGelu) {
            launchKernel<false, true>(
                input,
                packedWeight,
                packedBias,
                output,
                grid,
                block,
                height,
                width,
                channelPairs,
                stream
            );
        } else {
            launchKernel<false, false>(
                input,
                packedWeight,
                packedBias,
                output,
                grid,
                block,
                height,
                width,
                channelPairs,
                stream
            );
        }
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // 命名空间 egcinet::plugins
