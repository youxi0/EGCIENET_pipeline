#include "plugins/Block1PackedDwconvKernel.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputsPerTile = 4;

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

// 一个 CTA 计算同一 batch、同一行的连续 4 个空间位置；一个线程负责
// 一个 channel-pair。blockIdx 直接表达 column-tile/row/batch，因此热路径
// 不再把线性线程编号除法、取模成 token、row 和 column。
__global__ void block1PackedDwconvHalf2Kernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ packedWeight,
    const __half2* __restrict__ packedBias,
    __half2* __restrict__ output,
    int32_t height,
    int32_t width,
    int32_t channelPairs
) {
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
        (static_cast<int64_t>(batchIndex) * height + row) * width *
        channelPairs;
    const int64_t tileOutputPairOffset =
        batchRowPairOffset +
        static_cast<int64_t>(tileColumn) * channelPairs + channelPair;

    // 绝大多数 CTA 都在图像内部，走无边界判断的快速路径。每一行只加载
    // 6 个不同输入和 3 个权重，完成 4 个输出的 12 次 half2 卷积累加。
    const bool isInterior =
        row > 0 && row + 1 < height && tileColumn > 0 &&
        tileColumn + kOutputsPerTile < width;
    if (isInterior) {
#pragma unroll
        for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
            const int32_t inputRow = row + kernelRow - 1;
            const int64_t inputRowPairOffset =
                (static_cast<int64_t>(batchIndex) * height + inputRow) * width *
                    channelPairs +
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
            if (inputRow < 0 || inputRow >= height) {
                continue;
            }

            const int64_t inputRowPairOffset =
                (static_cast<int64_t>(batchIndex) * height + inputRow) * width *
                    channelPairs +
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

            const int32_t firstInputColumn = tileColumn - 1;
            float2 input0 = zero;
            float2 input1 = zero;
            float2 input2 = zero;
            float2 input3 = zero;
            float2 input4 = zero;
            float2 input5 = zero;
            if (firstInputColumn >= 0 && firstInputColumn < width) {
                input0 = loadHalf2AsFloat2(
                    inputRowStart + firstInputColumn * channelPairs
                );
            }
            if (firstInputColumn + 1 >= 0 && firstInputColumn + 1 < width) {
                input1 = loadHalf2AsFloat2(
                    inputRowStart + (firstInputColumn + 1) * channelPairs
                );
            }
            if (firstInputColumn + 2 >= 0 && firstInputColumn + 2 < width) {
                input2 = loadHalf2AsFloat2(
                    inputRowStart + (firstInputColumn + 2) * channelPairs
                );
            }
            if (firstInputColumn + 3 >= 0 && firstInputColumn + 3 < width) {
                input3 = loadHalf2AsFloat2(
                    inputRowStart + (firstInputColumn + 3) * channelPairs
                );
            }
            if (firstInputColumn + 4 >= 0 && firstInputColumn + 4 < width) {
                input4 = loadHalf2AsFloat2(
                    inputRowStart + (firstInputColumn + 4) * channelPairs
                );
            }
            if (firstInputColumn + 5 >= 0 && firstInputColumn + 5 < width) {
                input5 = loadHalf2AsFloat2(
                    inputRowStart + (firstInputColumn + 5) * channelPairs
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

    // 这里只写回 DWConv FP16；exact erf-GELU 继续由 TensorRT/Myelin 处理。
    output[tileOutputPairOffset] =
        __floats2half2_rn(accumulator0.x, accumulator0.y);
    if (tileColumn + 1 < width) {
        output[tileOutputPairOffset + channelPairs] =
            __floats2half2_rn(accumulator1.x, accumulator1.y);
    }
    if (tileColumn + 2 < width) {
        output[tileOutputPairOffset + 2 * channelPairs] =
            __floats2half2_rn(accumulator2.x, accumulator2.y);
    }
    if (tileColumn + 3 < width) {
        output[tileOutputPairOffset + 3 * channelPairs] =
            __floats2half2_rn(accumulator3.x, accumulator3.y);
    }
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

    const dim3 grid(
        static_cast<uint32_t>((width + kOutputsPerTile - 1) / kOutputsPerTile),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(batch)
    );
    const dim3 block(static_cast<uint32_t>(channelPairs));

    block1PackedDwconvHalf2Kernel<<<grid, block, 0, stream>>>(
        static_cast<const __half2*>(input),
        static_cast<const __half2*>(packedWeight),
        static_cast<const __half2*>(packedBias),
        static_cast<__half2*>(output),
        height,
        width,
        channelPairs
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // 命名空间 egcinet::plugins
