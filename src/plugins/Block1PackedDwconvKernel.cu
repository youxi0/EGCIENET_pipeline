#include "plugins/Block1PackedDwconvKernel.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kThreadsPerBlock = 256;

// 一个线程计算同一 token 的两个相邻通道。线程束内相邻线程读取连续的
// input half2、packedWeight half2 和 packedBias half2。
__global__ void block1PackedDwconvHalf2Kernel(
    const __half2* input,
    const __half2* packedWeight,
    const __half2* packedBias,
    __half2* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channelPairs
) {
    // 步骤 1：把线程编号映射为 batch、空间 token 和 channel pair。
    const int64_t tokenCount = static_cast<int64_t>(height) * width;
    const int64_t totalPairs =
        static_cast<int64_t>(batch) * tokenCount * channelPairs;
    const int64_t linearPair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearPair >= totalPairs) {
        return;
    }

    const int32_t channelPair =
        static_cast<int32_t>(linearPair % channelPairs);
    const int64_t tokenLinear = linearPair / channelPairs;
    const int32_t token = static_cast<int32_t>(tokenLinear % tokenCount);
    const int32_t batchIndex = static_cast<int32_t>(tokenLinear / tokenCount);
    const int32_t row = token / width;
    const int32_t column = token - row * width;

    // 步骤 2：bias 已按 channel pair 打包，一次连续读取即可初始化两个 FP32 累加器。
    float2 accumulator = __half22float2(packedBias[channelPair]);

    // 步骤 3：遍历 3x3 邻域。packedWeight 的第一维是 kernelIndex，
    // 第二维是 channelPair，所以 warp 在每个 kernelIndex 上执行合并读取。
#pragma unroll
    for (int32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
        const int32_t inputRow = row + kernelRow - 1;
        if (inputRow < 0 || inputRow >= height) {
            continue;
        }

#pragma unroll
        for (int32_t kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {
            const int32_t inputColumn = column + kernelColumn - 1;
            if (inputColumn < 0 || inputColumn >= width) {
                continue;
            }

            const int32_t kernelIndex = kernelRow * 3 + kernelColumn;
            const int64_t inputToken =
                static_cast<int64_t>(inputRow) * width + inputColumn;
            const int64_t inputPairOffset =
                (static_cast<int64_t>(batchIndex) * tokenCount + inputToken)
                    * channelPairs
                + channelPair;
            const int64_t weightPairOffset =
                static_cast<int64_t>(kernelIndex) * channelPairs + channelPair;

            const float2 inputValues = __half22float2(input[inputPairOffset]);
            const float2 weightValues =
                __half22float2(packedWeight[weightPairOffset]);
            accumulator.x =
                fmaf(inputValues.x, weightValues.x, accumulator.x);
            accumulator.y =
                fmaf(inputValues.y, weightValues.y, accumulator.y);
        }
    }

    // 步骤 4：这里只写回 DWConv 的 FP16 结果；后续 GELU 继续由 TensorRT 处理。
    output[linearPair] =
        __floats2half2_rn(accumulator.x, accumulator.y);
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
    const int64_t totalPairs =
        static_cast<int64_t>(batch) * height * width * channelPairs;
    const int32_t grid = static_cast<int32_t>(
        (totalPairs + kThreadsPerBlock - 1) / kThreadsPerBlock
    );

    block1PackedDwconvHalf2Kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        static_cast<const __half2*>(input),
        static_cast<const __half2*>(packedWeight),
        static_cast<const __half2*>(packedBias),
        static_cast<__half2*>(output),
        batch,
        height,
        width,
        channelPairs
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // 命名空间 egcinet::plugins
