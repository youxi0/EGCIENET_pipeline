#include "plugins/Block1FusedDwconvGeluKernel.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kThreadsPerBlock = 256;
constexpr float kInvSqrtTwo = 0.70710678118654752440F;

// 与 PyTorch nn.GELU 默认实现保持一致：0.5 * x * (1 + erf(x / sqrt(2)))。
__device__ __forceinline__ float exactGelu(float value) {
    return 0.5F * value * (1.0F + erff(value * kInvSqrtTwo));
}

// 每个线程同时处理相邻的两个通道。token-major 布局中相邻通道天然连续，
// 因而可以使用 half2 合并访存；空间邻居则通过 token 下标直接定位，
// 整个过程不需要真正生成 NCHW Transpose 中间张量。
__global__ void block1FusedDwconvGeluHalf2Kernel(
    const __half* input,
    const __half* weight,
    const __half* bias,
    __half* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channels
) {
    // 步骤 1：把线程编号映射为 batch、 token 和channel pair。
    const int32_t channelPairs = channels / 2;
    const int64_t tokenCount = static_cast<int64_t>(height) * width;
    const int64_t totalPairs = static_cast<int64_t>(batch) * tokenCount * channelPairs;
    const int64_t linearPair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearPair >= totalPairs) {
        return;
    }

    const int32_t channelPair = static_cast<int32_t>(linearPair % channelPairs);
    const int64_t tokenLinear = linearPair / channelPairs;
    const int32_t token = static_cast<int32_t>(tokenLinear % tokenCount);
    const int32_t batchIndex = static_cast<int32_t>(tokenLinear / tokenCount);
    const int32_t row = token / width;
    const int32_t column = token - row * width;
    const int32_t channel = channelPair * 2;

    // 步骤 2：使用 FP32 累加器，并以两个通道各自的 bias 作为初值。
    float2 accumulator = __half22float2(
        __halves2half2(bias[channel], bias[channel + 1])
    );

    // 步骤 3：遍历 3x3 邻域。越界位置等价于零填充，直接跳过即可。
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
            // 计算当前 kernel 元素在权重张量中的下标，以及对应的输入 token 下标和偏移。
            const int32_t kernelIndex = kernelRow * 3 + kernelColumn;
            const int64_t inputToken =
                static_cast<int64_t>(inputRow) * width + inputColumn;
            const int64_t inputOffset =
                (static_cast<int64_t>(batchIndex) * tokenCount + inputToken) * channels + channel;

            // 一次读取两个连续通道，并分别乘以各自的 depthwise 权重。
            const __half2 inputPair =
                *reinterpret_cast<const __half2*>(input + inputOffset);
            const __half2 weightPair = __halves2half2(
                weight[channel * 9 + kernelIndex],
                weight[(channel + 1) * 9 + kernelIndex]
            );
            const float2 inputValues = __half22float2(inputPair);
            const float2 weightValues = __half22float2(weightPair);
            accumulator.x = fmaf(inputValues.x, weightValues.x, accumulator.x);
            accumulator.y = fmaf(inputValues.y, weightValues.y, accumulator.y);
        }
    }

    // 步骤 4：在 FP32 中计算精确 GELU，最后只进行一次 FP16 舍入并写回。
    const __half2 result = __floats2half2_rn(
        exactGelu(accumulator.x),
        exactGelu(accumulator.y)
    );
    *reinterpret_cast<__half2*>(
        output + (tokenLinear * channels + channel)
    ) = result;
}

// 奇数通道的通用回退 kernel。当前 block1 的 C=256 会走上面的 half2 路径，
// 保留标量路径是为了避免插件实现被偶数通道这一隐含条件绑死。
__global__ void block1FusedDwconvGeluHalfKernel(
    const __half* input,
    const __half* weight,
    const __half* bias,
    __half* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channels
) {
    // 步骤 1：每个线程对应一个 [batch, token, channel] 输出元素。
    const int64_t tokenCount = static_cast<int64_t>(height) * width;
    const int64_t totalElements = static_cast<int64_t>(batch) * tokenCount * channels;
    const int64_t linearIndex =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearIndex >= totalElements) {
        return;
    }

    const int32_t channel = static_cast<int32_t>(linearIndex % channels);
    const int64_t tokenLinear = linearIndex / channels;
    const int32_t token = static_cast<int32_t>(tokenLinear % tokenCount);
    const int32_t batchIndex = static_cast<int32_t>(tokenLinear / tokenCount);
    const int32_t row = token / width;
    const int32_t column = token - row * width;

    // 步骤 2：以 bias 初始化 FP32 累加器，再直接按 token 下标访问 3x3 邻域。
    float accumulator = __half2float(bias[channel]);
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
            const int64_t inputOffset =
                (static_cast<int64_t>(batchIndex) * tokenCount + inputToken) * channels + channel;
            accumulator = fmaf(
                __half2float(input[inputOffset]),
                __half2float(weight[channel * 9 + kernelIndex]),
                accumulator
            );
        }
    }

    // 步骤 3：融合精确 GELU，并将最终结果转换回 FP16。
    output[linearIndex] = __float2half_rn(exactGelu(accumulator));
}

} // namespace

int32_t launchBlock1FusedDwconvGelu(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channels,
    cudaStream_t stream
) noexcept {
    // 步骤 1：在提交 kernel 前拒绝空指针和非法维度。
    if (input == nullptr || weight == nullptr || bias == nullptr || output == nullptr ||
        batch <= 0 || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }

    // 步骤 2：偶数通道优先使用 half2；只有奇数通道才回退到标量实现。
    if ((channels & 1) == 0) {
        const int64_t totalPairs = static_cast<int64_t>(batch) * height * width * (channels / 2);
        const int32_t grid = static_cast<int32_t>(
            (totalPairs + kThreadsPerBlock - 1) / kThreadsPerBlock
        );
        block1FusedDwconvGeluHalf2Kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<__half*>(output),
            batch,
            height,
            width,
            channels
        );
    } else {
        const int64_t totalElements = static_cast<int64_t>(batch) * height * width * channels;
        const int32_t grid = static_cast<int32_t>(
            (totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock
        );
        block1FusedDwconvGeluHalfKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<__half*>(output),
            batch,
            height,
            width,
            channels
        );
    }

    // 步骤 3：只检查 kernel 启动状态，不在 enqueue 中同步 CUDA stream。
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // namespace egcinet::plugins
