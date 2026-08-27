#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

template <
    int32_t kInputWidth,
    int32_t kChannels,
    int32_t kElementsPerVector,
    int32_t kKernelHeight,
    int32_t kKernelWidth,
    int32_t kStrideHeight,
    int32_t kStrideWidth,
    int32_t kOutputWidth>
__global__ void packSpatialReductionWindowsKernel(
    const uint4* __restrict__ input,
    uint4* __restrict__ packedInput
) {
    constexpr int32_t kChannelVectors = kChannels / kElementsPerVector;
    constexpr int32_t kWindowVectors =
        kKernelHeight * kKernelWidth * kChannelVectors;

    const int32_t outputToken = static_cast<int32_t>(blockIdx.x);
    constexpr int32_t kSpatialOutputTokens = 11 * 11;
    if (outputToken >= kSpatialOutputTokens) {
        const uint4 zero{};
        for (int32_t windowVector = static_cast<int32_t>(threadIdx.x);
             windowVector < kWindowVectors;
             windowVector += static_cast<int32_t>(blockDim.x)) {
            packedInput[outputToken * kWindowVectors + windowVector] = zero;
        }
        return;
    }
    const int32_t outputRow = outputToken / kOutputWidth;
    const int32_t outputColumn = outputToken - outputRow * kOutputWidth;
    const int32_t inputRowOrigin = outputRow * kStrideHeight;
    const int32_t inputColumnOrigin = outputColumn * kStrideWidth;

    for (int32_t windowVector = static_cast<int32_t>(threadIdx.x);
         windowVector < kWindowVectors;
         windowVector += static_cast<int32_t>(blockDim.x)) {
        const int32_t windowPixel = windowVector / kChannelVectors;
        const int32_t channelVector =
            windowVector - windowPixel * kChannelVectors;
        const int32_t kernelRow = windowPixel / kKernelWidth;
        const int32_t kernelColumn =
            windowPixel - kernelRow * kKernelWidth;
        const int32_t inputToken =
            (inputRowOrigin + kernelRow) * kInputWidth +
            inputColumnOrigin + kernelColumn;

        packedInput[outputToken * kWindowVectors + windowVector] =
            input[inputToken * kChannelVectors + channelVector];
    }
}

union Half8Vector {
    uint4 storage;
    half values[8];
};

union Int8x8Vector {
    uint2 storage;
    int8_t values[8];
};

union Float4Vector {
    float4 storage;
    float values[4];
};

union Int8x4Vector {
    uint32_t storage;
    int8_t values[4];
};

__device__ __forceinline__ int8_t quantizeInt8(
    float value,
    float inverseScale
) {
    int32_t quantized = __float2int_rn(value * inverseScale);
    quantized = quantized < -128 ? -128 : quantized;
    quantized = quantized > 127 ? 127 : quantized;
    return static_cast<int8_t>(quantized);
}

template <
    int32_t kInputWidth,
    int32_t kChannels,
    int32_t kKernelHeight,
    int32_t kKernelWidth,
    int32_t kStrideHeight,
    int32_t kStrideWidth,
    int32_t kOutputWidth>
__global__ void quantizePackHalfSpatialReductionWindowsKernel(
    const uint4* __restrict__ input,
    uint2* __restrict__ packedInput,
    float inverseScale
) {
    constexpr int32_t kElementsPerVector = 8;
    constexpr int32_t kChannelVectors = kChannels / kElementsPerVector;
    constexpr int32_t kWindowVectors =
        kKernelHeight * kKernelWidth * kChannelVectors;
    constexpr int32_t kSpatialOutputTokens = 11 * 11;
    const int32_t outputToken = static_cast<int32_t>(blockIdx.x);

    for (int32_t windowVector = static_cast<int32_t>(threadIdx.x);
         windowVector < kWindowVectors;
         windowVector += static_cast<int32_t>(blockDim.x)) {
        Int8x8Vector quantized{};
        if (outputToken < kSpatialOutputTokens) {
            const int32_t outputRow = outputToken / kOutputWidth;
            const int32_t outputColumn =
                outputToken - outputRow * kOutputWidth;
            const int32_t windowPixel = windowVector / kChannelVectors;
            const int32_t channelVector =
                windowVector - windowPixel * kChannelVectors;
            const int32_t kernelRow = windowPixel / kKernelWidth;
            const int32_t kernelColumn =
                windowPixel - kernelRow * kKernelWidth;
            const int32_t inputToken =
                (outputRow * kStrideHeight + kernelRow) * kInputWidth +
                outputColumn * kStrideWidth + kernelColumn;
            Half8Vector source{};
            source.storage =
                input[inputToken * kChannelVectors + channelVector];
#pragma unroll
            for (int32_t element = 0; element < kElementsPerVector; ++element) {
                quantized.values[element] = quantizeInt8(
                    __half2float(source.values[element]), inverseScale
                );
            }
        }
        packedInput[outputToken * kWindowVectors + windowVector] =
            quantized.storage;
    }
}

template <
    int32_t kInputWidth,
    int32_t kChannels,
    int32_t kKernelHeight,
    int32_t kKernelWidth,
    int32_t kStrideHeight,
    int32_t kStrideWidth,
    int32_t kOutputWidth>
__global__ void quantizePackFloatSpatialReductionWindowsKernel(
    const float4* __restrict__ input,
    uint32_t* __restrict__ packedInput,
    float inverseScale
) {
    constexpr int32_t kElementsPerVector = 4;
    constexpr int32_t kChannelVectors = kChannels / kElementsPerVector;
    constexpr int32_t kWindowVectors =
        kKernelHeight * kKernelWidth * kChannelVectors;
    constexpr int32_t kSpatialOutputTokens = 11 * 11;
    const int32_t outputToken = static_cast<int32_t>(blockIdx.x);

    for (int32_t windowVector = static_cast<int32_t>(threadIdx.x);
         windowVector < kWindowVectors;
         windowVector += static_cast<int32_t>(blockDim.x)) {
        Int8x4Vector quantized{};
        if (outputToken < kSpatialOutputTokens) {
            const int32_t outputRow = outputToken / kOutputWidth;
            const int32_t outputColumn =
                outputToken - outputRow * kOutputWidth;
            const int32_t windowPixel = windowVector / kChannelVectors;
            const int32_t channelVector =
                windowVector - windowPixel * kChannelVectors;
            const int32_t kernelRow = windowPixel / kKernelWidth;
            const int32_t kernelColumn =
                windowPixel - kernelRow * kKernelWidth;
            const int32_t inputToken =
                (outputRow * kStrideHeight + kernelRow) * kInputWidth +
                outputColumn * kStrideWidth + kernelColumn;
            Float4Vector source{};
            source.storage =
                input[inputToken * kChannelVectors + channelVector];
#pragma unroll
            for (int32_t element = 0; element < kElementsPerVector; ++element) {
                quantized.values[element] =
                    quantizeInt8(source.values[element], inverseScale);
            }
        }
        packedInput[outputToken * kWindowVectors + windowVector] =
            quantized.storage;
    }
}

__global__ void dequantizeSpatialReductionOutputKernel(
    const int32_t* __restrict__ accumulator,
    const half* __restrict__ bias,
    const float* __restrict__ dequantScale,
    half* __restrict__ output,
    int32_t outputTokens,
    int32_t outputChannels,
    int32_t accumulatorTokens
) {
    constexpr int32_t kTile = 32;
    constexpr int32_t kBlockRows = 8;
    __shared__ int32_t accumulatorTile[kTile][kTile + 1];

    const int32_t inputToken =
        static_cast<int32_t>(blockIdx.x) * kTile +
        static_cast<int32_t>(threadIdx.x);
    const int32_t inputChannelBase =
        static_cast<int32_t>(blockIdx.y) * kTile +
        static_cast<int32_t>(threadIdx.y);

#pragma unroll
    for (int32_t row = 0; row < kTile; row += kBlockRows) {
        const int32_t channel = inputChannelBase + row;
        if (channel < outputChannels && inputToken < accumulatorTokens) {
            accumulatorTile[threadIdx.y + row][threadIdx.x] =
                accumulator[channel * accumulatorTokens + inputToken];
        }
    }
    __syncthreads();

    const int32_t outputChannel =
        static_cast<int32_t>(blockIdx.y) * kTile +
        static_cast<int32_t>(threadIdx.x);
    if (outputChannel >= outputChannels) {
        return;
    }
    const float scale = dequantScale[outputChannel];
    const float channelBias = __half2float(bias[outputChannel]);
    const int32_t outputTokenBase =
        static_cast<int32_t>(blockIdx.x) * kTile +
        static_cast<int32_t>(threadIdx.y);

#pragma unroll
    for (int32_t row = 0; row < kTile; row += kBlockRows) {
        const int32_t outputToken = outputTokenBase + row;
        if (outputToken < outputTokens) {
            const float value = static_cast<float>(
                                    accumulatorTile[threadIdx.x]
                                                   [threadIdx.y + row]
                                ) *
                                    scale +
                                channelBias;
            output[outputToken * outputChannels + outputChannel] =
                __float2half_rn(value);
        }
    }
}

} // namespace

int32_t launchPackSpatialReductionWindows(
    int32_t stage,
    int32_t int8Mode,
    bool floatInput,
    float activationScale,
    const void* input,
    void* packedInput,
    cudaStream_t stream
) noexcept {
    if (input == nullptr || packedInput == nullptr || int8Mode < 0 ||
        int8Mode > 2 || (int8Mode == 2 && activationScale <= 0.0F)) {
        return -1;
    }

    constexpr int32_t kSpatialOutputTokens = 11 * 11;
    constexpr int32_t kInt8GemmTokens = 128;
    const int32_t gridBlocks = int8Mode != 0
        ? kInt8GemmTokens
        : kSpatialOutputTokens;
    const float inverseScale = int8Mode == 2
        ? 1.0F / activationScale
        : 0.0F;
    switch (stage) {
    case 1:
        if (int8Mode == 2 && floatInput) {
            quantizePackFloatSpatialReductionWindowsKernel<
                88, 64, 8, 8, 8, 8, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const float4*>(input),
                    static_cast<uint32_t*>(packedInput),
                    inverseScale
                );
        } else if (int8Mode == 2) {
            quantizePackHalfSpatialReductionWindowsKernel<
                88, 64, 8, 8, 8, 8, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint2*>(packedInput),
                    inverseScale
                );
        } else if (int8Mode == 1) {
            packSpatialReductionWindowsKernel<88, 64, 16, 8, 8, 8, 8, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint4*>(packedInput)
                );
        } else {
            packSpatialReductionWindowsKernel<88, 64, 8, 8, 8, 8, 8, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint4*>(packedInput)
                );
        }
        break;
    case 2:
        if (int8Mode == 2 && floatInput) {
            quantizePackFloatSpatialReductionWindowsKernel<
                44, 128, 4, 4, 4, 4, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const float4*>(input),
                    static_cast<uint32_t*>(packedInput),
                    inverseScale
                );
        } else if (int8Mode == 2) {
            quantizePackHalfSpatialReductionWindowsKernel<
                44, 128, 4, 4, 4, 4, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint2*>(packedInput),
                    inverseScale
                );
        } else if (int8Mode == 1) {
            packSpatialReductionWindowsKernel<44, 128, 16, 4, 4, 4, 4, 11>
                <<<gridBlocks, 128, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint4*>(packedInput)
                );
        } else {
            packSpatialReductionWindowsKernel<44, 128, 8, 4, 4, 4, 4, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint4*>(packedInput)
                );
        }
        break;
    case 3:
        if (int8Mode == 2 && floatInput) {
            quantizePackFloatSpatialReductionWindowsKernel<
                22, 320, 2, 2, 2, 2, 11>
                <<<gridBlocks, 256, 0, stream>>>(
                    static_cast<const float4*>(input),
                    static_cast<uint32_t*>(packedInput),
                    inverseScale
                );
        } else if (int8Mode == 2) {
            quantizePackHalfSpatialReductionWindowsKernel<
                22, 320, 2, 2, 2, 2, 11>
                <<<gridBlocks, 160, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint2*>(packedInput),
                    inverseScale
                );
        } else if (int8Mode == 1) {
            packSpatialReductionWindowsKernel<22, 320, 16, 2, 2, 2, 2, 11>
                <<<gridBlocks, 128, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint4*>(packedInput)
                );
        } else {
            packSpatialReductionWindowsKernel<22, 320, 8, 2, 2, 2, 2, 11>
                <<<gridBlocks, 160, 0, stream>>>(
                    static_cast<const uint4*>(input),
                    static_cast<uint4*>(packedInput)
                );
        }
        break;
    default:
        return -1;
    }

    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int32_t launchDequantizeSpatialReductionOutput(
    const void* accumulator,
    const void* bias,
    const void* dequantScale,
    void* output,
    int32_t outputTokens,
    int32_t outputChannels,
    int32_t accumulatorTokens,
    cudaStream_t stream
) noexcept {
    if (accumulator == nullptr || bias == nullptr || dequantScale == nullptr ||
        output == nullptr || outputTokens <= 0 || outputChannels <= 0 ||
        accumulatorTokens < outputTokens) {
        return -1;
    }

    constexpr int32_t kTile = 32;
    constexpr int32_t kBlockRows = 8;
    const dim3 block(kTile, kBlockRows);
    const dim3 grid(
        (outputTokens + kTile - 1) / kTile,
        (outputChannels + kTile - 1) / kTile
    );
    dequantizeSpatialReductionOutputKernel<<<grid, block, 0, stream>>>(
        static_cast<const int32_t*>(accumulator),
        static_cast<const half*>(bias),
        static_cast<const float*>(dequantScale),
        static_cast<half*>(output),
        outputTokens,
        outputChannels,
        accumulatorTokens
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // namespace egcinet::plugins
