#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputWidth = 11;
constexpr int32_t kOutputTokens = kOutputWidth * kOutputWidth;
constexpr int32_t kDefaultSharedMemoryBytes = 48 * 1024;

template <
    int32_t kInputWidthValue,
    int32_t kChannelsValue,
    int32_t kKernelWidthValue,
    int32_t kStrideValue,
    int32_t kCtaTokensValue,
    int32_t kCtaChannelsValue,
    int32_t kKStageValue,
    int32_t kChannelTilesPerWarpValue>
struct SpatialReductionConfiguration {
    static constexpr int32_t kInputWidth = kInputWidthValue;
    static constexpr int32_t kChannels = kChannelsValue;
    static constexpr int32_t kKernelWidth = kKernelWidthValue;
    static constexpr int32_t kStride = kStrideValue;
    static constexpr int32_t kCtaTokens = kCtaTokensValue;
    static constexpr int32_t kCtaChannels = kCtaChannelsValue;
    static constexpr int32_t kKStage = kKStageValue;
    static constexpr int32_t kChannelTilesPerWarp =
        kChannelTilesPerWarpValue;
    static constexpr int32_t kChannelsPerWarp =
        8 * kChannelTilesPerWarp;
    static constexpr int32_t kPackedK =
        kKernelWidth * kKernelWidth * kChannels;
    static constexpr int32_t kChannelWarps =
        kCtaChannels / kChannelsPerWarp;
    static constexpr int32_t kCtaWarps =
        (kCtaTokens / 16) * kChannelWarps;
    static constexpr int32_t kCtaThreads = kCtaWarps * 32;
    static constexpr int32_t kInputTileBytes = kCtaTokens * kKStage;
    static constexpr int32_t kWeightTileBytes = kCtaChannels * kKStage;

    static_assert(kCtaTokens % 16 == 0);
    static_assert(kCtaChannels % 8 == 0);
    static_assert(kChannels % kCtaChannels == 0);
    static_assert(kCtaChannels % kChannelsPerWarp == 0);
    static_assert(kKStage % 32 == 0);
    static_assert(
        ((kKStage / 16) & (kKStage / 16 - 1)) == 0,
        "XOR shared-memory swizzle requires a power-of-two vector stride"
    );
    static_assert(kPackedK % kKStage == 0);
    static_assert(kCtaThreads <= 1024);
};

// Block1 only launches four CTAs, one per Orin SM. A larger K stage therefore
// cuts the number of CTA-wide synchronization points without reducing useful
// inter-CTA occupancy.
using Block1Configuration =
    SpatialReductionConfiguration<88, 64, 8, 8, 32, 64, 512, 1>;
using Block2Configuration =
    SpatialReductionConfiguration<44, 128, 4, 4, 32, 64, 128, 1>;
// Block3 covers all 320 output channels in one CTA. Each warp owns four
// adjacent N=8 MMA tiles, so a 32-token input window is constructed once
// instead of once for each of five 64-channel CTAs.
using Block3Configuration =
    SpatialReductionConfiguration<22, 320, 2, 2, 32, 320, 128, 4>;

union Half8Vector {
    uint4 storage;
    half values[8];
};

union Int8x8Vector {
    uint2 storage;
    int8_t values[8];
};

union Int8x4Vector {
    uint32_t storage;
    int8_t values[4];
};

template <typename Configuration>
__device__ __forceinline__ int32_t sharedOffset(
    int32_t row,
    int32_t logicalColumn
) {
    const int32_t physicalVector =
        (logicalColumn >> 4) ^ (row & 7);
    return row * Configuration::kKStage + physicalVector * 16 +
        (logicalColumn & 15);
}

__device__ __forceinline__ int8_t quantizeInt8(
    float value,
    float inverseScale
) {
    int32_t quantized = __float2int_rn(value * inverseScale);
    quantized = quantized < -128 ? -128 : quantized;
    quantized = quantized > 127 ? 127 : quantized;
    return static_cast<int8_t>(quantized);
}

__device__ __forceinline__ void mmaInt8M16N8K32(
    int32_t (&accumulator)[4],
    const uint32_t (&inputFragment)[4],
    const uint32_t (&weightFragment)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(accumulator[0]), "+r"(accumulator[1]),
          "+r"(accumulator[2]), "+r"(accumulator[3])
        : "r"(inputFragment[0]), "r"(inputFragment[1]),
          "r"(inputFragment[2]), "r"(inputFragment[3]),
          "r"(weightFragment[0]), "r"(weightFragment[1])
    );
}

__device__ __forceinline__ void copyGlobalToShared16Cached(
    void* sharedDestination,
    const void* globalSource,
    int32_t sourceBytes
) {
    const uint32_t sharedAddress = static_cast<uint32_t>(
        __cvta_generic_to_shared(sharedDestination)
    );
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(sharedAddress), "l"(globalSource), "r"(sourceBytes)
    );
}

__device__ __forceinline__ void copyGlobalToShared16L2(
    void* sharedDestination,
    const void* globalSource
) {
    const uint32_t sharedAddress = static_cast<uint32_t>(
        __cvta_generic_to_shared(sharedDestination)
    );
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(sharedAddress), "l"(globalSource)
    );
}

template <typename Configuration>
__device__ __forceinline__ int32_t inputElementOffset(
    int32_t outputToken,
    int32_t packedColumn
) {
    const int32_t outputRow = outputToken / kOutputWidth;
    const int32_t outputColumn = outputToken - outputRow * kOutputWidth;
    const int32_t windowPixel = packedColumn / Configuration::kChannels;
    const int32_t inputChannel =
        packedColumn - windowPixel * Configuration::kChannels;
    const int32_t kernelRow =
        windowPixel / Configuration::kKernelWidth;
    const int32_t kernelColumn =
        windowPixel - kernelRow * Configuration::kKernelWidth;
    const int32_t inputToken =
        (outputRow * Configuration::kStride + kernelRow) *
            Configuration::kInputWidth +
        outputColumn * Configuration::kStride + kernelColumn;
    return inputToken * Configuration::kChannels + inputChannel;
}

template <typename Configuration>
__device__ __forceinline__ void copyInputStage(
    const int8_t* input,
    int8_t* inputTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    float /* inverseScale */
) {
    constexpr int32_t kVectorsPerToken = Configuration::kKStage / 16;
    constexpr int32_t kCopies =
        Configuration::kCtaTokens * kVectorsPerToken;
    constexpr int32_t kIterations =
        (kCopies + Configuration::kCtaThreads - 1) /
        Configuration::kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * Configuration::kCtaThreads;
        if (copy >= kCopies) {
            continue;
        }
        const int32_t tileToken = copy / kVectorsPerToken;
        const int32_t vector = copy - tileToken * kVectorsPerToken;
        const int32_t outputToken = ctaTokenBase + tileToken;
        const int32_t logicalColumn = vector * 16;
        const void* source = input;
        int32_t sourceBytes = 0;
        if (outputToken < kOutputTokens) {
            source = input + inputElementOffset<Configuration>(
                outputToken, kBase + logicalColumn
            );
            sourceBytes = 16;
        }
        copyGlobalToShared16Cached(
            inputTile + sharedOffset<Configuration>(tileToken, logicalColumn),
            source,
            sourceBytes
        );
    }
}

template <typename Configuration>
__device__ __forceinline__ void copyInputStage(
    const half* input,
    int8_t* inputTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    float inverseScale
) {
    constexpr int32_t kElements = 8;
    constexpr int32_t kVectorsPerToken =
        Configuration::kKStage / kElements;
    constexpr int32_t kCopies =
        Configuration::kCtaTokens * kVectorsPerToken;
    constexpr int32_t kIterations =
        (kCopies + Configuration::kCtaThreads - 1) /
        Configuration::kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * Configuration::kCtaThreads;
        if (copy >= kCopies) {
            continue;
        }
        const int32_t tileToken = copy / kVectorsPerToken;
        const int32_t vector = copy - tileToken * kVectorsPerToken;
        const int32_t outputToken = ctaTokenBase + tileToken;
        const int32_t logicalColumn = vector * kElements;
        Int8x8Vector quantized{};
        if (outputToken < kOutputTokens) {
            Half8Vector source{};
            source.storage = *reinterpret_cast<const uint4*>(
                input + inputElementOffset<Configuration>(
                    outputToken, kBase + logicalColumn
                )
            );
#pragma unroll
            for (int32_t element = 0; element < kElements; ++element) {
                quantized.values[element] = quantizeInt8(
                    __half2float(source.values[element]), inverseScale
                );
            }
        }
        *reinterpret_cast<uint2*>(
            inputTile +
            sharedOffset<Configuration>(tileToken, logicalColumn)
        ) = quantized.storage;
    }
}

template <typename Configuration>
__device__ __forceinline__ void copyInputStage(
    const float* input,
    int8_t* inputTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    float inverseScale
) {
    constexpr int32_t kElements = 4;
    constexpr int32_t kVectorsPerToken =
        Configuration::kKStage / kElements;
    constexpr int32_t kCopies =
        Configuration::kCtaTokens * kVectorsPerToken;
    constexpr int32_t kIterations =
        (kCopies + Configuration::kCtaThreads - 1) /
        Configuration::kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * Configuration::kCtaThreads;
        if (copy >= kCopies) {
            continue;
        }
        const int32_t tileToken = copy / kVectorsPerToken;
        const int32_t vector = copy - tileToken * kVectorsPerToken;
        const int32_t outputToken = ctaTokenBase + tileToken;
        const int32_t logicalColumn = vector * kElements;
        Int8x4Vector quantized{};
        if (outputToken < kOutputTokens) {
            const float4 source = *reinterpret_cast<const float4*>(
                input + inputElementOffset<Configuration>(
                    outputToken, kBase + logicalColumn
                )
            );
            quantized.values[0] = quantizeInt8(source.x, inverseScale);
            quantized.values[1] = quantizeInt8(source.y, inverseScale);
            quantized.values[2] = quantizeInt8(source.z, inverseScale);
            quantized.values[3] = quantizeInt8(source.w, inverseScale);
        }
        *reinterpret_cast<uint32_t*>(
            inputTile +
            sharedOffset<Configuration>(tileToken, logicalColumn)
        ) = quantized.storage;
    }
}

template <typename Configuration>
__device__ __forceinline__ void copyWeightStage(
    const int8_t* weight,
    int8_t* weightTile,
    int32_t kBase,
    int32_t channelBlock
) {
    constexpr int32_t kVectorsPerChannel = Configuration::kKStage / 16;
    constexpr int32_t kCopies =
        Configuration::kCtaChannels * kVectorsPerChannel;
    constexpr int32_t kIterations =
        (kCopies + Configuration::kCtaThreads - 1) /
        Configuration::kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * Configuration::kCtaThreads;
        if (copy >= kCopies) {
            continue;
        }
        const int32_t weightChannel = copy / kVectorsPerChannel;
        const int32_t vector = copy - weightChannel * kVectorsPerChannel;
        const int32_t globalWeightChannel =
            channelBlock * Configuration::kCtaChannels + weightChannel;
        copyGlobalToShared16L2(
            weightTile + sharedOffset<Configuration>(
                weightChannel, vector * 16
            ),
            weight + globalWeightChannel * Configuration::kPackedK + kBase +
                vector * 16
        );
    }
}

template <typename Configuration, typename InputType>
__device__ __forceinline__ void copyStage(
    const InputType* input,
    const int8_t* weight,
    int8_t* inputTile,
    int8_t* weightTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    int32_t channelBlock,
    float inverseScale
) {
    copyInputStage<Configuration>(
        input, inputTile, kBase, ctaTokenBase, inverseScale
    );
    copyWeightStage<Configuration>(
        weight, weightTile, kBase, channelBlock
    );
}

template <typename Configuration>
__device__ __forceinline__ void loadInputFragment(
    const int8_t* inputTile,
    int32_t warpTokenGroup,
    int32_t lane,
    int32_t kOffset,
    uint32_t (&inputFragment)[4]
) {
    const int32_t inputMatrix = lane >> 3;
    const int32_t inputRow = warpTokenGroup * 16 +
        (inputMatrix & 1) * 8 + (lane & 7);
    const int32_t inputColumn = kOffset + (inputMatrix >> 1) * 16;
    const uint32_t inputAddress = static_cast<uint32_t>(
        __cvta_generic_to_shared(
            inputTile + sharedOffset<Configuration>(inputRow, inputColumn)
        )
    );
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(inputFragment[0]), "=r"(inputFragment[1]),
          "=r"(inputFragment[2]), "=r"(inputFragment[3])
        : "r"(inputAddress)
    );
}

template <typename Configuration>
__device__ __forceinline__ void loadWeightFragment(
    const int8_t* weightTile,
    int32_t warpTileChannelBase,
    int32_t lane,
    int32_t kOffset,
    uint32_t (&weightFragment)[2]
) {
    const int32_t weightMatrix = (lane >> 3) & 1;
    const int32_t weightChannel = warpTileChannelBase + (lane & 7);
    const uint32_t weightAddress = static_cast<uint32_t>(
        __cvta_generic_to_shared(
            weightTile + sharedOffset<Configuration>(
                weightChannel, kOffset + weightMatrix * 16
            )
        )
    );
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(weightFragment[0]), "=r"(weightFragment[1])
        : "r"(weightAddress)
    );
}

template <typename Configuration>
__device__ __forceinline__ void loadFragments(
    const int8_t* inputTile,
    const int8_t* weightTile,
    int32_t warpTokenGroup,
    int32_t warpTileChannelBase,
    int32_t lane,
    int32_t kOffset,
    uint32_t (&inputFragment)[4],
    uint32_t (&weightFragment)[2]
) {
    loadInputFragment<Configuration>(
        inputTile, warpTokenGroup, lane, kOffset, inputFragment
    );
    loadWeightFragment<Configuration>(
        weightTile,
        warpTileChannelBase,
        lane,
        kOffset,
        weightFragment
    );
}

template <typename Configuration, typename InputType>
__global__ void fusedInt8SpatialReductionKernel(
    const InputType* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ bias,
    const float* __restrict__ dequantScale,
    half* __restrict__ output,
    float inverseScale
) {
    constexpr int32_t kSharedMemoryBytes = 2 * (
        Configuration::kInputTileBytes + Configuration::kWeightTileBytes
    );
    constexpr bool kUseDynamicSharedMemory =
        kSharedMemoryBytes > kDefaultSharedMemoryBytes;
    extern __shared__ __align__(32) int8_t dynamicSharedMemory[];
    __shared__ __align__(32) int8_t staticSharedMemory[
        kUseDynamicSharedMemory ? 1 : kSharedMemoryBytes
    ];
    int8_t* const sharedMemory = kUseDynamicSharedMemory
        ? dynamicSharedMemory
        : staticSharedMemory;
    int8_t* const inputTiles = sharedMemory;
    int8_t* const weightTiles =
        sharedMemory + 2 * Configuration::kInputTileBytes;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t warp = thread >> 5;
    const int32_t lane = thread & 31;
    const int32_t laneGroup = lane >> 2;
    const int32_t threadInGroup = lane & 3;
    const int32_t warpTokenGroup = warp / Configuration::kChannelWarps;
    const int32_t warpTileChannelBase =
        (warp - warpTokenGroup * Configuration::kChannelWarps) *
        Configuration::kChannelsPerWarp;
    const int32_t channelBlock = static_cast<int32_t>(blockIdx.y);
    const int32_t warpChannelBase =
        channelBlock * Configuration::kCtaChannels + warpTileChannelBase;
    const int32_t ctaTokenBase =
        static_cast<int32_t>(blockIdx.x) * Configuration::kCtaTokens;

    int32_t accumulator[Configuration::kChannelTilesPerWarp][4]{};
    copyStage<Configuration>(
        input,
        weight,
        inputTiles,
        weightTiles,
        0,
        ctaTokenBase,
        channelBlock,
        inverseScale
    );
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

#pragma unroll
    for (int32_t stage = 0;
         stage < Configuration::kPackedK / Configuration::kKStage;
         ++stage) {
        const int32_t currentBuffer = stage & 1;
        const int32_t nextStage = stage + 1;
        if (nextStage <
            Configuration::kPackedK / Configuration::kKStage) {
            const int32_t nextBuffer = currentBuffer ^ 1;
            copyStage<Configuration>(
                input,
                weight,
                inputTiles + nextBuffer * Configuration::kInputTileBytes,
                weightTiles + nextBuffer * Configuration::kWeightTileBytes,
                nextStage * Configuration::kKStage,
                ctaTokenBase,
                channelBlock,
                inverseScale
            );
            asm volatile("cp.async.commit_group;\n" ::);
        }

        const int8_t* currentInputTile =
            inputTiles + currentBuffer * Configuration::kInputTileBytes;
        const int8_t* currentWeightTile =
            weightTiles + currentBuffer * Configuration::kWeightTileBytes;
        uint32_t inputFragments[2][4];
        if constexpr (Configuration::kChannelTilesPerWarp == 1) {
            uint32_t weightFragments[2][2];
            loadFragments<Configuration>(
                currentInputTile,
                currentWeightTile,
                warpTokenGroup,
                warpTileChannelBase,
                lane,
                0,
                inputFragments[0],
                weightFragments[0]
            );

#pragma unroll
            for (int32_t fragment = 0;
                 fragment < Configuration::kKStage / 32;
                 ++fragment) {
                const int32_t currentFragment = fragment & 1;
                const int32_t nextFragment = currentFragment ^ 1;
                if (fragment + 1 < Configuration::kKStage / 32) {
                    loadFragments<Configuration>(
                        currentInputTile,
                        currentWeightTile,
                        warpTokenGroup,
                        warpTileChannelBase,
                        lane,
                        (fragment + 1) * 32,
                        inputFragments[nextFragment],
                        weightFragments[nextFragment]
                    );
                }
                mmaInt8M16N8K32(
                    accumulator[0],
                    inputFragments[currentFragment],
                    weightFragments[currentFragment]
                );
            }
        } else {
            loadInputFragment<Configuration>(
                currentInputTile,
                warpTokenGroup,
                lane,
                0,
                inputFragments[0]
            );
#pragma unroll
            for (int32_t fragment = 0;
                 fragment < Configuration::kKStage / 32;
                 ++fragment) {
                const int32_t currentFragment = fragment & 1;
                const int32_t nextFragment = currentFragment ^ 1;
                if (fragment + 1 < Configuration::kKStage / 32) {
                    loadInputFragment<Configuration>(
                        currentInputTile,
                        warpTokenGroup,
                        lane,
                        (fragment + 1) * 32,
                        inputFragments[nextFragment]
                    );
                }
                uint32_t weightFragments[
                    Configuration::kChannelTilesPerWarp
                ][2];
#pragma unroll
                for (int32_t channelTile = 0;
                     channelTile < Configuration::kChannelTilesPerWarp;
                     ++channelTile) {
                    loadWeightFragment<Configuration>(
                        currentWeightTile,
                        warpTileChannelBase + channelTile * 8,
                        lane,
                        fragment * 32,
                        weightFragments[channelTile]
                    );
                }
#pragma unroll
                for (int32_t channelTile = 0;
                     channelTile < Configuration::kChannelTilesPerWarp;
                     ++channelTile) {
                    mmaInt8M16N8K32(
                        accumulator[channelTile],
                        inputFragments[currentFragment],
                        weightFragments[channelTile]
                    );
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    const int32_t upperToken = ctaTokenBase + warpTokenGroup * 16 +
        laneGroup;
    const int32_t lowerToken = upperToken + 8;
#pragma unroll
    for (int32_t channelTile = 0;
         channelTile < Configuration::kChannelTilesPerWarp;
         ++channelTile) {
        const int32_t channel = warpChannelBase + channelTile * 8 +
            threadInGroup * 2;
        const float2 scales = *reinterpret_cast<const float2*>(
            dequantScale + channel
        );
        const half2 biases = *reinterpret_cast<const half2*>(bias + channel);
        const float biasLowerChannel = __low2float(biases);
        const float biasUpperChannel = __high2float(biases);
        if (upperToken < kOutputTokens) {
            const half2 value = __halves2half2(
                __float2half_rn(
                    static_cast<float>(accumulator[channelTile][0]) *
                        scales.x + biasLowerChannel
                ),
                __float2half_rn(
                    static_cast<float>(accumulator[channelTile][1]) *
                        scales.y + biasUpperChannel
                )
            );
            *reinterpret_cast<half2*>(
                output + upperToken * Configuration::kChannels + channel
            ) = value;
        }
        if (lowerToken < kOutputTokens) {
            const half2 value = __halves2half2(
                __float2half_rn(
                    static_cast<float>(accumulator[channelTile][2]) *
                        scales.x + biasLowerChannel
                ),
                __float2half_rn(
                    static_cast<float>(accumulator[channelTile][3]) *
                        scales.y + biasUpperChannel
                )
            );
            *reinterpret_cast<half2*>(
                output + lowerToken * Configuration::kChannels + channel
            ) = value;
        }
    }
}

template <typename Configuration, typename InputType>
int32_t launchKernel(
    const void* input,
    const void* weight,
    const void* bias,
    const void* dequantScale,
    void* output,
    float inverseScale,
    cudaStream_t stream
) noexcept {
    constexpr int32_t kTokenBlocks =
        (kOutputTokens + Configuration::kCtaTokens - 1) /
        Configuration::kCtaTokens;
    constexpr int32_t kChannelBlocks =
        Configuration::kChannels / Configuration::kCtaChannels;
    constexpr int32_t kSharedMemoryBytes = 2 * (
        Configuration::kInputTileBytes + Configuration::kWeightTileBytes
    );
    constexpr int32_t kDynamicSharedMemoryBytes =
        kSharedMemoryBytes > kDefaultSharedMemoryBytes
        ? kSharedMemoryBytes
        : 0;
    if constexpr (kDynamicSharedMemoryBytes != 0) {
        static const cudaError_t attributeStatus = cudaFuncSetAttribute(
            fusedInt8SpatialReductionKernel<Configuration, InputType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSharedMemoryBytes
        );
        if (attributeStatus != cudaSuccess) {
            return -1;
        }
    }
    fusedInt8SpatialReductionKernel<Configuration, InputType><<<
        dim3(kTokenBlocks, kChannelBlocks),
        Configuration::kCtaThreads,
        kDynamicSharedMemoryBytes,
        stream
    >>>(
        static_cast<const InputType*>(input),
        static_cast<const int8_t*>(weight),
        static_cast<const half*>(bias),
        static_cast<const float*>(dequantScale),
        static_cast<half*>(output),
        inverseScale
    );
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace

int32_t launchFusedInt8SpatialReduction(
    int32_t stage,
    int32_t int8Mode,
    bool floatInput,
    float activationScale,
    const void* input,
    const void* weight,
    const void* bias,
    const void* dequantScale,
    void* output,
    cudaStream_t stream
) noexcept {
    if (input == nullptr || weight == nullptr || bias == nullptr ||
        dequantScale == nullptr || output == nullptr) {
        return -1;
    }
    switch (stage) {
    case 1: {
        if (int8Mode != 2 || activationScale <= 0.0F) {
            return -1;
        }
        const float inverseScale = 1.0F / activationScale;
        return floatInput
            ? launchKernel<Block1Configuration, float>(
                  input,
                  weight,
                  bias,
                  dequantScale,
                  output,
                  inverseScale,
                  stream
              )
            : launchKernel<Block1Configuration, half>(
                  input,
                  weight,
                  bias,
                  dequantScale,
                  output,
                  inverseScale,
                  stream
              );
    }
    case 2:
        return int8Mode == 1
            ? launchKernel<Block2Configuration, int8_t>(
                  input, weight, bias, dequantScale, output, 0.0F, stream
              )
            : -1;
    case 3:
        return int8Mode == 1
            ? launchKernel<Block3Configuration, int8_t>(
                  input, weight, bias, dequantScale, output, 0.0F, stream
              )
            : -1;
    default:
        return -1;
    }
}

} // namespace egcinet::plugins
