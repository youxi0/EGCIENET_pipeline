#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputWidth = 11;
constexpr int32_t kOutputTokens = kOutputWidth * kOutputWidth;
constexpr int32_t kInputWidth = 88;
constexpr int32_t kChannels = 64;
constexpr int32_t kKernelWidth = 8;
constexpr int32_t kStride = 8;
constexpr int32_t kCtaTokens = 32;
constexpr int32_t kCtaChannels = 64;
constexpr int32_t kKStage = 512;
constexpr int32_t kChannelsPerWarp = 8;
constexpr int32_t kPackedK = 4096;
constexpr int32_t kChannelWarps = 8;
constexpr int32_t kCtaThreads = 512;
constexpr int32_t kInputTileBytes = kCtaTokens * kKStage;
constexpr int32_t kWeightTileBytes = kCtaChannels * kKStage;
constexpr int32_t kSharedMemoryBytes =
    2 * (kInputTileBytes + kWeightTileBytes);

static_assert(kSharedMemoryBytes == 96 * 1024);

// Block1 只有 4 个 CTA，恰好每个 Orin SM 分配 1 个 CTA。K-stage 取 512
// 可以将全 CTA 同步次数减半，同时不会损失 CTA 之间的实际并行度。

// 向量联合体用于显式控制一次读写的字节数，避免编译器生成零散标量访存。
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

// 对 shared memory 的 K 维向量编号与行号做 XOR swizzle，降低 ldmatrix
// 读取输入和权重 fragment 时的 bank conflict。
__device__ __forceinline__ int32_t sharedOffset(
    int32_t row,
    int32_t logicalColumn
) {
    const int32_t physicalVector =
        (logicalColumn >> 4) ^ (row & 7);
    return row * kKStage + physicalVector * 16 +
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

// 单条 PTX 指令完成 16x8x32 的 INT8 Tensor Core MMA，并在 INT32 中累加。
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

// Block1 的权重通过 L2 异步搬入 shared memory，避免占用输入量化路径的 L1。
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

// 将输出 token 与 packed K 列还原为原始 BNC 输入中的元素下标。由于 SR
// 的卷积核大小等于步长，各输出窗口互不重叠，不需要额外处理 padding。
__device__ __forceinline__ int32_t inputElementOffset(
    int32_t outputToken,
    int32_t packedColumn
) {
    const int32_t outputRow = outputToken / kOutputWidth;
    const int32_t outputColumn = outputToken - outputRow * kOutputWidth;
    const int32_t windowPixel = packedColumn / kChannels;
    const int32_t inputChannel =
        packedColumn - windowPixel * kChannels;
    const int32_t kernelRow =
        windowPixel / kKernelWidth;
    const int32_t kernelColumn =
        windowPixel - kernelRow * kKernelWidth;
    const int32_t inputToken =
        (outputRow * kStride + kernelRow) *
            kInputWidth +
        outputColumn * kStride + kernelColumn;
    return inputToken * kChannels + inputChannel;
}

// Block1 的 FP16 输入在写入 shared memory 时完成逐元素 INT8 量化，不再
// 生成独立的全尺寸量化张量。
__device__ __forceinline__ void copyInputStage(
    const half* input,
    int8_t* inputTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    float inverseScale
) {
    constexpr int32_t kElements = 8;
    constexpr int32_t kVectorsPerToken =
        kKStage / kElements;
    constexpr int32_t kCopies =
        kCtaTokens * kVectorsPerToken;
    constexpr int32_t kIterations =
        (kCopies + kCtaThreads - 1) /
        kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * kCtaThreads;
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
                input + inputElementOffset(
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
            sharedOffset(tileToken, logicalColumn)
        ) = quantized.storage;
    }
}

// Block1 首层可能由 TensorRT 传入 FP32，采用 float4 读取并在 pack 阶段
// 完成与 FP16 路径相同的 INT8 量化。
__device__ __forceinline__ void copyInputStage(
    const float* input,
    int8_t* inputTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    float inverseScale
) {
    constexpr int32_t kElements = 4;
    constexpr int32_t kVectorsPerToken =
        kKStage / kElements;
    constexpr int32_t kCopies =
        kCtaTokens * kVectorsPerToken;
    constexpr int32_t kIterations =
        (kCopies + kCtaThreads - 1) /
        kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * kCtaThreads;
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
                input + inputElementOffset(
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
            sharedOffset(tileToken, logicalColumn)
        ) = quantized.storage;
    }
}

// 权重按输出通道连续存储；每个线程以 16 字节 cp.async 搬运当前 K-stage。
__device__ __forceinline__ void copyWeightStage(
    const int8_t* weight,
    int8_t* weightTile,
    int32_t kBase,
    int32_t channelBlock
) {
    constexpr int32_t kVectorsPerChannel = kKStage / 16;
    constexpr int32_t kCopies =
        kCtaChannels * kVectorsPerChannel;
    constexpr int32_t kIterations =
        (kCopies + kCtaThreads - 1) /
        kCtaThreads;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        const int32_t copy =
            thread + iteration * kCtaThreads;
        if (copy >= kCopies) {
            continue;
        }
        const int32_t weightChannel = copy / kVectorsPerChannel;
        const int32_t vector = copy - weightChannel * kVectorsPerChannel;
        const int32_t globalWeightChannel =
            channelBlock * kCtaChannels + weightChannel;
        copyGlobalToShared16L2(
            weightTile + sharedOffset(
                weightChannel, vector * 16
            ),
            weight + globalWeightChannel * kPackedK + kBase +
                vector * 16
        );
    }
}

template <typename InputType>
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
    copyInputStage(
        input, inputTile, kBase, ctaTokenBase, inverseScale
    );
    copyWeightStage(
        weight, weightTile, kBase, channelBlock
    );
}

// ldmatrix 根据 warp 内 lane 映射，从 swizzle 后的 shared memory 读取 MMA
// 所需的 A/B fragment。
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
            inputTile + sharedOffset(inputRow, inputColumn)
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
            weightTile + sharedOffset(
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
    loadInputFragment(
        inputTile, warpTokenGroup, lane, kOffset, inputFragment
    );
    loadWeightFragment(
        weightTile,
        warpTileChannelBase,
        lane,
        kOffset,
        weightFragment
    );
}

template <typename InputType>
__global__ void fusedInt8SpatialReductionKernel(
    const InputType* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ bias,
    const float* __restrict__ dequantScale,
    half* __restrict__ output,
    float inverseScale
) {
    // Block1 固定使用 96 KiB opt-in dynamic shared memory 保存双缓冲 tile。
    extern __shared__ __align__(32) int8_t sharedMemory[];
    int8_t* const inputTiles = sharedMemory;
    int8_t* const weightTiles =
        sharedMemory + 2 * kInputTileBytes;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t warp = thread >> 5;
    const int32_t lane = thread & 31;
    const int32_t laneGroup = lane >> 2;
    const int32_t threadInGroup = lane & 3;
    const int32_t warpTokenGroup = warp / kChannelWarps;
    const int32_t warpTileChannelBase =
        (warp - warpTokenGroup * kChannelWarps) *
        kChannelsPerWarp;
    const int32_t channelBlock = static_cast<int32_t>(blockIdx.y);
    const int32_t warpChannelBase =
        channelBlock * kCtaChannels + warpTileChannelBase;
    const int32_t ctaTokenBase =
        static_cast<int32_t>(blockIdx.x) * kCtaTokens;

    int32_t accumulator[4]{};
    copyStage(
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
         stage < kPackedK / kKStage;
         ++stage) {
        const int32_t currentBuffer = stage & 1;
        const int32_t nextStage = stage + 1;
        if (nextStage <
            kPackedK / kKStage) {
            const int32_t nextBuffer = currentBuffer ^ 1;
            copyStage(
                input,
                weight,
                inputTiles + nextBuffer * kInputTileBytes,
                weightTiles + nextBuffer * kWeightTileBytes,
                nextStage * kKStage,
                ctaTokenBase,
                channelBlock,
                inverseScale
            );
            asm volatile("cp.async.commit_group;\n" ::);
        }

        const int8_t* currentInputTile =
            inputTiles + currentBuffer * kInputTileBytes;
        const int8_t* currentWeightTile =
            weightTiles + currentBuffer * kWeightTileBytes;
        uint32_t inputFragments[2][4];
        uint32_t weightFragments[2][2];
        loadFragments(
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
             fragment < kKStage / 32;
             ++fragment) {
            const int32_t currentFragment = fragment & 1;
            const int32_t nextFragment = currentFragment ^ 1;
            if (fragment + 1 < kKStage / 32) {
                loadFragments(
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
                accumulator,
                inputFragments[currentFragment],
                weightFragments[currentFragment]
            );
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    const int32_t upperToken = ctaTokenBase + warpTokenGroup * 16 +
        laneGroup;
    const int32_t lowerToken = upperToken + 8;
    // Epilogue 直接应用逐通道反量化尺度和 FP16 bias，并以 half2 写回 BNC，
    // 不落地 INT32 accumulator 或单独启动反量化 kernel。
    const int32_t channel = warpChannelBase + threadInGroup * 2;
    const float2 scales = *reinterpret_cast<const float2*>(
        dequantScale + channel
    );
    const half2 biases = *reinterpret_cast<const half2*>(bias + channel);
    const float biasLowerChannel = __low2float(biases);
    const float biasUpperChannel = __high2float(biases);
    if (upperToken < kOutputTokens) {
        const half2 value = __halves2half2(
            __float2half_rn(
                static_cast<float>(accumulator[0]) * scales.x +
                    biasLowerChannel
            ),
            __float2half_rn(
                static_cast<float>(accumulator[1]) * scales.y +
                    biasUpperChannel
            )
        );
        *reinterpret_cast<half2*>(
            output + upperToken * kChannels + channel
        ) = value;
    }
    if (lowerToken < kOutputTokens) {
        const half2 value = __halves2half2(
            __float2half_rn(
                static_cast<float>(accumulator[2]) * scales.x +
                    biasLowerChannel
            ),
            __float2half_rn(
                static_cast<float>(accumulator[3]) * scales.y +
                    biasUpperChannel
            )
        );
        *reinterpret_cast<half2*>(
            output + lowerToken * kChannels + channel
        ) = value;
    }
}

// Block1 保留 FP16/FP32 两个输入实例，grid 与 shared memory 均为固定配置。
template <typename InputType>
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
        (kOutputTokens + kCtaTokens - 1) /
        kCtaTokens;
    constexpr int32_t kChannelBlocks =
        kChannels / kCtaChannels;
    static const cudaError_t attributeStatus = cudaFuncSetAttribute(
        fusedInt8SpatialReductionKernel<InputType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedMemoryBytes
    );
    if (attributeStatus != cudaSuccess) {
        return -1;
    }
    fusedInt8SpatialReductionKernel<InputType><<<
        dim3(kTokenBlocks, kChannelBlocks),
        kCtaThreads,
        kSharedMemoryBytes,
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

} // 匿名命名空间

int32_t launchBlock1FusedSpatialReduction(
    bool floatInput,
    float activationScale,
    const void* input,
    const void* weight,
    const void* bias,
    const void* dequantScale,
    void* output,
    cudaStream_t stream
) noexcept {
    if (activationScale <= 0.0F) {
        return -1;
    }
    const float inverseScale = 1.0F / activationScale;
    return floatInput
        ? launchKernel<float>(
              input,
              weight,
              bias,
              dequantScale,
              output,
              inverseScale,
              stream
          )
        : launchKernel<half>(
              input,
              weight,
              bias,
              dequantScale,
              output,
              inverseScale,
              stream
          );
}

} // 命名空间 egcinet::plugins
