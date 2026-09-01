#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace egcinet::plugins {
namespace {

constexpr int32_t kOutputWidth = 11;
constexpr int32_t kOutputTokens = kOutputWidth * kOutputWidth;
constexpr int32_t kInputWidth = 44;
constexpr int32_t kChannels = 128;
constexpr int32_t kKernelWidth = 4;
constexpr int32_t kStride = 4;
constexpr int32_t kCtaTokens = 32;
constexpr int32_t kCtaChannels = 64;
constexpr int32_t kKStage = 128;
constexpr int32_t kChannelsPerWarp = 8;
constexpr int32_t kPackedK = 2048;
constexpr int32_t kChannelWarps = 8;
constexpr int32_t kCtaThreads = 512;
constexpr int32_t kInputTileBytes = kCtaTokens * kKStage;
constexpr int32_t kWeightTileBytes = kCtaChannels * kKStage;
constexpr int32_t kSharedMemoryBytes =
    2 * (kInputTileBytes + kWeightTileBytes);

// Block2 复用 Q 投影产生的 INT8 激活；每个 CTA 计算 32 个 token 和
// 64 个输出通道，K 维按 128 个元素分段装入双缓冲 shared memory。

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

// 输入窗口使用 L1+L2 缓存，并支持越界 token 的零字节 cp.async；权重只走
// L2，避免一次性权重流量挤占输入窗口需要复用的 L1 容量。
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

// Block2 从 Q 分支复用 INT8 激活，可以直接异步搬入输入 tile。
__device__ __forceinline__ void copyInputStage(
    const int8_t* input,
    int8_t* inputTile,
    int32_t kBase,
    int32_t ctaTokenBase
) {
    constexpr int32_t kVectorsPerToken = kKStage / 16;
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
        const int32_t logicalColumn = vector * 16;
        const void* source = input;
        int32_t sourceBytes = 0;
        if (outputToken < kOutputTokens) {
            source = input + inputElementOffset(
                outputToken, kBase + logicalColumn
            );
            sourceBytes = 16;
        }
        copyGlobalToShared16Cached(
            inputTile + sharedOffset(tileToken, logicalColumn),
            source,
            sourceBytes
        );
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

__device__ __forceinline__ void copyStage(
    const int8_t* input,
    const int8_t* weight,
    int8_t* inputTile,
    int8_t* weightTile,
    int32_t kBase,
    int32_t ctaTokenBase,
    int32_t channelBlock
) {
    copyInputStage(input, inputTile, kBase, ctaTokenBase);
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

__global__ void fusedInt8SpatialReductionKernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ bias,
    const float* __restrict__ dequantScale,
    half* __restrict__ output
) {
    // Block2 的 24 KiB 双缓冲 tile 直接使用静态 shared memory。
    __shared__ __align__(32) int8_t sharedMemory[kSharedMemoryBytes];
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
        channelBlock
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
                channelBlock
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

// 启动固定形状的 Block2 kernel，不申请 dynamic shared memory。
int32_t launchKernel(
    const void* input,
    const void* weight,
    const void* bias,
    const void* dequantScale,
    void* output,
    cudaStream_t stream
) noexcept {
    constexpr int32_t kTokenBlocks =
        (kOutputTokens + kCtaTokens - 1) /
        kCtaTokens;
    constexpr int32_t kChannelBlocks =
        kChannels / kCtaChannels;
    fusedInt8SpatialReductionKernel<<<
        dim3(kTokenBlocks, kChannelBlocks),
        kCtaThreads,
        0,
        stream
    >>>(
        static_cast<const int8_t*>(input),
        static_cast<const int8_t*>(weight),
        static_cast<const half*>(bias),
        static_cast<const float*>(dequantScale),
        static_cast<half*>(output)
    );
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // 匿名命名空间

int32_t launchBlock2FusedSpatialReduction(
    const void* input,
    const void* weight,
    const void* bias,
    const void* dequantScale,
    void* output,
    cudaStream_t stream
) noexcept {
    return launchKernel(
        input, weight, bias, dequantScale, output, stream
    );
}

} // 命名空间 egcinet::plugins
