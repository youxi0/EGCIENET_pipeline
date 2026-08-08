#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

__device__ float readScore(const void* modelMask, size_t elementSize, size_t index) {
    if (elementSize == sizeof(__half)) {
        return __half2float(static_cast<const __half*>(modelMask)[index]);
    }
    return static_cast<const float*>(modelMask)[index];
}

// NCHW 每个类别是一个连续平面；对单个类别分数图做双线性采样。
__device__ float bilinearClassScore(
    const void* modelMask,
    size_t elementSize,
    int classId,
    int modelWidth,
    int modelHeight,
    float sourceX,
    float sourceY
) {
    sourceX = fminf(fmaxf(sourceX, 0.0f), static_cast<float>(modelWidth - 1));
    sourceY = fminf(fmaxf(sourceY, 0.0f), static_cast<float>(modelHeight - 1));

    const int x0 = static_cast<int>(floorf(sourceX));
    const int y0 = static_cast<int>(floorf(sourceY));
    const int x1 = min(x0 + 1, modelWidth - 1);
    const int y1 = min(y0 + 1, modelHeight - 1);
    const float dx = sourceX - static_cast<float>(x0);
    const float dy = sourceY - static_cast<float>(y0);
    const size_t classOffset = static_cast<size_t>(classId) *
        static_cast<size_t>(modelWidth) * static_cast<size_t>(modelHeight);
    const size_t row0 = static_cast<size_t>(y0) * static_cast<size_t>(modelWidth);
    const size_t row1 = static_cast<size_t>(y1) * static_cast<size_t>(modelWidth);

    const float topLeft = readScore(
        modelMask, elementSize, classOffset + row0 + static_cast<size_t>(x0));
    const float topRight = readScore(
        modelMask, elementSize, classOffset + row0 + static_cast<size_t>(x1));
    const float bottomLeft = readScore(
        modelMask, elementSize, classOffset + row1 + static_cast<size_t>(x0));
    const float bottomRight = readScore(
        modelMask, elementSize, classOffset + row1 + static_cast<size_t>(x1));

    const float top = topLeft * (1.0f - dx) + topRight * dx;
    const float bottom = bottomLeft * (1.0f - dx) + bottomRight * dx;
    return top * (1.0f - dy) + bottom * dy;
}

__global__ void segPostprocessKernel(
    const void* modelMask,
    size_t modelElementSize,
    int modelChannels,
    int modelWidth,
    int modelHeight,
    std::uint8_t* classMask,
    int outputWidth,
    int outputHeight
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outputWidth || y >= outputHeight) {
        return;
    }

    const float scaleX = static_cast<float>(modelWidth) / static_cast<float>(outputWidth);
    const float scaleY = static_cast<float>(modelHeight) / static_cast<float>(outputHeight);
    const float sourceX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
    const float sourceY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

    int bestClass = 0;
    float bestScore = -CUDART_INF_F;
    for (int classId = 0; classId < modelChannels; ++classId) {
        const float score = bilinearClassScore(
            modelMask,
            modelElementSize,
            classId,
            modelWidth,
            modelHeight,
            sourceX,
            sourceY);
        // NaN 比较恒为 false；若所有通道均无效，像素安全回落到 background。
        if (score > bestScore) {
            bestScore = score;
            bestClass = classId;
        }
    }
    classMask[static_cast<size_t>(y) * static_cast<size_t>(outputWidth) +
        static_cast<size_t>(x)] = static_cast<std::uint8_t>(bestClass);
}

} // namespace

void launchSegPostprocessKernel(
    const void* modelMask,
    size_t modelElementSize,
    int modelChannels,
    int modelWidth,
    int modelHeight,
    std::uint8_t* classMask,
    int outputWidth,
    int outputHeight,
    cudaStream_t stream
) {
    const dim3 block(16, 16);
    const dim3 grid(
        (outputWidth + block.x - 1) / block.x,
        (outputHeight + block.y - 1) / block.y
    );

    segPostprocessKernel<<<grid, block, 0, stream>>>(
        modelMask,
        modelElementSize,
        modelChannels,
        modelWidth,
        modelHeight,
        classMask,
        outputWidth,
        outputHeight
    );
}
