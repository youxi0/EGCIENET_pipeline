#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

__device__ float readProbability(
    const void* modelMask,
    size_t elementSize,
    int index
) {
    if (elementSize == sizeof(__half)) {
        return __half2float(static_cast<const __half*>(modelMask)[index]);
    }
    return static_cast<const float*>(modelMask)[index];
}

__device__ float bilinearProbability(
    const void* modelMask,
    size_t elementSize,
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

    const float topLeft = readProbability(
        modelMask, elementSize, y0 * modelWidth + x0);
    const float topRight = readProbability(
        modelMask, elementSize, y0 * modelWidth + x1);
    const float bottomLeft = readProbability(
        modelMask, elementSize, y1 * modelWidth + x0);
    const float bottomRight = readProbability(
        modelMask, elementSize, y1 * modelWidth + x1);

    const float top = topLeft * (1 - dx) + topRight * dx;
    const float bottom = bottomLeft * (1 - dx) + bottomRight * dx;
    return top * (1 - dy) + bottom * dy;
}

__global__ void segPostprocessKernel(
    const void* modelMask,
    size_t modelElementSize,
    int modelWidth,
    int modelHeight,
    float* probabilityMask,
    std::uint8_t* binaryMask,
    int outputWidth,
    int outputHeight,
    float threshold
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
    const float probability = fminf(fmaxf(
        bilinearProbability(
            modelMask,
            modelElementSize,
            modelWidth,
            modelHeight,
            sourceX,
            sourceY
        ),
        0.0f),
        1.0f);

    const int outputIndex = y * outputWidth + x;
    probabilityMask[outputIndex] = probability;
    binaryMask[outputIndex] = probability >= threshold ? 255 : 0;
}

} // namespace

void launchSegPostprocessKernel(
    const void* modelMask,
    size_t modelElementSize,
    int modelWidth,
    int modelHeight,
    float* probabilityMask,
    std::uint8_t* binaryMask,
    int outputWidth,
    int outputHeight,
    float threshold,
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
        modelWidth,
        modelHeight,
        probabilityMask,
        binaryMask,
        outputWidth,
        outputHeight,
        threshold
    );
}
