#include "visualize/Visualizer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

__global__ void visualizeKernel(
    unsigned char* image,
    size_t imageStep,
    int imageWidth,
    int imageHeight,
    const std::uint8_t* classMask,
    VisualizerConfig config
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageWidth || y >= imageHeight) {
        return;
    }

    const std::uint8_t classId = classMask[
        static_cast<size_t>(y) * static_cast<size_t>(imageWidth) +
        static_cast<size_t>(x)];
    if (classId == 0 || classId >= egcinet::segmentation::kClassCount) {
        return;
    }

    const BgrColor color = config.classColors[classId];
    const float keep = 1.0f - config.alpha;
    unsigned char* pixel = image + static_cast<size_t>(y) * imageStep + x * 3;
    pixel[0] = static_cast<unsigned char>(
        fminf(fmaxf(pixel[0] * keep + color.b * config.alpha, 0.0f), 255.0f));
    pixel[1] = static_cast<unsigned char>(
        fminf(fmaxf(pixel[1] * keep + color.g * config.alpha, 0.0f), 255.0f));
    pixel[2] = static_cast<unsigned char>(
        fminf(fmaxf(pixel[2] * keep + color.r * config.alpha, 0.0f), 255.0f));
}

} // namespace

void launchVisualizeKernel(
    unsigned char* image,
    size_t imageStep,
    int imageWidth,
    int imageHeight,
    const std::uint8_t* classMask,
    VisualizerConfig config,
    cudaStream_t stream
) {
    const dim3 block(16, 16);
    const dim3 grid(
        (imageWidth + block.x - 1) / block.x,
        (imageHeight + block.y - 1) / block.y
    );

    visualizeKernel<<<grid, block, 0, stream>>>(
        image,
        imageStep,
        imageWidth,
        imageHeight,
        classMask,
        config
    );
}
