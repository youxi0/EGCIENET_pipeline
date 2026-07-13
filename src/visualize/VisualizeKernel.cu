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
    const float* probabilityMask,
    const std::uint8_t* binaryMask,
    std::uint8_t colorB,
    std::uint8_t colorG,
    std::uint8_t colorR,
    float alpha
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageWidth || y >= imageHeight) {
        return;
    }

    const int maskIndex = y * imageWidth + x;
    if (binaryMask[maskIndex] == 0) {
        return;
    }

    // mask 内至少使用一半透明度，概率越高覆盖越明显，保证区域清晰可见。
    const float probability = fminf(fmaxf(probabilityMask[maskIndex], 0.0f), 1.0f);
    const float blend = alpha * (0.5f + 0.5f * probability);
    const float keep = 1.0f - blend;
    unsigned char* pixel = image + static_cast<size_t>(y) * imageStep + x * 3;

    pixel[0] = static_cast<unsigned char>(
        fminf(fmaxf(pixel[0] * keep + colorB * blend, 0.0f), 255.0f));
    pixel[1] = static_cast<unsigned char>(
        fminf(fmaxf(pixel[1] * keep + colorG * blend, 0.0f), 255.0f));
    pixel[2] = static_cast<unsigned char>(
        fminf(fmaxf(pixel[2] * keep + colorR * blend, 0.0f), 255.0f));
}

} // namespace

void launchVisualizeKernel(
    unsigned char* image,
    size_t imageStep,
    int imageWidth,
    int imageHeight,
    const float* probabilityMask,
    const std::uint8_t* binaryMask,
    std::uint8_t colorB,
    std::uint8_t colorG,
    std::uint8_t colorR,
    float alpha,
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
        probabilityMask,
        binaryMask,
        colorB,
        colorG,
        colorR,
        alpha
    );
}
