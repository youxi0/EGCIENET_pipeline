#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

__device__ float bilinearSample(
    const unsigned char* src,
    int srcW,
    int srcH,
    int srcStep,
    float srcX,
    float srcY,
    int channel
) {
    srcX = fminf(fmaxf(srcX, 0.0f), static_cast<float>(srcW - 1));
    srcY = fminf(fmaxf(srcY, 0.0f), static_cast<float>(srcH - 1));

    const int x1 = static_cast<int>(floorf(srcX));
    const int y1 = static_cast<int>(floorf(srcY));
    const int x2 = min(x1 + 1, srcW - 1);
    const int y2 = min(y1 + 1, srcH - 1);

    const float dx = srcX - static_cast<float>(x1);
    const float dy = srcY - static_cast<float>(y1);

    const float v11 = static_cast<float>(src[y1 * srcStep + x1 * 3 + channel]);
    const float v12 = static_cast<float>(src[y1 * srcStep + x2 * 3 + channel]);
    const float v21 = static_cast<float>(src[y2 * srcStep + x1 * 3 + channel]);
    const float v22 = static_cast<float>(src[y2 * srcStep + x2 * 3 + channel]);

    const float top = v11 * (1.0f - dx) + v12 * dx;
    const float bottom = v21 * (1.0f - dx) + v22 * dx;

    return top * (1.0f - dy) + bottom * dy;
}

__global__ void preprocessKernel(
    const unsigned char* src,
    int srcW,
    int srcH,
    int srcStep,
    void* dst,
    int dstW,
    int dstH,
    size_t dstElementSize,
    float meanB,
    float meanG,
    float meanR,
    float stdB,
    float stdG,
    float stdR
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) {
        return;
    }

    const float scaleX = static_cast<float>(srcW) / static_cast<float>(dstW);
    const float scaleY = static_cast<float>(srcH) / static_cast<float>(dstH);
    const float srcX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
    const float srcY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

    const float b = (bilinearSample(src, srcW, srcH, srcStep, srcX, srcY, 0) - meanB) / stdB;
    const float g = (bilinearSample(src, srcW, srcH, srcStep, srcX, srcY, 1) - meanG) / stdG;
    const float r = (bilinearSample(src, srcW, srcH, srcStep, srcX, srcY, 2) - meanR) / stdR;

    const int dstIndex = y * dstW + x;
    const int area = dstW * dstH;

    if (dstElementSize == sizeof(__half)) {
        __half* halfDst = static_cast<__half*>(dst);
        halfDst[0 * area + dstIndex] = __float2half_rn(b);
        halfDst[1 * area + dstIndex] = __float2half_rn(g);
        halfDst[2 * area + dstIndex] = __float2half_rn(r);
    } else {
        float* floatDst = static_cast<float*>(dst);
        floatDst[0 * area + dstIndex] = b;
        floatDst[1 * area + dstIndex] = g;
        floatDst[2 * area + dstIndex] = r;
    }
}

void launchPreprocessKernel(
    const unsigned char* src,
    int srcW,
    int srcH,
    int srcStep,
    void* dst,
    int dstW,
    int dstH,
    size_t dstElementSize,
    float meanB,
    float meanG,
    float meanR,
    float stdB,
    float stdG,
    float stdR,
    cudaStream_t stream
) {
    const dim3 block(16, 16);
    const dim3 grid(
        (dstW + block.x - 1) / block.x,
        (dstH + block.y - 1) / block.y
    );

    preprocessKernel<<<grid, block, 0, stream>>>(
        src,
        srcW,
        srcH,
        srcStep,
        dst,
        dstW,
        dstH,
        dstElementSize,
        meanB,
        meanG,
        meanR,
        stdB,
        stdG,
        stdR
    );
}
