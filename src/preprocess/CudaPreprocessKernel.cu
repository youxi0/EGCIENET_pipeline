#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

// 判断这个点是不是 padding 区域
// 如果是 padding，填 114 / 255
// 如果不是 padding，反算到原图 srcX / srcY
// 读取 OpenCV BGR 像素
// BGR -> RGB
// uint8 -> float
// HWC -> CHW
// 写入 TensorRT 输入 buffer


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

    int x1 = static_cast<int>(floorf(srcX));
    int y1 = static_cast<int>(floorf(srcY));
    int x2 = min(x1 + 1, srcW - 1);
    int y2 = min(y1 + 1, srcH - 1);

    float dx = srcX - x1;
    float dy = srcY - y1;

    float v11 = static_cast<float>(src[y1 * srcStep + x1 * 3 + channel]);
    float v12 = static_cast<float>(src[y1 * srcStep + x2 * 3 + channel]);
    float v21 = static_cast<float>(src[y2 * srcStep + x1 * 3 + channel]);
    float v22 = static_cast<float>(src[y2 * srcStep + x2 * 3 + channel]);

    float top = v11 * (1.0f - dx) + v12 * dx;
    float bottom = v21 * (1.0f - dx) + v22 * dx;

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
    float scale,
    int padX,
    int padY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) {
        return;
    }

    int dstIndex = y * dstW + x;
    int area = dstW * dstH;

    float r = 114.0f / 255.0f;
    float g = 114.0f / 255.0f;
    float b = 114.0f / 255.0f;

    int resizedW = static_cast<int>(roundf(srcW * scale));
    int resizedH = static_cast<int>(roundf(srcH * scale));

    bool inValidRegion =
        x >= padX &&
        x < padX + resizedW &&
        y >= padY &&
        y < padY + resizedH;

    if (inValidRegion) {
        //每个像素占据1，所以取中心就是srcx + 0.5 = （x + 0.5）/ scale
        float srcX = (static_cast<float>(x - padX) + 0.5f) / scale - 0.5f;
        float srcY = (static_cast<float>(y - padY) + 0.5f) / scale - 0.5f;

        b = bilinearSample(src, srcW, srcH, srcStep, srcX, srcY, 0) / 255.0f;
        g = bilinearSample(src, srcW, srcH, srcStep, srcX, srcY, 1) / 255.0f;
        r = bilinearSample(src, srcW, srcH, srcStep, srcX, srcY, 2) / 255.0f;
    }
    // 修改点: FP16 engine的输入binding可能是half，这里按真实元素大小写入，避免显存布局错误。
    if (dstElementSize == sizeof(__half)) {
        __half* halfDst = static_cast<__half*>(dst);
        halfDst[0 * area + dstIndex] = __float2half_rn(r);
        halfDst[1 * area + dstIndex] = __float2half_rn(g);
        halfDst[2 * area + dstIndex] = __float2half_rn(b);
    } else {
        float* floatDst = static_cast<float*>(dst);
        floatDst[0 * area + dstIndex] = r;
        floatDst[1 * area + dstIndex] = g;
        floatDst[2 * area + dstIndex] = b;
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
    float scale,
    int padX,
    int padY,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
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
        scale,
        padX,
        padY
    );
}
